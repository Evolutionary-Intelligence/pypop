"""This part of code is referenced from:
    https://github.com/hncszyq/ShadowAttack/blob/master/gtsrb.py
"""
# -*- coding: utf-8 -*-

import gc
import pickle
import time
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from utils import SmoothCrossEntropyLoss
from utils import draw_shadow
from utils import shadow_edge_blur
from utils import judge_mask_type
from utils import load_mask

with open('params.json', 'r') as config:
    params = json.load(config)
    class_n = params['GTSRB']['class_n']
    device = params['device']
    position_list, _ = load_mask()

loss_fun = SmoothCrossEntropyLoss(smoothing=0.1)


class TrafficSignDataset(torch.utils.data.Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        _x = transforms.ToTensor()(self.x[item])
        _y = self.y[item]
        return _x, _y


class GtsrbCNN(nn.Module):

    def __init__(self, n_class):

        super().__init__()
        self.color_map = nn.Conv2d(3, 3, (1, 1), stride=(1, 1), padding=0)
        self.module1 = nn.Sequential(
            nn.Conv2d(3, 32, (5, 5), stride=(1, 1), padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, (5, 5), stride=(1, 1), padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )
        self.module2 = nn.Sequential(
            nn.Conv2d(32, 64, (5, 5), stride=(1, 1), padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, (5, 5), stride=(1, 1), padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )
        self.module3 = nn.Sequential(
            nn.Conv2d(64, 128, (5, 5), stride=(1, 1), padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, (5, 5), stride=(1, 1), padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(14336, 1024, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )
        self.fc3 = nn.Linear(1024, n_class, bias=True)

    def forward(self, x):

        x = self.color_map(x)
        branch1 = self.module1(x)
        branch2 = self.module2(branch1)
        branch3 = self.module3(branch2)

        branch1 = branch1.reshape(branch1.shape[0], -1)
        branch2 = branch2.reshape(branch2.shape[0], -1)
        branch3 = branch3.reshape(branch3.shape[0], -1)
        concat = torch.cat([branch1, branch2, branch3], 1)

        out = self.fc1(concat)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


def pre_process_image(image):

    image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
    image[:, :, 1] = cv2.equalizeHist(image[:, :, 1])
    image[:, :, 2] = cv2.equalizeHist(image[:, :, 2])
    image = image / 255. - .5
    return image


def transform_image(image, ang_range, shear_range, trans_range, preprocess):

    # Rotation
    ang_rot = np.random.uniform(ang_range) - ang_range / 2
    rows, cols, ch = image.shape
    rot_m = cv2.getRotationMatrix2D((cols / 2, rows / 2), ang_rot, 1)

    # Translation
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    tr_y = trans_range * np.random.uniform() - trans_range / 2
    trans_m = np.float32([[1, 0, tr_x], [0, 1, tr_y]])

    # Shear
    pts1 = np.float32([[5, 5], [20, 5], [5, 20]])

    pt1 = 5 + shear_range * np.random.uniform() - shear_range / 2
    pt2 = 20 + shear_range * np.random.uniform() - shear_range / 2

    pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])

    shear_m = cv2.getAffineTransform(pts1, pts2)

    image = cv2.warpAffine(image, rot_m, (cols, rows))
    image = cv2.warpAffine(image, trans_m, (cols, rows))
    image = cv2.warpAffine(image, shear_m, (cols, rows))

    image = pre_process_image(image) if preprocess else image

    return image


def gen_extra_data(x_train, y_train, n_each, ang_range,
                   shear_range, trans_range, randomize_var, preprocess=True):

    x_arr, y_arr = [], []
    n_train = len(x_train)
    for i in range(n_train):
        for i_n in range(n_each):
            img_trf = transform_image(x_train[i],
                                      ang_range, shear_range, trans_range,
                                      preprocess)
            x_arr.append(img_trf)
            y_arr.append(y_train[i])

    x_arr = np.array(x_arr, dtype=np.float32())
    y_arr = np.array(y_arr, dtype=np.float32())

    if randomize_var == 1:
        len_arr = np.arange(len(y_arr))
        np.random.shuffle(len_arr)
        x_arr[len_arr] = x_arr
        y_arr[len_arr] = y_arr

    return x_arr, y_arr


def weights_init(m):

    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.05)
        nn.init.constant_(m.bias, 0.05)


def model_epoch(training_model, data_loader, train=False, optimizer=None):

    loss = acc = 0.0

    for data_batch in data_loader:
        train_predict = training_model(data_batch[0].to(device))
        batch_loss = loss_fun(train_predict, data_batch[1].to(device))
        if train:
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        acc += (torch.argmax(train_predict.cpu(), dim=1) == data_batch[1]).sum()
        loss += batch_loss.item() * len(data_batch[1])

    return acc, loss


def training(training_model, train_data, train_labels, test_data, test_labels, adv_train=False):

    num_epoch, batch_size = 25, 64
    optimizer = torch.optim.Adam(
        training_model.parameters(), lr=0.001, weight_decay=1e-5)

    for epoch in range(num_epoch):

        extra_train, extra_labels = adversarial_augmentation(
            train_data, train_labels) if adv_train else (train_data, train_labels)

        train_set = TrafficSignDataset(extra_train, extra_labels)
        test_set = TrafficSignDataset(test_data, test_labels)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        epoch_start_time = time.time()

        training_model.train()
        train_acc, train_loss = model_epoch(
            training_model, train_loader, train=True, optimizer=optimizer)

        training_model.eval()
        with torch.no_grad():
            test_acc, test_loss = model_epoch(training_model, test_loader)

        print(f'[{epoch+1}/{num_epoch}] {round(time.time() - epoch_start_time, 2)}', end=' ')
        print(f'Train Acc: {round(float(train_acc / train_set.__len__()), 4)}', end=' ')
        print(f'Loss: {round(float(train_loss / train_set.__len__()), 4)}', end=' | ')
        print(f'Test Acc: {round(float(test_acc / test_set.__len__()), 4)}', end=' ')
        print(f'Loss: {round(float(test_loss / test_set.__len__()), 4)}')

        del extra_train, extra_labels, train_set, train_loader
        gc.collect()

    torch.save(training_model.state_dict(),
               f'./model/{"adv_" if adv_train else ""}model_gtsrb.pth')


def adversarial_augmentation(ori_data_train, ori_labels_train):

    num_data = ori_data_train.shape[0]
    data_train = np.zeros((num_data * 2, 32, 32, 3), np.uint8)
    labels_train = np.zeros(num_data * 2, np.int)
    data_train[0::2] = ori_data_train
    labels_train[0::2] = labels_train[1::2] = ori_labels_train

    for i in range(0, num_data * 2, 2):
        pos_list = position_list[judge_mask_type("GTSRB", labels_train[i])]
        shadow_image, shadow_area = draw_shadow(
            np.random.uniform(-16, 48, 6), data_train[i], pos_list, np.random.uniform(0.2, 0.7))
        data_train[i + 1] = shadow_edge_blur(shadow_image, shadow_area, 3)

    data_train = data_train.astype(np.float32)
    for i in range(num_data * 2):
        data_train[i] = pre_process_image(data_train[i].astype(np.uint8))

    return data_train, labels_train


def train_model(adv_train=False):

    with open('./dataset/GTSRB/train.pkl', 'rb') as f:
        train = pickle.load(f)
        train_data, train_labels = train['data'], train['labels']
    with open('./dataset/GTSRB/test.pkl', 'rb') as f:
        test = pickle.load(f)
        test_data, test_labels = test['data'], test['labels']

    processed_train = np.array([
        pre_process_image(train_data[i]) for i in range(len(train_data))],
        dtype=np.float32) if not adv_train else train_data
    processed_test = np.array([
        pre_process_image(test_data[i]) for i in range(len(test_data))],
        dtype=np.float32)
    augment_data_train, augment_data_labels = gen_extra_data(
        train_data, train_labels, 10, 30, 5, 5, 1, preprocess=not adv_train)

    image_train = np.concatenate([processed_train, augment_data_train], 0)
    label_train = np.concatenate([train_labels, augment_data_labels], 0)
    image_test, label_test = processed_test, test_labels

    training_model = GtsrbCNN(n_class=class_n).to(device).apply(weights_init)
    training(training_model, image_train, label_train, image_test, label_test, adv_train)


def test_model(adv_model=False):

    trained_model = GtsrbCNN(n_class=class_n).to(device)
    trained_model.load_state_dict(
        torch.load(f'./model/{"adv_" if adv_model else ""}model_gtsrb.pth',
                   map_location=torch.device(device)))

    with open('./dataset/GTSRB/test.pkl', 'rb') as f:
        test = pickle.load(f)
        test_data, test_labels = test['data'], test['labels']

    test_data = np.array([
        pre_process_image(test_data[i]) for i in range(len(test_data))],
        dtype=np.float32)

    test_set = TrafficSignDataset(test_data, test_labels)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    trained_model.eval()
    with torch.no_grad():
        test_acc, _ = model_epoch(trained_model, test_loader)

    print(f'Test Acc: {round(float(test_acc / test_set.__len__()), 4)}')


def test_single_image(img_path, label, adv_model=False):

    trained_model = GtsrbCNN(n_class=class_n).to(device)
    trained_model.load_state_dict(
        torch.load(f'./model/{"adv_" if adv_model else ""}model_gtsrb.pth',
                   map_location=torch.device(device)))
    trained_model.eval()

    img = cv2.imread(img_path)
    img = cv2.resize(img, (32, 32))
    img = pre_process_image(img).astype(np.float32)
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0).to(device)

    predict = torch.softmax(trained_model(img)[0], 0)
    index = int(torch.argmax(predict).data)
    confidence = float(predict[index].data)

    print(f'Correct: {index==label}', end=' ')
    print(f'Predict: {index} Confidence: {confidence*100}%')

    return index, index == label


if __name__ == '__main__':

    # model training
    # train_model(adv_train=False)

    # model testing
    # test_model(adv_model=False)

    # test a single image
    test_single_image('./tmp/adv_img.png', 1, adv_model=False)
