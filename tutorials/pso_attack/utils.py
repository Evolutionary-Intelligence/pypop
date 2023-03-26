# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
import torch
import random
import pickle
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F
from shapely.geometry import Polygon
from itertools import permutations


def load_gtsrb(database_path):

    train_path = os.path.join(database_path, 'train')
    test_path = os.path.join(database_path, 'test')
    train_img_paths = os.listdir(train_path)
    test_img_paths = os.listdir(test_path)

    train_data = np.zeros((len(train_img_paths), 32, 32, 3), dtype=np.uint8)
    train_labels = np.zeros((len(train_img_paths),), dtype=np.int)
    test_data = np.zeros((len(test_img_paths), 32, 32, 3), dtype=np.uint8)
    test_labels = np.zeros((len(test_img_paths),), dtype=np.int)

    for i, path in enumerate(train_img_paths):
        img_path = os.path.join(train_path, path)
        c, _ = map(int, path[:-4].split('_'))
        train_data[i] = cv2.imread(img_path)
        train_labels[i] = c

    for i, path in enumerate(test_img_paths):
        img_path = os.path.join(test_path, path)
        c, _ = map(int, path[:-4].split('_'))
        test_data[i] = cv2.imread(img_path)
        test_labels[i] = c

    return train_data, train_labels, test_data, test_labels


def load_mask():

    position_list, mask_list = [], []
    for mask_file in sorted(os.listdir("./mask")):
        with open(f"./mask/{mask_file}", "rb") as mf:
            mask_list.append(pickle.load(mf))
            position_list.append(np.where(mask_list[-1] == 255))

    return position_list, mask_list


def pre_process_image(image):
    image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
    image[:, :, 1] = cv2.equalizeHist(image[:, :, 1])
    image[:, :, 2] = cv2.equalizeHist(image[:, :, 2])
    image = image / 255. - .5
    return image.astype(np.float32)


def judge_inside(vertices, p):

    a, b, c = vertices
    cross1 = np.cross(b - a, p - a) <= 0
    cross2 = np.cross(c - b, p - b) <= 0
    cross3 = np.cross(a - c, p - c) <= 0
    return ~(cross1 ^ cross2) & ~(cross2 ^ cross3)


def contains(vertices, p):

    vertices = np.append(vertices, vertices[0].reshape(1, 2), 0)
    res = np.zeros(p.shape[0], dtype=np.bool)
    x = p[0:, 0]
    for i in range(len(vertices) - 1):
        (x1, _), (x2, _) = vertices[i], vertices[i + 1]
        vector1 = vertices[i + 1] - vertices[i]
        vector2 = p - vertices[i]
        cross = np.cross(vector1, vector2)
        res ^= ((x1 <= x) & (x <= x2) & (cross <= 0)) | ((x2 <= x) & (x <= x1) & (cross >= 0))

    return res


def polygon_correction(vertices):

    vertices = np.reshape(vertices, (-1, 2))
    if Polygon(vertices).is_valid:
        return np.ravel(vertices)
    for candidate_position in permutations(vertices):
        if Polygon(candidate_position).is_valid:
            return np.ravel(candidate_position)


def brightness(attack_image, sign_mask):

    attack_image = cv2.cvtColor(attack_image, cv2.COLOR_BGR2LAB)
    return np.average(attack_image[sign_mask == 255][:, 0])


def shadow_edge_blur(image, shadow_area, coefficient):

    blurred_img = cv2.GaussianBlur(image, (coefficient, coefficient), 0)
    gray_img = cv2.cvtColor(shadow_area, cv2.COLOR_BGR2GRAY)
    threshold = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _mask = np.zeros_like(image, dtype=np.uint8)
    cv2.drawContours(_mask, contours, -1, (255, 255, 255), coefficient)
    return np.where(_mask == np.array([255, 255, 255]), blurred_img, image)


def judge_mask_type(database, label):

    if database == "GTSRB":
        # circle mask
        if label in [0,  1,  2,  3,
                     4,  5,  6,  7,
                     8,  9,  10, 15,
                     16, 17, 32, 33,
                     34, 35, 36, 37,
                     38, 39, 40, 41,
                     42]:
            return 0
        # triangle mask
        if label in [11, 18, 19, 20,
                     21, 22, 23, 24,
                     25, 26, 27, 28,
                     29, 30, 31]:
            return 6
        # inverse triangle mask
        if label in [13]:
            return 1
        # octagon mask
        if label in [14]:
            return 2
        # rhombus mask
        if label in [12]:
            return 5

    elif database == "LISA":
        # rhombus mask
        if label in [0, 2, 3, 4, 7, 13, 14]:
            return 5
        # rectangle mask
        if label in [1, 6, 8, 9, 10, 11]:
            return 4
        # inverse triangle mask
        if label in [15]:
            return 1
        # octagon mask
        if label in [12]:
            return 2
        # pentagon mask
        if label in [5]:
            return 3


def draw_shadow(position, image, pos_list, coefficient):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    position = np.reshape(position, (-1, 2))
    shadow_area = np.zeros_like(image, dtype=np.uint8)

    if position.shape[0] == 3:
        judge_array = judge_inside(position, np.transpose(pos_list, [1, 0]))
    else:
        judge_array = contains(position, np.transpose(pos_list, [1, 0]))
    inside_list = np.where(judge_array == 1)
    x_list, y_list = pos_list[0][inside_list], pos_list[1][inside_list]
    shadow_area[x_list, y_list] = 255
    image[x_list, y_list, 0] = np.round(image[x_list, y_list, 0] * coefficient).astype(np.uint8)

    image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
    return image, shadow_area


def motion_blur(image, size=12, angle=45):

    if size == 0:
        return image

    k = np.zeros((size, size), dtype=np.float32)
    k[(size - 1) // 2, :] = np.ones(size, dtype=np.float32)
    k = cv2.warpAffine(k, cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), angle, 1.0), (size, size))
    k = k * (1.0 / np.sum(k))

    return cv2.filter2D(image, -1, k)


def random_param_generator(num, w, h):

    motion_degree = [random.randint(1, 10) for _ in range(num)] + [0]
    motion_angle = [random.uniform(0, 360) for _ in range(num)] + [0]
    size_mul = [random.uniform(0.1, 1) for _ in range(num)] + [1]
    brightness_mul = [random.uniform(0.6, 1.4) for _ in range(num)] + [1]
    shadow_mul = [random.uniform(0.4, 0.8) for _ in range(num)] + [0.6]
    shadow_move = [[random.uniform(-15, 15), random.uniform(-15, 15)] for _ in range(num)] + [[0, 0]]
    perspective_mat = [np.float32(
        [[w * random.uniform(0, 0.1), h * random.uniform(0, 0.1)],
         [w * (1 - random.uniform(0, 0.1)), h * random.uniform(0, 0.1)],
         [w * (1 - random.uniform(0, 0.1)), h * (1 - random.uniform(0, 0.1))],
         [w * random.uniform(0, 0.1), h * (1 - random.uniform(0, 0.1))]]) for _ in range(num)] + \
        [np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])]

    return motion_degree, motion_angle, size_mul, brightness_mul, shadow_mul, shadow_move, perspective_mat


def image_transformation(image, position, pos_list, motion_degree, motion_angle, size_mul,
                         brightness_mul, shadow_mul, shadow_move, perspective_mat, pre_process):
    transform_num = len(motion_degree)
    h, w, _ = image.shape
    res_images = []

    for i in range(transform_num):
        # shadow
        shadow_position = position.copy()
        shadow_position[0::2] += shadow_move[i][0]
        shadow_position[1::2] += shadow_move[i][1]
        adv_img, shadow_area = draw_shadow(shadow_position, image, pos_list, shadow_mul[i])
        adv_img = shadow_edge_blur(adv_img, shadow_area, 5)

        # random resize
        adv_img = cv2.resize(adv_img, (int(w * size_mul[i]), int(h * size_mul[i])))
        adv_img = cv2.resize(adv_img, (w, h))

        # random brightness adjustment
        if brightness_mul[i] != 1:
            adv_img = cv2.cvtColor(adv_img, cv2.COLOR_BGR2LAB).astype(np.int32)
            adv_img[:, :, 0] = np.clip(adv_img[:, :, 0] * brightness_mul[i], 0, 255)
            adv_img = cv2.cvtColor(adv_img.astype(np.uint8), cv2.COLOR_LAB2BGR)

        # random perspective transformation
        before = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
        after = perspective_mat[i]
        if (before - after).sum() != 0:
            matrix = cv2.getPerspectiveTransform(before, after)
            adv_img = cv2.warpPerspective(adv_img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
            adv_img = adv_img[int(min(after[0:2, 1])): int(max(after[2:4, 1])),
                              int(min(after[0::3, 0])): int(max(after[1:3, 0]))]

        # random motion blur
        adv_img = motion_blur(adv_img, motion_degree[i], motion_angle[i])
        res_images.append(adv_img)

    for i in range(transform_num):
        res_images[i] = cv2.resize(res_images[i], (32, 32))
        res_images[i] = pre_process(res_images[i])

    return torch.stack(res_images, dim=0)


class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.1):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes, smoothing=0.1):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size()[0], n_classes),
                                  device=targets.device) \
                .fill_(smoothing / (n_classes - 1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1. - smoothing)
        return targets

    def forward(self, inputs, targets):
        targets = SmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
                                                         self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss
