"""This part of code is referenced from:
    https://github.com/hncszyq/ShadowAttack/blob/master/gtsrb.py
"""
# -*- coding: utf-8 -*-

import os
import cv2
import json
import torch
import argparse
import numpy as np
import gtsrb
from gtsrb import GtsrbCNN
from utils import shadow_edge_blur
from utils import draw_shadow
from utils import image_transformation
from utils import load_mask
from utils import pre_process_image
from utils import random_param_generator
from torchvision import transforms

from pypop7.optimizers.pso.spso import SPSO as Solver


def objective(position, args):
    r"""
    Fitness function in PSO.

    Args:
        position: The coordinates of the vertices of the polygonal shadow area.

    Returns:
        confidence: The lower the value, the more successful the attack.
        success: Whether the attack is successful.
        predict: The model output.
    """
    if args.get('physical'):
        img = image_transformation(
            args.get('image'), position, args.get('coord'),
            *args.get('rand_param'), args.get('pre_process')).to(device)
    else:
        img, shadow_area = draw_shadow(position, args.get('image'), args.get('coord'), args.get('coefficient'))
        img = shadow_edge_blur(img, shadow_area, 3)
        img = args.get('pre_process')(img).unsqueeze(0).to(device)

    with torch.no_grad():
        predict = torch.softmax(args.get('model')(img), 1)
        predict = torch.mean(predict, dim=0)

    if args.get('targeted'):
        target = args.get("target")
        confidence = float(1 - predict[target])
    else:
        confidence = float(predict[args.get('label')])
    return confidence


with open('params.json', 'rb') as f:
    params = json.load(f)
    class_n_gtsrb = params['GTSRB']['class_n']
    device = params['device']
    position_list, mask_list = load_mask()

parser = argparse.ArgumentParser(description="Adversarial attack by shadow")
parser.add_argument("--shadow_level", type=float, default=0.43,
                    help="shadow coefficient k")
parser.add_argument("--attack_db", type=str, default="GTSRB",
                    help="the target dataset should be specified for a digital attack")
parser.add_argument("--attack_type", type=str, default="digital",
                    help="digital attack or physical attack")
parser.add_argument("--image_path", type=str, default="./xxx",
                    help="a file path to the target image should be specified for a physical attack")
parser.add_argument("--mask_path", type=str, default="./xxx",
                    help="a file path to the mask should be specified for a physical attack")
parser.add_argument("--image_label", type=int, default=0,
                    help="a ground truth should be specified for a physical attack")
parser.add_argument("--polygon", type=int, default=3,
                    help="shadow shape: n-sided polygon")
parser.add_argument("--n_try", type=int, default=5,
                    help="n-random-start strategy: retry n times")
parser.add_argument("--target_model", type=str, default="normal",
                    help="attack normal model or robust model")

args = parser.parse_args()
shadow_level = args.shadow_level
target_model = args.target_model
attack_db = args.attack_db
attack_type = args.attack_type
image_path = args.image_path
mask_path = args.mask_path
image_label = args.image_label
polygon = args.polygon
n_try = args.n_try

assert attack_db in ['GTSRB']
model = GtsrbCNN(n_class=class_n_gtsrb).to(device)
model.load_state_dict(
    torch.load(f'./model/{"adv_" if target_model == "robust" else ""}model_gtsrb.pth',
               map_location=torch.device(device)))
pre_process = transforms.Compose([
    pre_process_image, transforms.ToTensor()])
model.eval()

assert attack_type in ['digital', 'physical']
if attack_type == 'digital':
    particle_size = 10
    iter_num = 100
    x_min, x_max = -16, 48
    max_speed = 1.5
else:
    particle_size = 10
    iter_num = 200
    x_min, x_max = -112, 336
    max_speed = 10.
    n_try = 1


def attack(attack_image, label, coords, targeted_attack=False, physical_attack=False, **parameters):
    r"""
    Physical-world adversarial attack by shadow.

    Args:
        attack_image: The image to be attacked.
        label: The ground-truth label of attack_image.
        coords: The coordinates of the points where mask == 1.
        targeted_attack: Targeted / Non-targeted attack.
        physical_attack: Physical / digital attack.

    Returns:
        adv_img: The generated adversarial image.
        succeed: Whether the attack is successful.
        num_query: Number of queries.
    """
    num_query = 0
    succeed = False
    global_best_solution = float('inf')
    global_best_position = None

    for attempt in range(n_try):

        if succeed:
            break

        print(f"try {attempt + 1}:", end=" ")
        pro = {'fitness_function': objective,
               'ndim_problem': polygon * 2,
               'lower_boundary': np.ones(polygon * 2) * x_min,
               'upper_boundary': np.ones(polygon * 2) * x_max}

        opt = {'seed_rng': 0,
               'max_function_evaluations': iter_num * polygon * 2 + particle_size,
               'n_individuals': particle_size,
               'cognition': 2,
               'society': 2,
               'max_ratio_v': 0.05,
               'verbose': 0,
               'saving_fitness': 0}
        h, w, _ = attack_image.shape
        transform_num = parameters.get("transform_num")
        fit_args = {'image': attack_image,
                    'coord': coords,
                    'model': model,
                    'coefficient': 0.43,
                    'targeted': targeted_attack,
                    'physical': physical_attack,
                    'target': parameters.get("target"),
                    'label': label,
                    'pre_process': pre_process,
                    'rand_param': random_param_generator(transform_num, w, h)}
        solver = Solver(pro, opt)
        res = solver.optimize(args=fit_args)
        best_solution, best_pos, succeed = res['best_so_far_y'], res['best_so_far_x'], res['success']

        if targeted_attack:
            best_solution = 1 - best_solution
        print(f"Best solution: {best_solution} {'succeed' if succeed else 'failed'}")
        if best_solution < global_best_solution:
            global_best_solution = best_solution
            global_best_position = best_pos
        # num_query += query

    adv_image, shadow_area = draw_shadow(
        global_best_position, attack_image, coords, shadow_level)
    adv_image = shadow_edge_blur(adv_image, shadow_area, 3)

    return adv_image, succeed, num_query


def attack_physical():
    global position_list

    mask_image = cv2.resize(
        cv2.imread(mask_path, cv2.IMREAD_UNCHANGED), (224, 224))
    target_image = cv2.resize(
        cv2.imread(image_path), (224, 224))
    pos_list = np.where(mask_image.sum(axis=2) > 0)

    # EOT is included in the first stage
    adv_img, _, _ = attack(target_image, image_label, pos_list,
                           physical_attack=True, transform_num=10)

    cv2.imwrite('./tmp/temp.bmp', adv_img)
    predict, failed = gtsrb.test_single_image(
        './tmp/temp.bmp', image_label, target_model == "robust")
    if failed:
        print('Attack failed! Try to run again.')

    # Predict stabilization
    adv_img, _, _ = attack(target_image, image_label, pos_list, targeted_attack=True,
                           physical_attack=True, target=predict, transform_num=10)

    cv2.imwrite('./tmp/adv_img.png', adv_img)
    predict, failed = gtsrb.test_single_image(
        './tmp/adv_img.png', image_label, target_model == "robust")
    if failed:
        print('Attack failed! Try to run again.')
    else:
        print('Attack succeed! Try to implement it in the real world.')

    cv2.imshow("Adversarial image", adv_img)
    cv2.waitKey(0)


if __name__ == '__main__':
    attack_physical()
