# -*- coding:utf-8 -*-
"""
@file name  : 04_cam_series_demo.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-06-156
@brief      : https://github.com/jacobgil/pytorch-grad-cam  学习与使用
安装：pip install grad_cam
"""
import cv2
import json
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
from matplotlib import pyplot as plt

def img_transform(img_in, transform):
    """
    将img进行预处理，并转换成模型输入所需的形式—— B*C*H*W
    :param img_roi: np.array
    :return:
    """
    img = img_in.copy()
    img = Image.fromarray(np.uint8(img))
    img = transform(img)
    img = img.unsqueeze(0)    # C*H*W --> B*C*H*W
    return img


def img_preprocess(img_in):
    """
    读取图片，转为模型可读的形式
    :param img_in: ndarray, [H, W, C]
    :return: PIL.image
    """
    img = img_in.copy()
    img = cv2.resize(img, (224, 224))
    img = img[:, :, ::-1]   # BGR --> RGB
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4948052, 0.48568845, 0.44682974], [0.24580306, 0.24236229, 0.2603115])
    ])
    img_input = img_transform(img, transform)
    return img_input


def cam_factory(cam_name_):
    return eval(cam_name_)


if __name__ == '__main__':

    path_img = "both.png"
    output_dir = "./Result"

    # 图片读取
    img = cv2.imread(path_img, 1)  # H*W*C
    img_input = img_preprocess(img)

    model = resnet50(pretrained=True)
    target_layers = [model.layer4[-1]]
    input_tensor = img_input

    cam_alg_list = "GradCAM,ScoreCAM,GradCAMPlusPlus,XGradCAM,EigenCAM,FullGrad".split(",")




    plt.tight_layout()
    # fig, axs = plt.subplots(2, 3, figsize=(9, 9))
    fig, axs = plt.subplots(2, 3)
    for idx, cam_name in enumerate(cam_alg_list):
        cam = cam_factory(cam_name)(model=model, target_layers=target_layers)
        # targets = [e.g ClassifierOutputTarget(281)]
        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)  # If targets is None, the highest scoring category
        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]
        # img_norm = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)
        img_norm = img/255.
        visualization = show_cam_on_image(img_norm, grayscale_cam, use_rgb=False)
        vis_rgb = cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB)

        im = axs.ravel()[idx].imshow(vis_rgb)
        axs.ravel()[idx].set_title(cam_name)
    plt.show()