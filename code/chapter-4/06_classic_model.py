# -*- coding:utf-8 -*-
"""
@file name  : 06_classic_model.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-02-13
@brief      : torchvision 经典模型学习

"""
import torch
import torch.nn as nn
from torchvision import models

model_alexnet = models.alexnet()

model_vgg16 = models.vgg16()

model_googlenet = models.googlenet()

model_resnet50 = models.resnet50()


for m in model_alexnet.modules():
    if isinstance(m, torch.nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
