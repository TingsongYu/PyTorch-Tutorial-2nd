# -*- coding:utf-8 -*-
"""
@file name  : 05_model_print.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-06-21
@brief      : 模型参数打印
"""
import torchvision.models as models
from torchinfo import summary

if __name__ == '__main__':
    resnet_50 = models.resnet50(pretrained=False)
    batch_size = 1

    summary(resnet_50, input_size=(batch_size, 3, 224, 224))

    col_names_ = ("input_size", "output_size", "num_params", "kernel_size", "mult_adds", "trainable",)
    # summary(resnet_50, input_size=(batch_size, 3, 224, 224), col_names=col_names_)

    # summary(resnet_50, input_size=(batch_size, 3, 224, 224), col_names=("input_size",))
    # summary(resnet_50, input_size=(batch_size, 3, 224, 224), row_settings=("ascii_only",))
    # summary(resnet_50, input_size=(batch_size, 3, 224, 224), verbose=1)
