# -*- coding:utf-8 -*-
"""
@file name  : 01_resnet_export_onnx.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-06-02
@brief      : resnet50 onnx导出
"""

import torchvision
import torch

model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
# model = torchvision.models.vgg16()

if __name__ == '__main__':

    op_set = 13
    dummy_data = torch.randn((1, 3, 224, 224))
    dummdy_data_128 = torch.randn((128, 3, 224, 224))

    # 固定 batch = 1
    torch.onnx.export(model, (dummy_data), "resnet50_bs_1.onnx",
                      opset_version=op_set, input_names=['input'],  output_names=['output'])

    # 固定 batch = 128
    torch.onnx.export(model, (dummdy_data_128), "resnet50_bs_128.onnx",
                      opset_version=op_set, input_names=['input'],  output_names=['output'])

    # 动态 batch
    torch.onnx.export(model, (dummy_data), "resnet50_bs_dynamic.onnx",
                      opset_version=op_set,  input_names=['input'], output_names=['output'],
                      dynamic_axes={"input": {0: "batch_axes"},
                                    "output": {0: "batch_axes"}})



