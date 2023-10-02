# -*- coding:utf-8 -*-
"""
@file name  : 01_resnet_export_onnx.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-06-02
@brief      : resnet50 onnx导出
"""
import os.path

import torchvision
import torch
import torch
import torchvision
import torch.nn as nn

ckpt_path = r"./Result/2023-09-25_22-09-35/checkpoint_best.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torchvision.models.resnet50(pretrained=False)

# 替换第一层： 因为预训练模型输入是3通道，而本案例是灰度图，输入是1通道
model.conv1 = nn.Conv2d(1, 64, (7, 7), stride=(2, 2), padding=(3, 3), bias=False)
num_ftrs = model.fc.in_features  # 替换最后一层
model.fc = nn.Linear(num_ftrs, 2)

state_dict = torch.load(ckpt_path)
model_sate_dict = state_dict['model_state_dict']
model.load_state_dict(model_sate_dict)  # 模型参数加载


if __name__ == '__main__':

    op_set = 13
    dummy_data = torch.randn((1, 3, 224, 224))

    # 固定 batch = 1
    out_dir = os.path.dirname(ckpt_path)
    path_out = os.path.join(out_dir, "resnet50_bs_1.onnx")
    torch.onnx.export(model, (dummy_data), path_out,
                      opset_version=op_set, input_names=['input'],  output_names=['output'])



