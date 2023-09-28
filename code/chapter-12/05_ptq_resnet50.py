# -*- coding:utf-8 -*-
"""
@file name  : 05_ptq_resnet50.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-09-24
@brief      : 采用 pytorch_quantization 库实现PTQ量化
参考自：https://github.com/NVIDIA/TensorRT/blob/main/quickstart/quantization_tutorial/qat-ptq-workflow.ipynb
"""

import tensorrt as trt

import pytorch_quantization
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules
from pytorch_quantization import calib
from tqdm import tqdm

print(pytorch_quantization.__version__)

import os
import numpy as np
import time

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision import models, datasets
from torch.utils.data import DataLoader
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import transforms


class PneumoniaDataset(Dataset):
    """
    数据目录组织结构为文件夹划分train/test，2个类别标签通过文件夹名称获得
    """

    def __init__(self, root_dir, transform=None):
        """
        获取数据集的路径、预处理的方法，此时只需要根目录即可，其余信息通过文件目录获取
        """
        self.root_dir = root_dir
        self.transform = transform
        self.img_info = []  # [(path, label), ... , ]
        self.label_array = None
        # 由于标签信息是string，需要一个字典转换为模型训练时用的int类型
        self.str_2_int = {"NORMAL": 0, "PNEUMONIA": 1}

        self._get_img_info()

    def __getitem__(self, index):
        """
        输入标量index, 从硬盘中读取数据，并预处理，to Tensor
        :param index:
        :return:
        """
        path_img, label = self.img_info[index]
        img = Image.open(path_img).convert('L')

        if self.transform is not None:
            if 'albumentations' in str(type(self.transform)):
                img = np.array(img)
                img = self.transform(image=img)['image']
            else:
                img = self.transform(img)

        return img, label

    def __len__(self):
        if len(self.img_info) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(
                self.root_dir))  # 代码具有友好的提示功能，便于debug
        return len(self.img_info)

    def _get_img_info(self):
        """
        实现数据集的读取，将硬盘中的数据路径，标签读取进来，存在一个list中
        path, label
        :return:
        """
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith("jpg") or file.endswith("jpeg"):
                    path_img = os.path.join(root, file)
                    sub_dir = os.path.basename(root)
                    label_int = self.str_2_int[sub_dir]
                    self.img_info.append((path_img, label_int))


def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
    model.cuda()


def collect_stats(model, data_loader, num_batches):
    """Feed data to the network and collect statistics"""
    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    # Feed data to the network for collecting stats
    for i, (image, _) in tqdm(enumerate(data_loader), total=num_batches):
        model(image.cuda())
        if i >= num_batches:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

quant_modules.initialize()  # 替换torch.nn的常用层，变为可量化的层

# ========================== 定义模型
ckpt_path = './ptq_qat_ckpt/cls_resnet50_checkpoint_best-2023-02-08_16-37-24.pth'  # 第八章的分类章节

model = torchvision.models.convnext_tiny(pretrained=True)
# 替换第一层： 因为预训练模型输入是3通道，而本案例是灰度图，输入是1通道
num_kernel = 96
model.features[0][0] = nn.Conv2d(1, num_kernel, (4, 4), stride=(4, 4))  # convnext base/ tiny
# 替换最后一层
num_ftrs = model.classifier[2].in_features
model.classifier[2] = nn.Linear(num_ftrs, 2)

state_dict = torch.load(ckpt_path)
model_sate_dict = state_dict['model_state_dict']
model.load_state_dict(model_sate_dict)  # 模型参数加载
model.to(device)


# model = torchvision.models.resnet50(pretrained=False)
# model.conv1 = nn.Conv2d(1, 64, (7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# num_ftrs = model.fc.in_features  # 替换最后一层
# model.fc = nn.Linear(num_ftrs, 2)
#
# # 加载预训练权重
# state_dict = torch.load(ckpt_path)
# model_sate_dict = state_dict['model_state_dict']
# model.load_state_dict(model_sate_dict)  # 模型参数加载
#
# model.to(device)

# =============================== 定义数据
normMean = [0.5]
normStd = [0.5]
input_size = (224, 224)
normTransform = transforms.Normalize(normMean, normStd)
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(input_size, padding=4),
    transforms.ToTensor(),
    normTransform
])

valid_transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    normTransform
])

# chest_xray.zip 解压，获得 chest_xray/train, chest_xray/test
# 数据可从 https://data.mendeley.com/datasets/rscbjbr9sj/2 下载
data_dir = r'G:\deep_learning_data\chest_xray'
train_dir = os.path.join(data_dir, 'train')
train_set = PneumoniaDataset(train_dir, transform=train_transform)
train_loader = DataLoader(dataset=train_set, batch_size=4, shuffle=True)


# Calibrate the model using max calibration technique.
with torch.no_grad():
    collect_stats(model, train_loader, num_batches=16)
    compute_amax(model, method="max")


torch.save(model.state_dict(), "./ptq_qat_ckpt/resnet50_ptq.pth")

def evaluate(model, dataloader, crit, epoch):
    total = 0
    correct = 0
    loss = 0.0
    class_probs = []
    class_preds = []
    model.eval()
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.cuda(), labels.cuda(non_blocking=True)
            out = model(data)
            loss += crit(out, labels)
            preds = torch.max(out, 1)[1]
            class_preds.append(preds)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return correct / total




