# -*- coding:utf-8 -*-
"""
@file name  : 02_sgd.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-06-01
@brief      : SGD使用教程
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image


def main():
    # 思考：如何实现你的模型训练？第一步干什么？第二步干什么？...第n步...
    # step 1/4 : 数据模块：构建dataset, dataloader，实现对硬盘中数据的读取及设定预处理方法
    # step 2/4 : 模型模块：构建神经网络，用于后续训练
    # step 3/4 : 优化模块：设定损失函数与优化器，用于在训练过程中对网络参数进行更新
    # step 4/4 : 迭代模块: 循环迭代地进行模型训练，数据一轮又一轮的喂给模型，不断优化模型，直到我们让它停止训练

    # step 1/4 : 数据模块
    class COVID19Dataset(Dataset):
        def __init__(self, root_dir, txt_path, transform=None):
            """
            获取数据集的路径、预处理的方法
            """
            self.root_dir = root_dir
            self.txt_path = txt_path
            self.transform = transform
            self.img_info = []  # [(path, label), ... , ]
            self.label_array = None
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
            # 读取txt，解析txt
            with open(self.txt_path, "r") as f:
                txt_data = f.read().strip()
                txt_data = txt_data.split("\n")

            self.img_info = [(os.path.join(self.root_dir, i.split()[0]), int(i.split()[2]))
                             for i in txt_data]
    # you can download the datasets from
    # https://pan.baidu.com/s/18BsxploWR3pbybFtNsw5fA  code：pyto
    root_dir = r"F:\pytorch-tutorial-2nd\data\datasets\covid-19-demo"  # path to datasets——covid-19-demo
    img_dir = os.path.join(root_dir, "imgs")
    path_txt_train = os.path.join(root_dir, "labels", "train.txt")
    path_txt_valid = os.path.join(root_dir, "labels", "valid.txt")
    transforms_func = transforms.Compose([
        transforms.Resize((8, 8)),
        transforms.ToTensor(),
    ])
    train_data = COVID19Dataset(root_dir=img_dir, txt_path=path_txt_train, transform=transforms_func)
    train_loader = DataLoader(dataset=train_data, batch_size=2)

    # step 2/4 : 模型模块
    class TinnyCNN(nn.Module):
        def __init__(self, cls_num=2):
            super(TinnyCNN, self).__init__()
            self.convolution_layer = nn.Conv2d(1, 1, kernel_size=(3, 3))
            self.fc = nn.Linear(36, cls_num)

        def forward(self, x):
            x = self.convolution_layer(x)
            x = x.view(x.size(0), -1)
            out = self.fc(x)
            return out

    model = TinnyCNN(2)

    # step 3/4 : 优化模块
    loss_f = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=50)
    # step 4/4 : 迭代模块
    for epoch in range(100):
        # 训练集训练
        model.train()
        for data, labels in train_loader:
            # forward & backward
            outputs = model(data)
            optimizer.zero_grad()

            # loss 计算
            loss = loss_f(outputs, labels)
            loss.backward()
            optimizer.step()

            # 计算分类准确率
            _, predicted = torch.max(outputs.data, 1)
            correct_num = (predicted == labels).sum()
            acc = correct_num / labels.shape[0]
            print("Epoch:{} Train Loss:{:.2f} Acc:{:.0%}".format(epoch, loss, acc))


if __name__ == "__main__":
    main()

