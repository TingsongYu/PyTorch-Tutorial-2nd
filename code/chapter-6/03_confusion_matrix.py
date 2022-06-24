# -*- coding:utf-8 -*-
"""
@file name  : 03_confusion_matrix.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-06-14
@brief      : 混淆矩阵绘制及训练曲线记录
"""
import torch
import numpy as np
import os
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from datetime import datetime
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from my_utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_bs = 128
valid_bs = 128
lr_init = 0.01
max_epoch = 200
print_interval = 100
classes_name = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# ------------------------------------  log ------------------------------------
result_dir = os.path.join("Result")

now_time = datetime.now()
time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')

log_dir = os.path.join(result_dir, time_str)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

writer = SummaryWriter(log_dir=log_dir)

# ------------------------------------ step 1/5 : 加载数据------------------------------------
normMean = [0.4948052, 0.48568845, 0.44682974]
normStd = [0.24580306, 0.24236229, 0.2603115]
normTransform = transforms.Normalize(normMean, normStd)
train_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    normTransform
])

valid_transform = transforms.Compose([
    transforms.ToTensor(),
    normTransform
])

# 构建MyDataset实例
# root变量下需要存放cifar-10-python.tar.gz 文件
# cifar-10-python.tar.gz可从 "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz" 下载
data_dir = r"F:\pytorch-tutorial-2nd\data\datasets\cifar10-office"
train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, transform=train_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, transform=valid_transform, download=True)

# 构建DataLoder
train_loader = DataLoader(dataset=train_set, batch_size=train_bs, shuffle=True)
valid_loader = DataLoader(dataset=test_set, batch_size=valid_bs)

# ------------------------------------ step 2/5 : 定义网络------------------------------------
model = resnet8()
model.to(device)
# ------------------------------------ step 3/5 : 定义损失函数和优化器 ------------------------------------
criterion = nn.CrossEntropyLoss()  # 选择损失函数
optimizer = optim.SGD(model.parameters(), lr=lr_init, momentum=0.9, dampening=0.1)  # 选择优化器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.1)  # 设置学习率下降策略
# ------------------------------------ step 4/5 : 训练 --------------------------------------------------
for epoch in range(max_epoch):

    # scheduler.step()  # 更新学习率

    class_num = len(classes_name)
    conf_mat = np.zeros((class_num, class_num))
    loss_sigma = []
    loss_avg = 0
    acc_avg = 0
    path_error = []
    label_list = []
    
    model.train()
    for i, data in enumerate(train_loader):
        # if i == 30 : break
        # 获取图片和标签
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # forward, backward, update weights
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 统计预测信息
        loss_sigma.append(loss.item())
        loss_avg = np.mean(loss_sigma)

        _, predicted = torch.max(outputs.data, 1)
        for j in range(len(labels)):
            cate_i = labels[j].cpu().numpy()
            pre_i = predicted[j].cpu().numpy()
            conf_mat[cate_i, pre_i] += 1.
        acc_avg = conf_mat.trace() / conf_mat.sum()

        # 每10个iteration 打印一次训练信息，loss为10个iteration的平均
        if i % print_interval == print_interval - 1:
            print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".
                  format(epoch + 1, max_epoch, i + 1, len(train_loader), loss_avg, acc_avg))
    # 学习率更新
    scheduler.step()

    # 记录训练loss
    writer.add_scalars('Loss_group', {'train_loss': loss_avg}, epoch)
    # 记录learning rate
    writer.add_scalar('learning rate', scheduler.get_last_lr()[0], epoch)
    # 记录Accuracy
    writer.add_scalars('Accuracy_group', {'train_acc': acc_avg}, epoch)

    conf_mat_figure = show_conf_mat(conf_mat, classes_name, "train", log_dir, epoch=epoch, verbose=epoch == max_epoch - 1)
    writer.add_figure('confusion_matrix_train', conf_mat_figure, global_step=epoch)


    # ------------------------------------ 观察模型在验证集上的表现 ------------------------------------
    class_num = len(classes_name)
    conf_mat = np.zeros((class_num, class_num))
    loss_sigma = []
    loss_avg = 0
    acc_avg = 0
    path_error = []
    label_list = []

    model.eval()
    for i, data in enumerate(valid_loader):
        # 获取图片和标签
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # forward
        outputs = model(inputs)
        outputs.detach_()
        # 计算loss
        loss = criterion(outputs, labels)

        # 统计预测信息
        loss_sigma.append(loss.item())
        loss_avg = np.mean(loss_sigma)

        _, predicted = torch.max(outputs.data, 1)
        for j in range(len(labels)):
            cate_i = labels[j].cpu().numpy()
            pre_i = predicted[j].cpu().numpy()
            conf_mat[cate_i, pre_i] += 1.
        acc_avg = conf_mat.trace() / conf_mat.sum()

    print('{} set Accuracy:{:.2%}'.format('Valid', conf_mat.trace() / conf_mat.sum()))
    # 记录Loss, accuracy
    writer.add_scalars('Loss_group', {'valid_loss': loss_avg}, epoch)
    writer.add_scalars('Accuracy_group', {'valid_acc': acc_avg}, epoch)
    # 保存混淆矩阵图
    conf_mat_figure = show_conf_mat(conf_mat, classes_name, "valid", log_dir, epoch=epoch, verbose=epoch == max_epoch - 1)
    writer.add_figure('confusion_matrix_valid', conf_mat_figure, global_step=epoch)
print('Finished Training')


