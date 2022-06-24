# -*- coding: utf-8 -*-
"""
# @file name  : 02_fintune-freeze.py
# @author     :  TingsongYu https://github.com/TingsongYu
# @date       : 2022-06-24
# @brief      : finetune方法之冻结特征提取层
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from datetime import datetime
from my_utils import *

BASEDIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("use device :{}".format(device))

label_name = {"ants": 0, "bees": 1}


class AntsDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.label_name = {"ants": 0, "bees": 1}
        self.data_info = self.get_img_info(data_dir)
        self.transform = transform

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.data_info)

    def get_img_info(self, data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            # 遍历类别
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))

                # 遍历图片
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    label = self.label_name[sub_dir]
                    data_info.append((path_img, int(label)))

        if len(data_info) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(data_dir))
        return data_info

# 参数设置
max_epoch = 25
BATCH_SIZE = 16
LR = 0.001
log_interval = 10
val_interval = 1
classes = 2
start_epoch = -1
lr_decay_step = 7
print_interval = 2

# ------------------------------------  log ------------------------------------
result_dir = os.path.join("Result")

now_time = datetime.now()
time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')

log_dir = os.path.join(result_dir, time_str)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

writer = SummaryWriter(log_dir=log_dir)

# ============================ step 1/5 数据 ============================
# https://download.pytorch.org/tutorial/hymenoptera_data.zip
data_dir = r"G:\deep_learning_data\hymenoptera_data"

train_dir = os.path.join(data_dir, "train")
valid_dir = os.path.join(data_dir, "val")

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

valid_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

# 构建MyDataset实例
train_data = AntsDataset(data_dir=train_dir, transform=train_transform)
valid_data = AntsDataset(data_dir=valid_dir, transform=valid_transform)

# 构建DataLoder
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)

# ============================ step 2/5 模型 ============================

# 1/3 构建模型
resnet18_ft = models.resnet18()

# 2/3 加载参数
# download resnet18-f37072fd.pth from: 
# https://download.pytorch.org/models/resnet18-f37072fd.pth
path_pretrained_model = r"F:\pytorch-tutorial-2nd\data\model_zoo\resnet18-f37072fd.pth"
state_dict_load = torch.load(path_pretrained_model)
resnet18_ft.load_state_dict(state_dict_load)

# 法1: 冻结卷积层
for param in resnet18_ft.parameters():
    param.requires_grad = True
print("conv1.weights[0, 0, ...]:\n {}".format(resnet18_ft.conv1.weight[0, 0, ...]))

# 替换fc层
num_ftrs = resnet18_ft.fc.in_features
resnet18_ft.fc = nn.Linear(num_ftrs, classes)

resnet18_ft.to(device)
# ============================ step 3/5 损失函数 ============================
criterion = nn.CrossEntropyLoss()                                                   # 选择损失函数

# =========================== step 4/5 优化器 ============================
optimizer = optim.SGD(resnet18_ft.parameters(), lr=LR, momentum=0.9)               # 选择优化器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=0.1)     # 设置学习率下降策略

# ============================ step 5/5 训练 ============================
train_curve = list()
valid_curve = list()

for epoch in range(start_epoch + 1, max_epoch):

    class_num = classes
    conf_mat = np.zeros((class_num, class_num))
    loss_sigma = []
    loss_avg = 0
    acc_avg = 0
    path_error = []
    label_list = []

    resnet18_ft.train()
    for i, data in enumerate(train_loader):

        # forward
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # forward, backward, update weights
        optimizer.zero_grad()
        outputs = resnet18_ft(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 统计分类情况
        # 统计预测信息
        loss_sigma.append(loss.item())
        loss_avg = np.mean(loss_sigma)

        _, predicted = torch.max(outputs.data, 1)
        for j in range(len(labels)):
            cate_i = labels[j].cpu().numpy()
            pre_i = predicted[j].cpu().numpy()
            conf_mat[cate_i, pre_i] += 1.
        acc_avg = conf_mat.trace() / conf_mat.sum()

        # 打印训练信息
        # 每10个iteration 打印一次训练信息，loss为10个iteration的平均
        if i % print_interval == print_interval - 1:
            print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".
                  format(epoch + 1, max_epoch, i + 1, len(train_loader), loss_avg, acc_avg))

            print("epoch:{} conv1.weights[0, 0, ...] :\n {}".format(epoch, resnet18_ft.conv1.weight[0, 0, ...]))

    scheduler.step()  # 更新学习率
    # 记录训练loss
    writer.add_scalars('Loss_group', {'train_loss': loss_avg}, epoch)
    # 记录learning rate
    writer.add_scalar('learning rate', scheduler.get_last_lr()[0], epoch)
    # 记录Accuracy
    writer.add_scalars('Accuracy_group', {'train_acc': acc_avg}, epoch)

    conf_mat_figure = show_conf_mat(conf_mat, list(label_name.keys()), "train", log_dir, epoch=epoch, verbose=epoch == max_epoch - 1)
    writer.add_figure('confusion_matrix_train', conf_mat_figure, global_step=epoch)

    # validate the model
    class_num = classes
    conf_mat = np.zeros((class_num, class_num))
    loss_sigma = []
    loss_avg = 0
    acc_avg = 0
    path_error = []
    label_list = []

    resnet18_ft.eval()
    with torch.no_grad():
        for j, data in enumerate(valid_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = resnet18_ft(inputs)
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
    conf_mat_figure = show_conf_mat(conf_mat, list(label_name.keys()), "valid", log_dir, epoch=epoch, verbose=epoch == max_epoch - 1)
    writer.add_figure('confusion_matrix_valid', conf_mat_figure, global_step=epoch)
print('Finished Training')


