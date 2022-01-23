# -*- coding:utf-8 -*-
"""
@file name  : 02_dataloader.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-01-22
@brief      : dataloader使用学习
"""
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import transforms


class AntsBeesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_info = []  # [(path, label), ... , ]
        self.label_array = None
        # 由于标签信息是string，需要一个字典转换为模型训练时用的int类型
        self.str_2_int = {"ants": 0, "bees": 1}
        self._get_img_info()

    def __getitem__(self, index):
        """
        输入标量index, 从硬盘中读取数据，并预处理，to Tensor
        :param index:
        :return:
        """
        path_img, label = self.img_info[index]
        img = Image.open(path_img)

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        if len(self.img_info) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(
                self.root_dir))  # 代码具有友好的提示功能，便于debug
        return len(self.img_info)

    def _get_img_info(self):
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith("jpg"):
                    path_img = os.path.join(root, file)
                    sub_dir = os.path.basename(root)
                    label_int = self.str_2_int[sub_dir]
                    self.img_info.append((path_img, label_int))


if __name__ == "__main__":
    # 链接：https://pan.baidu.com/s/1X11v5XEbdrgdgsVAESCVrA
    # 提取码：4wx1
    root_dir = r"E:\pytorch-tutorial-2nd\data\datasets\mini-hymenoptera_data\train"
    # =========================== 配合 Dataloader ===================================
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 来自ImageNet数据集统计值
    transforms_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    train_set = AntsBeesDataset(root_dir, transform=transforms_train)  # 加入transform

    train_loader_bs2 = DataLoader(dataset=train_set, batch_size=2, shuffle=True)
    train_loader_bs3 = DataLoader(dataset=train_set, batch_size=3, shuffle=True)
    train_loader_bs2_drop = DataLoader(dataset=train_set, batch_size=2, shuffle=True, drop_last=True)

    for i, (inputs, target) in enumerate(train_loader_bs2):
        print(i, inputs.shape, target.shape, target)
    for i, (inputs, target) in enumerate(train_loader_bs3):
        print(i, inputs.shape, target.shape, target)
    for i, (inputs, target) in enumerate(train_loader_bs2_drop):
        print(i, inputs.shape, target.shape, target)

