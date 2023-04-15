# -*- coding:utf-8 -*-
"""
@file name  : brain_mri_dataset.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-03-03
@brief      : brain mri 数据集读取
"""
import os
import random

import pandas as pd
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from skimage.io import imread


# 读取中文路径的图片
def cv_imread(path_file):
    cv_img = cv2.imdecode(np.fromfile(path_file, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    return cv_img


class BrainMRIDataset(Dataset):
    def __init__(self, path_csv, transforms_=None):
        self.df = pd.read_csv(path_csv)
        self.transforms = transforms_

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = cv_imread(self.df.iloc[idx, 1])
        mask = cv_imread(self.df.iloc[idx, 2])
        mask[mask == 255] = 1  # 转换为0, 1 二分类标签

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        return image, mask.long()


if __name__ == "__main__":
    root_dir_train = r"../data_train.csv"  # path to your data
    root_dir_valid = r"../data_val.csv"  # path to your data

    train_set = BrainMRIDataset(root_dir_train)
    valid_set = BrainMRIDataset(root_dir_valid)

    train_loader = DataLoader(dataset=train_set, batch_size=2, shuffle=True)
    for i, (inputs, target) in enumerate(train_loader):
        print(i, inputs.shape, inputs, target.shape, target)
