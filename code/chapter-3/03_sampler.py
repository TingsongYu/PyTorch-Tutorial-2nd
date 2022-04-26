# -*- coding:utf-8 -*-
"""
@file name  : 03_sampler.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-01-21
@brief      : WeightedRandomSampler使用，初认识
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from torchvision.transforms import transforms


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


if __name__ == "__main__":
    root_dir = r"E:\pytorch-tutorial-2nd\data\datasets\covid-19-demo"  # path to datasets——covid-19-demo
    img_dir = os.path.join(root_dir, "imgs")
    path_txt_train = os.path.join(root_dir, "labels", "train.txt")

    # 设置 dataset
    normalize = transforms.Normalize([0.5], [0.5])
    transforms_train = transforms.Compose([
        transforms.Resize((4, 4)),
        transforms.ToTensor(),
        normalize
    ])
    train_data = COVID19Dataset(root_dir=img_dir, txt_path=path_txt_train, transform=transforms_train)

    # 第一步：计算每个类的采样权重
    weights = torch.tensor([1, 5], dtype=torch.float)

    # 第二步：生成每个样本的采样权重
    train_targets = [sample[1] for sample in train_data.img_info]
    samples_weights = weights[train_targets]

    # 第三步：实例化WeightedRandomSampler
    sampler_w = WeightedRandomSampler(
        weights=samples_weights,
        num_samples=len(samples_weights),
        replacement=True)

    # 设置 dataloader
    train_loader = DataLoader(dataset=train_data, batch_size=2, sampler=sampler_w)
    for epoch in range(10):
        for i, (inputs, target) in enumerate(train_loader):
            print(target.shape, target)
    print("\n由于是有放回采样，并且样本1的采样概率比0高5倍，可以看到很多次出现[1, 1]")









