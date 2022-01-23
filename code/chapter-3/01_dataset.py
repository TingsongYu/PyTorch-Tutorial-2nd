# -*- coding:utf-8 -*-
"""
@file name  : 01_datasets.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-01-18
@brief      : dataset编写示例
"""
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import transforms


class COVID19Dataset_2(Dataset):
    """
    对应数据集形式-2： 数据的划分及标签在文件夹中体现
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
        self.str_2_int = {"no-finding": 0, "covid-19": 1}

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
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith("png") or file.endswith("jpeg"):
                    path_img = os.path.join(root, file)
                    sub_dir = os.path.basename(root)
                    label_int = self.str_2_int[sub_dir]
                    self.img_info.append((path_img, label_int))


class COVID19Dataset_3(Dataset):
    """
    对应数据集形式-3： 数据的划分及标签在csv中
    """

    def __init__(self, root_dir, path_csv, mode, transform=None):
        """
        获取数据集的路径、预处理的方法。由于数据划分体现在同一份文件中，因此需要设计 train/valid模式
        :param root_dir:
        :param path_csv:
        :param mode: str, train/valid
        :param transform:
        """
        self.root_dir = root_dir
        self.path_csv = path_csv
        self.mode = mode
        self.transform = transform
        self.img_info = []  # [(path, label), ... , ]
        self.label_array = None
        # 由于标签信息是string，需要一个字典转换为模型训练时用的int类型
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
        df = pd.read_csv(self.path_csv)
        df.drop(df[df["set-type"] != self.mode].index, inplace=True)  # 只要对应的数据
        df.reset_index(inplace=True)    # 非常重要！ pandas的drop不会改变index
        # 遍历表格，获取每张样本信息
        for idx in range(len(df)):
            path_img = os.path.join(self.root_dir, df.loc[idx, "img-name"])
            label_int = int(df.loc[idx, "label"])
            self.img_info.append((path_img, label_int))


if __name__ == "__main__":

    # =========================== COVID19Dataset_2 ===================================
    # you can download dataset from
    # 链接：https://pan.baidu.com/s/1szfefHgGMeyh6IyfDggLzQ
    # 提取码：ruzz
    root_dir_train = r"E:\pytorch-tutorial-2nd\data\datasets\covid-19-dataset-2\train"  # path to your data
    root_dir_valid = r"E:\pytorch-tutorial-2nd\data\datasets\covid-19-dataset-2\valid"  # path to your data
    train_set = COVID19Dataset_2(root_dir_train)
    valid_set = COVID19Dataset_2(root_dir_valid)

    # =========================== COVID19Dataset_3 ===================================
    root_dir = r"E:\pytorch-tutorial-2nd\data\datasets\covid-19-dataset-3\imgs"  # path to your data
    path_csv = r"E:\pytorch-tutorial-2nd\data\datasets\covid-19-dataset-3\dataset-meta-data.csv"  # path to your data
    train_set = COVID19Dataset_3(root_dir, path_csv, "train")
    print(len(train_set), next(iter(train_set)))
    print(len(valid_set), next(iter(valid_set)))  # 思考，为什么返回的是 PIL.Image.Image ？

    # =========================== 配合 Dataloader ===================================
    normalize = transforms.Normalize([0.5], [0.5])
    transforms_train = transforms.Compose([
        transforms.Resize((4, 4)),
        transforms.ToTensor(),
        normalize
    ])

    train_set = COVID19Dataset_3(root_dir, path_csv, "train", transform=transforms_train)  # 加入transform
    train_loader = DataLoader(dataset=train_set, batch_size=2, shuffle=True)
    for i, (inputs, target) in enumerate(train_loader):
        print(i, inputs.shape, inputs, target.shape, target)
