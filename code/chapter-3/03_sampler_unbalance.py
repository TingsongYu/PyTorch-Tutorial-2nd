# -*- coding:utf-8 -*-
"""
@file name  : 03_sampler_unbalance.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-01-21
@brief      : WeightedRandomSampler  使用于不均衡数据集
"""
import os
import shutil
import collections
import torch
import random
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from torchvision.transforms import transforms


def make_fake_data(base_num):
    """
    制作虚拟数据集，
    :return:
    """
    root_dir = r"G:\deep_learning_data\cifar10\cifar10_train"
    out_dir = r"E:\pytorch-tutorial-2nd\data\datasets\cifar-unbalance"

    import random
    for i in range(10):
        sample_num = (i + 1) * base_num

        sub_dir = os.path.join(root_dir, str(i))
        path_imgs = os.listdir(sub_dir)
        random.shuffle(path_imgs)

        out_sub_dir = os.path.join(out_dir, str(i))
        if not os.path.exists(out_sub_dir):
            os.makedirs(out_sub_dir)

        for j in range(sample_num):
            file_name = path_imgs[j]
            path_img = os.path.join(sub_dir, file_name)
            shutil.copy(path_img, out_sub_dir)
    print("done")


class CifarDataset(Dataset):
    names = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    cls_num = len(names)

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_info = []      # 定义list用于存储样本路径、标签
        self._get_img_info()

    def __getitem__(self, index):
        path_img, label = self.img_info[index]
        img = Image.open(path_img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        if len(self.img_info) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(
                self.root_dir))   # 代码具有友好的提示功能，便于debug
        return len(self.img_info)

    def _get_img_info(self):
        for root, dirs, _ in os.walk(self.root_dir):
            # 遍历类别
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.png'), img_names))
                # 遍历图片
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.abspath(os.path.join(root, sub_dir, img_name))
                    label = int(sub_dir)
                    self.img_info.append((path_img, int(label)))
        random.shuffle(self.img_info)   # 将数据顺序打乱


if __name__ == "__main__":
    # make_fake_data()
    # 链接：https://pan.baidu.com/s/1ST85f8qgyKQucvKCBKbzug
    # 提取码：vf4j
    root_dir = r"E:\pytorch-tutorial-2nd\data\datasets\cifar-unbalance"

    normalize = transforms.Normalize([0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010])
    transforms_train = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        normalize
    ])
    train_data = CifarDataset(root_dir=root_dir, transform=transforms_train)

    # 第一步：计算各类别的采样权重
    # 计算每个类的样本数量
    train_targets = [sample[1] for sample in train_data.img_info]
    label_counter = collections.Counter(train_targets)
    class_sample_counts = [label_counter[k] for k in sorted(label_counter)]  # 需要特别注意，此list的顺序！
    # 计算权重，利用倒数即可
    weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
    # weights = 12345. / torch.tensor(class_sample_counts, dtype=torch.float)

    # 第二步：生成每个样本的采样权重
    samples_weights = weights[train_targets]

    # 第三步：实例化WeightedRandomSampler
    sampler_w = WeightedRandomSampler(
        weights=samples_weights,
        num_samples=len(samples_weights),
        replacement=True)

    # 配置dataloader
    train_loader_sampler = DataLoader(dataset=train_data, batch_size=16, sampler=sampler_w)
    train_loader = DataLoader(dataset=train_data, batch_size=16)

    def show_sample(loader):
        for epoch in range(10):
            label_count = []
            for i, (inputs, target) in enumerate(loader):
                label_count.extend(target.tolist())
            print(collections.Counter(label_count))


    show_sample(train_loader)
    print("\n接下来运用sampler\n")
    show_sample(train_loader_sampler)





