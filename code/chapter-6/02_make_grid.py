# -*- coding:utf-8 -*-
"""
@file name  : 02_make_grid.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-06-09
@brief      : make_grid 函数学习
"""
import os
import torch
import cv2
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter


def resize_img_keep_ratio(img_, target_size):
    """
    按比例缩放图像，并填充至指定大小
    :param img_:
    :param target_size:
    :return:
    """
    old_size = img_.shape[0:2]  # 原始图像大小
    ratio = min(float(target_size[i]) / (old_size[i]) for i in range(len(old_size)))  # 计算原始图像宽高与目标图像大小的比例，并取其中的较小值
    new_size = tuple([int(i * ratio) for i in old_size])  # 根据上边求得的比例计算在保持比例前提下得到的图像大小
    img = cv2.resize(img_, (new_size[1], new_size[0]))  # 根据上边的大小进行放缩
    pad_w = target_size[1] - new_size[1]  # 计算需要填充的像素数目（图像的宽这一维度上）
    pad_h = target_size[0] - new_size[0]  # 计算需要填充的像素数目（图像的高这一维度上）
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)
    img_new = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, (0, 0, 0))
    return img_new


if __name__ == "__main__":

    writer = SummaryWriter(comment="grid images", filename_suffix="grid_img")
    # ============================== mode 1 ==============================
    # download dataset from
    # 链接：https://pan.baidu.com/s/1szfefHgGMeyh6IyfDggLzQ
    # 提取码：ruzz
    data_dir = r"F:\pytorch-tutorial-2nd\data\datasets\covid-19-dataset-3\imgs"  # path to your data
    name_list = os.listdir(data_dir)
    path_list = [os.path.join(data_dir, name) for name in name_list]

    PATCH_SIZE = (500, 500)
    img_list = []
    for path_img in path_list:
        img_hwc = cv2.imread(path_img)
        img_resize = resize_img_keep_ratio(img_hwc, PATCH_SIZE)
        img_tensor = torch.from_numpy(img_resize)
        img_tensor = img_tensor.transpose(0, 2).transpose(1, 2)  # HWC --> CHW
        img_list.append(img_tensor)
    img_grid = vutils.make_grid(img_list, normalize=False, scale_each=False)
    writer.add_image("X-ray", img_grid)

    # ============================== mode 2 ==============================
    for step in range(10):
        dummy_img = torch.rand(32, 3, 64, 64)  # (B x C x H x W)
        img_grid = vutils.make_grid(dummy_img, normalize=True, scale_each=True)
        writer.add_image('Image', img_grid, step)

    writer.close()















