# -*- coding:utf-8 -*-
"""
@file name  : 02_summarywriter.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-06-07
@brief      : tensorboard的writer 使用
"""
import os
import cv2
import numpy as np
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":

    # ================================ add_scalar ================================
    writer = SummaryWriter(comment="add_scalar", filename_suffix="_test_tensorboard")
    x = range(100)
    for i in x:
        writer.add_scalar('y=2x', i * 2, i)
        writer.add_scalar('Loss/train', np.random.random(), i)
        writer.add_scalar('Loss/Valid', np.random.random(), i)
    writer.close()

    # ================================ add_scalars ================================
    writer = SummaryWriter(comment="add_scalars", filename_suffix="_test_tensorboard")
    for i in range(100):
        writer.add_scalars('Loss_curve', {'train_loss': np.random.random(),
                                          'valid_loss': np.random.random()}, i)
    writer.close()

    # ================================ add_histogram ================================
    writer = SummaryWriter(comment="add_histogram", filename_suffix="_test_tensorboard")
    for i in range(10):
        x = np.random.random(1000)
        writer.add_histogram('distribution centers', x + i, i)
    writer.close()

    # ================================ add_image ================================
    writer = SummaryWriter(comment="add_image", filename_suffix="_test_tensorboard")

    img = np.zeros((3, 100, 100))
    img[0] = np.arange(0, 10000).reshape(100, 100) / 10000
    img[1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000

    img_HWC = np.zeros((100, 100, 3))
    img_HWC[:, :, 0] = np.arange(0, 10000).reshape(100, 100) / 10000
    img_HWC[:, :, 1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000

    writer.add_image('my_image-shape:{}'.format(img.shape), img, 0)
    print(img.shape)

    writer.add_image('my_image_HWC-shape:{}'.format(img_HWC.shape), img_HWC, 0, dataformats='HWC')
    print(img_HWC.shape)

    # download dataset from
    # 链接：https://pan.baidu.com/s/1szfefHgGMeyh6IyfDggLzQ
    # 提取码：ruzz
    path_img = r"F:\pytorch-tutorial-2nd\data\datasets\covid-19-dataset-3\imgs\ryct.2020200028.fig1a.jpeg"
    img_opencv = cv2.imread(path_img)
    writer.add_image('img_opencv_HWC-shape:{}'.format(img_opencv.shape), img_opencv, 0, dataformats='HWC')
    writer.close()

    # ================================ add_images ================================
    writer = SummaryWriter(comment="add_images", filename_suffix="_test_tensorboard")

    img_batch = np.zeros((16, 3, 100, 100))
    for i in range(16):
        img_batch[i, 0] = np.arange(0, 10000).reshape(100, 100) / 10000 / 16 * i
        img_batch[i, 1] = (1 - np.arange(0, 10000).reshape(100, 100) / 10000) / 16 * i

    writer.add_images('add_images', img_batch, 0)
    writer.close()

    # ================================ add_mesh ================================
    import torch
    vertices_tensor = torch.as_tensor([
        [1, 1, 1],
        [-1, -1, 1],
        [1, -1, -1],
        [-1, 1, -1],
    ], dtype=torch.float).unsqueeze(0)
    colors_tensor = torch.as_tensor([
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 0, 255],
    ], dtype=torch.int).unsqueeze(0)
    faces_tensor = torch.as_tensor([
        [0, 2, 3],
        [0, 3, 1],
        [0, 1, 2],
        [1, 3, 2],
    ], dtype=torch.int).unsqueeze(0)

    writer = SummaryWriter(comment="add_mesh", filename_suffix="_test_tensorboard")
    writer.add_mesh('add_mesh', vertices=vertices_tensor, colors=colors_tensor, faces=faces_tensor)
    writer.close()

    # ================================ add_hparams ================================
    writer = SummaryWriter(comment="add_hparams", filename_suffix="_test_tensorboard")
    for i in range(5):
        writer.add_hparams({'lr': 0.1 * i, 'bsize': i},
                           {'hparam/accuracy': 10 * i, 'hparam/loss': 10 * i})
    writer.close()
