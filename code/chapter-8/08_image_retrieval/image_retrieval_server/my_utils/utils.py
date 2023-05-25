# -*- coding=utf-8 -*-
"""
# @file name  : utils.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-04-30
@brief      : 基础函数
"""
import os
import cv2
import numpy as np


def get_file_path(root_dir: str, extentions: list):
    """
    获取文件夹下所有图片
    :param root: 文件夹
    :param extentions: list， 想要的文件的后缀
    :return: list
    """
    assert os.path.isdir(root_dir), '%s is not a valid directory' % root_dir
    assert isinstance(extentions, list), '%s is not a list' % extentions

    img_path_list = []

    def is_image_file(filename):
        return any(filename.endswith(extension) for extension in extentions)  # any

    for root, _, fnames in sorted(os.walk(root_dir)):
        for fname in fnames:
            if is_image_file(fname):
                img_path = os.path.join(root, fname)
                img_path_list.append(img_path)
    return img_path_list


def cv_imread(path_file):
    cv_img = cv2.imdecode(np.fromfile(path_file, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    return cv_img

if __name__ == '__main__':
    dir_name = r'G:\deep_learning_data\coco128\images\train2017'

    img_ext = 'jpg JPG jpeg JPEG png PNG'.split()

    img_path_list = get_file_path(dir_name, img_ext)
    print(img_path_list[0], len(img_path_list))















