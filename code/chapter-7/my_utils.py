# -*- coding:utf-8 -*-
"""
@file name  : 03_utils.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-06-14
@brief      : 训练所需的函数
"""
import numpy as np
import os
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def validate(net, data_loader, set_name, classes_name, device):
    """
    对一批数据进行预测，返回混淆矩阵以及Accuracy
    :param net:
    :param data_loader:
    :param set_name:  eg: 'valid' 'train' 'tesst
    :param classes_name:
    :return:
    """
    net.eval()
    cls_num = len(classes_name)
    conf_mat = np.zeros([cls_num, cls_num])

    for data in data_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = net(inputs)
        outputs.detach_()

        _, predicted = torch.max(outputs.data, 1)

        # 统计混淆矩阵
        for i in range(len(labels)):
            cate_i = labels[i].cpu().numpy()
            pre_i = predicted[i].cpu().numpy()
            conf_mat[cate_i, pre_i] += 1.0

    for i in range(cls_num):
        print('class:{:<10}, total num:{:<6}, correct num:{:<5}  Recall: {:.2%} Precision: {:.2%}'.format(
            classes_name[i], np.sum(conf_mat[i, :]), conf_mat[i, i], conf_mat[i, i] / (1 + np.sum(conf_mat[i, :])),
                                                                conf_mat[i, i] / (1 + np.sum(conf_mat[:, i]))))

    print('{} set Accuracy:{:.2%}'.format(set_name, np.trace(conf_mat) / np.sum(conf_mat)))

    return conf_mat, '{:.2}'.format(np.trace(conf_mat) / np.sum(conf_mat))


def show_conf_mat(confusion_mat, classes, set_name, out_dir, epoch=999, verbose=False, perc=False):
    """
    混淆矩阵绘制并保存图片
    :param confusion_mat:  nd.array
    :param classes: list or tuple, 类别名称
    :param set_name: str, 数据集名称 train or valid or test?
    :param out_dir:  str, 图片要保存的文件夹
    :param epoch:  int, 第几个epoch
    :param verbose: bool, 是否打印精度信息
    :param perc: bool, 是否采用百分比，图像分割时用，因分类数目过大
    :return:
    """
    cls_num = len(classes)

    # 归一化
    confusion_mat_tmp = confusion_mat.copy()
    for i in range(len(classes)):
        confusion_mat_tmp[i, :] = confusion_mat[i, :] / confusion_mat[i, :].sum()

    # 设置图像大小
    if cls_num < 10:
        figsize = 6
    elif cls_num >= 100:
        figsize = 30
    else:
        figsize = np.linspace(6, 30, 91)[cls_num-10]

    fig, ax = plt.subplots(figsize=(int(figsize), int(figsize*1.3)))

    # 获取颜色
    cmap = plt.cm.get_cmap('Greys')  # 更多颜色: http://matplotlib.org/examples/color/colormaps_reference.html
    plt_object = ax.imshow(confusion_mat_tmp, cmap=cmap)
    cbar = plt.colorbar(plt_object, ax=ax, fraction=0.03)
    cbar.ax.tick_params(labelsize='12')

    # 设置文字
    xlocations = np.array(range(len(classes)))
    ax.set_xticks(xlocations)
    ax.set_xticklabels(list(classes), rotation=60)  # , fontsize='small'
    ax.set_yticks(xlocations)
    ax.set_yticklabels(list(classes))
    ax.set_xlabel('Predict label')
    ax.set_ylabel('True label')
    ax.set_title("Confusion_Matrix_{}_{}".format(set_name, epoch))

    # 打印数字
    if perc:
        cls_per_nums = confusion_mat.sum(axis=0)
        conf_mat_per = confusion_mat / cls_per_nums
        for i in range(confusion_mat_tmp.shape[0]):
            for j in range(confusion_mat_tmp.shape[1]):
                ax.text(x=j, y=i, s="{:.0%}".format(conf_mat_per[i, j]), va='center', ha='center', color='red',
                         fontsize=10)
    else:
        for i in range(confusion_mat_tmp.shape[0]):
            for j in range(confusion_mat_tmp.shape[1]):
                ax.text(x=j, y=i, s=int(confusion_mat[i, j]), va='center', ha='center', color='red', fontsize=10)
    # 保存
    fig.savefig(os.path.join(out_dir, "Confusion_Matrix_{}.png".format(set_name)))
    plt.close()

    if verbose:
        for i in range(cls_num):
            print('class:{:<10}, total num:{:<6}, correct num:{:<5}  Recall: {:.2%} Precision: {:.2%}'.format(
                classes[i], np.sum(confusion_mat[i, :]), confusion_mat[i, i],
                confusion_mat[i, i] / (1e-9 + np.sum(confusion_mat[i, :])),
                confusion_mat[i, i] / (1e-9 + np.sum(confusion_mat[:, i]))))

    return fig