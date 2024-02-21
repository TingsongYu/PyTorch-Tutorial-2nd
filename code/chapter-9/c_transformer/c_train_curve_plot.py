# -*- coding:utf-8 -*-
"""
@file name  : c_train_curve_plot.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2024-02-21
@brief      : 绘制loss曲线、acc曲线
"""

import os
import sys

BASE_DIR = os.path.dirname(__file__)
PRJ_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(PRJ_DIR)

import os
import matplotlib.pyplot as plt
import pandas as pd


def read_log_file(file_name):
    data = pd.read_csv(file_name, header=0, delimiter=",")
    return data


def plot_data(train_data, valid_data, type="loss"):
    plt.figure(figsize=(10, 5))
    plt.plot(train_data['epoch'], train_data[type], label=f'Train {type}', marker='o')
    plt.plot(valid_data['epoch'], valid_data[type], label=f'Valid {type}', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel(type)
    plt.title(f'{type} vs Epoch')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    path_train_log = os.path.join(BASE_DIR, "result", "train.log")
    path_valid_log = os.path.join(BASE_DIR, "result", "valid.log")

    train_data = read_log_file(path_train_log)
    valid_data = read_log_file(path_valid_log)

    plot_data(train_data, valid_data, 'loss')
    plot_data(train_data, valid_data, 'accuracy')
