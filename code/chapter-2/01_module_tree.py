# -*- coding:utf-8 -*-
"""
@file name  : 01_module_tree.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2021-12-28
@brief      : PyTorch 模块结构观察
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
print("your torch library in here:{}".format(torch.__path__))
print("your nn library in here:{}".format(nn.__path__))
print("your optim library in here:{}".format(optim.__path__))

# print(type(DataLoader), type(nn))
# print("your DataLoader library in here:{}".format(DataLoader.__path__))  # 思考：这行代码为什么报错？
# print("your Dataset library in here:{}".format(Dataset.__path__))        # 思考：这行代码为什么报错？



