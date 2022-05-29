# -*- coding:utf-8 -*-
"""
@file name  : 01_loss_function.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-05-29
@brief      : loss function实现流程剖析
"""
import torch
import torch.nn as nn


if __name__ == "__main__":
    # ========================== L1 Loss ==========================
    output = torch.ones(2, 2, requires_grad=True) * 0.5
    target = torch.ones(2, 2)

    params = "none mean sum".split()

    for p in params:
        loss_func = nn.L1Loss(reduction=p)
        loss_tmp = loss_func(output, target)
        print("reduction={}:loss={}, shape:{}".format(p, loss_tmp, loss_tmp.shape))

    print("\n")
    # ========================== CrossEntropy Loss ==========================
    import torch
    import torch.nn as nn
    import numpy as np
    import math

    params = "none mean sum".split()

    output = torch.ones(2, 3, requires_grad=True) * 0.5
    target = torch.from_numpy(np.array([0, 1])).type(torch.LongTensor)

    for p in params:
        loss_func = nn.CrossEntropyLoss(reduction=p)
        loss_tmp = loss_func(output, target)
        print("reduction={}:loss={}, shape:{}".format(p, loss_tmp, loss_tmp.shape))

    # ----------------------- 手动计算 ------------------------------------
    # 熟悉计算公式，手动计算第一个样本
    output = output[0].detach().numpy()
    output_1 = output[0]  # 第一个样本的输出值
    target_1 = target[0].numpy()
    # 第一项
    x_class = output[target_1]
    # 第二项
    exp = math.e
    sigma_exp_x = pow(exp, output[0]) + pow(exp, output[1]) + pow(exp, output[2])
    log_sigma_exp_x = math.log(sigma_exp_x)
    # 两项相加
    loss_1 = -x_class + log_sigma_exp_x
    print("\n手动计算，第一个样本的loss:{}".format(loss_1))

    # ----------------------- weight ------------------------------------
    weight = torch.from_numpy(np.array([0.6, 0.2, 0.2])).float()
    loss_f = nn.CrossEntropyLoss(weight=weight, reduction="none")
    output = torch.ones(2, 3, requires_grad=True) * 0.5  # 假设一个三分类任务，batchsize为2个，假设每个神经元输出都为0.5
    target = torch.from_numpy(np.array([0, 1])).type(torch.LongTensor)
    loss = loss_f(output, target)
    print('\n\nCrossEntropy loss: weight')
    print('loss: ', loss)  #
    print('原始loss值为1.0986, 第一个样本是第0类，weight=0.6,所以输出为1.0986*0.6 =', 1.0986 * 0.6)

    # ----------------------- ignore_index ------------------------------------
    loss_f_1 = nn.CrossEntropyLoss(weight=None, reduction="none", ignore_index=1)
    loss_f_2 = nn.CrossEntropyLoss(weight=None, reduction="none", ignore_index=2)

    output = torch.ones(3, 3, requires_grad=True) * 0.5  # 假设一个三分类任务，batchsize为3个，假设每个神经元输出都为0.5
    target = torch.from_numpy(np.array([0, 1, 2])).type(torch.LongTensor)

    loss_1 = loss_f_1(output, target)
    loss_2 = loss_f_2(output, target)

    print('\n\nCrossEntropy loss: ignore_index')
    print('\nignore_index = 1: ', loss_1)  # 类别为1的样本的loss为0
    print('ignore_index = 2: ', loss_2)  # 类别为2的样本的loss为0

