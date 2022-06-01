# -*- coding:utf-8 -*-
"""
@file name  : 02——optimizer.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-05-31
@brief      : 优化器基类属性与方法
"""
import torch
import torch.optim as optim


if __name__ == "__main__":
    # =================================== 参数组 ========================================
    w1 = torch.randn(2, 2)
    w1.requires_grad = True

    w2 = torch.randn(2, 2)
    w2.requires_grad = True

    w3 = torch.randn(2, 2)
    w3.requires_grad = True

    # 一个参数组
    optimizer_1 = optim.SGD([w1, w3], lr=0.1)
    print('len(optimizer.param_groups): ', len(optimizer_1.param_groups))
    print(optimizer_1.param_groups, '\n')

    # 两个参数组
    optimizer_2 = optim.SGD([{'params': w1, 'lr': 0.1},
                             {'params': w2, 'lr': 0.001}])
    print('len(optimizer.param_groups): ', len(optimizer_2.param_groups))
    print(optimizer_2.param_groups)

    # =================================== zero_grad ========================================
    print("\n\n")
    w1 = torch.randn(2, 2)
    w1.requires_grad = True

    w2 = torch.randn(2, 2)
    w2.requires_grad = True

    optimizer = optim.SGD([w1, w2], lr=0.001, momentum=0.9)

    optimizer.param_groups[0]['params'][0].grad = torch.randn(2, 2)

    print('参数w1的梯度：')
    print(optimizer.param_groups[0]['params'][0].grad, '\n')  # 参数组，第一个参数(w1)的梯度

    optimizer.zero_grad()
    print('执行zero_grad()之后，参数w1的梯度：')
    print(optimizer.param_groups[0]['params'][0].grad)  # 参数组，第一个参数(w1)的梯度
    # ------------------------- state_dict -------------------------
    print("\n\n")
    print("state dict:{}".format(optimizer.state_dict()))

    # =================================== add_param_group ========================================
    print("\n\n")
    w1 = torch.randn(2, 2)
    w1.requires_grad = True

    w2 = torch.randn(2, 2)
    w2.requires_grad = True

    w3 = torch.randn(2, 2)
    w3.requires_grad = True

    # 一个参数组
    optimizer_1 = optim.SGD([w1, w2], lr=0.1)
    print('当前参数组个数: ', len(optimizer_1.param_groups))
    print(optimizer_1.param_groups, '\n')

    # 增加一个参数组
    print('增加一组参数 w3\n')
    optimizer_1.add_param_group({'params': w3, 'lr': 0.001, 'momentum': 0.8})

    print('当前参数组个数: ', len(optimizer_1.param_groups))
    print(optimizer_1.param_groups, '\n')

