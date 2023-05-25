# -*- coding:utf-8 -*-
"""
@file name  : 00_lsh_demo.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-04-18
@brief      : LSH(Locality-Sensitive Hashing) 算法demo，有助于理解哈希函数（高斯分布）
"""
import numpy as np
import random


def getHash(v, x, b, w):
    """
    获取哈希值
    :param v:
    :param x:
    :param b:
    :param w:
    :return:
    """
    return (v.dot(x) + b) // w


def dealOneBuket(dataSet):
    """
    将数据采用一个随机的哈希函数进行映射，获取映射后数据
    :param dataSet:
    :return:
    """
    k = dataSet.shape[1]
    b = random.uniform(0, w)
    x = np.random.random(k)
    buket = []
    for data in dataSet:
        h = getHash(data, x, b, w)
        buket.append(h)
    return buket


if __name__ == "__main__":
    dataSet = [[8, 7, 6, 4, 8, 9], [7, 8, 5, 8, 9, 7], [3, 2, 0, 1, 2, 3], [3, 3, 2, 3, 3, 3], [21, 21, 22, 99, 2, 12],
               [1, 1, 1, 0, 1, 0], [1, 1, 1, 1, 1, 0]]
    dataSet = np.array(dataSet)

    w = 4
    hash_funcs_num = 4

    for _ in range(hash_funcs_num):
        print(dealOneBuket(dataSet))

