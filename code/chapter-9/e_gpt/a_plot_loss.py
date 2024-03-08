# -*- coding:utf-8 -*-
"""
@file name  : a_plot_loss.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2024-03-02
@brief      : Loss可视化
"""
import matplotlib.pyplot as plt
import numpy as np

path_file = "model/20240307_055849.log"
# 从log文件中提取loss值
loss_values = []
with open(path_file, "r", encoding="utf8") as file:
    lines = file.readlines()
    for line in lines:
        if "loss:" in line:
            loss_value = float(line.split(" ")[3][5:])  # loss:4.49
            loss_values.append(loss_value)

# 绘制折线图
x = np.arange(1, len(loss_values) + 1) * 100
plt.plot(loss_values)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Loss over time')
plt.show()
