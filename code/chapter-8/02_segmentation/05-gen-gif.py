# -*- coding:utf-8 -*-
"""
本代码主要由 chatGPT 编写！
本代码主要由 chatGPT 编写！
本代码主要由 chatGPT 编写！
"""
import imageio
import os
import cv2

# 读取文件夹中的所有图像文件
image_folder = 'gif_images'
images = []
for filename in os.listdir(image_folder):
    img = cv2.imread(os.path.join(image_folder, filename))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images.append(img)

# 生成GIF图像
imageio.mimsave('animation.gif', images, fps=2)
