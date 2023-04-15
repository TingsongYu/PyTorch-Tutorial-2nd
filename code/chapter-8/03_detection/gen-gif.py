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
# video_path = './yolov5-master/runs/detect/exp2/DJI_0690.mp4'
video_path = '../tracking/track_video-0048.mp4'
images = []

scale = 1
capture = cv2.VideoCapture(video_path)  # 打开视频

while True:
    _, im = capture.read()  # 读取帧
    if im is None:
        break
    im = cv2.resize(im, None, fx=scale, fy=scale)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # im = im[100:-100, 200:-200, :]
    images.append(im)

# 生成GIF图像
images = images[::3][:32]
imageio.mimsave('animation.gif', images, fps=4)
