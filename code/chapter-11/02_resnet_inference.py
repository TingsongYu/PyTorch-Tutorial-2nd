# -*- coding:utf-8 -*-
"""
@file name  : 02_resnet_inference.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-06-03
@brief      : resnet50 推理示例
"""
import json
from PIL import Image
import time
import numpy as np
import onnxruntime as ort
from tqdm import tqdm
import matplotlib.pyplot as plt

print(ort.get_device())


def preprocess(input_data):
    # convert the input data into the float32 input
    img_data = input_data.astype('float32')

    # normalize
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):
        norm_img_data[i, :, :] = (img_data[i, :, :] / 255 - mean_vec[i]) / stddev_vec[i]

    # add batch channel
    norm_img_data = norm_img_data.reshape(1, 3, 224, 224).astype('float32')
    return norm_img_data


def softmax(x):
    x = x.reshape(-1)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def postprocess(result):
    return softmax(np.array(result))


def load_class_names(p_clsnames, p_clsnames_cn):
    """
    加载标签名
    :param p_clsnames:
    :param p_clsnames_cn:
    :return:
    """
    with open(p_clsnames, "r") as f:
        class_names = json.load(f)
    with open(p_clsnames_cn, encoding='UTF-8') as f:  # 设置文件对象
        class_names_cn = f.readlines()
    return class_names, class_names_cn


if __name__ == '__main__':

    path_img = r'G:\deep_learning_data\coco128\images\train2017\000000000081.jpg'
    path_classnames = "imagenet1000.json"
    path_classnames_cn = "imagenet_classnames.txt"

    # load class names
    cls_n, cls_n_cn = load_class_names(path_classnames, path_classnames_cn)
    # 初始化模型
    ort_session = ort.InferenceSession('resnet50_bs_1.onnx', providers=['CUDAExecutionProvider'])

    # 图片读取
    image = Image.open(path_img).resize((224, 224))
    img_rgb = np.array(image)
    image_data = img_rgb.transpose(2, 0, 1)
    input_data = preprocess(image_data)

    # 推理
    raw_result = ort_session.run([], {'input': input_data})
    res = postprocess(raw_result)  # 后处理 softmax

    def topk(array, k=1):
        index = array.argsort()[::-1][:k]
        return index

    top5_idx = topk(res, k=5)

    # 结果可视化
    pred_str, pred_cn = cls_n[top5_idx[0]], cls_n_cn[top5_idx[0]]
    print("img: {} is: {}, {}".format(path_img, pred_str, pred_cn))
    plt.imshow(img_rgb)
    plt.title("predict:{}".format(pred_str))
    text_str = [cls_n[t] for t in top5_idx]
    for idx in range(len(top5_idx)):
        plt.text(5, 15+idx*15, "top {}:{}".format(idx+1, text_str[idx]), bbox=dict(fc='yellow'))
    # plt.savefig("tmp.png")
    plt.show()


