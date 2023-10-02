# -*- coding:utf-8 -*-
"""
@file name  : 06_trt_resnet50_execute_v2.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-09-30
@brief      : 基于context.execute_v2()进行trt推理
"""
import pycuda.autoinit
import json
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from cuda import cudart
from collections import OrderedDict, namedtuple


def topk(array, k=1):
    index = array.argsort()[::-1][:k]
    return index


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
    # norm_img_data = np.repeat(norm_img_data, 32, axis=0)  # 手动设置多batch 数据
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


def init_model(model_path):

    # 创建logger：日志记录器
    logger = trt.Logger(trt.Logger.WARNING)

    # 创建runtime并反序列化生成engine
    with open(model_path, 'rb') as ff, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(ff.read())

    # 创建context，是一个模型的实例，将用于推理
    context = engine.create_execution_context()
    # context.set_binding_shape(0, [1, 3, 224, 224])
    # context.set_input_shape("input", [1, 3, 224, 224])  # 绑定输入张量的形状，'data'是input的名字

    # 获取输入、输出命名信息
    n_io = engine.num_io_tensors
    l_tensor_name = [engine.get_tensor_name(ii) for ii in range(n_io)]  # umber like TensorRT 8.4 or before
    n_input = [engine.get_tensor_mode(l_tensor_name[ii]) for ii in range(n_io)].count(trt.TensorIOMode.INPUT)
    n_output = [engine.get_tensor_mode(l_tensor_name[ii]) for ii in range(n_io)].count(trt.TensorIOMode.OUTPUT)

    return context, engine, l_tensor_name, n_input, n_output, n_io


def model_infer(context, img_chw_array, l_tensor_name, n_input, n_output, n_io):

    batch_size = img_chw_array.shape[0]
    context.set_input_shape("input", [batch_size, 3, 224, 224])  # 绑定输入张量的形状，'data'是input的名字

    # =============================== step1: 初始化host地址（创建数据则自动有内存分配） ===============================
    buffer_h = []
    # 输入数据
    buffer_h.append(np.ascontiguousarray(img_chw_array))
    # 输出数据
    for i in range(n_input, n_io):
        buffer_h.append(np.empty(shape=context.get_tensor_shape(l_tensor_name[i]),
                                 dtype=trt.nptype(engine.get_tensor_dtype(l_tensor_name[i]))))
    # =============================== step2: 初始化device地址（显存空间） ===============================
    buffer_d = []
    for i in range(n_io):
        buffer_d.append(cudart.cudaMalloc(buffer_h[i].nbytes)[1])  # 申请显存，并获取显存地址

    # =============================== step3: 将数据从host拷贝到device（gpu）中 ===============================
    for i in range(n_input):
        cudart.cudaMemcpy(buffer_d[i], buffer_h[i].ctypes.data, buffer_h[i].nbytes,
                          cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    # =============================== step4: 告知context，输出block需存放到指定device地址 =====================
    #  set address of all input and output data in device buffer
    for i in range(n_io):
        context.set_tensor_address(l_tensor_name[i], int(buffer_d[i]))

    # =============================== step5: context执行推理 =====================
    context.execute_async_v3(0)  # 输出数据在device上，推理输出存放在指定的显存中

    # =============================== step6: 从device取结果到host =====================
    h_output = []
    for i in range(n_input, n_io):
        cudart.cudaMemcpy(buffer_h[i].ctypes.data, buffer_d[i], buffer_h[i].nbytes,
                          cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        h_output.append(buffer_h[i])

    # =============================== step7: 释放显存 =====================
    for b in buffer_d:  # free the GPU memory buffer after all work
        cudart.cudaFree(b)
    return h_output


if __name__ == '__main__':

    path_img = r'G:\deep_learning_data\coco128\images\train2017\000000000081.jpg'
    path_classnames = "../chapter-11/imagenet1000.json"
    path_classnames_cn = "../chapter-11/imagenet_classnames.txt"
    model_path = 'resnet50_bs_1.engine'
    # model_path = 'resnet50_bs_dynamic_1-32-64.engine'

    # load class names
    cls_n, cls_n_cn = load_class_names(path_classnames, path_classnames_cn)

    # 图片读取
    image = Image.open(path_img).resize((224, 224))
    img_rgb = np.array(image)
    image_data = img_rgb.transpose(2, 0, 1)
    input_data = preprocess(image_data)

    # 初始化模型
    context, engine, l_tensor_name, n_input, n_output, n_io = init_model(model_path)

    # 推理
    infer_num = 2000
    for i in tqdm(range(infer_num), total=infer_num):
        h_output = model_infer(context, input_data, l_tensor_name, n_input, n_output, n_io)
    res = postprocess(h_output[0])  # 后处理 softmax
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



