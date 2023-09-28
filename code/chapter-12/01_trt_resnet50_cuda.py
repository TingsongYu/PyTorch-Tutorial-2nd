# -*- coding:utf-8 -*-
"""
@file name  : 01_trt_resnet50_cuda.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-08-19
@brief      : resnet50 推理示例。采用cuda库进行数据流管理，要求python版本大于3.7。也是官方后续推荐的形式。
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

    return context, engine

def model_infer(context, engine, img_chw_array):

    # n_io 是I/O的数量，即输入输出的张量个数，在本案例中，resnet有一个输入‘input'，一个输出'output'，因此是2
    # l_tensor_name：即输入输出张量的名字。这里为'input', 'output'
    n_io = engine.num_io_tensors  # since TensorRT 8.5, the concept of Binding is replaced by I/O Tensor, all the APIs with "binding" in their name are deprecated
    l_tensor_name = [engine.get_tensor_name(ii) for ii in range(n_io)]  # get a list of I/O tensor names of the engine, because all I/O tensor in Engine and Excution Context are indexed by name, not binding number like TensorRT 8.4 or before

    # 内存、显存的申请
    h_input = np.ascontiguousarray(img_chw_array)

    batch_size = h_input.shape[0]
    context.set_input_shape("input", [batch_size, 3, 224, 224])  # 绑定输入张量的形状，'data'是input的名字
    # context.set_binding_shape(0, [1, 3, 224, 224])  # 旧的接口，通过int来设置，有时候忘记输入张量的名称，可以用旧接口

    #
    h_output_shape = context.get_tensor_shape(l_tensor_name[1])
    if h_output_shape[0] == -1:
        h_output_shape[0] = h_input.shape[0]
    h_output = np.empty(h_output_shape, dtype=trt.nptype(engine.get_tensor_dtype(l_tensor_name[1])))
    # h_output = np.empty(context.get_tensor_shape(l_tensor_name[1]),
    #                     dtype=trt.nptype(engine.get_tensor_dtype(l_tensor_name[1])))

    d_input = cudart.cudaMalloc(h_input.nbytes)[1]
    d_output = cudart.cudaMalloc(h_output.nbytes)[1]

    # 分配地址
    context.set_tensor_address(l_tensor_name[0], d_input)  # 'input'
    context.set_tensor_address(l_tensor_name[1], d_output)  # 'output'

    # 数据拷贝
    cudart.cudaMemcpy(d_input, h_input.ctypes.data, h_input.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    # 推理
    context.execute_async_v3(0)  # do inference computation

    # 数据拷贝
    cudart.cudaMemcpy(h_output.ctypes.data, d_output, h_output.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    # 释放显存
    cudart.cudaFree(d_input)
    cudart.cudaFree(d_output)

    return h_output


if __name__ == '__main__':

    path_img = r'G:\deep_learning_data\coco128\images\train2017\000000000081.jpg'
    path_classnames = "../chapter-11/imagenet1000.json"
    path_classnames_cn = "../chapter-11/imagenet_classnames.txt"
    # model_path = 'resnet50_bs_1.engine'
    model_path = 'resnet50_bs_dynamic_1-32-64.engine'

    # load class names
    cls_n, cls_n_cn = load_class_names(path_classnames, path_classnames_cn)

    # 图片读取
    image = Image.open(path_img).resize((224, 224))
    img_rgb = np.array(image)
    image_data = img_rgb.transpose(2, 0, 1)
    input_data = preprocess(image_data)

    # 初始化模型
    context, engine = init_model(model_path)

    # 推理
    for i in tqdm(range(2000)):
        h_output = model_infer(context, engine, input_data)
    res = postprocess(h_output)  # 后处理 softmax
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



