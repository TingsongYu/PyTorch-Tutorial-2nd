# -*- coding:utf-8 -*-
"""
@file name  : 04_build_resnet50_by_api.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-09-07
@brief      : 采用 TensorRT API 搭建resnet50
参考自：https://github.com/wang-xinyu/tensorrtx/tree/master/resnet
"""
import os
import struct
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


def load_wts_file(file):
    """
    加载wts文件，获得权重字典
    :param file:
    :return:
    """
    print(f"Loading weights: {file}")

    assert os.path.exists(file), 'Unable to load weight file.'

    weight_map = {}
    with open(file, "r") as f:
        lines = [line.strip() for line in f]
    count = int(lines[0])
    assert count == len(lines) - 1
    for i in range(1, count + 1):
        splits = lines[i].split(" ")
        name = splits[0]
        cur_count = int(splits[1])
        assert cur_count + 2 == len(splits)
        values = []
        for j in range(2, len(splits)):
            # hex string to bytes to float
            values.append(struct.unpack(">f", bytes.fromhex(splits[j])))
        weight_map[name] = np.array(values, dtype=np.float32)

    return weight_map


def addBatchNorm2d(network, weight_map, input, layer_name):
    gamma = weight_map[layer_name + ".weight"]
    beta = weight_map[layer_name + ".bias"]
    mean = weight_map[layer_name + ".running_mean"]
    var = weight_map[layer_name + ".running_var"]
    var = np.sqrt(var + 1e-5)

    scale = gamma / var
    shift = -mean / var * gamma + beta
    return network.add_scale(input=input,
                             mode=trt.ScaleMode.CHANNEL,
                             shift=shift,
                             scale=scale)

def bottleneck(network, weight_map, input, in_channels, out_channels, stride, layer_name):

    conv1 = network.add_convolution_nd(input=input,
                                    num_output_maps=out_channels,
                                    kernel_shape=(1, 1),
                                    kernel=weight_map[layer_name +
                                                      "conv1.weight"],
                                    bias=trt.Weights())
    assert conv1

    bn1 = addBatchNorm2d(network, weight_map, conv1.get_output(0),
                         layer_name + "bn1")
    assert bn1

    relu1 = network.add_activation(bn1.get_output(0),
                                   type=trt.ActivationType.RELU)
    assert relu1

    conv2 = network.add_convolution_nd(input=relu1.get_output(0),
                                    num_output_maps=out_channels,
                                    kernel_shape=(3, 3),
                                    kernel=weight_map[layer_name +
                                                      "conv2.weight"],
                                    bias=trt.Weights())
    assert conv2
    conv2.stride_nd = (stride, stride)
    conv2.padding_nd = (1, 1)

    bn2 = addBatchNorm2d(network, weight_map, conv2.get_output(0),
                         layer_name + "bn2")
    assert bn2

    relu2 = network.add_activation(bn2.get_output(0),
                                   type=trt.ActivationType.RELU)
    assert relu2

    conv3 = network.add_convolution_nd(input=relu2.get_output(0),
                                    num_output_maps=out_channels * 4,
                                    kernel_shape=(1, 1),
                                    kernel=weight_map[layer_name +
                                                      "conv3.weight"],
                                    bias=trt.Weights())
    assert conv3

    bn3 = addBatchNorm2d(network, weight_map, conv3.get_output(0),
                         layer_name + "bn3")
    assert bn3

    if stride != 1 or in_channels != 4 * out_channels:
        conv4 = network.add_convolution_nd(
            input=input,
            num_output_maps=out_channels * 4,
            kernel_shape=(1, 1),
            kernel=weight_map[layer_name + "downsample.0.weight"],
            bias=trt.Weights())
        assert conv4
        conv4.stride_nd = (stride, stride)

        bn4 = addBatchNorm2d(network, weight_map, conv4.get_output(0),
                             layer_name + "downsample.1")
        assert bn4

        ew1 = network.add_elementwise(bn4.get_output(0), bn3.get_output(0),
                                      trt.ElementWiseOperation.SUM)
    else:
        ew1 = network.add_elementwise(input, bn3.get_output(0),
                                      trt.ElementWiseOperation.SUM)
    assert ew1

    relu3 = network.add_activation(ew1.get_output(0),
                                   type=trt.ActivationType.RELU)
    assert relu3

    return relu3


def build_model_by_trt_api(network, model_path):
    weight_map = load_wts_file(model_path)  # 权重字典

    data = network.add_input("data", trt.float32, (1, 3, 224, 224))
    assert data

    conv1 = network.add_convolution_nd(input=data,
                                    num_output_maps=64,
                                    kernel_shape=(7, 7),
                                    kernel=weight_map["conv1.weight"],
                                    bias=trt.Weights())
    assert conv1
    conv1.stride_nd = (2, 2)
    conv1.padding_nd = (3, 3)

    bn1 = addBatchNorm2d(network, weight_map, conv1.get_output(0), "bn1")
    assert bn1

    relu1 = network.add_activation(bn1.get_output(0),
                                   type=trt.ActivationType.RELU)
    assert relu1

    pool1 = network.add_pooling_nd(input=relu1.get_output(0),
                                window_size=trt.DimsHW(3, 3),
                                type=trt.PoolingType.MAX)
    assert pool1
    pool1.stride_nd = (2, 2)
    pool1.padding_nd = (1, 1)

    x = bottleneck(network, weight_map, pool1.get_output(0), 64, 64, 1,
                   "layer1.0.")
    x = bottleneck(network, weight_map, x.get_output(0), 256, 64, 1,
                   "layer1.1.")
    x = bottleneck(network, weight_map, x.get_output(0), 256, 64, 1,
                   "layer1.2.")

    x = bottleneck(network, weight_map, x.get_output(0), 256, 128, 2,
                   "layer2.0.")
    x = bottleneck(network, weight_map, x.get_output(0), 512, 128, 1,
                   "layer2.1.")
    x = bottleneck(network, weight_map, x.get_output(0), 512, 128, 1,
                   "layer2.2.")
    x = bottleneck(network, weight_map, x.get_output(0), 512, 128, 1,
                   "layer2.3.")

    x = bottleneck(network, weight_map, x.get_output(0), 512, 256, 2,
                   "layer3.0.")
    x = bottleneck(network, weight_map, x.get_output(0), 1024, 256, 1,
                   "layer3.1.")
    x = bottleneck(network, weight_map, x.get_output(0), 1024, 256, 1,
                   "layer3.2.")
    x = bottleneck(network, weight_map, x.get_output(0), 1024, 256, 1,
                   "layer3.3.")
    x = bottleneck(network, weight_map, x.get_output(0), 1024, 256, 1,
                   "layer3.4.")
    x = bottleneck(network, weight_map, x.get_output(0), 1024, 256, 1,
                   "layer3.5.")

    x = bottleneck(network, weight_map, x.get_output(0), 1024, 512, 2,
                   "layer4.0.")
    x = bottleneck(network, weight_map, x.get_output(0), 2048, 512, 1,
                   "layer4.1.")
    x = bottleneck(network, weight_map, x.get_output(0), 2048, 512, 1,
                   "layer4.2.")

    pool2 = network.add_pooling_nd(x.get_output(0),
                                window_size=trt.DimsHW(7, 7),
                                type=trt.PoolingType.AVERAGE)
    assert pool2
    pool2.stride_nd = (1, 1)

    # --------------------------------- V 10.x 删除了add_fully_connected方法，v8.6.1版本可用 -----------
    # input: <tensorrt.tensorrt.ITensor at 0x2e82f233db0>
    # kernel: nd.array, shape == (2048000, 1)
    # bias : ndarray, shape == (1000, 1)
    # fc1 = network.add_fully_connected(input=pool2.get_output(0),
    #                                   num_outputs=1000,
    #                                   kernel=weight_map['fc.weight'],
    #                                   bias=weight_map['fc.bias'])
    # --------------------------------- V 10.x 删除了add_fully_connected方法，v8.6.1版本可用 -----------

    # ---------------------------------V 10.x 版本删除了add_fully_connected方法--------------------
    # 需要通过add_matrix_multiply实现权重乘法，再通过add_elementwise实现bias的加法
    # 其中还需要设置数据的shape，因此把全连接的操作封装为一个函数add_matmul_as_fc
    # 这个部分参考自TensorRT的github。TRT这方面做得不够友好，在官方api文档中，找不到add_fully_connected被删除的说明及替代方法
    # 记录整个debug过程:
    # 0. AttributeError: 'tensorrt.tensorrt.INetworkDefinition' object has no attribute 'add_fully_connected'
    # 1. 各种文档找'add_fully_connected'，无果；
    # 2. 官方文档查阅 .INetworkDefinition' 支持的方法，的确没了add_fully_connected，猜测肯定有对应方法替换，找到最相关的
    # add_matrix_multiply
    # 3. TRT github仓库搜索add_matrix_multiply， sample.py，发现了FC层实现方法。
    # https://github.com/NVIDIA/TensorRT/blob/c0c633cc629cc0705f0f69359f531a192e524c0f/samples/python/network_api_pytorch_mnist/sample.py

    def add_matmul_as_fc(net, input, outputs, w, b):
        assert len(input.shape) >= 3
        m = 1 if len(input.shape) == 3 else input.shape[0]
        k = int(np.prod(input.shape) / m)  # 输入大小： 2048
        assert np.prod(input.shape) == m * k
        n = int(w.size / k)  # 输出大小： 1000
        assert w.size == n * k
        assert b.size == n

        input_reshape = net.add_shuffle(input)
        input_reshape.reshape_dims = trt.Dims2(m, k)

        filter_const = net.add_constant(trt.Dims2(n, k), w)
        mm = net.add_matrix_multiply(
            input_reshape.get_output(0),
            trt.MatrixOperation.NONE,
            filter_const.get_output(0),
            trt.MatrixOperation.TRANSPOSE,
        )

        bias_const = net.add_constant(trt.Dims2(1, n), b)
        bias_add = net.add_elementwise(mm.get_output(0), bias_const.get_output(0), trt.ElementWiseOperation.SUM)

        output_reshape = net.add_shuffle(bias_add.get_output(0))
        output_reshape.reshape_dims = trt.Dims4(m, n, 1, 1)
        return output_reshape
    fc_w = weight_map["fc.weight"]
    fc_b = weight_map["fc.bias"]
    fc1 = add_matmul_as_fc(network, pool2.get_output(0), 1000, fc_w, fc_b)

    # 实现加法 +bias

    assert fc1

    fc1.get_output(0).name = "prob"
    network.mark_output(fc1.get_output(0))

    return network


def init_model(model_path):

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = build_model_by_trt_api(network, model_path)
    config = builder.create_builder_config()
    # -------------------- V 10.x 以前 可用build_engine方法 -------------------
    # engine = builder.build_engine(network, config)
    # -------------------- V 10.x 以前 可用build_engine方法 -------------------

    # -------------------- V 10.x 删除了 build_engine方法 -------------------
    # 参考自：    # https://github.com/NVIDIA/TensorRT/blob/c0c633cc629cc0705f0f69359f531a192e524c0f/samples/python/network_api_pytorch_mnist/sample.py
    # 替代方式如下：
    plan = builder.build_serialized_network(network, config)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    context.set_input_shape("data", [1, 3, 224, 224])  # 绑定输入张量的形状

    return context, engine

def model_infer(context, engine, img_chw_array):

    # n_io 是I/O的数量，即输入输出的张量个数，在本案例中，resnet有一个输入‘input'，一个输出'output'，因此是2
    # l_tensor_name：即输入输出张量的名字。这里为'input', 'output'
    n_io = engine.num_io_tensors  # since TensorRT 8.5, the concept of Binding is replaced by I/O Tensor, all the APIs with "binding" in their name are deprecated
    l_tensor_name = [engine.get_tensor_name(ii) for ii in range(n_io)]  # get a list of I/O tensor names of the engine, because all I/O tensor in Engine and Excution Context are indexed by name, not binding number like TensorRT 8.4 or before

    # 内存、显存的申请
    h_input = np.ascontiguousarray(img_chw_array)

    batch_size = h_input.shape[0]
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
    model_path = 'resnet50.wts'

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



