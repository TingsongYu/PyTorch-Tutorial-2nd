# -*- coding:utf-8 -*-
"""
@file name  : 02_trt_resnet50_pycuda.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-06-03
@brief      : resnet50 推理示例。cuda数据流采用pycuda进行管理。（python3.7之后推荐用cuda库进行管理！）
              详情见：02_trt_resnet50_cuda.py
"""
import pycuda.autoinit
import json
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm


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


def init_model(model_path):
    # 创建logger：日志记录器
    logger = trt.Logger(trt.Logger.WARNING)
    # 创建runtime并反序列化生成engine
    with open(model_path, 'rb') as ff, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(ff.read())
    # 创建cuda流
    stream = cuda.Stream()
    # 创建context，是一个模型的实例，将用于推理
    context = engine.create_execution_context()
    # 分配CPU锁页内存和GPU显存
    try:
        # TRT v8.5之前可用
        input_shape = context.get_binding_shape(0)
        output_shape = context.get_binding_shape(1)
    except Exception as e:
        print("Exception: ", e)
        print('since TensorRT 8.5, the concept of Binding is replaced by I/O Tensor,'
              ' all the APIs with "binding" in their name are deprecated')
        # since TensorRT 8.5, the concept of Binding is replaced by I/O Tensor,
        # all the APIs with "binding" in their name are deprecated
        n_io = engine.num_io_tensors
        # get a list of I/O tensor names of the engine, because all I/O tensor in Engine and
        # Excution Context are indexed by name, not binding number like TensorRT 8.4 or before
        l_tensor_name = [engine.get_tensor_name(ii) for ii in range(n_io)]  # 获取tensor的名称 ['input', 'output']
        input_shape = context.get_tensor_shape(l_tensor_name[0])  # 输入tensor的名称
        output_shape = context.get_tensor_shape(l_tensor_name[1])  # 输出tensor的名称

    h_input = cuda.pagelocked_empty(trt.volume(input_shape), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)

    return context, h_input, h_output, d_input, d_output, stream


def model_infer(context, h_input, h_output, d_input, d_output, stream, img_chw_array):
    # 图像数据迁到 input buffer
    np.copyto(h_input, img_chw_array.ravel())
    # 数据迁移, H2D
    cuda.memcpy_htod_async(d_input, h_input, stream)
    # 推理
    try:
        context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)  # v8.6.1 接口可用
    except Exception as e:
        print(e)
        print("*"*20)
        print("当前代码仅在trt v8.6.1上通过，更高版本的接口发生变化，已尝试在v10.0.0.6无法运行，请参考官方文档debug，或者采用cuda形式。"
              "当前TRT版本为:{}".format(trt.__version__))
        print("*"*20)
        context.execute_async_v3(stream_handle=stream.handle)  # v10.0.0.6的接口，但是strea_handle的id不正确 todo： 未解决
        context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    # 数据迁移，D2H
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()
    return h_output


if __name__ == '__main__':

    path_img = r'G:\deep_learning_data\coco128\images\train2017\000000000081.jpg'
    path_classnames = "../chapter-11/imagenet1000.json"
    path_classnames_cn = "../chapter-11/imagenet_classnames.txt"
    model_path = 'resnet50_bs_1.engine'  # 推理速度是dynamic batch的一倍左右。
    # model_path = 'resnet50_bs_dynamic_1-32-64.engine'

    # load class names
    cls_n, cls_n_cn = load_class_names(path_classnames, path_classnames_cn)

    # 图片读取
    image = Image.open(path_img).resize((224, 224))
    img_rgb = np.array(image)
    image_data = img_rgb.transpose(2, 0, 1)
    input_data = preprocess(image_data)

    # 初始化模型
    context, h_input, h_output, d_input, d_output, stream = init_model(model_path)

    # 推理
    for i in tqdm(range(3000)):
        h_output = model_infer(context, h_input, h_output, d_input, d_output, stream, input_data)
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



