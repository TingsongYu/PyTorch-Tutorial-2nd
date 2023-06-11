# -*- coding:utf-8 -*-
"""
@file name  : 06_graph_optimization.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-06-04
@brief      : resnet50 计算图优化
"""

import time
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

from tqdm import tqdm
import matplotlib.pyplot as plt

import onnx
from onnxconverter_common import float16

print(ort.get_device())


def speed_test(bs, model, model_name):
    print(f"start: bs {bs}, model_name {model_name}")
    inp = np.random.randn(bs, 3, 224, 224).astype(np.float32)
    if model_name == 'float16':
        inp = inp.astype(np.float16)

    loop_times = datasize / bs

    # warmup
    _ = model.run(['output'], {'input': inp})

    time_s = time.time()
    for i in tqdm(range(int(loop_times))):
        _ = model.run(['output'], {'input': inp})
    time_consumed = time.time() - time_s

    latency = time_consumed / loop_times * 1000
    throughput = 1 / (time_consumed / datasize)

    print("model_name: {} bs: {} latency: {:.1f} ms, throughput: {:.0f} frame / s".format(
        model_name, bs, latency, throughput))
    return latency, throughput


if __name__ == '__main__':

    datasize = 1280

    path_model_float32 = 'resnet50_bs_dynamic.onnx'
    path_model_graph_optim = 'resnet50_bs_dynamic_ENABLE_EXTENDED.onnx'

    # path_model_float32 = 'vgg16_bs_dynamic.onnx'
    # path_model_graph_optim = 'vgg16_bs_dynamic_ENABLE_EXTENDED.onnx'

    # step1: 设置sess_options
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED  # Set level
    # sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    # sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.optimized_model_filepath = path_model_graph_optim  # 保存到磁盘

    # step2： 实例化session
    _ = ort.InferenceSession(path_model_float32, sess_options, providers=['CUDAExecutionProvider'])

    # load model
    ort_session_graph_optim = ort.InferenceSession(path_model_graph_optim, providers=['CUDAExecutionProvider'])
    ort_session_raw = ort.InferenceSession(path_model_float32, providers=['CUDAExecutionProvider'])

    # 测试动态 batch size
    bs_list = list(map(lambda x: 2**x,  range(0, 8)))

    # =================== 需要分开测，因为onnx不会释放显存，导致后面的模型效率变慢！ ===================
    # model_names = ['float32_raw']
    # model_list = [ort_session_raw]
    # model_container = dict(zip(model_names, model_list))

    # =================== 需要分开测，因为onnx不会释放显存，导致后面的模型效率变慢！ ===================
    model_names = ['graph_optim']
    model_list = [ort_session_graph_optim]
    model_container = dict(zip(model_names, model_list))

    for model_name in model_names:
        info_dict = {}
        for bs in bs_list:
            latency, throughput = speed_test(bs, model_container[model_name], model_name)
            info_dict[model_name + str(bs)] = (latency, throughput)

        throughput_list = [v[1] for v in info_dict.values()]
        plt.plot(bs_list, throughput_list, marker='o', linestyle='--')
        for a, b in zip(bs_list, throughput_list):
            plt.text(a, b, f'{b:.2f}', ha='center', va='bottom', fontsize=10)
        plt.title(f'model name:{model_name}, Throughput frame/s')
        plt.show()


