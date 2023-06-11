# -*- coding:utf-8 -*-
"""
@file name  : 08_thread_management.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-06-04
@brief      : cpu模式下的，多线程
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

    # step1: 设置sess_options
    sess_options4 = ort.SessionOptions()
    # sess_options4.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    sess_options4.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options4.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # sess_options4.inter_op_num_threads = 4
    sess_options4.intra_op_num_threads = 4

    sess_options2 = ort.SessionOptions()
    sess_options2.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    sess_options2.intra_op_num_threads = 2

    # step2： 实例化session
    ort_session_thread_4 = ort.InferenceSession(path_model_float32, sess_options4, providers=['CPUExecutionProvider'])
    ort_session_thread_2 = ort.InferenceSession(path_model_float32, sess_options2, providers=['CPUExecutionProvider'])

    bs_list = list(map(lambda x: 2 ** x, range(0, 7)))

    model_names = ['inter_4', 'inter_2']
    model_list = [ort_session_thread_4, ort_session_thread_2]
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


