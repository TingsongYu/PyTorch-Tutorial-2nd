# -*- coding:utf-8 -*-
"""
@file name  : 05_onnx_quantization.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-06-04
@brief      : resnet50 量化效率评估
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
    path_model_float16 = 'resnet50_bs_dynamic_float16.onnx'
    path_model_int8 = 'resnet50_bs_dynamic_int8.onnx'

    # float16
    model = onnx.load(path_model_float32)
    model_fp16 = float16.convert_float_to_float16(model)
    onnx.save(model_fp16, path_model_float16)

    # int8
    ort_session_float32 = ort.InferenceSession(path_model_float32, providers=['CUDAExecutionProvider'])
    _ = quantize_dynamic(path_model_float32, path_model_int8, weight_type=QuantType.QUInt8)

    # load model
    ort_session_float16 = ort.InferenceSession(path_model_float16, providers=['CUDAExecutionProvider'])
    ort_session_int8 = ort.InferenceSession(path_model_int8, providers=['CUDAExecutionProvider'])

    # 测试动态 batch size
    bs_list = list(map(lambda x: 2**x,  range(0, 8)))

    # model_names = ['float32', 'int8']
    # model_list = [ort_session_float32, ort_session_int8]

    model_names = ['float16', 'int8']
    model_list = [ort_session_float16, ort_session_int8]

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


