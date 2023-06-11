# -*- coding:utf-8 -*-
"""
@file name  : 02_onnxruntime_speed_benckmark.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-06-02
@brief      : resnet50 固定bs1， bs128， 动态bs的推理速度评估
使用说明：需要将bs1, bs128 与 动态bs的实验分开跑，即需要运行两次。（在下边手动注释，切换模型）
原因：onnxruntime不会自动释放显存，导致显存不断增长，6G的显存扛不住。
"""

import time
import numpy as np
import onnxruntime as ort
from tqdm import tqdm
import matplotlib.pyplot as plt

print(ort.get_device())


def speed_test(bs, model, model_name):
    print(f"start: bs {bs}, model_name {model_name}")
    inp = np.random.randn(bs, 3, 224, 224).astype(np.float32)
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

    # Load the ONNX model

    ort_session_bs1 = ort.InferenceSession('resnet50_bs_1.onnx', providers=['CUDAExecutionProvider'])
    ort_session_bs128 = ort.InferenceSession('resnet50_bs_128.onnx', providers=['CUDAExecutionProvider'])
    ort_session_dynamic = ort.InferenceSession('resnet50_bs_dynamic.onnx', providers=['CUDAExecutionProvider'])

    # 测试固定 batch size, 由于onnx不会释放显存，所以把3个模型拆开推理
    bs_list = [1, 128]
    model_names = ['bs1', 'bs128']
    model_list = [ort_session_bs1, ort_session_bs128]
    model_container = dict(zip(model_names, model_list))

    # 测试动态 batch size
    # bs_list = list(map(lambda x: 2**x,  range(0, 8)))
    # model_names = ['bs_dynamic']
    # model_list = [ort_session_dynamic]
    # model_container = dict(zip(model_names, model_list))

    info_dict = {}
    for model_name in model_names:
        for bs in bs_list:
            if bs != 1 and model_name == 'bs1':
                continue
            if bs != 128 and model_name == 'bs128':
                continue
            latency, throughput = speed_test(bs, model_container[model_name], model_name)

            info_dict[model_name + str(bs)] = (latency, throughput)

    throughput_list = [v[1] for v in info_dict.values()]
    plt.plot(bs_list, throughput_list, marker='o', linestyle='--')
    for a, b in zip(bs_list, throughput_list):
        plt.text(a, b, f'{b:.2f}', ha='center', va='bottom', fontsize=10)
    plt.title('Throughput frame/s')
    plt.show()


