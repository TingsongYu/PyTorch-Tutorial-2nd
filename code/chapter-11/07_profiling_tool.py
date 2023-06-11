# -*- coding:utf-8 -*-
"""
@file name  : 07_profiling_tool.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-06-04
@brief      : profiling tool 使用
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


if __name__ == '__main__':

    datasize = 1280

    path_model_float32 = 'resnet50_bs_dynamic.onnx'

    # step1: 设置sess_options
    sess_options = ort.SessionOptions()
    sess_options.enable_profiling = True

    # step2： 实例化session
    ort_session_raw = ort.InferenceSession(path_model_float32, sess_options, providers=['CUDAExecutionProvider'])

    inp = np.random.randn(1, 3, 224, 224).astype(np.float32)

    # warmup
    _ = ort_session_raw.run(['output'], {'input': inp})
    _ = ort_session_raw.run(['output'], {'input': inp})


