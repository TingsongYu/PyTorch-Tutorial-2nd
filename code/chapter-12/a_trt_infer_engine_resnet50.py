# -*- coding:utf-8 -*-
"""
@file name  : a_trt_infer_engine_resnet50
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-10-01
@brief      : TRT 推理类模板，实现图像分类
"""
import copy

import cv2
import json
import tensorrt as trt
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from cuda import cudart
from typing import List, Tuple


class TensorRTInfer(object):
    """
    基于分类模型的TRT推理类编写
    对外提供inference，实现对np.ndarray的推理
    若需修改为分割、检测等其他任务，只需修改预处理、后处理等模块，中间关于tensorrt的 init_model, model_infer无需修改
    """
    def __init__(self, path_engine: str, mean_vec: List[float], std_vec: List[float], inp_size: Tuple[int, int],
                 path_classnames: str, path_classnames_cn: str):
        self.path_engine = path_engine
        self.mean_vec = np.array(mean_vec)  # [0.485, 0.456, 0.406]
        self.std_vec = np.array(std_vec)  # [0.229, 0.224, 0.225]
        self.inp_size = inp_size

        self.class_names, self.class_names_cn = self.load_class_names(path_classnames, path_classnames_cn)
        self.init_model(path_engine)

    def inference(self, inp_data: np.ndarray, is_vis: bool = True) -> [List[np.ndarray], np.ndarray, str, str]:
        """
        对外推理接口，目前基于resnet50，实现分类任务
        :param inp_data: 待处理的图像矩阵
        :param is_vis: 是否进行可视化
        :return: 模型输出的list（如果模型有多个输出，通过list包装），分类概率，分类名称， 分类名称中文
        """
        inp_data_bak = copy.copy(inp_data)
        # step1：预处理
        inp_data = self.preprocess(inp_data)
        # step2: 推理
        output_list = self.model_inference(inp_data)
        # step3: 后处理
        pred_prob = self.postprocess(output_list[0])  # 后处理 softmax
        top5_idx = self.topk(pred_prob, k=5)
        pred_str, pred_cn = self.class_names[top5_idx[0]], self.class_names_cn[top5_idx[0]]
        # step4: 可视化
        if is_vis:
            self.visualize(inp_data_bak, top5_idx)
        return pred_prob, top5_idx, pred_str, pred_cn

    def preprocess(self, input_data: np.ndarray) -> np.ndarray:
        """
        对输入图像矩阵进行预处理，变为TRT模型可接受的形式。
        包括：resize， 轴变换，数据类型转换，数据标准化
        :param input_data:
        :return:
        """
        # resize
        input_data = cv2.resize(input_data, self.inp_size)
        # transpose: hwc -> chw
        input_data = input_data.transpose(2, 0, 1)
        # to float32
        img_data = input_data.astype('float32')
        # normalize
        norm_img_data = np.zeros(img_data.shape).astype('float32')
        for i in range(img_data.shape[0]):
            norm_img_data[i, :, :] = (img_data[i, :, :] / 255 - self.mean_vec[i]) / self.std_vec[i]
        norm_img_data = norm_img_data.reshape(-1, 3, 224, 224).astype('float32')
        return norm_img_data

    def init_model(self, model_path: str):
        """
        TRT 推理引擎初始化，包括logger， engine， context的创建
        :param model_path:
        :return:
        """

        logger = trt.Logger(trt.Logger.WARNING)  # 创建logger：日志记录器

        # 创建runtime并反序列化生成engine
        with open(model_path, 'rb') as ff, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(ff.read())

        # 创建context，是一个模型的实例，将用于推理
        self.context = engine.create_execution_context()

        # 获取输入、输出命名信息
        n_io = engine.num_io_tensors
        l_tensor_name = [engine.get_tensor_name(ii) for ii in range(n_io)]  # umber like TensorRT 8.4 or before
        n_input = [engine.get_tensor_mode(l_tensor_name[ii]) for ii in range(n_io)].count(trt.TensorIOMode.INPUT)
        n_output = [engine.get_tensor_mode(l_tensor_name[ii]) for ii in range(n_io)].count(trt.TensorIOMode.OUTPUT)

        # 所有变量转self，参考yolov5的common.py。这样初始化属性很方便，少写很多self！
        self.__dict__.update(locals())  # 缺点：IDE找不到变量定义，不方便跳转

    def model_inference(self, img_chw_array: np.ndarray) -> List[np.ndarray]:
        """
        TRT 引擎推理实现。几乎与模型无关。若要修改，注意输入输出的名称。
        本函数hardcode在于self.context.set_input_shape。这里在onnx导出时，输入数据叫做 'input',若要修改，需要关注命名
        :param img_chw_array:
        :return: 返回list，里面包装的是模型输出的各个数据。
        """

        batch_size, c, h, w = img_chw_array.shape

        self.context.set_input_shape("input", [batch_size, 3, 224, 224])  # 绑定输入张量的形状，'data'是input的名字

        # =============================== step1: 初始化host地址（创建数据则自动有内存分配） ===============================
        buffer_h = []
        # 输入数据
        buffer_h.append(np.ascontiguousarray(img_chw_array))
        # 输出数据
        for i in range(self.n_input, self.n_io):
            buffer_h.append(np.empty(shape=self.context.get_tensor_shape(self.l_tensor_name[i]),
                                     dtype=trt.nptype(self.engine.get_tensor_dtype(self.l_tensor_name[i]))))
        # =============================== step2: 初始化device地址（显存空间） ===============================
        buffer_d = []
        for host_data in buffer_h:
            buffer_d.append(cudart.cudaMalloc(host_data.nbytes)[1])  # 申请显存，并获取显存地址

        # =============================== step3: 将数据从host拷贝到device（gpu）中 ===============================
        for i in range(self.n_input):
            cudart.cudaMemcpy(buffer_d[i], buffer_h[i].ctypes.data, buffer_h[i].nbytes,
                              cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

        # =============================== step4: 告知context，输出block需存放到指定device地址 =====================
        #  set address of all input and output data in device buffer
        for i in range(self.n_io):
            self.context.set_tensor_address(self.l_tensor_name[i], int(buffer_d[i]))

        # =============================== step5: context执行推理 =====================
        self.context.execute_async_v3(0)  # 输出数据在device上，推理输出存放在指定的显存中

        # =============================== step6: 从device取结果到host =====================
        h_output = []
        for i in range(self.n_input, self.n_io):
            cudart.cudaMemcpy(buffer_h[i].ctypes.data, buffer_d[i], buffer_h[i].nbytes,
                              cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
            h_output.append(buffer_h[i])

        # =============================== step7: 释放显存 =====================
        for b in buffer_d:  # free the GPU memory buffer after all work
            cudart.cudaFree(b)
        return h_output

    def postprocess(self, result):
        return self.softmax(np.array(result))

    def visualize(self, img_raw: np.ndarray, top5_idx):
        pred_str, pred_cn = self.class_names[top5_idx[0]], self.class_names_cn[top5_idx[0]]
        plt.imshow(img_raw)
        plt.title("predict:{}".format(pred_str))
        text_str = [self.class_names[t] for t in top5_idx]
        for idx in range(len(top5_idx)):
            plt.text(5, 15 + idx * 15, "top {}:{}".format(idx + 1, text_str[idx]), bbox=dict(fc='yellow'))
        plt.show()

    @staticmethod
    def softmax(x):
        x = x.reshape(-1)
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    @staticmethod
    def topk(array, k=1):
        index = array.argsort()[::-1][:k]
        return index

    @staticmethod
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
    path_classnames = "../chapter-11/imagenet1000.json"
    path_classnames_cn = "../chapter-11/imagenet_classnames.txt"
    # model_path = 'resnet50_bs_1.engine'
    model_path = 'resnet50_bs_dynamic_1-32-64.engine'

    # 模型参数配置
    mean_vec = [0.485, 0.456, 0.406]
    std_vec = [0.229, 0.224, 0.225]
    inp_size = (224, 224)

    # 模型实例化
    trt_engine = TensorRTInfer(model_path, mean_vec, std_vec, inp_size, path_classnames, path_classnames_cn)

    # 模型推理
    img = Image.open(path_img)
    img = np.array(img)
    for i in tqdm(range(2000), total=2000):
        pred, top5_idx, pred_str, pred_cn = trt_engine.inference(img, is_vis=False)

    trt_engine.visualize(img, top5_idx)
    print("img: {} is: {}, {}".format(path_img, pred_str, pred_cn))




