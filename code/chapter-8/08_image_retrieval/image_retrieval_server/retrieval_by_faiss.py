# -*- coding:utf-8 -*-
"""
@file name  : 01_retrieval_by_faiss.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-04-30
@brief      : 基于faiss构建检索功能模块，包含IndexModule； CLIPModel； ImageRetrievalModule
"""
import os
import shutil
import faiss
import matplotlib.pyplot as plt
import torch
import cv2
import clip
from PIL import Image
import pickle
import time
import os
import numpy as np

from config.base_config import CFG
from my_utils.utils import cv_imread

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

res = faiss.StandardGpuResources()  # 由于pq除了8bit之外，gpu不支持，因此用cpu进行实验
gpu_id = 0


class IndexModule(object):
    """
    索引器，将faiss功能封装到该类中，对外提供基于向量的检索功能
    """
    def __init__(self, index_string, feat_dim, feat_mat):
        self.index_string = index_string
        self.feat_dim = feat_dim
        self.index = None
        self._init_index(feat_mat)

    def _init_index(self, feat_mat):
        """
        初始化faiss索引器， 包括训练及数据添加
        :param feat_mat: ndarray，  shape is N x feat_dim, CLIP的feat_dim是512维， 需要是float32
        :return:
        """
        assert len(feat_mat.shape) == 2, f'feat_mat must be 2 dim, but got {feat_mat.shape}'
        if feat_mat.dtype != np.float32:
            feat_mat = feat_mat.astype(np.float32)
            print(f'feat_mat dtype is not float32, is {feat_mat.dtype}. convert done!')

        index = faiss.index_factory(self.feat_dim, self.index_string)  # PQ{}{}分别是子向量个数及量化bit数
        self.index = faiss.index_cpu_to_gpu(res, gpu_id, index)

        s1 = time.time()
        self.index.train(feat_mat)
        s2 = time.time()

        self.index.add(feat_mat)

        log_info = f'training time:{s2-s1} s, train set:{feat_mat.shape}'
        print(log_info)

    def feat_retrieval(self, feat_query, topk):
        """
        执行特征检索，这里未编写批量查询的代码
        :param feat_query: 1x512的矩阵
        :param topk:
        :return: distance与ids分别是L2距离值， 结果图片的索引序号
        """
        assert feat_query.shape == (1, CFG.feat_dim), f'query vec must be 1x512, but got {feat_query.shape}'
        assert isinstance(topk, int), f'topk should be int, but got {type(topk)}'

        distance, ids = self.index.search(feat_query, topk)  # noqa: E741
        ids = ids.squeeze(0)
        distance = distance.squeeze(0)
        return distance, ids


class CLIPModel(object):
    """
    特征提取器，将clip模型封装到这里，对外提供特征提取功能
    """
    def __init__(self, clip_backbone_type, device):
        self.device = device
        self.clip_backbone_type = clip_backbone_type
        self.model, self.preprocess = clip.load(clip_backbone_type, device=device, jit=False)

    def encode_image_by_path(self, path_img):
        """
        对图像进行编码，接收的是图片路径
        :param path_img:
        :return:
        """
        # read img
        image_bgr = cv_imread(path_img)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = self.preprocess(Image.fromarray(image_rgb)).unsqueeze(0).to(device)
        with torch.no_grad():
            img_feat_vec = self.model.encode_image(image)
            img_feat_vec /= img_feat_vec.norm(dim=-1, keepdim=True)
            img_feat_vec = img_feat_vec.cpu().numpy()  # 1x512向量

        return img_feat_vec

    def encode_image_by_ndarray(self, image_rgb):
        """
        对图像进行编码，接收的是图像数组
        :param img_rgb:
        :return:
        """
        assert image_rgb.ndim == 3, 'image_rgb 必须要是3d-array，但传入的是:{}维的！！'.format(image_rgb.ndim)
        # read img
        image = self.preprocess(Image.fromarray(image_rgb)).unsqueeze(0).to(device)
        with torch.no_grad():
            img_feat_vec = self.model.encode_image(image)
            img_feat_vec /= img_feat_vec.norm(dim=-1, keepdim=True)
            img_feat_vec = img_feat_vec.cpu().numpy()  # 1x512向量

        return img_feat_vec

    def encode_text_by_string(self, text):
        """
        对图像进行编码，接收的是文本描述
        :param text:
        :return:
        """
        token = clip.tokenize([text]).to(self.device)
        # # 一定要Normalization！https://github.com/rom1504/clip-retrieval/blob/main/clip_retrieval/clip_back.py#L226
        feat_text = self.model.encode_text(token)
        feat_text /= feat_text.norm(dim=-1, keepdim=True)
        feat_text = feat_text.detach().cpu().numpy()  # 1x512向量

        return feat_text


class ImageRetrievalModule(object):
    """
    图像检索模块，组装特征提取器、索引器，对外实现query的特征提取、索引
    """
    def __init__(self, index_string, feat_dim, feat_mat, map_dict, backbone, device):
        self.index_model = IndexModule(index_string, feat_dim, feat_mat)
        self.clip_model = CLIPModel(backbone, device)
        self.map_dict = map_dict

    def retrieval_func(self, query_info, topk):
        """
        根据查询信息进行图像检索，可以输入text或者图片路径
        :param query_info:
        :param topk:
        :return:
        """
        if os.path.exists(query_info):
            feat_vec = self.clip_model.encode_image_by_path(query_info)
        elif type(query_info) == str:
            feat_vec = self.clip_model.encode_text_by_string(query_info)
        elif type(query_info) == np.ndarray:
            feat_vec = self.clip_model.encode_image_by_ndarray(query_info)

        feat_vec = feat_vec.astype(np.float32)
        distance_, id_ = self.index_model.feat_retrieval(feat_vec, topk)

        result_path_list = [self.map_dict.get(id_tmp, 'None') for id_tmp in id_]
        return distance_, id_, result_path_list

    def visual_result(self, query_info, distances, ids, path_list):
        """
        绘制检索结果图像
        :param query_info: str, 图片路径或者text
        :param distances:
        :param ids:
        :param path_list:
        :return: None
        """
        plt.figure(figsize=(12, 12))
        subplot_num = int(np.floor(np.sqrt(len(ids))) + 1)

        if os.path.exists(query_info):
            img_ = Image.open(query_info)
            plt.subplot(subplot_num, subplot_num, np.square(subplot_num))
            plt.imshow(img_)
            plt.title('query image')
        else:
            plt.subplot(subplot_num, subplot_num, np.square(subplot_num))
            plt.text(.1, .1, query_info, fontsize=12)

        for ii, (distance, id, path_) in enumerate(zip(distances, ids, path_list)):
            if id == -1:
                continue

            img_ = Image.open(path_)
            plt.subplot(subplot_num, subplot_num, ii + 1)
            plt.imshow(img_)
            plt.title(str(distance))

        plt.subplots_adjust(wspace=0.5)
        plt.subplots_adjust(hspace=0.5)
        plt.show()


def main(query):
    """
    测试图像检索功能。
    前提：image_feature_extract.py已经执行完毕，在data/文件夹下有对应的pkl文件
    :param query: string， text or path_img
    :return:
    """

    with open(CFG.feat_mat_path, 'rb') as f:
        feat_mat = pickle.load(f)
    with open(CFG.map_dict_path, 'rb') as f:
        map_dict = pickle.load(f)

    # 初始化图像检索模块
    ir_model = ImageRetrievalModule(CFG.index_string, CFG.feat_dim, feat_mat, map_dict,
                                    CFG.clip_backbone_type, CFG.device)

    # 调用图像检索功能
    distance_result, index_result, path_list = ir_model.retrieval_func(query, CFG.topk)

    # 可视化结果
    ir_model.visual_result(query, distance_result, index_result, path_list)


if __name__ == '__main__':
    # 设置query，可选文本，或者是图像

    # path_img = '000000000154.jpg'
    # query = path_img
    query = 'an image of a {}'.format('car')  # zebra

    main(query)

