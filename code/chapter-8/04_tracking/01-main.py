# -*- coding:utf-8 -*-
"""
@file name  : main.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-03-21
@brief      : 采用yolov5，基于区域撞线机制，实现双向目标计数
"""

import numpy as np
import tracker
import cv2
import copy
from detector import Detector


class BoundaryType(object):
    """
    用于边界区域的mask像素填充，basecae：1和2, 由于用了插值，导致2的边界有一圈1，使得计数出错。
    采用了最近邻插值也会导致问题所在，为此，修改两个边界的索引像素，让它们差距大一些就好
    """
    inner = 68  # 内边界索引，用于矩阵像素赋值。从Outner-->inner，表示进入；蓝色区域
    outer = 168  # 外边界索引； 黄色区域


class CountBoundary(object):
    def __init__(self, point_set, mark_index, color, img_raw_shape, img_in_shape):
        """
        :param point_set:  list， # 边界点集， [(x, y), (x1, y1), ...] 要求是左上角开始，顺时针设置
        :param mark_index:  int，索引用的编号，用于区分是哪一个边界
        :param color:  list， [b, g, r]
        :param img_raw_shape: tuple, (w, h)，创建mask
        :param img_in_shape: tuple, (w, h)，缩放mask尺寸，模型输入输出的尺寸
        """
        self.point_set = point_set
        self.mark_index = mark_index
        self.color = color
        self.img_raw_shape = img_raw_shape
        self.img_in_shape = img_in_shape

        self.id_container = dict()  # 通过字典管理，key是track_id, value是进入边界的总数
        self.total_num = 0

        self._init_mask()

    def _init_mask(self):
        ndarray_pts = np.array(self.point_set, np.int32)
        mask_raw_ = np.zeros((self.img_raw_shape[1], self.img_raw_shape[0]), dtype=np.uint8)
        polygon_line_mask = cv2.fillPoly(mask_raw_, [ndarray_pts], color=self.mark_index)  # 绘制mask
        polygon_line_mask = polygon_line_mask[:, :, np.newaxis]  # 扩充维度
        self.mask = cv2.resize(polygon_line_mask, self.img_in_shape, cv2.INTER_NEAREST)  # 缩放到模型输入尺寸
        self.mask = self.mask[:, :, np.newaxis]

        mask_ = copy.deepcopy(self.mask)
        self.mask_color = np.array(mask_ * self.color, np.uint8)  # 可视化用的

    def register_tracks(self, dets_id_list):
        for track_id in dets_id_list:
            self.add_id(track_id)  # x1, y1, x2, y2, label, track_id = bbox   # string, not int!

    def remove_tracks(self, dets_id_list):
        for track_id in dets_id_list:
            self.del_id(track_id)

    def add_id(self, id_):
        self.total_num += 1
        self.id_container[id_] = self.total_num

    def del_id(self, id_):
        self.id_container.pop(id_)


class BaseCounter(object):
    def __init__(self, point_set, img_raw_shape, img_in_shape):

        self.inner_boundary = CountBoundary(point_set[0], BoundaryType.inner, [255, 0, 0], img_raw_shape, img_in_shape)
        self.outer_boundary = CountBoundary(point_set[1], BoundaryType.outer, [0, 255, 255], img_raw_shape, img_in_shape)

        self.area_mask = self.inner_boundary.mask + self.outer_boundary.mask
        self.color_img = self.inner_boundary.mask_color + self.outer_boundary.mask_color  # 用于外部绘图，size为img_inp

        self.inner_total = 0
        self.outer_total = 0

    def counting(self, tracks):
        if len(tracks) == 0:
            return

        # 获取目标在mask上的像素，0， 1， 2组成的一个list
        index_x = [int((bbox[0]+bbox[2])/2) for bbox in tracks]  # x1, y1, x2, y2, label, track_id
        index_y = [int((bbox[1]+bbox[3])/2) for bbox in tracks]  # x1, y1, x2, y2, label, track_id
        index_yx = (index_y, index_x)  # numpy 是，yx
        bbox_area_list = self.area_mask[index_yx]  # 获取bbox在图像中区域的索引，1,2分别表示在边界区域. [int,]

        # ======================== 先处理inner区域 ====================================
        inner_tracks_currently_ids = self.get_currently_ids_by_area(tracks, bbox_area_list, BoundaryType.inner)
        # ↑这行有问题，为什么id-13的坐标是在outer的，但是返回的索引是1 ？
        outer_tracks_history_ids = list(self.outer_boundary.id_container.keys())  # 获取历史帧经过outer区域的目标的id

        # 当前与历史的交集，认为是目标从outer已经到达inner，可以计数，并且删除。
        outer_2_inner_tracks_id = self.intersection(inner_tracks_currently_ids, outer_tracks_history_ids)
        only_at_inner_tracks_id = self.difference(inner_tracks_currently_ids, outer_tracks_history_ids)
        self.outer_boundary.remove_tracks(outer_2_inner_tracks_id)  # 删除outer中已计数的id
        self.inner_boundary.register_tracks(only_at_inner_tracks_id)  # 注册仅inner有的id

        if len(outer_2_inner_tracks_id):
            self.inner_total += len(outer_2_inner_tracks_id)
            print('inner: {}， append: {}'.format(self.inner_total, outer_2_inner_tracks_id))

        # ======================== 处理outer区域 ====================================
        # 这部分代码是可以再抽象的，让inter与outer共用一个函数，但为了方便理解，就让它重复吧 2023年3月25日20:16:37 by TingsongYu
        outer_tracks_currently_ids = self.get_currently_ids_by_area(tracks, bbox_area_list, BoundaryType.outer)
        inner_tracks_history_ids = list(self.inner_boundary.id_container.keys())  # 获取历史帧经过output区域的目标

        # 当前与历史的交集， 存在则认为目标从inner已经到达outer，可以计数，并且删除。
        inner_2_outer_tracks_id = self.intersection(outer_tracks_currently_ids, inner_tracks_history_ids)
        only_at_outer_tracks_id = self.difference(outer_tracks_currently_ids, inner_tracks_history_ids)
        self.inner_boundary.remove_tracks(inner_2_outer_tracks_id)  # 删除inner中已计数的id
        self.outer_boundary.register_tracks(only_at_outer_tracks_id)  # 注册仅outer有的id

        if len(inner_2_outer_tracks_id):
            self.outer_total += len(inner_2_outer_tracks_id)
            print('outer: {}， append: {}'.format(self.outer_total, inner_2_outer_tracks_id))

    @staticmethod
    def get_currently_ids_by_area(tracks, bbox_area_list_, area_index):
        """
        判断跟踪框列表中，在区域1或2的框，的 track_id， 返回list
        :param tracks: list, 目标跟踪的输出
        :param bbox_area_list_: list, [int,] 目标位置对应于区域索引矩阵的索引，用于判断目标在区域1， 区域2，还是区域0
        :param area_index: int， 用于判断位于区域1，还是2。
        :return: list, [str,]
        """
        area_bbox_index = np.argwhere(bbox_area_list_.squeeze() == area_index).squeeze()  # 进入边界区域 的bbox
        area_tracks = np.array(tracks)[area_bbox_index]
        if len(area_tracks.shape) == 1:
            area_tracks = area_tracks[np.newaxis, :]
        area_tracks_currently_ids = list(area_tracks[:, -1])  # 获取当前帧在output区域的目标
        return area_tracks_currently_ids

    @staticmethod
    def intersection(aa, bb):
        return list(set(aa).intersection(set(bb)))

    @staticmethod
    def difference(aa, bb):
        return list(set(aa).difference(set(bb)))  # a中有，b没有


def main():
    path_video = r'G:\虎门大桥车流\DJI_0049.MP4'
    # path_video = r'G:\DJI_0690.MP4'
    path_output_video = 'track_video.mp4'
    path_yolov5_ckpt = r'F:\pytorch-tutorial-2nd\code\chapter-8\03_detection\yolov5-master\best.pt'
    # outer_point_set = [(1772, 1394), (2088, 1388), (2102, 1494), (1730, 1452)]
    # inner_point_set = [(1696, 1542), (2160, 1548), (2152, 1616), (1664, 1628)]
    # 0048
    # outer_point_set = [(845, 626), (1175, 630), (1175, 661), (838, 648)]
    # inner_point_set = [(822, 736), (1231, 735), (1227, 776), (796, 776)]
    # 0049
    outer_point_set = [(616, 666), (1235, 655), (1245, 715), (600, 701)]
    inner_point_set = [(560, 808), (1238, 812), (1243, 848), (556, 837)]

    capture = cv2.VideoCapture(path_video)  # 打开视频
    w_raw, h_raw = int(capture.get(3)), int(capture.get(4))
    raw_size_wh = (w_raw, h_raw)  # w, h
    in_size_wh = (1280, 720)

    # 获取视频的帧率和帧数
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # 创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path_output_video, fourcc, fps, in_size_wh)

    counter = BaseCounter([inner_point_set, outer_point_set], raw_size_wh, in_size_wh)
    detector = Detector(path_yolov5_ckpt)  # 初始化 yolov5

    draw_text_postion = (int(in_size_wh[0] * 0.01), int(in_size_wh[1] * 0.05))

    while True:
        _, im = capture.read()  # 读取帧
        if im is None:
            break

        # 检测
        im = cv2.resize(im, in_size_wh)  # im为HWC的 ndarray
        bboxes = detector.detect(im)  # bboxes是list，[(坐标(原尺寸), 分类字符串, 概率tensor), ]

        # 跟踪
        if len(bboxes) > 0:
            list_bboxs = tracker.update(bboxes, im)  # 跟踪器跟踪
            output_image_frame = tracker.draw_bboxes(im, list_bboxs, line_thickness=None)  # 画框
        else:
            output_image_frame = im
            list_bboxs = []

        # 撞线计数
        counter.counting(list_bboxs)  # 撞线计数

        # 图片可视化
        text_draw = "In: {}, Out: {}".format(counter.inner_total, counter.outer_total)
        output_image_frame = cv2.add(output_image_frame, counter.color_img)  # 输出图片
        output_image_frame = cv2.putText(img=output_image_frame, text=text_draw,
                                         org=draw_text_postion,
                                         fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                         fontScale=1, color=(255, 255, 255), thickness=2)
        out.write(output_image_frame)
        cv2.imshow('demo', output_image_frame)
        cv2.waitKey(1)

    capture.release()
    out.release()
    cv2.destroyAllWindows()
    print(text_draw)


if __name__ == '__main__':
    main()


