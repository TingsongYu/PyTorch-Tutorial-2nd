# -*- coding:utf-8 -*-
"""
@file name  : 03_utils.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-06-14
@brief      : 训练所需的函数
"""
import random
import numpy as np
import os
import time
import cv2

import torchmetrics
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from datetime import datetime
import logging
import segmentation_models_pytorch as smp


def makedir(dir_):
    if not os.path.exists(dir_):
        os.makedirs(dir_)


def cv_imread(path_file):
    cv_img = cv2.imdecode(np.fromfile(path_file, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    return cv_img


def cv_imwrite(path_file, img):
    _ = cv2.imencode(".jpg", img)[1].tofile(path_file)
    return True


class ModelTrainer(object):

    @staticmethod
    def train_one_epoch(data_loader, model, loss_f, optimizer, device, args, logger):
        model.train()
        end = time.time()

        loss_m = AverageMeter()
        miou_m = AverageMeter()
        acc_m = AverageMeter()
        batch_time_m = AverageMeter()

        last_idx = len(data_loader) - 1
        for batch_idx, data in enumerate(data_loader):

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # forward & backward
            outputs = model(inputs)
            optimizer.zero_grad()

            if 'BCE' in loss_f._get_name():
                loss = loss_f(outputs.squeeze(), labels.float())
            else:
                loss = loss_f(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

            # 计算miou
            outputs = (outputs.sigmoid() > 0.5).float()
            labels = labels.unsqueeze(dim=1)
            # Shape of the mask should be [bs, num_classes, h, w] ,for binary segmentation num_classes = 1
            tp, fp, fn, tn = smp.metrics.get_stats(outputs.long(), labels, mode="binary")
            # iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")
            iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro-imagewise")  # 由于大量阴性图片的存在，因此macro-imagewise的iou要高，且合理
            acc = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro-imagewise")

            # 记录指标
            loss_m.update(loss.item(), inputs.size(0))  # 因update里： self.sum += val * n， 因此需要传入batch数量
            miou_m.update(iou_score.item(), outputs.size(0))
            acc_m.update(acc.item(), outputs.size(0))

            # 打印训练信息
            batch_time_m.update(time.time() - end)
            end = time.time()
            if batch_idx % args.print_freq == args.print_freq - 1:
                logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'miou: {miou.val:>7.4f} ({miou.avg:>7.4f})  '
                    'acc: {acc.val:>7.4f} ({acc.avg:>7.4f})'.format(
                        "train", batch_idx, last_idx, batch_time=batch_time_m,
                        loss=loss_m, miou=miou_m, acc=acc_m))  # val是当次传进去的值，avg是整体平均值。
        return loss_m, miou_m, acc_m

    @staticmethod
    def evaluate(data_loader, model, loss_f, device):
        model.eval()

        loss_m = AverageMeter()
        miou_m = AverageMeter()
        acc_m = AverageMeter()

        for i, data in enumerate(data_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # forwar
            outputs = model(inputs)

            # outputs_3d = outputs.squeeze()
            if 'BCE' in loss_f._get_name():
                loss = loss_f(outputs.squeeze(), labels.float())
            else:
                loss = loss_f(outputs.squeeze(), labels)

            # 计算miou
            outputs = outputs.sigmoid()
            outputs = (outputs > 0.5).float()
            labels = labels.unsqueeze(dim=1)

            tp, fp, fn, tn = smp.metrics.get_stats(outputs.long(), labels, mode="binary")
            # iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")
            iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro-imagewise")  # 由于大量阴性图片的存在，因此macro-imagewise的iou要高，且合理
            acc = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro-imagewise")

            # 记录指标
            loss_m.update(loss.item(), inputs.size(0))  # 因update里： self.sum += val * n， 因此需要传入batch数量
            miou_m.update(iou_score.item(), outputs.size(0))
            acc_m.update(acc.item(), outputs.size(0))

        return loss_m, miou_m, acc_m



class Logger(object):
    def __init__(self, path_log):
        log_name = os.path.basename(path_log)
        self.log_name = log_name if log_name else "root"
        self.out_path = path_log

        log_dir = os.path.dirname(self.out_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def init_logger(self):
        logger = logging.getLogger(self.log_name)
        logger.setLevel(level=logging.INFO)

        # 配置文件Handler
        file_handler = logging.FileHandler(self.out_path, 'w')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # 配置屏幕Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        # console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

        # 添加handler
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger


def make_logger(out_dir):
    """
    在out_dir文件夹下以当前时间命名，创建日志文件夹，并创建logger用于记录信息
    :param out_dir: str
    :return:
    """
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%Y-%m-%d_%H-%M-%S')
    log_dir = os.path.join(out_dir, time_str)  # 根据config中的创建时间作为文件夹名
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # 创建logger
    path_log = os.path.join(log_dir, "log.log")
    logger = Logger(path_log)
    logger = logger.init_logger()
    return logger, log_dir


def setup_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)     # cpu
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True       # 训练集变化不大时使训练加速，是固定cudnn最优配置，如卷积算法


class AverageMeter:
    """Computes and stores the average and current value
    Hacked from https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/metrics.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k
    Hacked from https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/metrics.py"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]