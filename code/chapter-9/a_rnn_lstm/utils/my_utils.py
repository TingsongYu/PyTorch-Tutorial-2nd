# -*- coding:utf-8 -*-
"""
@file name  : my_utils.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-06-14
@brief      : 训练所需的函数
"""
import random
import numpy as np
import os
import time

import torchmetrics
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from datetime import datetime
import logging


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.LSTM:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])


def load_glove_vectors(glove_file_path, word2idx):
    """
    加载预训练词向量权重
    :param glove_file_path:
    :param word2idx:
    :return:
    """
    with open(glove_file_path, 'r', encoding='utf-8') as f:
        vectors = {}
        for line in f:
            split = line.split()
            word = split[0]
            if word in word2idx:
                vector = torch.FloatTensor([float(num) for num in split[1:]])
                vectors[word] = vector
        return vectors


def show_conf_mat(confusion_mat, classes, set_name, out_dir, epoch=999, verbose=False, perc=False, save=True):
    """
    混淆矩阵绘制并保存图片
    :param confusion_mat:  nd.array
    :param classes: list or tuple, 类别名称
    :param set_name: str, 数据集名称 train or valid or test?
    :param out_dir:  str, 图片要保存的文件夹
    :param epoch:  int, 第几个epoch
    :param verbose: bool, 是否打印精度信息
    :param perc: bool, 是否采用百分比，图像分割时用，因分类数目过大
    :return:
    """
    cls_num = len(classes)

    # 归一化
    confusion_mat_tmp = confusion_mat.copy()
    for i in range(len(classes)):
        confusion_mat_tmp[i, :] = confusion_mat[i, :] / confusion_mat[i, :].sum()

    # 设置图像大小
    if cls_num < 10:
        figsize = 6
    elif cls_num >= 100:
        figsize = 30
    else:
        figsize = np.linspace(6, 30, 91)[cls_num-10]

    fig, ax = plt.subplots(figsize=(int(figsize), int(figsize*1.3)))

    # 获取颜色
    cmap = plt.cm.get_cmap('Greys')  # 更多颜色: http://matplotlib.org/examples/color/colormaps_reference.html
    plt_object = ax.imshow(confusion_mat_tmp, cmap=cmap)
    cbar = plt.colorbar(plt_object, ax=ax, fraction=0.03)
    cbar.ax.tick_params(labelsize='12')

    # 设置文字
    xlocations = np.array(range(len(classes)))
    ax.set_xticks(xlocations)
    ax.set_xticklabels(list(classes), rotation=60)  # , fontsize='small'
    ax.set_yticks(xlocations)
    ax.set_yticklabels(list(classes))
    ax.set_xlabel('Predict label')
    ax.set_ylabel('True label')
    ax.set_title("Confusion_Matrix_{}_{}".format(set_name, epoch))

    # 打印数字
    if perc:
        cls_per_nums = confusion_mat.sum(axis=0)
        conf_mat_per = confusion_mat / cls_per_nums
        for i in range(confusion_mat_tmp.shape[0]):
            for j in range(confusion_mat_tmp.shape[1]):
                ax.text(x=j, y=i, s="{:.0%}".format(conf_mat_per[i, j]), va='center', ha='center', color='red',
                         fontsize=10)
    else:
        for i in range(confusion_mat_tmp.shape[0]):
            for j in range(confusion_mat_tmp.shape[1]):
                ax.text(x=j, y=i, s=int(confusion_mat[i, j]), va='center', ha='center', color='red', fontsize=10)
    # 保存
    if save:
        fig.savefig(os.path.join(out_dir, "Confusion_Matrix_{}.png".format(set_name)))
    plt.close()

    if verbose:
        for i in range(cls_num):
            print('class:{:<10}, total num:{:<6}, correct num:{:<5}  Recall: {:.2%} Precision: {:.2%}'.format(
                classes[i], np.sum(confusion_mat[i, :]), confusion_mat[i, i],
                confusion_mat[i, i] / (1e-9 + np.sum(confusion_mat[i, :])),
                confusion_mat[i, i] / (1e-9 + np.sum(confusion_mat[:, i]))))

    return fig


class ModelTrainer(object):

    @staticmethod
    def train_one_epoch(data_loader, model, loss_f, optimizer, scheduler, epoch_idx, device, args, logger, classes):
        model.train()
        end = time.time()

        class_num = len(classes)
        conf_mat = np.zeros((class_num, class_num))

        loss_m = AverageMeter()
        top1_m = AverageMeter()
        batch_time_m = AverageMeter()

        last_idx = len(data_loader) - 1
        for batch_idx, data in enumerate(data_loader):

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # forward & backward
            outputs = model(inputs)
            optimizer.zero_grad()

            loss = loss_f(outputs.cpu(), labels.cpu())
            loss.backward()
            optimizer.step()

            # 计算accuracy
            acc1 = accuracy(outputs, labels, topk=(1,))

            _, predicted = torch.max(outputs.data, 1)
            for j in range(len(labels)):
                cate_i = labels[j].cpu().numpy()
                pre_i = predicted[j].cpu().numpy()
                conf_mat[cate_i, pre_i] += 1.

            # 记录指标
            loss_m.update(loss.item(), inputs.size(0))  # 因update里： self.sum += val * n， 因此需要传入batch数量
            top1_m.update(acc1[0].item(), outputs.size(0))

            # 打印训练信息
            batch_time_m.update(time.time() - end)
            end = time.time()
            if batch_idx % args.print_freq == args.print_freq - 1:
                logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f}) '.format(
                        "train", batch_idx, last_idx, batch_time=batch_time_m,
                        loss=loss_m, top1=top1_m))  # val是当次传进去的值，avg是整体平均值。
        return loss_m, top1_m, conf_mat

    @staticmethod
    def evaluate(data_loader, model, loss_f, device, classes):
        model.eval()

        class_num = len(classes)
        conf_mat = np.zeros((class_num, class_num))

        loss_m = AverageMeter()
        top1_m = AverageMeter()

        for i, data in enumerate(data_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_f(outputs.cpu(), labels.cpu())

            # 计算accuracy
            acc1, = accuracy(outputs, labels, topk=(1,))

            _, predicted = torch.max(outputs.data, 1)

            for j in range(len(labels)):
                cate_i = labels[j].cpu().numpy()
                pre_i = predicted[j].cpu().numpy()
                conf_mat[cate_i, pre_i] += 1.

            # 记录指标
            loss_m.update(loss.item(), inputs.size(0))  # 因update里： self.sum += val * n， 因此需要传入batch数量
            top1_m.update(acc1.item(), outputs.size(0))

        return loss_m, top1_m, conf_mat


class ModelTrainerEnsemble(ModelTrainer):
    @staticmethod
    def average(outputs):
        """Compute the average over a list of tensors with the same size."""
        return sum(outputs) / len(outputs)

    @staticmethod
    def evaluate(data_loader, models, loss_f, device, classes):

        class_num = len(classes)
        conf_mat = np.zeros((class_num, class_num))

        loss_m = AverageMeter()
        top1_m = torchmetrics.Accuracy().to(device)

        # top1 acc group
        top1_group = []
        for model_idx in range(len(models)):
            top1_group.append(torchmetrics.Accuracy().to(device))

        for i, data in enumerate(data_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = []
            for model_idx, model in enumerate(models):
                output_single = F.softmax(model(inputs), dim=1)
                outputs.append(output_single)
                # 计算单个模型acc
                top1_group[model_idx](output_single, labels)
                # 计算单个模型loss

            # 计算acc 组
            output_avg = ModelTrainerEnsemble.average(outputs)
            top1_m(output_avg, labels)

            # loss 组
            loss = loss_f(output_avg.cpu(), labels.cpu())
            loss_m.update(loss.item(), inputs.size(0))

        return loss_m, top1_m.compute(), top1_group, conf_mat


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