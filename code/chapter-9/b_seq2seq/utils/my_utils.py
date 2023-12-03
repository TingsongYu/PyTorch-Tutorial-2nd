# -*- coding:utf-8 -*-
"""
@file name  : 03_utils.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-06-14
@brief      : 训练所需的函数
"""
from tqdm import tqdm
import random
import numpy as np
import os
import time
import math
import collections
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from datetime import datetime
import logging
# from torchtext.data.metrics import bleu_score


def translate_sentence(model, sentence, german, english, device, max_length=50):
    # spacy_ger = spacy.load("de")

    tokens = [token.lower() for token in sentence]
    tokens.insert(0, german.init_token)
    tokens.append(german.eos_token)

    text_to_indices = [german.vocab.stoi[token] for token in tokens]

    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)  # 获取输入idx

    # Build encoder hidden, cell state
    with torch.no_grad():
        hidden, cell = model.Encoder_LSTM(sentence_tensor)

    outputs = [english.vocab.stoi[""]]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.Decoder_LSTM(previous_word, hidden, cell)
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # Model predicts it's the end of the sentence
        if output.argmax(1).item() == english.vocab.stoi[""]:
            break

    translated_sentence = [english.vocab.itos[idx] for idx in outputs]
    return translated_sentence[1:]


# def bleu(data, model, german, english, device):
#     targets = []
#     outputs = []
#
#     for example in data:
#         src = vars(example)["src"]
#         trg = vars(example)["trg"]
#
#         prediction = translate_sentence(model, src, german, english, device)
#         prediction = prediction[:-1]  # remove  token
#
#         targets.append([trg])
#         outputs.append(prediction)
#
#     return bleu_score(outputs, targets)


def idx2token(idx_matrix, vocab_inv):
    """
    将索引转换为token，并且以一个句子一个list的形式返回。
    :param idx_matrix:
    :param vocab_inv:
    :return:
    """
    sentence_list = []
    for idx_list in idx_matrix:
        sentence = [vocab_inv[idx] for idx in idx_list]
        sentence_list.append(sentence)
    return sentence_list


def bleu(pred_seq, label_seq, k):
    """
    计算一个样本的bleu
    :param pred_seq:
    :param label_seq:
    :param k:
    :return:
    """
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        if len_pred + 1 > n:
            score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score


def bleu_batch(sen_l_trg, sen_l_pred, k=4):
    """
    计算一个batch的bleu
    :param sen_l_trg:
    :param sen_l_pred:
    :param k:
    :return:
    """
    bleu_list = []
    for i in range(len(sen_l_pred)):
        bleu_tmp = bleu(' '.join(sen_l_pred[i]), ' '.join(sen_l_trg[i]), k=k)
        bleu_list.append(bleu_tmp)
    return np.mean(bleu_list)


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""
    # pred的形状：(batch_size,num_steps,vocab_size)
    # label的形状：(batch_size,num_steps)
    # valid_len的形状：(batch_size,)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction = 'none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss


def sequence_mask(tensor_inp, valid_len, value=0):
    """
    遮住后续的元素，让其变为0
    X = torch.tensor([[1, 2, 3], [4, 5, 6]])
    sequence_mask(X, torch.tensor([1, 2]))
    >> tensor([[1, 0, 0],
        [4, 5, 0]])
        第一行，保留1个元素，第二行保留2个元素。

    :param tensor_inp: 原张量矩阵
    :param valid_len: 每一行，保留几个元素
    :param value: 元素填充值
    :return:
    """
    maxlen = tensor_inp.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=tensor_inp.device)[None, :] < valid_len[:, None]
    tensor_inp[~mask] = value
    return tensor_inp


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
    def train_one_epoch(data_loader, model, loss_f, optimizer, scheduler, epoch_idx, device, args, logger):
        model.train()
        end = time.time()

        class_num = len(data_loader.dataset.vocab_fra)  # 目标语言的词表长度

        loss_m = AverageMeter()
        bleu_m = AverageMeter()
        batch_time_m = AverageMeter()

        last_idx = len(data_loader) - 1
        vocab_fra_inv = {value: key for key, value in data_loader.dataset.vocab_fra.items()}
        for batch_idx, data in enumerate(data_loader):
            src, src_len, trg, trg_len = [x.to(device) for x in data]
            bs = src.shape[0]
            # src.shape == [64, 50]
            # trg.shape == [64, 50]
            # bos.shape == [bs, 1]
            bos = torch.tensor([data_loader.dataset.vocab_fra['<bos>']] * trg.shape[0], device=device).reshape(-1, 1)

            # dec_trg.shape == trg.shape
            dec_trg = torch.cat([bos, trg[:, :-1]], 1)  # 用于教师教学

            # forward & backward
            src, dec_trg = src.permute(1, 0), dec_trg.permute(1, 0)
            output = model(src, dec_trg)

            outputs = output[1:, :, :].reshape(-1, output.shape[2])
            trg = dec_trg[1:, :].reshape(-1)

            optimizer.zero_grad()

            # outputs
            loss = loss_f(outputs, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            loss_value = loss.item()

            optimizer.step()

            # 评价指标计算
            # 1. 获取预测的index
            # 2. 将index转换为token字符
            # 3. 调用blue函数计算得到单个样本的token
            output_idx = outputs.argmax(dim=1).cpu().numpy()
            trg_idx = trg.cpu().numpy()
            output_idx = output_idx[:, np.newaxis].reshape(-1, bs)
            trg_idx = trg_idx[:, np.newaxis].reshape(-1, bs)

            output_sentence = idx2token(output_idx, vocab_fra_inv)
            trg_sentence = idx2token(trg_idx, vocab_fra_inv)
            bleu_value = bleu_batch(trg_sentence, output_sentence)
            bleu_m.update(bleu_value, src.size(0))  # 因update里： self.sum += val * n， 因此需要传入batch数量

            # 记录指标
            loss_m.update(loss_value, src.size(0))  # 因update里： self.sum += val * n， 因此需要传入batch数量

            # 打印训练信息
            batch_time_m.update(time.time() - end)
            end = time.time()
            if batch_idx % args.print_freq == args.print_freq - 1:
                logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'BLEU: {bleu_m.val:>7.4f} ({bleu_m.avg:>7.4f}) '.format(
                        "train", batch_idx, last_idx, batch_time=batch_time_m,
                        loss=loss_m, bleu_m=bleu_m))  # val是当次传进去的值，avg是整体平均值。
        return loss_m, bleu_m

    @staticmethod
    def evaluate(data_loader, model, loss_f, device):
        model.eval()

        end = time.time()

        loss_m = AverageMeter()
        bleu_m = AverageMeter()
        batch_time_m = AverageMeter()

        vocab_fra_inv = {value: key for key, value in data_loader.dataset.vocab_fra.items()}

        for batch_idx, data in tqdm(enumerate(data_loader), total=len(data_loader), ):
            src, src_len, trg, trg_len = [x.to(device) for x in data]
            bs, max_step = src.shape[:2]

            src = src.permute(1, 0)
            # Build encoder hidden, cell state
            with torch.no_grad():
                hidden, cell = model.Encoder_LSTM(src)  # bs at dim1, h.shape == [2, 13, 1024]

            outputs = [data_loader.dataset.vocab_fra['<bos>']]  # 1
            for _ in range(max_step):
                previous_word = torch.tensor([outputs[-1]], dtype=torch.int64).to(device)  # shape: [bs, ]
                with torch.no_grad():
                    output, hidden, cell = model.Decoder_LSTM(previous_word, hidden, cell)
                    best_guess = output.argmax(1)[0].item()

                outputs.append(best_guess)

                if best_guess == data_loader.dataset.vocab_fra['<eos>']:
                    break

            translated_sentence = [vocab_fra_inv[idx] for idx in outputs]

            # 评价指标计算
            trg_idx = trg.cpu().numpy()
            trg_sentence = idx2token(trg_idx, vocab_fra_inv)

            bleu_value = bleu_batch(trg_sentence, [translated_sentence])
            bleu_m.update(bleu_value, src.size(0))  # 因update里： self.sum += val * n， 因此需要传入batch数量

            # 打印训练信息
            batch_time_m.update(time.time() - end)
            end = time.time()

        return loss_m, bleu_m


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