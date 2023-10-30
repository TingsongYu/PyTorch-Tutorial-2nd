# -*- coding:utf-8 -*-
"""
@file name  : train_rnn_lstm.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-10-08
@brief      : RNN 文本分类
"""
import os
import sys
import time
import datetime
import numpy as np
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(__file__)
PRJ_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(PRJ_DIR)

from datasets.aclImdb_dataset import AclImdbDataset
from models.rnn import RNNTextClassifier, LSTMTextClassifier
import utils.my_utils as utils


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    parser.add_argument("--data-path", default=r"G:\deep_learning_data\aclImdb_v1\aclImdb", type=str,
                        help="dataset path")
    parser.add_argument("--glove-file-path", default="G:\deep_learning_data\glove.6B\glove.6B.100d.txt", type=str,
                        help="预训练词向量文件")
    parser.add_argument("--model-mode", default="lstm", type=str, help="模型类型，rnn还是lstm")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("-b", "--batch-size", default=64, type=int, help="the total batch size is $NGPU x batch_size")
    parser.add_argument("--epochs", default=100, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers")
    parser.add_argument("--random-seed", default=42, type=int, help="random seed")
    parser.add_argument("--lr", default=0.01, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument("--lr-step-size", default=20, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--print-freq", default=50, type=int, help="print frequency")
    parser.add_argument("--output-dir", default="./Result", type=str, help="path to save outputs")
    parser.add_argument('--is-freeze', action='store_true', default=False, help='是否冻结embedding层')

    return parser


if __name__ == "__main__":
    classes = ['neg', 'pos']
    # root_dir = r'G:\deep_learning_data\aclImdb_v1\aclImdb'
    result_dir = os.path.join(BASE_DIR, "result")
    input_size = 100  # embedding size
    hidden_size = 128  # hidden state size
    num_layers = 2  # RNN层数
    text_max_len = 500  # 一个句子token长度
    cls_num = 2  # 分类类别

    args = get_args_parser().parse_args()
    logger, log_dir = utils.make_logger(result_dir)
    writer = SummaryWriter(log_dir=log_dir)

    # ------------------------------------ step1: dataset ------------------------------------
    vocab_path = os.path.join(BASE_DIR, 'result', "aclImdb_vocab.npy")  # 通过 a_gen_vocabulary.py 获得

    train_set = AclImdbDataset(args.data_path, vocab_path, is_train=True, max_len=text_max_len)
    valid_set = AclImdbDataset(args.data_path, vocab_path, is_train=False, max_len=text_max_len)
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    valid_loader = DataLoader(dataset=valid_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # ------------------------------------ step2: model ------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_mode == "rnn":
        model = RNNTextClassifier(input_size, hidden_size, cls_num, num_layers, vocab_len=len(train_set.vocab),
                                  device=device)
    elif args.model_mode == "lstm":
        model = LSTMTextClassifier(len(train_set.vocab), input_size, hidden_size, num_layers)
    else:
        logger.error(f"model mode is not recognize! got {args.model_mode}")
    model.apply(utils.init_weights)

    if args.glove_file_path:
        word2idx = np.load(vocab_path, allow_pickle=True).item()  # 词表顺序仍旧根据训练集统计得到的词表顺序
        glove_vectors = utils.load_glove_vectors(args.glove_file_path, word2idx)  # 加载存在的token的vector，不存在的token不加载。
        # 将GloVe预训练词向量放到embedding层中
        counter = 0
        for word, idx in word2idx.items():
            if word in glove_vectors:
                model.embedding.weight.data[idx] = glove_vectors[word]
                counter += 1
        # model.embedding.weight.data.copy_(embeds)
        if args.is_freeze:
            model.embedding.weight.requires_grad = False
        logger.info(f'加载了{counter}个预训练词向量.')
    model.to(device)

    # ------------------------------------ step3: optimizer, lr scheduler ------------------------------------
    criterion = nn.CrossEntropyLoss()  # 选择损失函数
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size,
                                                gamma=args.lr_gamma)  # 设置学习率下降策略
    # ------------------------------------ step4: iteration ------------------------------------
    best_acc, best_epoch = 0, 0
    logger.info(args)
    logger.info("Start training")
    start_time = time.time()
    epoch_time_m = utils.AverageMeter()
    end = time.time()
    for epoch in range(args.epochs):
        # 训练
        loss_m_train, acc_m_train, mat_train = \
            utils.ModelTrainer.train_one_epoch(train_loader, model, criterion, optimizer, scheduler,
                                               epoch, device, args, logger, classes)
        # 验证
        loss_m_valid, acc_m_valid, mat_valid = \
            utils.ModelTrainer.evaluate(valid_loader, model, criterion, device, classes)

        epoch_time_m.update(time.time() - end)
        end = time.time()

        lr_current = scheduler.get_last_lr()[0]
        logger.info(
            'Epoch: [{:0>3}/{:0>3}]  '
            'Time: {epoch_time.val:.3f} ({epoch_time.avg:.3f})  '
            'Train Loss avg: {loss_train.avg:>6.4f}  '
            'Valid Loss avg: {loss_valid.avg:>6.4f}  '
            'Train Acc@1 avg:  {top1_train.avg:>7.4f}   '
            'Valid Acc@1 avg: {top1_valid.avg:>7.4f}    '
            'LR: {lr}'.format(
                epoch, args.epochs, epoch_time=epoch_time_m, loss_train=loss_m_train, loss_valid=loss_m_valid,
                top1_train=acc_m_train, top1_valid=acc_m_valid, lr=lr_current))

        # 学习率更新
        scheduler.step()

        # 记录
        writer.add_scalars('Loss_group', {'train_loss': loss_m_train.avg,
                                          'valid_loss': loss_m_valid.avg}, epoch)
        writer.add_scalars('Accuracy_group', {'train_acc': acc_m_train.avg,
                                              'valid_acc': acc_m_valid.avg}, epoch)
        conf_mat_figure_train = utils.show_conf_mat(mat_train, classes, "train", log_dir, epoch=epoch,
                                                    verbose=epoch == args.epochs - 1, save=True)
        conf_mat_figure_valid = utils.show_conf_mat(mat_valid, classes, "valid", log_dir, epoch=epoch,
                                                    verbose=epoch == args.epochs - 1, save=True)
        writer.add_figure('confusion_matrix_train', conf_mat_figure_train, global_step=epoch)
        writer.add_figure('confusion_matrix_valid', conf_mat_figure_valid, global_step=epoch)
        writer.add_scalar('learning rate', lr_current, epoch)

        # ------------------------------------ 模型保存 ------------------------------------
        if best_acc < acc_m_valid.avg or epoch == args.epochs - 1:
            best_epoch = epoch if best_acc < acc_m_valid.avg else best_epoch
            best_acc = acc_m_valid.avg if best_acc < acc_m_valid.avg else best_acc
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
                "best_acc": best_acc}
            pkl_name = "checkpoint_{}.pth".format(epoch) if epoch == args.epochs - 1 else "checkpoint_best.pth"
            path_checkpoint = os.path.join(log_dir, pkl_name)
            torch.save(checkpoint, path_checkpoint)
            logger.info(f'save ckpt done! best acc:{best_acc}, epoch:{epoch}')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))
