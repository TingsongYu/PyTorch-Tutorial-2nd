# -*- coding:utf-8 -*-
"""
@file name  : train_seq2seq.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-10-30
@brief      : 机器翻译训练代码
"""
import os
import sys
import time
import datetime
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(__file__)
PRJ_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(PRJ_DIR)

from datasets.nmt_en_cmn_dataset import NMTDataset
import utils.my_utils as utils


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    parser.add_argument("--data-path", default=r"G:\deep_learning_data\machine_transfer\cmn-eng", type=str,
                        help="dataset path")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("-b", "--batch-size", default=13, type=int, help="the total batch size is $NGPU x batch_size")
    parser.add_argument("--epochs", default=100, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("-j", "--workers", default=0, type=int, metavar="N", help="number of data loading workers")
    parser.add_argument("--random-seed", default=42, type=int, help="random seed")
    parser.add_argument("--lr", default=0.001, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument("--lr-step-size", default=20, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--print-freq", default=50, type=int, help="print frequency")
    parser.add_argument("--output-dir", default="./result", type=str, help="path to save outputs")
    parser.add_argument('--is-freeze', action='store_true', default=False, help='是否冻结embedding层')

    return parser


if __name__ == "__main__":
    from models.seq2seq import EncoderLSTM, DecoderLSTM, Seq2Seq
    result_dir = os.path.join(BASE_DIR, "result")
    max_len = 20

    encoder_embedding_size = 300
    hidden_size = 1024
    num_layers = 2
    encoder_dropout = 0.5
    decoder_embedding_size = 300
    hidden_size = 1024
    num_layers = 2
    decoder_dropout = 0.5

    args = get_args_parser().parse_args()
    logger, log_dir = utils.make_logger(result_dir)
    writer = SummaryWriter(log_dir=log_dir)

    # ------------------------------------ step1: dataset ------------------------------------
    vocab_path_en = os.path.join(BASE_DIR, 'result', "vocab_en.npy")
    vocab_path_fra = os.path.join(BASE_DIR, 'result', "vocab_cmn.npy")

    root_dir = args.data_path
    path_txt_train = os.path.join(root_dir, 'train.txt')
    path_txt_test = os.path.join(root_dir, 'test.txt')

    train_set = NMTDataset(path_txt_train, vocab_path_en, vocab_path_fra, max_len=max_len)
    test_set = NMTDataset(path_txt_test, vocab_path_en, vocab_path_fra, max_len=max_len)
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    valid_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=args.workers)

    # ------------------------------------ step2: model ------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src_vocab_len, trg_vocab_len = len(train_set.vocab_en), len(train_set.vocab_fra)
    input_size_encoder, input_size_decoder, output_size = src_vocab_len, trg_vocab_len, trg_vocab_len

    encoder_lstm = EncoderLSTM(input_size_encoder, encoder_embedding_size,
                               hidden_size, num_layers, encoder_dropout).to(device)
    decoder_lstm = DecoderLSTM(input_size_decoder, decoder_embedding_size,
                               hidden_size, num_layers, decoder_dropout, output_size).to(device)
    model = Seq2Seq(encoder_lstm, decoder_lstm, device).to(device)

    # model.apply(utils.init_weights)
    model.to(device)

    # ------------------------------------ step3: optimizer, lr scheduler ------------------------------------
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 选择损失函数
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
        loss_m_train, bleu_m_train = \
            utils.ModelTrainer.train_one_epoch(train_loader, model, criterion, optimizer, scheduler,
                                               epoch, device, args, logger)
        # 验证
        loss_m_valid, bleu_m_valid = \
            utils.ModelTrainer.evaluate(valid_loader, model, criterion, device)

        epoch_time_m.update(time.time() - end)
        end = time.time()

        lr_current = scheduler.get_last_lr()[0]
        logger.info(
            'Epoch: [{:0>3}/{:0>3}]  '
            'Time: {epoch_time.val:.3f} ({epoch_time.avg:.3f})  '
            'Train Loss avg: {loss_train.avg:>6.4f}  '
            'Valid Loss avg: {loss_valid.avg:>6.4f}  '
            'Train BLEU avg:  {bleu_train.avg:>7.4f}   '
            'Valid BLEU avg: {bleu_valid.avg:>7.4f}    '
            'LR: {lr}'.format(
                epoch, args.epochs, epoch_time=epoch_time_m, loss_train=loss_m_train, loss_valid=loss_m_valid,
                bleu_train=bleu_m_train, bleu_valid=bleu_m_valid, lr=lr_current))

        # 学习率更新
        scheduler.step()

        # 记录
        writer.add_scalars('Loss_group', {'train_loss': loss_m_train.avg,
                                          'valid_loss': loss_m_valid.avg}, epoch)
        writer.add_scalars('bleu_group', {'train_bleu': bleu_m_train.avg,
                                              'valid_acc': bleu_m_valid.avg}, epoch)
        writer.add_scalar('learning rate', lr_current, epoch)

        # ------------------------------------ 模型保存 ------------------------------------
        if best_acc < bleu_m_valid.avg or epoch == args.epochs - 1:
            best_epoch = epoch if best_acc < bleu_m_valid.avg else best_epoch
            best_acc = bleu_m_valid.avg if best_acc < bleu_m_valid.avg else best_acc
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
                "best_bleu": best_acc}
            pkl_name = "checkpoint_{}.pth".format(epoch) if epoch == args.epochs - 1 else "checkpoint_best.pth"
            path_checkpoint = os.path.join(log_dir, pkl_name)
            torch.save(checkpoint, path_checkpoint)
            logger.info(f'save ckpt done! best acc:{best_acc}, epoch:{epoch}')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))
