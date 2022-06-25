# -*- coding:utf-8 -*-
"""
@file name  : 04_train_script.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-06-25
@brief      : 分类任务训练脚本
"""
import os
import time
import datetime
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import my_utils as utils


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    parser.add_argument("--data-path", default=r"F:\pytorch-tutorial-2nd\data\datasets\cifar10-office", type=str, help="dataset path")
    parser.add_argument("--model", default="resnet8", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=128, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=200, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--random-seed", default=42, type=int, help="random seed")
    parser.add_argument("--lr", default=0.01, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument("--lr-step-size", default=80, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--print-freq", default=100, type=int, help="print frequency")
    parser.add_argument("--output-dir", default="./Result", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")

    return parser


def main(args):
    device = args.device
    data_dir = args.data_path
    result_dir = args.output_dir
    # ------------------------------------  log ------------------------------------
    logger, log_dir = utils.make_logger(result_dir)
    writer = SummaryWriter(log_dir=log_dir)
    # ------------------------------------ step1: dataset ------------------------------------

    normMean = [0.4948052, 0.48568845, 0.44682974]
    normStd = [0.24580306, 0.24236229, 0.2603115]
    normTransform = transforms.Normalize(normMean, normStd)
    train_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        normTransform
    ])

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        normTransform
    ])

    # root变量下需要存放cifar-10-python.tar.gz 文件
    # cifar-10-python.tar.gz可从 "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz" 下载
    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, transform=train_transform, download=True)
    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, transform=valid_transform, download=True)

    # 构建DataLoder
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    valid_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, num_workers=args.workers)

    # ------------------------------------ tep2: model ------------------------------------
    model = utils.resnet8()
    model.to(device)

    # ------------------------------------ step3: optimizer, lr scheduler ------------------------------------
    criterion = nn.CrossEntropyLoss()  # 选择损失函数
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)  # 选择优化器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)  # 设置学习率下降策略

    # ------------------------------------ step4: iteration ------------------------------------
    best_acc, best_epoch = 0, 0
    logger.info(args)
    logger.info("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # 训练
        loss_train, acc_train, mat_train = \
            utils.ModelTrainer.train_one_epoch(train_loader, model, criterion, optimizer, scheduler,
                                               epoch, device, args, logger, classes)
        # 验证
        loss_valid, acc_valid, mat_valid = \
            utils.ModelTrainer.evaluate(valid_loader, model, criterion, device, classes)

        logger.info("Epoch[{:0>3}/{:0>3}] Train Acc: {:.2%} Valid Acc:{:.2%} Train loss:{:.4f} Valid loss:{:.4f} LR:{}". \
                    format(epoch + 1, args.epochs, acc_train, acc_valid, loss_train, loss_valid,
                           optimizer.param_groups[0]["lr"]))

        # 学习率更新
        scheduler.step()
        # 记录
        writer.add_scalars('Loss_group', {'train_loss': loss_train,
                                          'valid_loss': loss_valid}, epoch)
        writer.add_scalars('Accuracy_group', {'train_acc': acc_train,
                                              'valid_acc': acc_valid}, epoch)
        conf_mat_figure_train = utils.show_conf_mat(mat_train, classes, "train", log_dir, epoch=epoch,
                                        verbose=epoch == args.epochs - 1, save=False)
        conf_mat_figure_valid = utils.show_conf_mat(mat_valid, classes, "valid", log_dir, epoch=epoch,
                                        verbose=epoch == args.epochs - 1, save=False)
        writer.add_figure('confusion_matrix_train', conf_mat_figure_train, global_step=epoch)
        writer.add_figure('confusion_matrix_valid', conf_mat_figure_valid, global_step=epoch)
        writer.add_scalar('learning rate', scheduler.get_last_lr()[0], epoch)

        # ------------------------------------ 模型保存 ------------------------------------
        if best_acc < acc_valid or epoch == args.epochs - 1:
            best_epoch = epoch if best_acc < acc_valid else best_epoch
            best_acc = acc_valid if best_acc < acc_valid else best_acc
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

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))


classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    utils.setup_seed(args.random_seed)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(args)





