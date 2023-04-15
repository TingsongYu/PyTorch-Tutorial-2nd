# -*- coding:utf-8 -*-
"""
@file name  : train_script.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-02-04
@brief      : 肺炎Xray图像分类训练脚本
"""
import os
import time
import datetime

import torchvision
import torch
import torch.nn as nn
import albumentations as A
import matplotlib
import torch.optim as optim
import segmentation_models_pytorch as smp
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
import utils.utils as utils
from datasets.brain_mri_dataset import BrainMRIDataset

import platform
if platform.system() == 'Linux':
    matplotlib.use('Agg')


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training", add_help=add_help)
    parser.add_argument("--encoder", default="resnet18", type=str, help="encoder type, eg: resnet18, mobilenet_v2")
    parser.add_argument("--model", default="unet", type=str, help="unet/unetpp/deeplabv3p")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=16, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=100, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 4)"
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
    parser.add_argument("--lr-step-size", default=20, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--print-freq", default=20, type=int, help="print frequency")
    parser.add_argument("--output-dir", default="./Result", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument('--autoaug', action='store_true', default=False, help='use torchvision autoaugment')
    parser.add_argument('--useplateau', action='store_true', default=False, help='use torchvision autoaugment')
    parser.add_argument('--lowlr', action='store_true', default=False, help='encoder lr divided by 10')
    parser.add_argument('--bce', action='store_true', default=False, help='bce loss')

    return parser


def main(args):
    device = args.device
    result_dir = args.output_dir
    # ------------------------------------  log ------------------------------------
    logger, log_dir = utils.make_logger(result_dir)
    writer = SummaryWriter(log_dir=log_dir)
    # ------------------------------------ step1: dataset ------------------------------------

    norm_mean = (0.485, 0.456, 0.406)
    norm_std = (0.229, 0.224, 0.225)
    PATCH_SIZE = 256
    train_transform = A.Compose([
        A.Resize(width=PATCH_SIZE, height=PATCH_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25),
        A.Normalize(norm_mean, norm_std, max_pixel_value=255.),
        ToTensorV2(),
    ])

    valid_transform = A.Compose([
        A.Resize(width=PATCH_SIZE, height=PATCH_SIZE),
        A.Normalize(norm_mean, norm_std, max_pixel_value=255.),
        ToTensorV2(),   # 仅数据转换，不会除以255
    ])

    train_set = BrainMRIDataset(path_train, train_transform)
    valid_set = BrainMRIDataset(path_valid, valid_transform)

    # 构建DataLoder
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    valid_loader = DataLoader(dataset=valid_set, batch_size=args.batch_size, num_workers=args.workers)

    # ------------------------------------ tep2: model ------------------------------------
    if args.model == 'unet':
        model = smp.Unet(encoder_name=args.encoder,  encoder_weights="imagenet",  in_channels=3, classes=1)
    elif args.model == 'unetpp':
        model = smp.UnetPlusPlus(encoder_name=args.encoder,  encoder_weights="imagenet",  in_channels=3, classes=1)
    elif args.model == 'deeplabv3p':
        model = smp.DeepLabV3Plus(encoder_name=args.encoder,  encoder_weights="imagenet",  in_channels=3, classes=1)
    else:
        print('model architecture is not accept, must be unet, unetpp, deeplabv3p')
    model.to(device)

    # ------------------------------------ step3: optimizer, lr scheduler ------------------------------------
    if args.bce:
        criterion = smp.losses.SoftBCEWithLogitsLoss()
    else:
        criterion = smp.losses.DiceLoss(mode='binary')

    if args.lowlr:
        # encoder 学习率小10倍
        encoder_params_id = list(map(id, model.encoder.parameters()))
        base_params = filter(lambda p: id(p) not in encoder_params_id, model.parameters())
        optimizer = optim.SGD([
            {'params': base_params, 'lr': args.lr},  # 0
            {'params': model.encoder.parameters(), 'lr': args.lr * 0.1}],
            momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)  # 选择优化器

    if args.useplateau:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               factor=0.1, patience=10, cooldown=5, mode='max')
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size,
                                            gamma=args.lr_gamma)  # 设置学习率下降策略
    # ------------------------------------ step4: iteration ------------------------------------
    best_miou, best_epoch = 0, 0
    logger.info(args)
    logger.info('model architecture :{}'.format(str(model)))
    logger.info("Start training")
    start_time = time.time()
    epoch_time_m = utils.AverageMeter()
    end = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # 训练
        loss_m_train, miou_m_train, acc_m_train = \
            utils.ModelTrainer.train_one_epoch(train_loader, model, criterion, optimizer, device, args, logger)
        # 验证
        loss_m_valid, miou_m_valid, acc_m_valid = \
            utils.ModelTrainer.evaluate(valid_loader, model, criterion, device)

        epoch_time_m.update(time.time() - end)
        end = time.time()

        lr_current = scheduler.optimizer.param_groups[0]['lr'] if args.useplateau else scheduler.get_last_lr()[0]
        logger.info(
            'Epoch: [{:0>3}/{:0>3}]  '
            'Time: {epoch_time.val:.3f} ({epoch_time.avg:.3f})  '
            'Train Loss avg: {loss_train.avg:>6.4f}  '
            'Valid Loss avg: {loss_valid.avg:>6.4f}  '
            'Train miou avg:  {iou_train.avg:>7.4f}   '
            'Valid miou avg: {iou_valid.avg:>7.4f}    '
            'LR: {lr}'.format(
                epoch, args.epochs, epoch_time=epoch_time_m, loss_train=loss_m_train, loss_valid=loss_m_valid,
                iou_train=miou_m_train, iou_valid=miou_m_valid, lr=lr_current))

        # 学习率更新
        if args.useplateau:
            scheduler.step(miou_m_valid.avg)
        else:
            scheduler.step()
        # 记录
        writer.add_scalars('Loss_group', {'train_loss': loss_m_train.avg,
                                          'valid_loss': loss_m_valid.avg}, epoch)
        writer.add_scalars('miou_group', {'train_miou': miou_m_train.avg,
                                              'valid_miou': miou_m_valid.avg}, epoch) 
        writer.add_scalar('learning rate', lr_current, epoch)

        # ------------------------------------ 模型保存 ------------------------------------
        if best_miou < miou_m_valid.avg:
            best_epoch = epoch if best_miou < miou_m_valid.avg else best_epoch
            best_miou = miou_m_valid.avg if best_miou < miou_m_valid.avg else best_miou
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
                "best_miou": best_miou}
            pkl_name = "checkpoint_{}.pth".format(epoch) if epoch == args.epochs - 1 else "checkpoint_best.pth"
            path_checkpoint = os.path.join(log_dir, pkl_name)
            torch.save(checkpoint, path_checkpoint)
            logger.info(f'save ckpt done! best miou:{best_miou}, epoch:{epoch}')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))
    logger.info("best miou: {}, epoch: {}".format(checkpoint['best_miou'], checkpoint['epoch']))


if __name__ == "__main__":
    # 数据可从 https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation 下载，并通过 01_parse_data.py 解析
    path_train = r"data_train.csv"  # path to your csv
    path_valid = r"data_val.csv"  # path to your csv
    # path_train = r"data_train_split_by_img.csv"  # path to your csv
    # path_valid = r"data_val_split_by_img.csv"  # path to your csv

    args = get_args_parser().parse_args()
    utils.setup_seed(args.random_seed)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(args)





