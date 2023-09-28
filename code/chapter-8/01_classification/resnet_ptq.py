# -*- coding:utf-8 -*-
"""
@file name  : resnet_ptq.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-09-25
@brief      : 肺炎Xray图像分类模型，resnet50 PTQ 量化
评估未量化前精度：
python resnet_ptq.py --mode evaluate
执行PTQ量化，并保存模型
python resnet_ptq.py --mode quantize --ptq-method max --num-data 512
python resnet_ptq.py --mode quantize --ptq-method entropy --num-data 512
python resnet_ptq.py --mode quantize --ptq-method mse --num-data 512
python resnet_ptq.py --mode quantize --ptq-method percentile --num-data 512

支持4种方法：max entropy mse percentile
https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/userguide.html
"""
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules
from pytorch_quantization import calib
from tqdm import tqdm


import os
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import matplotlib

matplotlib.use('Agg')

import utils.my_utils as utils
from datasets.pneumonia_dataset import PneumoniaDataset


def collect_stats(model, data_loader, num_batches):
    """
    前向传播，获得统计数据，并进行量化
    :param model:
    :param data_loader:
    :param num_batches:
    :return:
    """
    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    # Feed data to the network for collecting stats
    for i, (image, _) in tqdm(enumerate(data_loader), total=len(data_loader)):
        model(image.cuda())
        if i >= num_batches:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()


def compute_amax(model, **kwargs):
    """
    根据统计值，计算amax，确定上限、下限。用于后续计算scale和Z值
    :param model:
    :param kwargs:
    :return:
    """
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
    model.cuda()


def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)
    parser.add_argument("--data-path", default=r"G:\deep_learning_data\chest_xray", type=str, help="dataset path")
    parser.add_argument("--ckpt-path", default=r"./Result/2023-09-26_01-47-40/checkpoint_best.pth", type=str, help="ckpt path")
    parser.add_argument("--model", default="resnet50", type=str,
                        help="model name; resnet50/convnext/convnext-tiny")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("--mode", default="quantize", type=str, help="quantize\evaluate\onnxexport")
    parser.add_argument("--num-data", default=512, type=int, help="量化校准数据batch数量")
    parser.add_argument("--output-dir", default="./Result", type=str, help="path to save outputs")
    parser.add_argument("--ptq-method", type=str, help="method for ptq; max; mse; entropy; percentile")

    return parser


def get_dataloader(args):
    data_dir = args.data_path
    normMean = [0.5]
    normStd = [0.5]
    input_size = (224, 224)
    normTransform = transforms.Normalize(normMean, normStd)
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(input_size, padding=4),
        transforms.ToTensor(),
        normTransform
    ])

    valid_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        normTransform
    ])

    # chest_xray.zip 解压，获得 chest_xray/train, chest_xray/test
    # 数据可从 https://data.mendeley.com/datasets/rscbjbr9sj/2 下载
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'test')
    train_set = PneumoniaDataset(train_dir, transform=train_transform)
    valid_set = PneumoniaDataset(valid_dir, transform=valid_transform)

    # 构建DataLoder
    train_loader = DataLoader(dataset=train_set, batch_size=8, shuffle=True, num_workers=2)
    valid_loader = DataLoader(dataset=valid_set, batch_size=8, num_workers=2)
    return train_loader, valid_loader


def get_model(args, logger, device):
    if args.model == 'resnet50':
        model = torchvision.models.resnet50()
    elif args.model == 'convnext':
        model = torchvision.models.convnext_base()
    elif args.model == 'convnext-tiny':
        model = torchvision.models.convnext_tiny()
    else:
        logger.error('unexpect model --> :{}'.format(args.model))

    model_name = model._get_name()

    if 'ResNet' in model_name:
        # 替换第一层： 因为预训练模型输入是3通道，而本案例是灰度图，输入是1通道
        model.conv1 = nn.Conv2d(1, 64, (7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = model.fc.in_features  # 替换最后一层
        model.fc = nn.Linear(num_ftrs, 2)
    elif 'ConvNeXt' in model_name:
        # 替换第一层： 因为预训练模型输入是3通道，而本案例是灰度图，输入是1通道
        num_kernel = 128 if args.model == 'convnext' else 96
        model.features[0][0] = nn.Conv2d(1, num_kernel, (4, 4), stride=(4, 4))  # convnext base/ tiny
        # 替换最后一层
        num_ftrs = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_ftrs, 2)

    # ------------------------- 加载训练权重
    state_dict = torch.load(args.ckpt_path)
    model_sate_dict = state_dict['model_state_dict']
    model.load_state_dict(model_sate_dict)  # 模型参数加载

    model.to(device)
    return model


def ptq(args):
    """
    进行PTQ量化，并且保存模型
    :param args:
    :return:
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ------------------------------------ step1: dataset ------------------------------------
    train_loader, valid_loader = get_dataloader(args)
    # ------------------------------------ tep2: model ------------------------------------
    model = get_model(args, logger, device)
    # ------------------------------------ step3: 前向推理校准、量化 ------------------------------------
    with torch.no_grad():
        collect_stats(model, train_loader, num_batches=args.num_data)  # 设置量化模块开关，并推理，同时统计激活值

        if args.ptq_method == 'percentile':
            compute_amax(model, method='percentile', percentile=99.9)  # 计算上限、下限，并计算scale 、Z值
        else:
            compute_amax(model, method=args.ptq_method)                     # 计算上限、下限，并计算scale 、Z值
        logger.info('PTQ 量化完成')
    # ------------------------------------ step4: 评估量化后精度  ------------------------------------
    classes = ["NORMAL", "PNEUMONIA"]
    criterion = nn.CrossEntropyLoss()  # 选择损失函数
    loss_m_valid, acc_m_valid, mat_valid = utils.ModelTrainer.evaluate(valid_loader, model, criterion, device, classes)
    logger.info('PTQ量化后模型ACC :{}'.format(acc_m_valid.avg))
    # ------------------------------------ step5: 保存ptq量化后模型 ------------------------------------
    dir_name = os.path.dirname(args.ckpt_path)
    ptq_ckpt_path = os.path.join(dir_name, "resnet50_ptq.pth")
    torch.save(model.state_dict(), ptq_ckpt_path)

    # 导出ONNX
    quant_nn.TensorQuantizer.use_fb_fake_quant = True
    for bs in [1, 32]:
        model_name = "resnet_50_ptq_bs{}_data-num{}_{}_{:.2%}.onnx".format(bs, args.num_data, args.ptq_method, acc_m_valid.avg / 100)
        onnx_path = os.path.join(dir_name, model_name)
        dummy_input = torch.randn(bs, 1, 224, 224, device='cuda')
        torch.onnx.export(model, dummy_input, onnx_path, opset_version=13, do_constant_folding=False,
                          input_names=['input'],  output_names=['output'])


def evaluate(args):
    """
    评估量化前模型精度
    :param args:
    :return:
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_dir = args.output_dir
    # ------------------------------------  log ------------------------------------
    logger, log_dir = utils.make_logger(result_dir)
    # ------------------------------------ step1: dataset ------------------------------------
    train_loader, valid_loader = get_dataloader(args)
    # ------------------------------------ tep2: model ------------------------------------
    model = get_model(args, logger, device)
    # ------------------------------------ step3: evaluate ------------------------------------
    classes = ["NORMAL", "PNEUMONIA"]
    criterion = nn.CrossEntropyLoss()  # 选择损失函数
    loss_m_valid, acc_m_valid, mat_valid =\
        utils.ModelTrainer.evaluate(valid_loader, model, criterion, device, classes)

    logger.info('PTQ量化前模型ACC :{}'.format(acc_m_valid.avg))


def pre_t_model_export(args):
    """
    导出fp32的onnx模型，用于效率对比
    :param args:
    :return:
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args, logger, device)
    dir_name = os.path.dirname(args.ckpt_path)

    for bs in [1, 32]:
        model_name = "resnet_50_fp32_bs{}.onnx".format(bs)
        onnx_path = os.path.join(dir_name, model_name)
        dummy_input = torch.randn(bs, 1, 224, 224, device='cuda')
        torch.onnx.export(model, dummy_input, onnx_path, opset_version=13, do_constant_folding=False,
                          input_names=['input'],  output_names=['output'])
        print('模型保存完成: {}'.format(onnx_path))

def main(args):
    if args.mode == 'quantize':
        quant_modules.initialize()  # 替换torch.nn的常用层，变为可量化的层
        ptq(args)
    elif args.mode == 'evaluate':
        evaluate(args)
    elif args.mode == 'onnxexport':
        pre_t_model_export(args)
    else:
        print("args.mode is not recognize! got :{}".format(args.mode))


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    result_dir = args.output_dir
    logger, log_dir = utils.make_logger(result_dir)

    # 不指定某一种ptq_method，则进行四种量化方法的对比实验
    if args.ptq_method:
        main(args)
    else:
        ptq_method_list = "max entropy mse percentile".split()
        for ptq_method in ptq_method_list:
            args.ptq_method = ptq_method
            main(args)


