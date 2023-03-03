# -*- coding:utf-8 -*-
"""
@file name  : inference_main.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-03-02
@brief      : 推理脚本
"""
import time
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import platform

if platform.system() == 'Linux':
    matplotlib.use('Agg')


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)
    parser.add_argument("--img-path", default=r"../../../data/imgs/person15_virus_46.jpeg", type=str, help="dataset path")
    parser.add_argument("--ckpt-path", default=r"./Result/2023-02-08_16-37-24/checkpoint_best.pth", type=str, help="ckpt path")
    parser.add_argument("--model", default="convnext-tiny", type=str,
                        help="model name; resnet50/convnext/convnext-tiny")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("--output-dir", default="./Result", type=str, help="path to save outputs")

    return parser


def main(args):
    device = args.device
    path_img = args.img_path
    result_dir = args.output_dir
    # ------------------------------------ step1: img preprocess ------------------------------------

    normMean = [0.5]
    normStd = [0.5]
    input_size = (224, 224)
    normTransform = transforms.Normalize(normMean, normStd)

    valid_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        normTransform
    ])

    img = Image.open(path_img).convert('L')
    img_tensor = valid_transform(img)
    img_tensor = img_tensor.to(device)

    # ------------------------------------ step2: model init ------------------------------------
    if args.model == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
    elif args.model == 'convnext':
        model = torchvision.models.convnext_base(pretrained=True)
    elif args.model == 'convnext-tiny':
        model = torchvision.models.convnext_tiny(pretrained=True)
    else:
        print('unexpect model --> :{}'.format(args.model))

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

    state_dict = torch.load(args.ckpt_path)
    model_sate_dict = state_dict['model_state_dict']
    model.load_state_dict(model_sate_dict)  # 模型参数加载

    model.to(device)
    model.eval()
    # ------------------------------------ step3: inference ------------------------------------
    with torch.no_grad():
        ss = time.time()
        for i in range(20):
            s = time.time()
            img_tensor_batch = img_tensor.unsqueeze(dim=0)
            bs = 128
            img_tensor_batch = img_tensor_batch.repeat(bs, 1, 1, 1)  # 128 or 100 or 1
            outputs = model(img_tensor_batch)
            outputs_prob = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs_prob.data, 1)
            pred_idx = predicted.cpu().data.numpy()[0]
            time_c = time.time() - s
            print('\r', 'model predict: {},  speed: {:.4f} s/batch, Throughput: {:.0f} frame/s'.format(
                classes[pred_idx], time_c, 1*bs/time_c), end='')
        print('\n', time.time()-ss)

    # ------------------------------------ step4: visualization ------------------------------------
    plt.imshow(img, cmap='Greys_r')
    plt.title("predict:{}".format(classes[pred_idx]))
    plt.text(50, 50, "predict: {}, probability: {:.1%}".format(
        classes[pred_idx], outputs_prob.cpu().data.numpy()[0, pred_idx]), bbox=dict(fc='yellow'))
    plt.show()


classes = ["NORMAL", "PNEUMONIA"]

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name()
    print('gpu name: {}'.format(gpu_name))
    main(args)
