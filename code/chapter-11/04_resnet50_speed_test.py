# -*- coding:utf-8 -*-
"""
@file name  : resnet50_speed_test
@author     : TingsongYu
@date       : 2023-05-24
@brief      : resnet50 在pytorch框架下 推理速度评估脚本
"""
import time
import pickle
import matplotlib.pyplot as plt
import torchvision
import torch
import numpy as np
import matplotlib
import platform

if platform.system() == 'Linux':
    matplotlib.use('Agg')


def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)
    parser.add_argument('--half', action='store_true', default=False)

    return parser

def main(args):
    # ------------------------------------ step2: model init ------------------------------------
    model = torchvision.models.resnet50(pretrained=False)
    if args.half:
        model.half()
    model.to(device)
    model.eval()
    # ------------------------------------ step3: inference ------------------------------------
    bs_list = list(map(lambda n: 2**n, range(2, 8)))
    speed_list = []
    throughput_list = []
    repeat_num = 10
    for bs in bs_list:
        img_tensor_batch = torch.randn(bs, 3, 224, 224)
        img_tensor_batch = img_tensor_batch.to(device)
        if args.half:
            img_tensor_batch = img_tensor_batch.half()

        with torch.no_grad():
            s = time.time()
            for i in range(repeat_num):
                _ = model(img_tensor_batch)
            time_c = time.time() - s

            speed = time_c/(bs*repeat_num)*1000  # ms
            throughput = (bs*repeat_num)/time_c
            print('bs: {} speed: {:.4f} s/batch, speed:{:.4f} ms/frame Throughput: {:.0f} frame/s'.format(
                bs, time_c/repeat_num, speed, throughput))

        speed_list.append(speed)
        throughput_list.append(throughput)

    # 绘图
    plt.subplot(2, 1, 1)
    plt.plot(bs_list, speed_list, marker='o', linestyle='--')
    for a, b in zip(bs_list, speed_list):
        plt.text(a, b, f'{b:.2f}', ha='center', va='bottom', fontsize=10)
    plt.title('Speed ms/frame')

    # 绘制第二幅图
    plt.subplot(2, 1, 2)
    plt.plot(bs_list, throughput_list, marker='o', linestyle='--')
    for a, b in zip(bs_list, throughput_list):
        plt.text(a, b, f'{b:.2f}', ha='center', va='bottom', fontsize=10)
    plt.title('Throughput frame/s')

    plt.suptitle(f'Resnet50 speed test in {gpu_name} imgsize 224-is half {args.half}')
    plt.subplots_adjust(hspace=0.5)
    # plt.show()
    plt.savefig(f'resnet50-speed-test-half-is-{args.half}.png')


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name()
    print('gpu name: {}'.format(gpu_name))
    main(args)
