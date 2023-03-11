# -*- coding:utf-8 -*-
"""
@file name  : inference_seg.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-03-05
@brief      : 推理脚本
"""
import os.path
import time
import cv2
import pandas as pd
import torchvision
import torchvision.transforms as transforms
import torch
import imageio
import torch.nn as nn
import matplotlib
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import utils.utils as utils
import platform

if platform.system() == 'Linux':
    matplotlib.use('Agg')


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training", add_help=add_help)
    parser.add_argument("--ckpt-path", default=r"./Result-exp1/2023-03-04_08-34-34/checkpoint_best.pth", type=str, help="ckpt path")
    parser.add_argument("--encoder", default="resnet18", type=str, help="encoder type, eg: resnet18, mobilenet_v2")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("--output-dir", default="./inference_result", type=str, help="path to save outputs")

    return parser


def main(args):
    device = args.device
    result_dir = args.output_dir
    df = pd.read_csv('data_val.csv')
    # ------------------------------------ step1: img preprocess ------------------------------------

    norm_mean = (0.485, 0.456, 0.406)
    norm_std = (0.229, 0.224, 0.225)
    PATCH_SIZE = 256

    valid_transform = A.Compose([
        A.Resize(width=PATCH_SIZE, height=PATCH_SIZE),
        A.Normalize(norm_mean, norm_std, max_pixel_value=255.),
        ToTensorV2(),   # 仅数据转换，不会除以255
    ])

    # ------------------------------------ step2: model init ------------------------------------
    model = smp.Unet(encoder_name=args.encoder, in_channels=3, classes=1)
    state_dict = torch.load(args.ckpt_path)
    model_sate_dict = state_dict['model_state_dict']
    model.load_state_dict(model_sate_dict)  # 模型参数加载

    model.to(device)
    model.eval()
    # ------------------------------------ step3: inference ------------------------------------
    with torch.no_grad():
        ss = time.time()
        for idx in range(len(df)):
            # 读取图片，转换图片
            path_img, path_mask = df.iloc[idx, 1], df.iloc[idx, 2]
            image = utils.cv_imread(path_img)
            mask = utils.cv_imread(path_mask)

            augmented = valid_transform(image=image, mask=image)
            img_tensor = augmented["image"]
            img_tensor = img_tensor.to(device)

            s = time.time()
            img_tensor_batch = img_tensor.unsqueeze(dim=0)
            bs = 1
            img_tensor_batch = img_tensor_batch.repeat(bs, 1, 1, 1)  # 128 or 100 or 1

            # 推理并获取预测概率
            outputs = model(img_tensor_batch)
            outputs_prob = (outputs.sigmoid() > 0.5).float()
            outputs_prob = outputs_prob.squeeze().cpu().numpy().astype('uint8')

            # 可视化
            output_contours, _ = cv2.findContours(outputs_prob, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            mask_contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            if output_contours:
                cv2.drawContours(image, output_contours, -1, (0, 255, 0), 1)
            if mask_contours:
                cv2.drawContours(image, mask_contours, -1, (0, 0, 255), 1)
            cv2.putText(image, "Red   - Ground Ture", (0, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(image, "Green - Model Predict", (0, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)

            utils.makedir(result_dir)
            path_save = os.path.join(result_dir, os.path.basename(path_img))
            utils.cv_imwrite(path_save, image)

            time_c = time.time() - s
            print('\r', 'speed: {:.4f} s/batch, Throughput: {:.0f} frame/s'.format(time_c, 1*bs/time_c), end='')

        print('\n', time.time()-ss)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name()
    print('gpu name: {}'.format(gpu_name))
    main(args)
