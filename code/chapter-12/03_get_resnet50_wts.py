# -*- coding:utf-8 -*-
"""
@file name  : 03_get_resnet50_wts.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-09-07
@brief      : resnet50 权重文件导出wts文件
参考自：https://github.com/wang-xinyu/pytorchx/tree/master/resnet
"""
import struct
import torch
import torchvision


if __name__ == '__main__':
    path_pth = "resnet50.pth"
    path_wts = "resnet50.wts"

    # step1 加载模型
    net = torchvision.models.resnet50(weights='IMAGENET1K_V1')
    net.eval()
    fake_img = torch.ones(2, 3, 224, 224)
    out = net(fake_img)
    print('resnet50 out:', out.shape)

    # step2 导出wts文件
    f = open(path_wts, 'w')
    f.write("{}\n".format(len(net.state_dict().keys())))
    for k, v in net.state_dict().items():
        print('key: ', k)
        print('value: ', v.shape)
        vr = v.reshape(-1).cpu().numpy()
        f.write("{} {}".format(k, len(vr)))
        for vv in vr:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())
        f.write("\n")
    print(f'wts已经保存于:{path_wts}')


