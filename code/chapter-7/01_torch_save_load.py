# -*- coding:utf-8 -*-
"""
@file name  : 01_torch_save_load.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-06-022
@brief      : 模型保存与加载
"""
import torch
import torchvision.models as models
from torchinfo import summary


if __name__ == "__main__":

    # ========================================= torch.save ==============================================
    path_state_dict = "resnet50_state_dict_2022.pth"
    resnet_50 = models.resnet50(pretrained=False)

    # 模拟训练，将模型参数进行修改
    print("训练前: ", resnet_50.conv1.weight[0, ...])
    for p in resnet_50.parameters():
        p.data.fill_(2022)
    print("训练后: ", resnet_50.conv1.weight[0, ...])

    # 保存
    net_state_dict = resnet_50.state_dict()
    torch.save(net_state_dict, path_state_dict)
    # ========================================= torch.load ==============================================

    resnet_50_new = models.resnet50(pretrained=False)

    print("初始化: ", resnet_50_new.conv1.weight[0, ...])
    state_dict = torch.load(path_state_dict)
    resnet_50_new.load_state_dict(state_dict)
    print("加载后: ", resnet_50_new.conv1.weight[0, ...])

    # ========================================= torchvision scripts ==============================================
    # https://github.com/pytorch/vision/blob/fa347eb9f38c1759b73677a11b17335191e3f602/references/classification/train.py
    # checkpoint = {
    #     "model": model_without_ddp.state_dict(),
    #     "optimizer": optimizer.state_dict(),
    #     "lr_scheduler": lr_scheduler.state_dict(),
    #     "epoch": epoch,
    # }
    # path_save = "model_{}.pth".format(epoch)
    # torch.save(checkpoint, path_save)
    # # ========================================= resume ==============================================
    # # resume
    # checkpoint = torch.load(path_save, map_location="cpu")
    # model.load_state_dict(checkpoint["model"])
    # optimizer.load_state_dict(checkpoint["optimizer"])
    # lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    # start_epoch = checkpoint["epoch"] + 1







