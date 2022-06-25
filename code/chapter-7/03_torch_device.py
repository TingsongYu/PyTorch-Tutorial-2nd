# -*- coding: utf-8 -*-
"""
# @file name  : 03_torch_device.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2022-06-25
# @brief      : torch.device 使用
"""
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    print("device_count: {}".format(torch.cuda.device_count()))

    print("current_device: ", torch.cuda.current_device())

    print(torch.cuda.get_device_capability(device=None))

    print(torch.cuda.get_device_name())

    print(torch.cuda.is_available())

    print(torch.cuda.get_arch_list())

    print(torch.cuda.get_device_properties(0))

    print(torch.cuda.mem_get_info(device=None))

    print(torch.cuda.memory_summary(device=None, abbreviated=False))

    print(torch.cuda.empty_cache())

