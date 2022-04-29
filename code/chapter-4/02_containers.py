# -*- coding:utf-8 -*-
"""
@file name  : 02_containers.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-01-30
@brief      : 熟悉常用容器：sequential, modulelist
"""
import torch
import torch.nn as nn
from torchvision.models import alexnet


if __name__ == "__main__":
    # ========================== Sequential ==========================
    model = alexnet(pretrained=False)
    fake_input = torch.randn((1, 3, 224, 224))
    output = model(fake_input)

    # ========================== ModuleList ==========================

    class MyModule(nn.Module):
        def __init__(self):
            super(MyModule, self).__init__()
            self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])
            # self.linears = [nn.Linear(10, 10) for i in range(10)]    # 观察model._modules，将会是空的

        def forward(self, x):
            for sub_layer in self.linears:
                x = sub_layer(x)
            return x

    model = MyModule()
    fake_input = torch.randn((32, 10))
    output = model(fake_input)
    print(output.shape)

    # ========================== ModuleDict ==========================
    class MyModule2(nn.Module):
        def __init__(self):
            super(MyModule2, self).__init__()
            self.choices = nn.ModuleDict({
                    'conv': nn.Conv2d(3, 16, 5),
                    'pool': nn.MaxPool2d(3)
            })
            self.activations = nn.ModuleDict({
                    'lrelu': nn.LeakyReLU(),
                    'prelu': nn.PReLU()
            })

        def forward(self, x, choice, act):
            x = self.choices[choice](x)
            x = self.activations[act](x)
            return x

    model2 = MyModule2()
    fake_input = torch.randn((1, 3, 7, 7))
    convout = model2(fake_input, "conv", "lrelu")
    poolout = model2(fake_input, "pool", "prelu")
    print(convout.shape, poolout.shape)