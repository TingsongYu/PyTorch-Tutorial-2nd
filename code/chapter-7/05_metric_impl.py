# -*- coding:utf-8 -*-
"""
@file name  : 05_metric_impl.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-06-26
@brief      : 自定义 metric
"""

import torch
from torchmetrics import Metric


class MyAccuracy(Metric):
    full_state_update: bool = False

    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        batch_size = target.size(0)
        _, pred = preds.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        self.correct += torch.sum(correct)
        self.total += batch_size

    def compute(self):
        return self.correct.float() / self.total


if __name__ == "__main__":
    from my_utils import setup_seed
    setup_seed(40)
    import torch

    metric = MyAccuracy()
    n_batches = 3
    for i in range(n_batches):
        preds = torch.randn(10, 5).softmax(dim=-1)
        target = torch.randint(5, (10,))
        acc = metric(preds, target)  # 单次计算，会调用.compute()
        print(f"Accuracy on batch {i}: {acc}")

    acc_avg = metric.compute()
    print(f"Accuracy on all data: {acc_avg}")
    metric.reset()





