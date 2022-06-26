# -*- coding:utf-8 -*-
"""
@file name  : 05_torchmetrics.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-06-26
@brief      : https://github.com/Lightning-AI/metrics 学习
"""


if __name__ == "__main__":
    from my_utils import setup_seed
    setup_seed(40)
    import torch
    import torchmetrics

    metric = torchmetrics.Accuracy()
    n_batches = 3
    for i in range(n_batches):
        preds = torch.randn(10, 5).softmax(dim=-1)
        target = torch.randint(5, (10,))
        acc = metric(preds, target)  # 单次计算，并记录本次信息。通过维护tp, tn, fp, fn来记录所有数据
        print(f"Accuracy on batch {i}: {acc}")

    acc_avg = metric.compute()
    print(f"Accuracy on all data: {acc_avg}")
    tp, tn, fp, fn = metric.tp, metric.tn, metric.fp, metric.fn
    print(tp, tn, fp, fn, sum([tp, tn, fp, fn]))
    metric.reset()




