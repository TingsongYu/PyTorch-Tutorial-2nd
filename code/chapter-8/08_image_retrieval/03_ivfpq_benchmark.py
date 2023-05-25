# -*- coding:utf-8 -*-
"""
@file name  : 02_pq_benchmark.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-04-29
@brief      : faiss中 IVF + Product Quantization 算法的性能，主要观察IVF中不同的probe、聚类中心的耗时和recall
sift数据下载：http://corpus-texmex.irisa.fr/         sift.tar.gz(161MB)
"""
from __future__ import print_function
import faiss
import pickle
from faiss_datasets import load_sift1M, evaluate
from matplotlib import pyplot as plt


def main():

    # 1 加载数据
    # sift数据下载：http://corpus-texmex.irisa.fr/         sift.tar.gz(161MB)
    xb, xq, xt, gt = load_sift1M()   # 数据下载：http://corpus-texmex.irisa.fr/，并解压与当前文件，修改文件夹为sift1M
    nq, d = xq.shape  # xb是基础数据，xq是查询数据，xt是训练数据，gt是标签

    # 2 创建索引器

    results_dict = {}
    res = faiss.StandardGpuResources()  # 由于pq除了8bit之外，gpu不支持，因此用cpu进行实验
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0

    for center_num in center_nums:

        pq_index_string = "IVF{},PQ32x8".format(center_num)
        index = faiss.index_factory(d, pq_index_string)  # PQ{}{}分别是子向量个数及量化bit数
        # index = faiss.index_cpu_to_gpu(res, 0, index)
        index.train(xt)
        index.add(xb)
        index.search(xq, 123)  # warmup

        for probe in probes:
            index.nprobe = probe
            t, r = evaluate(index, xq, gt, 100)  # recall选择100
            results_key = '{}_probe_{}'.format(pq_index_string, probe)
            results_dict[results_key] = [t, r[1], r[10], r[100]]
            print("{}: {:.3f} ms, recalls = {:.4f}, {:.4f}, {:.4f}".format(results_key, t, r[1], r[10], r[100]))

    with open(path_pkl, 'wb') as f:
        pickle.dump(results_dict, f)


def plot_curve():

    with open(path_pkl, 'rb') as f:
        results_dict = pickle.load(f)

    for idx, center_num in enumerate(center_nums):

        time_list_sub, r1_l, r10_l, r100_l = [], [], [], []
        for probe in probes:
            pq_index_string = "IVF{},PQ32x8_probe_{}".format(center_num, probe)
            t, r1, r10, r100 = results_dict[pq_index_string]
            time_list_sub.append(t)
            r1_l.append(r1)
            r10_l.append(r10)
            r100_l.append(r100)

        x = range(len(probes))
        plt.subplot(3, 2, idx+1)
        plt.plot(x, r1_l, label='r1')
        plt.plot(x, r10_l, label='r10')
        plt.plot(x, r100_l, label='r100')
        plt.legend()
        plt.xticks(x, probes)
        plt.xlabel('nprobe')
        plt.ylabel('recall')
        plt.ylim(0, 1.2)
        plt.title(f'{center_num} centroid')

    plt.subplots_adjust(wspace=0.5)
    plt.subplots_adjust(hspace=1)
    plt.suptitle(f'IVFxxxx,PQ32x8  center_nums: {center_nums}')
    plt.show()

    # 耗时统计
    for idx, center_num in enumerate(center_nums):
        time_list, r1_l, r10_l, r100_l = [], [], [], []
        for probe in probes:
            pq_index_string = "IVF{},PQ32x8_probe_{}".format(center_num, probe)
            t, r1, r10, r100 = results_dict[pq_index_string]
            time_list.append(t)
        x = range(len(probes))
        plt.plot(x, time_list, label='IVF{},PQ32x8'.format(center_num))
    plt.legend()
    plt.xticks(x, probes)
    plt.xlabel('nprobe')
    plt.ylabel('time pass')
    plt.title(f'time pass')
    plt.show()

    a = 1


if __name__ == '__main__':

    center_nums = [128, 256, 512, 1024, 2048, 4096]  # pq.nbits == 8 is supported
    # center_nums = [1024, 2048, 3072, 4096]  # pq.nbits == 8 is supported
    probes = [1, 2, 4, 8, 16, 32, 64]

    path_pkl = 'results_dict_ivf{}pq16x8.pkl'

    main()
    # plot_curve()


