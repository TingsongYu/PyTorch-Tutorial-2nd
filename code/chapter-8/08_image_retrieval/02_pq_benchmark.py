# -*- coding:utf-8 -*-
"""
@file name  : 02_pq_benchmark.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-04-29
@brief      : faiss中Product Quantization 算法的性能评估，观察不同子段数量、不同量化bit的速度、recall
结论：8 bit速度快，应该是针对性有优化； 16个字段性能较好，并且时间不会提升太多，32个字段性能更高一点点，但是时间是16子段的一倍！
果然论文中的参数好用，PQ16x8
sift数据下载：http://corpus-texmex.irisa.fr/         sift.tar.gz(161MB)
"""

from __future__ import print_function
import faiss
import pickle
from matplotlib import pyplot as plt

from faiss_datasets import load_sift1M, evaluate


def main():

    # 1 加载数据
    # sift数据下载：http://corpus-texmex.irisa.fr/         sift.tar.gz(161MB)
    xb, xq, xt, gt = load_sift1M()   # 数据下载：http://corpus-texmex.irisa.fr/，并解压与当前文件，修改文件夹为sift1M
    nq, d = xq.shape  # xb是基础数据，xq是查询数据，xt是训练数据，gt是标签

    # 2 创建索引器
    sub_vectors = [4, 8, 16, 32]  # pq.nbits == 8 is supported
    # sub_vectors = [4, 8]  # pq.nbits == 8 is supported
    bits = list(range(6, 10))  # GPU: only pq.nbits == 8 is supported, 本想着对比不同bit的结果，但发现bit==8时gpu才支持
    results_dict = {}
    res = faiss.StandardGpuResources()  # 由于pq除了8bit之外，gpu不支持，因此用cpu进行实验
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0

    for vectors_num in sub_vectors:
        for pq_bit in bits:

            pq_index_string = "PQ{}x{}".format(vectors_num, pq_bit)
            index = faiss.index_factory(d, pq_index_string)  # PQ{}{}分别是子向量个数及量化bit数
            # index = faiss.index_cpu_to_gpu(res, 0, index)
            index.train(xt)
            index.add(xb)
            index.search(xq, 123)  # warmup

            t, r = evaluate(index, xq, gt, 100)  # recall选择100
            results_dict[pq_index_string] = [t, r[1], r[10], r[100]]
            print("{}: {:.3f} ms, recalls = {:.4f}, {:.4f}, {:.4f}".format(pq_index_string, t, r[1], r[10], r[100]))

    with open(path_pkl, 'wb') as f:
        pickle.dump(results_dict, f)


def plot_curve():

    with open(path_pkl, 'rb') as f:
        results_dict = pickle.load(f)

    sub_vectors = [4, 8, 16, 32]  # pq.nbits == 8 is supported
    bits = list(range(6, 10))  # GPU: only pq.nbits == 8 is supported, 本想着对比不同bit的结果，但发现bit==8时gpu才支持

    for idx, vectors_num in enumerate(sub_vectors):

        time_list, r1_l, r10_l, r100_l = [], [], [], []
        for pq_bit in bits:
            pq_index_string = "PQ{}x{}".format(vectors_num, pq_bit)
            t, r1, r10, r100 = results_dict[pq_index_string]
            time_list.append(t)
            r1_l.append(r1)
            r10_l.append(r10)
            r100_l.append(r100)

        x = range(len(bits))
        plt.subplot(2, 2, idx+1)
        plt.plot(x, r1_l, label='r1')
        plt.plot(x, r10_l, label='r10')
        plt.plot(x, r100_l, label='r100')
        plt.legend()
        plt.xticks(x, bits)
        plt.xlabel('pq bits')
        plt.ylabel('recall')
        plt.ylim(0, 1.2)
        plt.title(f'{vectors_num} parts')

    plt.subplots_adjust(wspace=0.5)
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle(f'sub_vectors: {sub_vectors}, bits: {bits}')
    plt.show()

    # 耗时图绘制
    for idx, vectors_num in enumerate(sub_vectors):
        time_list, r1_l, r10_l, r100_l = [], [], [], []
        for pq_bit in bits:
            pq_index_string = "PQ{}x{}".format(vectors_num, pq_bit)
            t, r1, r10, r100 = results_dict[pq_index_string]
            time_list.append(t)

        x = range(len(bits))
        plt.plot(x, time_list, label='PQ{}'.format(vectors_num))
    plt.legend()
    plt.xticks(x, bits)
    plt.xlabel('pq bits')
    plt.ylabel('time pass')
    plt.title(f'time pass')
    plt.show()


if __name__ == '__main__':

    path_pkl = 'results_dict_pq4-8-16-32--6789bit.pkl'

    # main()  # 先推理，获得结果保存为pkl文件
    plot_curve()  # 再绘图，这样节省debug时间


