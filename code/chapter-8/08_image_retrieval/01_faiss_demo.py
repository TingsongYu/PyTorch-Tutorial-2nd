# -*- coding:utf-8 -*-
"""
@file name  : 00_lsh_demo.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-04-21
@brief      : faiss库安装后的初步调用
"""
import time
import numpy as np
import faiss

# ============================ step 0: 数据构建 ============================

np.random.seed(1234)

d = 64                           # dimension
nb = 100000                      # database size
nq = 10000                       # nb of queries
xb = np.random.random((nb, d)).astype('float32')
xq = np.random.random((nq, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
xq[:, 0] += np.arange(nq) / 1000.

# ============================ step 1: 构建索引器 ============================
index = faiss.IndexFlatL2(d)
index.add(xb)

# ============================ step 2: 索引 ============================
k = 4  # top_k number
for i in range(5):
    s = time.time()
    D, I = index.search(xq, k)
    print("{}*{}量级的精确检索，耗时:{:.3f}s".format(nb, nq, time.time()-s))

# ============================ step 3: 检查索引结果 ============================
print('D.shape: {}, D[0, ...]: {}'.format(D.shape, D[0]))
print('I.shape: {}, I[0, ...]: {}'.format(I.shape, I[0]))
# D是查询向量与topk向量的距离，distance
# I是与查询向量最近的向量的id，此处有10万数据，index在0-99999之间。

# ============================ step 4: gpu 加速 ============================
res = faiss.StandardGpuResources()  # 1. 获取gpu资源
index_flat = faiss.IndexFlatL2(d)  # 2. 创建cpu索引器
gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)  # 3. 迁移至gpu

gpu_index_flat.add(xb)         # add vectors to the index
k = 4                          # we want to see 4 nearest neighbors
for i in range(5):
    s = time.time()
    D, I = gpu_index_flat.search(xq, k)
    print("{}*{}量级的精确检索，耗时:{:.3f}s".format(nb, nq, time.time()-s))

print('D.shape: {}, D[0, ...]: {}'.format(D.shape, D[0]))
print('I.shape: {}, I[0, ...]: {}'.format(I.shape, I[0]))