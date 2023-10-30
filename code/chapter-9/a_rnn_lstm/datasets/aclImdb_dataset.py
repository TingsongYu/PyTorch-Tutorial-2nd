# -*- coding:utf-8 -*-
"""
@file name  : aclImdb_dataset.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-10-8
@brief      : aclImdb 数据集读取
"""
import sys
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader

BASE_DIR = os.path.dirname(__file__)
PRJ_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(PRJ_DIR)

from a_gen_vocabulary import text_split


class WordToIndex(object):
    def __init__(self):
        self.PAD_TAG = "PAD"
        self.UNK = 0

    def encode(self, sentence, vocab_dict, max_len=None):
        if max_len is not None:    # 补齐，切割 句子固定长度
            if max_len > len(sentence):
                sentence = sentence + [self.PAD_TAG]*(max_len-len(sentence))
            if max_len < len(sentence):
                sentence = sentence[:max_len]
        return [vocab_dict.get(word, self.UNK) for word in sentence]

    @staticmethod
    def decode(ws_inverse, indices):
        return [ws_inverse.get(idx) for idx in indices]


class AclImdbDataset(Dataset):
    def __init__(self, root_dir, vocab_path, is_train=True, max_len=200):
        sub_dir = "train" if is_train else "test"
        self.data_dir = os.path.join(root_dir, sub_dir)
        self.vocab_path = vocab_path
        self.max_len = max_len

        self.word2index = WordToIndex()
        self._init_vocab()
        self._get_file_info()

    def __getitem__(self, item):
        # 读取文件路径
        file_path = self.total_file_path[item]
        # 获取 label
        label = 0 if os.path.basename(os.path.dirname(file_path)) == "neg" else 1    # neg -> 0; pos -> 1

        # tokenize & encode to index
        token_list = text_split(open(file_path, encoding='utf-8').read())  # 切分
        token_idx_list = self.word2index.encode(token_list, self.vocab, self.max_len)

        return np.array(token_idx_list), label

    def __len__(self):
        return len(self.total_file_path)

    def _get_file_info(self):
        # 获取所有文件的路径
        self.data_dir_list = [os.path.join(self.data_dir, "pos"),  os.path.join(self.data_dir, "neg")]
        self.total_file_path = []
        for dir_tmp in self.data_dir_list:
            self.file_name_list = os.listdir(dir_tmp)
            self.file_path_list = [os.path.join(dir_tmp, file_name) for file_name in self.file_name_list
                                   if file_name.endswith("txt")]
            self.total_file_path.extend(self.file_path_list)

    def _init_vocab(self):
        # 加载词表字典
        self.vocab = np.load(self.vocab_path, allow_pickle=True).item()


if __name__ == "__main__":
    root_dir = r'G:\deep_learning_data\aclImdb_v1\aclImdb'
    vocab_path = os.path.join(BASE_DIR, '..', 'result', "aclImdb_vocab.npy")

    train_set = AclImdbDataset(root_dir, vocab_path, is_train=True, max_len=200)
    valid_set = AclImdbDataset(root_dir, vocab_path, is_train=False, max_len=200)

    train_loader = DataLoader(dataset=train_set, batch_size=10, shuffle=True)
    for i, (inputs, target) in enumerate(train_loader):
        print(i, inputs.shape, inputs, target.shape, target)
