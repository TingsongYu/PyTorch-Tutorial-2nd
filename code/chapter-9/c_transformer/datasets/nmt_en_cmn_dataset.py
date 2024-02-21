# -*- coding:utf-8 -*-
"""
@file name  : nmt_en_cmn_dataset.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-10-19
@brief      : nmt 神经机器翻译 英语 --> 中文 数据集读取
"""
import sys
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader

BASE_DIR = os.path.dirname(__file__)
PRJ_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(PRJ_DIR)


from b_gen_vocabulary import read_data_nmt, text_preprocess, text_split


class WordToIndex(object):
    def __init__(self):
        self.PAD_TAG = "<pad>"
        self.BOS_TAG = "<bos>"  # 用BOS表示开始
        self.EOS_TAG = "<eos>"  # 用EOS表示结束
        self.UNK_TAG = "<unk>"  # 用UNK表示结束
        self.PAD = 0
        self.BOS = 1
        self.EOS = 2
        self.UNK = 3

    def encode(self, sentence, vocab_dict, max_len=None):
        if max_len is not None:    # 补齐，切割 句子固定长度
            if max_len > len(sentence):
                sentence = sentence + [self.EOS_TAG] + [self.PAD_TAG]*(max_len-len(sentence)-1)  # 加入结束符, 填充 <pad>
            if max_len < len(sentence):
                sentence = sentence[:max_len]
        return [vocab_dict.get(word, self.UNK) for word in sentence]

    @staticmethod
    def decode(ws_inverse, indices):
        return [ws_inverse.get(idx) for idx in indices]


class NMTDataset(Dataset):
    def __init__(self, path_txt, vocab_path_en, vocab_path_fra, max_len=32):
        self.path_txt = path_txt
        self.vocab_path_en = vocab_path_en
        self.vocab_path_fra = vocab_path_fra
        self.max_len = max_len

        self.word2index = WordToIndex()
        self._init_vocab()
        self._get_file_info()

    def __getitem__(self, item):

        # 获取切分好的句子list，一个元素是一个词
        sentence_src, sentence_trg = self.source_list[item], self.target_list[item]
        # 进行填充， 增加结束符，索引转换
        token_idx_src = self.word2index.encode(sentence_src, self.vocab_en, self.max_len)
        token_idx_trg = self.word2index.encode(sentence_trg, self.vocab_fra, self.max_len)
        str_len, trg_len = len(sentence_src) + 1, len(sentence_trg) + 1  # 有效长度， +1是填充的结束符 <eos>.

        return np.array(token_idx_src, dtype=np.int64), str_len,  np.array(token_idx_trg, dtype=np.int64), trg_len

    def __len__(self):
        return len(self.source_list)

    def _get_file_info(self):

        text_raw = read_data_nmt(self.path_txt)
        text_clean = text_preprocess(text_raw)
        self.source_list, self.target_list = text_split(text_clean)

    def _init_vocab(self):
        # 加载词表字典
        self.vocab_en = np.load(self.vocab_path_en, allow_pickle=True).item()
        self.vocab_fra = np.load(self.vocab_path_fra, allow_pickle=True).item()


if __name__ == "__main__":
    import torch

    path_txt = r'G:\deep_learning_data\machine_transfer\cmn-eng\test.txt'
    vocab_path_en = os.path.join(BASE_DIR, '..', 'result', "vocab_en.npy")
    vocab_path_fra = os.path.join(BASE_DIR, '..', 'result', "vocab_cmn.npy")

    test_set = NMTDataset(path_txt, vocab_path_en, vocab_path_fra, max_len=20)
    train_loader = DataLoader(dataset=test_set, batch_size=2, shuffle=True, num_workers=2)
    for i, (src, src_len, trg, trg_len) in enumerate(train_loader):
        print(i, src.shape, src_len, trg.shape, trg_len)
        s_ = src[0]
        len_ = int(torch.count_nonzero(s_))
        print(f'句子长度为:{len_}， 第{len_}个token是{s_[len_-1]}， 第{len_+1}个token是:{s_[len_]}，表明最后一个token是结束符<eos>')

