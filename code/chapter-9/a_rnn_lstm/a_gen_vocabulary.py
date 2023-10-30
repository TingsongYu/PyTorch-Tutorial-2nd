
"""
@file name  : a_gen_bocabulary.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-10-08
@brief      : 影评文本数据词表构建
"""
import re
import os
from tqdm import tqdm
import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt


def plot_word_frequency(word_count_dict, hist_size=100):
    words = list(word_count_dict.keys())[:hist_size]
    frequencies = list(word_count_dict.values())[:hist_size]
    # 设置图形大小
    plt.figure(figsize=(10, 6))
    plt.bar(words, frequencies)
    plt.title('Word Frequency')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=90)
    path_out = 'word_frequency.jpg'
    plt.savefig(path_out)
    print(f'保存词频统计图:{path_out}')


# 数据清洗规则
def text_split(content: str) -> List[str]:
    """
    对原始文本进行token化，包含一系列预处理清洗操作
    :param content:
    :return:
    """
    content = re.sub(r"([.!?])", r" \1", content)  # 在 .!? 之前添加一个空格
    content = re.sub(r"[^a-zA-Z.!?]+", r" ", content)  # 去除掉不是大小写字母及 .!? 符号的数据
    token = [i.strip().lower() for i in content.split()]  # 全部转换为小写，然后去除两边空格，将字符串转换成list,
    return token


class Vocabulary:
    UNK_TAG = "UNK"  # 遇到未知字符，用UNK表示
    PAD_TAG = "PAD"  # 用PAD补全句子长度
    UNK = 0  # UNK字符对应的数字
    PAD = 1  # PAD字符对应的数字

    def __init__(self):
        self.inverse_vocab = None
        self.vocabulary = {self.UNK_TAG: self.UNK, self.PAD_TAG: self.PAD}
        self.count = {}  # 统计词频

    def fit(self, sentence_: List[str]):
        """
        统计词频
        """
        for word in sentence_:
            self.count[word] = self.count.get(word, 0) + 1

    def build_vocab(self, min=0, max=None, max_vocab_size=None) -> Tuple[dict, dict]:
        # 词频截断，词频大于或者小于一定数值时，舍弃
        if min is not None:
            self.count = {word: value for word, value in self.count.items() if value > min}
        if max is not None:
            self.count = {word: value for word, value in self.count.items() if value < max}
        # 选择词表大小，根据词频排序后截断
        if max_vocab_size is not None:
            raw_len = len(self.count.items())
            vocab_size = max_vocab_size if raw_len > max_vocab_size else raw_len
            print('原始词表长度:{}，截断后长度:{}'.format(raw_len, vocab_size))
            temp = sorted(self.count.items(), key=lambda x: x[-1], reverse=True)[:vocab_size]
            self.count = dict(temp)

        # 建立词表： token -> index
        for word in self.count:
            self.vocabulary[word] = len(self.vocabulary)
        # 词表翻转：index -> token
        self.inverse_vocab = dict(zip(self.vocabulary.values(), self.vocabulary.keys()))

        return self.vocabulary, self.inverse_vocab

    def __len__(self):
        return len(self.vocabulary)


if __name__ == '__main__':
    max_vocab_size = 20000
    path = r'G:\deep_learning_data\aclImdb_v1\aclImdb\train'
    BASE_DIR = os.path.dirname(__file__)
    out_dir = os.path.join(BASE_DIR, 'result')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    vocab_path = os.path.join(out_dir, "aclImdb_vocab.npy")
    vocab_inv_path = os.path.join(out_dir, "aclImdb_vocab_inv.npy")

    # 统计词频
    vocab_hist = Vocabulary()
    temp_data_path = [os.path.join(path, "pos"), os.path.join(path, "neg")]  # 训练集中包含 正类数据pos 负类数据neg
    for data_path in temp_data_path:
        file_paths = [os.path.join(data_path, file_name) for file_name in os.listdir(data_path) if file_name.endswith("txt")]
        for file_path in tqdm(file_paths):
            sentence = text_split(open(file_path, encoding='utf-8').read())
            vocab_hist.fit(sentence)

    # 建立词表
    vocab, inverse_vocab = vocab_hist.build_vocab(max_vocab_size=(max_vocab_size - 2))  # 2 是 unk 和 pad

    # 保存词表
    np.save(vocab_path, vocab)
    np.save(vocab_inv_path, inverse_vocab)

    # 词表、词频可视化
    print(len(vocab))
    word_count = vocab_hist.count
    plot_word_frequency(word_count)



