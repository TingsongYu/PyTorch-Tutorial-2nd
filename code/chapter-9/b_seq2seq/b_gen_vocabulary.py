
"""
@file name  : b_gen_vocabulary.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-10-19
@brief      : 英语-法语翻译的文本数据词表构建，保存到result/vocab_en.npy ; vocab_fra.npy
"""
import jieba
import re
import os
from tqdm import tqdm
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'SimHei'

def plot_sentence_length(text_list, hist_size=50, path_out=None):
    plt.clf()
    text_len_list = [len(sentence) for sentence in text_list]
    plt.hist(text_len_list, bins=max(text_len_list))
    # 设置图形大小
    plt.title('sentence length Frequency')
    plt.xlabel('length')
    plt.ylabel('Frequency')
    # plt.xticks(rotation=90)
    if path_out:
        plt.savefig(path_out)
        print(f'保存统计图:{path_out}')

def plot_word_frequency(word_count_dict, hist_size=100, path_out=None):
    words = list(word_count_dict.keys())[:hist_size]
    frequencies = list(word_count_dict.values())[:100]
    # 设置图形大小
    plt.figure(figsize=(10, 6))
    plt.bar(words, frequencies)
    plt.title('Word Frequency')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=90)
    if path_out:
        plt.savefig(path_out)
        print(f'保存词频统计图:{path_out}')


def read_data_nmt(path_file_):
    with open(os.path.join(path_file_), 'r', encoding='utf-8') as f:
        return f.read()  # 不是list，而是一个str


def text_preprocess(text):
    """
    数据预处理
    1 用空格代替不间断空格（non-breaking space）
    2 使用小写字母替换大写字母
    3 在单词和标点符号之间插入空格
    :param text: str， 原始txt文件，用string方式读取进来
    :return:
    """
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 使用空格替换不间断空格
    # 使用小写字母替换大写字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()

    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)


def text_split(text: str) -> Tuple[List, List]:
    """
    分词，这里采用空格切分，需要返回target和source两个list
    :param text: str 原始文
    :return:
    """
    source, target = [], []
    # 遍历每一行
    for i, line in enumerate(text.split('\n')):
        # 按照\t进行切分
        parts = line.split('\t')
        # 如果是3部分才算，一个是英语，一个是中文
        # 一行样本：Go.	Va !	CC-BY 2.0 (France) Attribution: tatoeba.org #2877272 (CM) & #1158250 (Wittydev)
        if len(parts) == 3:
            source.append(parts[0].split(' '))
            target.append(list(jieba.cut(parts[1])))  # 分词
    return source, target


class Vocabulary:
    PAD_TAG = "<pad>"  # 用PAD补全句子长度
    BOS_TAG = "<bos>"  # 用BOS表示开始
    EOS_TAG = "<eos>"  # 用EOS表示结束
    UNK_TAG = "<unk>"  # 用EOS表示结束
    PAD = 0  # PAD字符对应的数字
    BOS = 1  # BOS字符对应的数字
    EOS = 2  # EOS字符对应的数字
    UNK = 3  # UNK字符对应的数字

    def __init__(self):
        self.inverse_vocab = None
        self.vocabulary = {self.BOS_TAG: self.BOS, self.EOS_TAG: self.EOS,
                           self.PAD_TAG: self.PAD, self.UNK_TAG: self.UNK}
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


def get_vocab(text_list, path_out):
    # 每个元素是list， 是一个切分好的句子
    vocab_hist = Vocabulary()
    for sentence in tqdm(text_list):
        vocab_hist.fit(sentence)
    vocab, inverse_vocab = vocab_hist.build_vocab(min=3, max_vocab_size=(max_vocab_size - 4))  # 3 是 pad\bos\eos\unk
    np.save(path_out, vocab)

    # 词表、词频可视化
    print(len(vocab))
    word_count = vocab_hist.count
    path_w_frequency_out = os.path.basename(path_out) + '_word_freq.jpg'
    path_l_frequency_out = os.path.basename(path_out) + '_length_freq.jpg'
    plot_word_frequency(word_count, hist_size=100, path_out=path_w_frequency_out)
    plot_sentence_length(text_list, hist_size=50, path_out=path_l_frequency_out)


if __name__ == '__main__':
    max_vocab_size = 3000

    data_dir = r"G:\deep_learning_data\machine_transfer\cmn-eng"
    path_raw = os.path.join(data_dir, "cmn.txt")
    path_train = os.path.join(data_dir, "train.txt")

    BASE_DIR = os.path.dirname(__file__)
    out_dir = os.path.join(BASE_DIR, 'result')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    vocab_path_en = os.path.join(out_dir, "vocab_en.npy")
    vocab_path_cmn = os.path.join(out_dir, "vocab_cmn.npy")

    # 加载数据
    text_raw = read_data_nmt(path_train)
    text_clean = text_preprocess(text_raw)
    source_list, target_list = text_split(text_clean)

    # 统计词频、保存词表
    get_vocab(source_list, vocab_path_en)
    get_vocab(target_list, vocab_path_cmn)



