# -*- coding:utf-8 -*-
"""
@file name  : rnn.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-10-08
@brief      : RNN 模型构建
"""
import os
import sys
import time
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import platform
from torch.utils.data import DataLoader
import torch.nn.functional as F


BASE_DIR = os.path.dirname(__file__)
PRJ_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(PRJ_DIR)


class LSTMTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, **kwargs):
        super(LSTMTextClassifier, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 将bidirectional设置为True以获取双向循环神经网络
        self.encoder = nn.LSTM(embed_size, num_hiddens, num_layers=num_layers, bidirectional=True)
        self.decoder = nn.Linear(4 * num_hiddens, 2)  # 一个output向量长度是2倍的hidden size，有两个output拼接，所以是4倍
        # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM

    def forward(self, inputs):
        # inputs的形状是（批量大小，时间步数）
        # 因为长短期记忆网络要求其输入的第一个维度是时间维，所以在获得词元表示之前，输入会被转置。
        # 输出形状为（时间步数，批量大小，词向量维度）
        embeddings = self.embedding(inputs.T)
        self.encoder.flatten_parameters()
        # 返回上一个隐藏层在不同时间步的隐状态，
        # outputs的形状是（时间步数，批量大小，2*隐藏单元数）
        outputs, _ = self.encoder(embeddings)
        # 连结初始和最终时间步的隐状态，作为全连接层的输入，
        # 其形状为（批量大小，4*隐藏单元数）
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1)
        outs = self.decoder(encoding)
        return outs


class RNNTextClassifier(nn.Module):
    def __init__(self, inp_size, hidden_size, n_class, layer_num, vocab_len, device):
        super(RNNTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_len, inp_size)

        self.device = device
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.rnn = nn.RNN(input_size=inp_size, hidden_size=hidden_size, num_layers=layer_num, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_class, bias=True)

    def forward(self, x):

        x_embed = self.embedding(x)  # [batch_size, max_len] -> [batch_size, text_len, embed_len]
        outputs, hidden = self.rnn(x_embed)
        last_hidden = hidden[-1].squeeze(0)  # [num_layers, bs, hidden_size]  ->  [bs, hidden_size]
        fc_output = self.fc(last_hidden)

        return fc_output

    def forward_bak(self, x):

        x_embed = self.embedding(x)  # [batch_size, max_len] -> [batch_size, text_len, embed_len]

        bs_, text_len, embed_len = x_embed.shape

        hidden_init = self.init_hidden(bs_)
        outputs, hidden = self.rnn(x_embed, hidden_init)
        # Extract the last hidden state
        last_hidden = hidden[-1].squeeze(0)  # [num_layers, bs, hidden_size]  ->  [bs, hidden_size]
        last_output = outputs[:, -1, :].squeeze(0)  # bs, sequence len, hidden_size]  ->  [bs, hidden_size]
        fc_output = self.fc(last_hidden)

        return fc_output

    def init_hidden(self, batch_size):
        # https://pytorch.org/docs/stable/generated/torch.nn.RNN.html#torch.nn.RNN
        hidden = torch.zeros(self.layer_num, batch_size, self.hidden_size)  # (D∗num_layers, N, H_out)
        hidden = hidden.to(self.device)
        return hidden


if __name__ == "__main__":
    # Example usage
    input_size = 768  # Size of input features (e.g., word embeddings)
    hidden_size = 128  # Number of hidden units in the RNN
    layer_num = 3  # Number of hidden layer in the RNN
    output_size = 4  # Number of output classes (binary classification)
    vocab_len = 20000

    # Create an instance of the RNNTextClassifier
    model = RNNTextClassifier(input_size, hidden_size, output_size, layer_num, vocab_len, 'cpu')

    batch_size = 16
    sequence_length = 200
    input_token_list = torch.randint(vocab_len, (batch_size, sequence_length))

    output = model(input_token_list)
    print(output.shape)
