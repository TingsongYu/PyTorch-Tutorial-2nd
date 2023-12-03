# -*- coding:utf-8 -*-
"""
@file name  : c_inference.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-11-15
@brief      : 机器翻译推理示例
"""
import os
import sys
import time
import numpy as np
import datetime
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(__file__)
PRJ_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(PRJ_DIR)

from datasets.nmt_en_cmn_dataset import NMTDataset
# import utils.my_utils as utils


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)
    parser.add_argument("--ckpt-path", default=r"./Result/2023-11-15_23-31-53/checkpoint_best.pth", type=str, help="ckpt path")
    parser.add_argument("--data-path", default=r"G:\deep_learning_data\machine_transfer\fra-eng", type=str,
                        help="dataset path")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("--random-seed", default=42, type=int, help="random seed")
    parser.add_argument("--output-dir", default="./result", type=str, help="path to save outputs")

    return parser


if __name__ == "__main__":
    from models.seq2seq import EncoderLSTM, DecoderLSTM, Seq2Seq
    result_dir = os.path.join(BASE_DIR, "result")
    max_len = 32

    encoder_embedding_size = 300
    hidden_size = 1024
    num_layers = 2
    encoder_dropout = 0.5
    decoder_embedding_size = 300
    hidden_size = 1024
    num_layers = 2
    decoder_dropout = 0.5

    args = get_args_parser().parse_args()

    # ------------------------------------ step1: dataset ------------------------------------
    root_dir = args.data_path

    vocab_path_en = os.path.join(BASE_DIR, 'result', "vocab_en.npy")
    vocab_path_fra = os.path.join(BASE_DIR, 'result', "vocab_fra.npy")

    vocab_en = np.load(vocab_path_en, allow_pickle=True).item()
    vocab_fra = np.load(vocab_path_fra, allow_pickle=True).item()

    path_txt_train = os.path.join(root_dir, 'train.txt')
    path_txt_test = os.path.join(root_dir, 'test.txt')

    train_set = NMTDataset(path_txt_train, vocab_path_en, vocab_path_fra, max_len=max_len)
    test_set = NMTDataset(path_txt_test, vocab_path_en, vocab_path_fra, max_len=max_len)

    vocab_fra_inv = {value: key for key, value in vocab_fra.items()}
    vocab_en_inv = {value: key for key, value in vocab_en.items()}

    # ------------------------------------ step2: model ------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src_vocab_len, trg_vocab_len = len(vocab_en), len(vocab_fra)
    input_size_encoder, input_size_decoder, output_size = src_vocab_len, trg_vocab_len, trg_vocab_len

    encoder_lstm = EncoderLSTM(input_size_encoder, encoder_embedding_size,
                               hidden_size, num_layers, encoder_dropout).to(device)
    decoder_lstm = DecoderLSTM(input_size_decoder, decoder_embedding_size,
                               hidden_size, num_layers, decoder_dropout, output_size).to(device)
    model = Seq2Seq(encoder_lstm, decoder_lstm, device).to(device)

    # 加载权重
    state_dict = torch.load(args.ckpt_path, map_location=torch.device('cpu'))
    model_sate_dict = state_dict['model_state_dict']
    model.load_state_dict(model_sate_dict)  # 模型参数加载

    model.to(device)
    model.eval()
    # ------------------------------------ step3: inference ------------------------------------
    max_step = 32

    for idx, data in enumerate(test_set):
        # 只推理10个样本
        if idx == 10:
            break
        src_idx, _,  trg_idx, _ = data

        # 获取文本索引
        src_idx_raw, trg_idx_raw = [], []
        for text_idx in src_idx:
            if text_idx == 0:
                break
            src_idx_raw.append(text_idx)

        for text_idx in trg_idx:
            if text_idx == 0:
                break
            trg_idx_raw.append(text_idx)

        # 转为tensor
        sentence_tensor = torch.LongTensor(src_idx_raw).unsqueeze(1).to(device)
        # Build encoder hidden, cell state
        with torch.no_grad():
            hidden, cell = model.Encoder_LSTM(sentence_tensor)  # bs at dim1, h.shape == [2, 13, 1024]

        outputs = [vocab_fra['<bos>']]  # 1
        for _ in range(max_step):
            previous_word = torch.tensor([outputs[-1]], dtype=torch.int64).to(device)  # shape: [bs, ]
            with torch.no_grad():
                output, hidden, cell = model.Decoder_LSTM(previous_word, hidden, cell)
                best_guess = output.argmax(1)[0].item()
            outputs.append(best_guess)

            if best_guess == vocab_fra['<eos>']:
                break

        translated_sentence = [vocab_fra_inv[idx] for idx in outputs]

        # 索引转为文本
        sentence_src = [vocab_en_inv.get(idx, '<unk>') for idx in src_idx_raw]
        sentence_trg = [vocab_fra_inv.get(idx, '<unk>') for idx in trg_idx_raw]

        translated_sentence_out = " ".join(translated_sentence[1:])
        sentence_src_out = " ".join(sentence_src)
        sentence_trg_out = " ".join(sentence_trg)
        print(f'输入: {sentence_src_out}\n标签:{sentence_trg_out}\n翻译:{translated_sentence_out}\n')
