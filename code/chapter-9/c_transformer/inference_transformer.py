"""
@file name  : inference_transformer.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2024-02-21
@brief      : Transformer模型推理代码
"""


import torch
import argparse
import os
import sys
import numpy as np
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(__file__)
PRJ_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(PRJ_DIR)

from datasets.nmt_en_cmn_dataset import NMTDataset
from transformer.Models import Transformer
from transformer.Translator import Translator


def main():
    # -------------------------- step 1: 加载数据 ----------------------
    # 加载词表
    vocab_en = np.load(vocab_path_en, allow_pickle=True).item()
    vocab_fra = np.load(vocab_path_fra, allow_pickle=True).item()
    vocab_en_inv = {value: key for key, value in vocab_en.items()}
    vocab_fra_inv = {value: key for key, value in vocab_fra.items()}

    # 加载数据
    train_set = NMTDataset(path_txt_train, vocab_path_en, vocab_path_fra, max_len=max_len)
    test_set = NMTDataset(path_txt_test, vocab_path_en, vocab_path_fra, max_len=max_len)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False, num_workers=0)
    valid_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    # -------------------------- step 2: 加载模型 ----------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载Transformer
    checkpoint = torch.load(path_model, map_location=device)
    model_opt = checkpoint['settings']
    src_vocab_len, trg_vocab_len = len(train_set.vocab_en), len(train_set.vocab_fra)  # 2522, 3000
    transformer = Transformer(
        trg_vocab_len,  # 训练时，若采用了embs_share_weight， 请将src的词表长度设置为trg的词表长度 2024年11月8日
        trg_vocab_len,
        src_pad_idx=train_set.word2index.PAD,
        trg_pad_idx=train_set.word2index.PAD,
        trg_emb_prj_weight_sharing=False,
        emb_src_trg_weight_sharing=False,
        d_k=model_opt.d_k,
        d_v=model_opt.d_v,
        d_model=model_opt.d_model,
        d_word_vec=model_opt.d_word_vec,
        d_inner=model_opt.d_inner_hid,
        n_layers=model_opt.n_layers,
        n_head=model_opt.n_head,
        dropout=model_opt.dropout).to(device)

    transformer.load_state_dict(checkpoint['model'])

    # 构建翻译器，实现自回归机制，完成逐个token的输出，直至遇到<eos>或者最大长度时，组成整个句子
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    translator = Translator(
        model=transformer,
        beam_size=beam_size,
        max_seq_len=max_seq_len,
        src_pad_idx=train_set.word2index.PAD,
        trg_pad_idx=train_set.word2index.PAD,
        trg_bos_idx=train_set.word2index.BOS,
        trg_eos_idx=train_set.word2index.EOS).to(device)
    print('[Info] Trained model state loaded.')

    # ------------------------------------ 逐个样本预测 ------------------------------------
    for idx, batch in enumerate(train_loader):
        if idx > pred_exmales:
            break
        src, src_len, trg, trg_len = [x.to(device) for x in batch]

        # 执行预测，获取序列的index
        pred_seq = translator.translate_sentence(src)

        # 将index转换为词
        sentence_pred = [vocab_fra_inv.get(idx, '<unk>') for idx in pred_seq[1:] if idx != train_set.word2index.EOS]
        pred_line = ' '.join(sentence_pred)

        # 打印源句子、目标句子、预测句子
        assert src.shape[0] == 1
        assert trg.shape[0] == 1
        sentence_src = [vocab_en_inv.get(idx.cpu().item(), '<unk>') for idx in src[0]
                        if not (idx.cpu().item() == train_set.word2index.PAD or idx.cpu().item() == train_set.word2index.EOS)]
        sentence_trg = [vocab_fra_inv.get(idx.cpu().item(), '<unk>') for idx in trg[0][:]
                        if not (idx.cpu().item() == train_set.word2index.PAD or idx.cpu().item() == train_set.word2index.EOS)]
        src_line = ' '.join(sentence_src)
        trg_line = ' '.join(sentence_trg)
        print("src:\t{}\ntrg:\t{}\npred:\t{}\n".format(src_line, trg_line, pred_line))


if __name__ == "__main__":
    vocab_path_en = os.path.join(BASE_DIR, 'result', "vocab_en.npy")
    vocab_path_fra = os.path.join(BASE_DIR, 'result', "vocab_cmn.npy")
    root_dir = r"G:\deep_learning_data\machine_transfer\cmn-eng"
    path_txt_train = os.path.join(root_dir, 'train.txt')
    path_txt_test = os.path.join(root_dir, 'test.txt')

    max_len = 32
    batch_size = 1  # 当前代码仅支持一个batch的预测
    pred_exmales = 10  # 预测样本的个数
    beam_size = 5
    max_seq_len = 20
    path_model = os.path.join(BASE_DIR, 'result', "model_acc_train_71_val_50.chkpt")
    path_model = os.path.join(BASE_DIR, 'result', "model_accu_51.090.chkpt")

    main()
