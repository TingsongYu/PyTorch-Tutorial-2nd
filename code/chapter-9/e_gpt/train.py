# -*- coding:utf-8 -*-
"""
@file name  : train.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2024-03-02
@brief      : GPT2 模型训练代码
使用注意事项！这里只适配了wiki2019数据集和baikeqa2018数据集，并且需要手动修改代码，在208行，需要根据数据来选择函数
build_files_baikeqa or  build_files;  (太懒了，不想写代码来识别两个数据集了)
参考自https://github.com/Morizeyao/GPT2-Chinese
"""
import transformers
import torch
import os
import json
import random
import time
import logging
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from torch.nn import DataParallel
from tokenizations.bpe_tokenizer import get_encoder

logger = logging.getLogger()

def init_logger(log_file=None, log_file_level=logging.NOTSET):
    if isinstance(log_file, Path):
        log_file = str(log_file)
    log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                   datefmt='%m/%d/%Y %H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        # file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger

def build_files_baikeqa(data_path, tokenized_data_path, num_pieces, full_tokenizer, min_length):
    all_files = []
    lines_total = []
    full_tokens = 0
    # 使用os.walk()遍历文件夹
    for root, dirs, files in os.walk(data_path):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)

    lines = []
    for path_single_file in all_files:
        with open(path_single_file, 'r', encoding='utf8') as f:
            for idx, line in enumerate(f):
                # if idx > 200:
                #     continue  # show data
                json_obj = json.loads(line)
                qa_pair = "[CLS]{}[SEP]{}[SEP]{}[SEP]".format(json_obj['desc'], json_obj['title'], json_obj['answer'])
                lines.append(qa_pair)
        lines_total.extend(lines)

    all_len = len(lines_total)
    if not os.path.exists(tokenized_data_path):
        os.mkdir(tokenized_data_path)
    for i in tqdm(range(num_pieces)):
        sublines = lines_total[all_len // num_pieces * i: all_len // num_pieces * (i + 1)]
        if i == num_pieces - 1:
            sublines.extend(lines_total[all_len // num_pieces * (i + 1):])  # 把尾部例子添加到最后一个piece
        sublines = [full_tokenizer.tokenize(line) for line in sublines if
                    len(line) > min_length]  # 只考虑长度超过min_length的句子
        sublines = [full_tokenizer.convert_tokens_to_ids(line) for line in sublines]
        full_line = []
        for subline in sublines:
            full_line.append(full_tokenizer.convert_tokens_to_ids('[MASK]'))  # 文章开头添加MASK表示文章开始
            full_line.extend(subline)
            # full_line.append(full_tokenizer.convert_tokens_to_ids('[CLS]'))  # 文章之间添加CLS表示文章结束
        with open(tokenized_data_path + 'tokenized_train_{}.txt'.format(i), 'w') as f:
            for id in full_line:
                f.write(str(id) + ' ')
        full_tokens += len(full_line)
    logger.info(f'读取了{len(all_files)}个文件，划分为{num_pieces}txt，总共有{full_tokens}个token')


def build_files(data_path, tokenized_data_path, num_pieces, full_tokenizer, min_length):
    all_files = []
    lines_total = []
    full_tokens = 0
    # 使用os.walk()遍历文件夹
    for root, dirs, files in os.walk(data_path):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)

    for path_single_file in all_files:
        with open(path_single_file, 'r', encoding='utf8') as f:
            lines = f.readlines()
            # lines = json.load(f)
            lines = [line.replace('\n', ' [SEP] ') for line in lines]  # 用[SEP]表示换行, 段落之间使用SEP表示段落结束
        lines_total.extend(lines)

    all_len = len(lines_total)
    if not os.path.exists(tokenized_data_path):
        os.mkdir(tokenized_data_path)
    for i in tqdm(range(num_pieces)):
        sublines = lines_total[all_len // num_pieces * i: all_len // num_pieces * (i + 1)]
        if i == num_pieces - 1:
            sublines.extend(lines_total[all_len // num_pieces * (i + 1):])  # 把尾部例子添加到最后一个piece
        sublines = [full_tokenizer.tokenize(line) for line in sublines if
                    len(line) > min_length]  # 只考虑长度超过min_length的句子
        sublines = [full_tokenizer.convert_tokens_to_ids(line) for line in sublines]
        full_line = []
        for subline in sublines:
            full_line.append(full_tokenizer.convert_tokens_to_ids('[MASK]'))  # 文章开头添加MASK表示文章开始
            full_line.extend(subline)
            full_line.append(full_tokenizer.convert_tokens_to_ids('[CLS]'))  # 文章之间添加CLS表示文章结束
        with open(tokenized_data_path + 'tokenized_train_{}.txt'.format(i), 'w') as f:
            for id in full_line:
                f.write(str(id) + ' ')
        full_tokens += len(full_line)
    logger.info(f'读取了{len(all_files)}个文件，划分为{num_pieces}txt，总共有{full_tokens}个token')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1,2,3', type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument('--model_config', default='config/model_config_small.json', type=str, required=False,
                        help='选择模型参数')
    parser.add_argument('--tokenizer_path', default='cache/vocab_small.txt', type=str, required=False, help='选择词库')
    parser.add_argument('--raw_data_path', default='data/demo_data', type=str, required=False, help='原始训练语料')
    # parser.add_argument('--raw_data_path', default=r'G:\deep_learning_data\baike2018qa', type=str, required=False, help='原始训练语料')
    # parser.add_argument('--raw_data_path', default=r'G:\deep_learning_data\wiki_zh_2019\wiki_zh', type=str, required=False, help='原始训练语料')
    parser.add_argument('--tokenized_data_path', default='data/tokenized/', type=str, required=False,
                        help='tokenized语料存放位置')
    parser.add_argument('--raw', action='store_true', help='是否先做tokenize')
    parser.add_argument('--epochs', default=5, type=int, required=False, help='训练循环')
    parser.add_argument('--batch_size', default=1, type=int, required=False, help='训练batch size')
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False, help='学习率')
    parser.add_argument('--warmup_steps', default=100, type=int, required=False, help='warm up步数')
    parser.add_argument('--log_step', default=10, type=int, required=False, help='多少步汇报一次loss，设置为gradient accumulation的整数倍')
    parser.add_argument('--stride', default=768, type=int, required=False, help='训练时取训练数据的窗口步长')
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='梯度积累')
    parser.add_argument('--fp16', action='store_true', help='混合精度')
    parser.add_argument('--fp16_opt_level', default='O1', type=str, required=False)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--num_pieces', default=100, type=int, required=False, help='将训练语料分成多少份')
    parser.add_argument('--min_length', default=128, type=int, required=False, help='最短收录文章长度')
    parser.add_argument('--output_dir', default='model/', type=str, required=False, help='模型输出路径')
    parser.add_argument('--pretrained_model', default='', type=str, required=False, help='模型训练起点路径')
    parser.add_argument('--writer_dir', default='tensorboard_summary/', type=str, required=False, help='Tensorboard路径')
    parser.add_argument('--segment', action='store_true', help='中文以词为单位')
    parser.add_argument('--bpe_token', action='store_true', help='subword')
    parser.add_argument('--encoder_json', default="tokenizations/encoder.json", type=str, help="encoder.json")
    parser.add_argument('--vocab_bpe', default="tokenizations/vocab.bpe", type=str, help="vocab.bpe")

    # -------------------------------- 基础内容配置  --------------------------------
    args = parser.parse_args()
    # 获取当前时间的时间戳
    log_file_name = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    os.makedirs(args.output_dir, exist_ok=True)
    init_logger(log_file=args.output_dir + f'/{log_file_name}.log')

    logger.info('args:\n' + args.__repr__())

    if args.segment:
        from tokenizations import tokenization_bert_word_level as tokenization_bert
    else:
        from tokenizations import tokenization_bert

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 此处设置程序使用哪些显卡


    if args.bpe_token:
        full_tokenizer = get_encoder(args.encoder_json, args.vocab_bpe)
    else:
        full_tokenizer = tokenization_bert.BertTokenizer(vocab_file=args.tokenizer_path)
    full_tokenizer.max_len = 999999
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info('using device:', device)

    raw_data_path = args.raw_data_path
    tokenized_data_path = args.tokenized_data_path
    raw = args.raw  # 选择是否从零开始构建数据集
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    warmup_steps = args.warmup_steps
    log_step = args.log_step
    stride = args.stride
    gradient_accumulation = args.gradient_accumulation
    fp16 = args.fp16  # 不支持半精度的显卡请勿打开
    fp16_opt_level = args.fp16_opt_level
    max_grad_norm = args.max_grad_norm
    num_pieces = args.num_pieces
    min_length = args.min_length
    output_dir = args.output_dir
    tb_writer = SummaryWriter(log_dir=args.writer_dir)
    assert log_step % gradient_accumulation == 0

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # -------------------------------- 原始数据处理为tokenized数据  --------------------------------
    if raw:
        logger.info('building files')
        # build_files_baikeqa(data_path=raw_data_path, tokenized_data_path=tokenized_data_path, num_pieces=num_pieces,
        #             full_tokenizer=full_tokenizer, min_length=10)
        build_files(data_path=raw_data_path, tokenized_data_path=tokenized_data_path, num_pieces=num_pieces,
                    full_tokenizer=full_tokenizer, min_length=min_length)
        logger.info('files built')
    # -------------------------------- 模型加载  --------------------------------
    model_config = transformers.modeling_gpt2.GPT2Config.from_json_file(args.model_config)
    logger.info('config:\n' + model_config.to_json_string())
    n_ctx = model_config.n_ctx  # 上下文最大长度， context length
    if not args.pretrained_model:
        model = transformers.modeling_gpt2.GPT2LMHeadModel(config=model_config)
    else:
        model = transformers.modeling_gpt2.GPT2LMHeadModel.from_pretrained(args.pretrained_model)
    model.train()
    model.to(device)

    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    logger.info('number of parameters: {}'.format(num_parameters))

    multi_gpu = False
    full_len = 0
    logger.info('calculating total steps')
    for i in tqdm(range(num_pieces)):
        with open(tokenized_data_path + 'tokenized_train_{}.txt'.format(i), 'r') as f:
            full_len += len([int(item) for item in f.read().strip().split()])  # token 数量
    total_steps = int(full_len / stride * epochs / batch_size / gradient_accumulation)  # stride是一条样本的长度。768.
    logger.info('full_len = {}, sample_seq_len = {}, total steps = {}'.format(full_len, stride, total_steps))

    # -------------------------------- 优化器、学习率 --------------------------------
    optimizer = transformers.AdamW(model.parameters(), lr=lr, correct_bias=True)
    warmup_steps = total_steps * 0.1  # warm up 10% steps  参考bert等文章，预热5-10%左右。
    # 学习率调整策略，从0增长到lr，再从lr开始衰减到0，总训练步数是total_steps
    scheduler = transformers.WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=total_steps)
    if fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)

    if torch.cuda.device_count() > 1:
        logger.info("Let's use", torch.cuda.device_count(), "GPUs!")
        model = DataParallel(model, device_ids=[int(i) for i in args.device.split(',')])
        multi_gpu = True

    # -------------------------------- 训练 --------------------------------
    logger.info('starting training')
    overall_step = 0
    running_loss = 0
    start_time = time.time()
    for epoch in range(epochs):
        x = np.linspace(0, num_pieces - 1, num_pieces, dtype=np.int32)
        random.shuffle(x)
        piece_num = 0

        # 对文本片段循环
        for i in x:
            with open(tokenized_data_path + 'tokenized_train_{}.txt'.format(i), 'r') as f:
                line = f.read().strip()
            tokens = line.split()
            tokens = [int(token) for token in tokens]
            start_point = 0
            # 从文本片段中生成batch_size个样本，采用最大样本。stride是两个样本的步长间隔。由此可知，样本间的overlap = n_ctx - stride
            samples = []
            while start_point < len(tokens) - n_ctx:
                samples.append(tokens[start_point: start_point + n_ctx])
                start_point += stride  # stride 是控制两个样本的步长间隔
            if start_point < len(tokens):
                samples.append(tokens[len(tokens)-n_ctx:])  # 处理剩余不足n_ctx的情况
            random.shuffle(samples)

            #  batch training
            for step in range(len(samples) // batch_size):  # drop last
                #  prepare data
                batch = samples[step * batch_size: (step + 1) * batch_size]
                batch_inputs = []
                for ids in batch:
                    int_ids = [int(x) for x in ids]
                    batch_inputs.append(int_ids)
                batch_inputs = torch.tensor(batch_inputs).long().to(device)

                #  forward pass
                outputs = model.forward(input_ids=batch_inputs, labels=batch_inputs)  # 输入与标签是同一个tensor
                loss, logits = outputs[:2]

                #  get loss
                if multi_gpu:
                    loss = loss.mean()
                if gradient_accumulation > 1:
                    loss = loss / gradient_accumulation

                #  loss backward
                if fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                #  optimizer step
                if (overall_step + 1) % gradient_accumulation == 0:
                    running_loss += loss.item()
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    lr_current = optimizer.param_groups[0]['lr']
                if (overall_step + 1) % log_step == 0:
                    # 计算已经过去的时间
                    elapsed_time = time.time() - start_time
                    average_time_per_step = elapsed_time / (overall_step + 1)
                    remaining_steps = total_steps - overall_step
                    remaining_time = remaining_steps * average_time_per_step / 60 / 60

                    tb_writer.add_scalar('loss', loss.item() * gradient_accumulation, overall_step)
                    l = f"epoch:{epoch+1}/{epochs}, piece:{piece_num+1}/{len(x)}, " \
                        f"step:{step+1}/{len(samples) // batch_size} " \
                        f"loss:{running_loss * gradient_accumulation / (log_step / gradient_accumulation):.2f} " \
                        f"lr_cur:{lr_current:.6f} " \
                        f"remaining:{remaining_time:.2f}h"
                    logger.info(l)
                    running_loss = 0
                overall_step += 1
            piece_num += 1

        logger.info('saving model for epoch {}'.format(epoch + 1))
        if not os.path.exists(output_dir + 'model_epoch{}'.format(epoch + 1)):
            os.mkdir(output_dir + 'model_epoch{}'.format(epoch + 1))
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(output_dir + 'model_epoch{}'.format(epoch + 1))
        # torch.save(scheduler.state_dict(), output_dir + 'model_epoch{}/scheduler.pt'.format(epoch + 1))
        # torch.save(optimizer.state_dict(), output_dir + 'model_epoch{}/optimizer.pt'.format(epoch + 1))
        logger.info('epoch {} finished'.format(epoch + 1))

    logger.info('training finished')
    if not os.path.exists(output_dir + 'final_model'):
        os.mkdir(output_dir + 'final_model')
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir + 'final_model')
    # torch.save(scheduler.state_dict(), output_dir + 'final_model/scheduler.pt')
    # torch.save(optimizer.state_dict(), output_dir + 'final_model/optimizer.pt')


if __name__ == '__main__':
    main()
