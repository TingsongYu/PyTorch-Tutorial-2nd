"""
@file name  : a_data_split.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-10-19
@brief      : 训练、测试数据集划分
"""
import os
import random


def split_dataset(input_file, train_file, valid_file, split_ratio_):

    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    random.shuffle(lines)

    split_index = int(len(lines) * split_ratio_)
    print("总共{}行，训练集:{}行， 测试集:{}行".format(len(lines), split_index, len(lines)-split_index))

    train_data = lines[:split_index]
    valid_data = lines[split_index:]

    with open(train_file, 'w', encoding='utf-8') as file:
        file.writelines(train_data)
    print(f"训练集保存成功，位于:{train_file}")

    with open(valid_file, 'w', encoding='utf-8') as file:
        file.writelines(valid_data)
    print(f"测试集保存成功，位于:{valid_file}")


if __name__ == '__main__':
    random.seed(42)

    data_dir = r"G:\deep_learning_data\machine_transfer\cmn-eng"
    path_raw = os.path.join(data_dir, "cmn.txt")
    path_train = os.path.join(data_dir, "train.txt")
    path_test = os.path.join(data_dir, "test.txt")
    split_ratio = 0.8

    split_dataset(path_raw, path_train, path_test, split_ratio)
