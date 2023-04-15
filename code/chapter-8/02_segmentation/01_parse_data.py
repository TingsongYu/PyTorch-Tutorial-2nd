# -*- coding:utf-8 -*-
"""
@file name  : 01_parse_data.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-03-04
@brief      : 解析数据，保存为dataframe形式，并划分数据集
@reference  : https://www.kaggle.com/code/truthisneverlinear/tumor-segmentation-91-accuracy-pytorch
"""
import os
import pandas as pd
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid import ImageGrid
from sklearn.model_selection import train_test_split


def cv_imread(path_file):
    cv_img = cv2.imdecode(np.fromfile(path_file, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    return cv_img


def data_parse():
    """
    根据目录结构，读取图片、标签的路径及患者id
    :return:
    """
    data = []

    # 获取图片信息，存储于dataframe
    for dir_ in os.listdir(data_dir):
        dir_path = os.path.join(data_dir, dir_)
        if os.path.isdir(dir_path):
            for filename in os.listdir(dir_path):
                img_path = os.path.join(dir_path, filename)
                data.append([dir_, img_path])
        else:
            print(f'This is not a dir --> {dir_path}')

    # 分别获取图片与标签的路径信息
    df = pd.DataFrame(data, columns=["patient", "image_path"])
    df_imgs = df[~df["image_path"].str.contains("mask")]
    df_imgs["mask_path"] = df_imgs["image_path"].apply(lambda x: x[:-4] + "_mask.tif")

    # 最终df，包含患者id，图片路径，标签路径
    dff = df_imgs

    # 新增一列判断是否有肿瘤
    def pos_neg_diagnosis(mask_path):
        return_value = 1 if np.max(cv_imread(mask_path)) > 0 else 0
        return return_value
    dff["diagnosis"] = dff["mask_path"].apply(lambda x: pos_neg_diagnosis(x))

    dff.to_csv(PATH_SAVE, index=False)


def data_analysis():
    """
    根据csv文件，分析正负图片比例，分析每个患者的正负图片比例
    :return:
    """

    dff = pd.read_csv(PATH_SAVE)

    ax = dff.diagnosis.value_counts().plot(kind="bar", stacked=True, figsize=(10, 6), color=["violet", "orange"])

    ax.set_xticklabels(["Positive", "Negative"], rotation=45, fontsize=12)
    ax.set_yticklabels("Total Images", fontsize=12)
    ax.set_title("Distribution of Data Grouped by Diagnosis", fontsize=18, y=1.05)

    for i, rows in enumerate(dff.diagnosis.value_counts().values):
        ax.annotate(int(rows), xy=(i, rows - 12), rotation=0, color="white", ha="center", verticalalignment='bottom',
                    fontsize=15, fontweight="bold")

    ax.text(1.2, 2550, f"Total {len(dff)} images", size=15, color="black", ha="center", va="center",
            bbox=dict(boxstyle="round", fc=("lightblue")))
    ax.set_facecolor((0.15, 0.15, 0.15))
    plt.show()
    # --------------------------------------- 观察正负图片数量比例 ------------------------------------------------------------
    patients_by_diagnosis = dff.groupby(["patient", "diagnosis"])["diagnosis"].size().unstack().fillna(0)
    patients_by_diagnosis.columns = ["Positive", "Negative"]
    ax = patients_by_diagnosis.plot(kind="bar", stacked=True, figsize=(18, 10), color=["violet", "springgreen"],
                                    alpha=0.85)
    ax.legend(fontsize=20, loc="upper left")
    ax.grid(False)
    ax.set_xlabel('Patients', fontsize=20)
    ax.set_ylabel('Total Images', fontsize=20)
    ax.set_title("Distribution of data grouped by patient and diagnosis", fontsize=25, y=1.005)
    plt.show()


def data_visual():
    """
    将图片、标签读取并可视化
    :return:
    """
    dff = pd.read_csv(PATH_SAVE)

    # masks
    sample_df = dff[dff["diagnosis"] == 1].sample(5).values

    sample_imgs = []

    for i, data in enumerate(sample_df):
        img = cv2.resize(cv2.imread(data[1]), (IMG_SHOW_SIZE, IMG_SHOW_SIZE))
        mask = cv2.resize(cv2.imread(data[2]), (IMG_SHOW_SIZE, IMG_SHOW_SIZE))
        sample_imgs.extend([img, mask])

    sample_img_arr = np.hstack(sample_imgs[::2])
    sample_mask_arr = np.hstack(sample_imgs[1::2])

    # Plot
    fig = plt.figure(figsize=(25., 25.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(2, 1),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )

    grid[0].imshow(sample_img_arr)
    grid[0].set_title("Images", fontsize=25)
    grid[0].axis("off")
    grid[0].grid(False)

    grid[1].imshow(sample_mask_arr)
    grid[1].set_title("Masks", fontsize=25, y=0.9)
    grid[1].axis("off")
    grid[1].grid(False)

    plt.show()


def data_split():
    """
    将数据划分为训练集、验证集，这里以dataframe形式存储
    :return:
    """
    dff = pd.read_csv(PATH_SAVE)

    # 需要根据患者维度划分，不可通过图片维度划分，以下代码可用于常见的csv划分
    grouped = dff.groupby('patient')
    # grouped = dff.groupby('image_path')  # bad method
    train_set, val_set = train_test_split(list(grouped), train_size=train_size, random_state=42)
    train_set, val_set = [ii[1] for ii in train_set], [ii[1] for ii in val_set]  # 提取dataframe
    train_df, val_df = pd.concat(train_set), pd.concat(val_set)  # 合并dataframe

    train_df.to_csv(PATH_SAVE_TRAIN, index=False)
    val_df.to_csv(PATH_SAVE_VAL, index=False)
    print(f"Train: {train_df.shape} \nVal: {val_df.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--data-path", default=r"G:\deep_learning_data\brain-seg\kaggle_3m",
                        type=str, help="dataset path")
    args = parser.parse_args()

    data_dir = args.data_path  # xxx/kaggle_3m
    PATH_SAVE = 'data_info.csv'
    # PATH_SAVE_TRAIN = 'data_train_split_by_img.csv'
    # PATH_SAVE_VAL = 'data_val_split_by_img.csv'
    PATH_SAVE_TRAIN = 'data_train.csv'
    PATH_SAVE_VAL = 'data_val.csv'
    IMG_SHOW_SIZE = 512  # 可视化时，图像大小
    train_size = 0.8  # 训练集划分比例，80%

    data_parse()  # 读取根目录下数据信息，存储为csv
    data_analysis()  # 分析数据数量、比例
    data_visual()  # 可视化原图与标签
    data_split()  # 划分训练集、验证集





