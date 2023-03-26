# -*- coding:utf-8 -*-
"""
@file name  : inference_main.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-03-02
@brief      : 推理脚本
"""

import argparse
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def visdrone2yolo(dir):
    def convert_box(size, box):
        # Convert VisDrone box to YOLO xywh box
        dw = 1. / size[0]
        dh = 1. / size[1]
        return (box[0] + box[2] / 2) * dw, (box[1] + box[3] / 2) * dh, box[2] * dw, box[3] * dh

    (dir / 'labels').mkdir(parents=True, exist_ok=True)  # make labels directory  pathlib库的作用！
    pbar = tqdm((dir / 'annotations').glob('*.txt'), desc=f'Converting {dir}')
    for f in pbar:
        img_size = Image.open((dir / 'images' / f.name).with_suffix('.jpg')).size
        lines = []
        with open(f, 'r') as file:  # read annotation.txt
            for row in [x.split(',') for x in file.read().strip().splitlines()]:
                if row[4] == '0':  # VisDrone 'ignored regions' class 0
                    continue
                cls = int(row[5]) - 1
                box = convert_box(img_size, tuple(map(int, row[:4])))
                lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")
                txt_path = str(f).replace(os.sep + 'annotations' + os.sep, os.sep + 'labels' + os.sep)
                with open(txt_path, 'w') as fl:
                    fl.writelines(lines)  # write label.txt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--data-path", default=r'G:\deep_learning_data\VisDrone',
                        type=str, help="dataset path")
    args = parser.parse_args()
    root_dir = args.data_path  # dataset root dir

    dataset_list = ['VisDrone2019-DET-train', 'VisDrone2019-DET-val', 'VisDrone2019-DET-test-dev']
    for name in dataset_list:
        visdrone2yolo(Path(root_dir, name))  # convert VisDrone annotations to YOLO labels

