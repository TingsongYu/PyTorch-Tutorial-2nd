# -*- coding:utf-8 -*-
"""
@file name  : 02_inference.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-04-11
@brief      : 模型推理

"""
import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.environ['HF_ENDPOINT'] = "https://ai.gitee.com/huggingface"
import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # project root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

# debug: windows下会报错：OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
import platform

if platform.system() == 'Windows':
    import os

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import clip
import glob
import cv2
import numpy as np
import skimage.io as io
import PIL.Image

from transformers import GPT2Tokenizer
from tqdm import tqdm
from my_models.models import *
from my_utils.utils import generate_beam, generate2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='./data/coco/oscar_split_ViT-B_32_train.pkl')
parser.add_argument('--prefix', default='coco_prefix', help='prefix for saved filenames')
parser.add_argument('--prefix_length', type=int, default=40)
parser.add_argument('--prefix_length_clip', type=int, default=40)
parser.add_argument('--only_prefix', dest='only_prefix', action='store_true')
parser.add_argument('--mapping_type', type=str, default='mlp', help='mlp/transformer')
parser.add_argument('--num_layers', type=int, default=8)
args = parser.parse_args()


class Predictor(object):
    def __init__(self, path_ckpt):
        self.device = device
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device, jit=False)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.prefix_length = 40

        model = ClipCaptionPrefix(args.prefix_length, clip_length=args.prefix_length, prefix_size=512,
                                  num_layers=args.num_layers, mapping_type=args.mapping_type)

        model.load_state_dict(torch.load(path_ckpt, map_location=torch.device("cpu")), strict=False)
        model = model.eval()
        model = model.to(self.device)
        self.model = model

    def predict(self, image, use_beam_search):
        """Run a single prediction on the model"""
        image = io.imread(image)
        pil_image = PIL.Image.fromarray(image)
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prefix = self.clip_model.encode_image(image).to(self.device, dtype=torch.float32)
            prefix_embed = self.model.clip_project(prefix).reshape(1, self.prefix_length, -1)
        if use_beam_search:
            return generate_beam(self.model, self.tokenizer, embed=prefix_embed)[0], pil_image
        else:
            return generate2(self.model, self.tokenizer, embed=prefix_embed), pil_image


def main():
    # download from :提取码：mqri](https://pan.baidu.com/s/1CuTDtCeT2-nIvRG7N4iKtw)
    ckpt_path = r'coco_prefix-009-2023-0411.pt'
    path_img = r'G:\deep_learning_data\coco_2014\images\val2014'
    # path_img = r'G:\deep_learning_data\coco_2017\images\train2017\train2017'
    out_dir = './inference_output2'

    # 获取路径
    img_paths = []
    if os.path.isfile(path_img):
        img_paths.append(path_img)
    if os.path.isdir(path_img):
        img_paths_tmp = glob.glob(os.path.join(path_img, '**/*.jpg'), recursive=True)
        img_paths.extend(img_paths_tmp)

    # 初始化模型
    predictor = Predictor(ckpt_path)

    for idx, path_img in tqdm(enumerate(img_paths)):
        caps, pil_image = predictor.predict(path_img, False)
        img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        cv2.putText(img_bgr, caps, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # 保存
        path_out = os.path.join(out_dir, os.path.basename(path_img))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        cv2.imwrite(path_out, img_bgr)


if __name__ == '__main__':
    main()
