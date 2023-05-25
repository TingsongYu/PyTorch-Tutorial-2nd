# -*- coding:utf-8 -*-
"""
@file name  : 02_flask_app.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-04-30
@brief      : web server base flask
"""
from flask import Flask, render_template, request, jsonify
from datetime import datetime
from retrieval_by_faiss import *

# 设置软连接，将图像数据文件夹链接到flask的静态文件的文件夹下，flask才能访问图片
img_root_dir = CFG.image_file_dir  # 图像数据库所在位置
static_file_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'img')
if not os.path.exists(static_file_dir):
    os.makedirs(static_file_dir)

print('***创建软连接，将图像数据文件夹链接到flask的静态文件的文件夹下，flask才能访问图片***\n\n')
print(f'windows需采用管理员权限打开cmd，执行以下命令：\nmklink /D {static_file_dir} {img_root_dir}')
print(f'linux系统，执行以下命令：\nln -s {img_root_dir} {static_file_dir}')
print('***创建软连接，将图像数据文件夹链接到flask的静态文件的文件夹下，flask才能访问图片***\n\n')

# 检索模块初始化
with open(CFG.feat_mat_path, 'rb') as f:
    feat_mat = pickle.load(f)
with open(CFG.map_dict_path, 'rb') as f:
    map_dict = pickle.load(f)

ir_model = ImageRetrievalModule(CFG.index_string, CFG.feat_dim, feat_mat, map_dict,
                                CFG.clip_backbone_type, CFG.device)


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    image_file = request.files['image']
    text = request.form['text']
    if image_file:
        def save_img(file, out_dir):
            now = datetime.now()
            current_time = now.strftime("%Y-%m-%d_%H-%M-%S")
            file_name = '{}-{}'.format(current_time, file.filename)
            path_img = os.path.join(out_dir, file_name)
            file.save(path_img)
            return path_img
        path_to_img = save_img(image_file, app.static_folder)
        query = path_to_img
    else:
        query = text

    # step2: 检索
    distance_result, index_result, path_list = ir_model.retrieval_func(query, CFG.topk)

    # step3: 结果封装
    results = []
    for distance, path in zip(distance_result, path_list):
        dict_ = {'path':  os.path.join('static', 'img', os.path.basename(path)),
                 'text': f'distance: {distance:.3f}'}
        results.append(dict_)

    # import random
    # names = ['000000000144.jpg', '000000000081.jpg', '000000000154.jpg']
    # name = random.choice(names)
    # paths = os.path.join('static', 'img', name)
    # path_dict = {'path': paths, 'text': 'result1'}
    # results = [path_dict] * 10

    return jsonify(results)


if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=5000)
    app.run(host='localhost', port=5000)
