import matplotlib
matplotlib.rcParams['font.family'] = 'SimHei'
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

# 读取txt文件并获取数据
def read_data_from_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        data = [tuple(map(int, line.strip().split(','))) for line in lines]
    return data

# 假设txt文件名为'gpu_usage_log.txt'，并放在当前目录下
# file_path = 'gpu_usage_log-240406.txt'
# file_path = 'gpu_usage_log-240409-7b-int4.txt'
# file_path = 'gpu_usage_log - chatglm3.txt'
file_path = 'gpu_usage_log -baichuan2.txt'
data = read_data_from_txt(file_path)

# 分解数据点
x_values, y_values_mb = zip(*data)
y_values_gb = [y / 1024 for y in y_values_mb]

# 绘制折线图，并在每个数据点上显示y轴的具体数据（以GB为单位）
plt.figure(figsize=(10, 6))
for x, y in zip(x_values, y_values_gb):
    plt.text(x, y, f'{y:.2f} GB', ha='right')

# 绘制折线图
plt.plot(x_values, y_values_gb, marker='o')
plt.title('文本长度与显存占用的关系-Baichuan2-7B-Int4')
# plt.title('文本长度与显存占用的关系-ChatGLM3-6B-Int4')
# plt.title('文本长度与显存占用的关系-Qwen-7B-Int4')
# plt.title('文本长度与显存占用的关系-Qwen-1.8B')
plt.xlabel('文本长度')
plt.ylabel('显存使用量')
# plt.grid(True)
plt.show()

