import re
import matplotlib.pyplot as plt

path_log = r"outputsbert/bert-cluener-2024-02-26_14-53-18.log"
# 正则表达式匹配一行中的step、train_loss和eval_loss
pattern = re.compile(r'step:(\d+)\ttrain_loss: (\d+\.\d+)\teval_loss: (\d+\.\d+)')

steps = []
train_losses = []
eval_losses = []

# 假设log文件名为"log.txt"，并且位于同一目录下
with open(path_log, 'r') as file:
    for line in file:
        match = pattern.search(line)
        if match:
            step, train_loss, eval_loss = match.groups()
            steps.append(int(step))
            train_losses.append(float(train_loss))
            eval_losses.append(float(eval_loss))

steps, train_losses, eval_losses = zip(*sorted(zip(steps, train_losses, eval_losses)))

# 绘制折线图
plt.figure(figsize=(10, 5))
plt.plot(steps, train_losses, label='train_loss')
plt.plot(steps, eval_losses, label='eval_loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training and Evaluation Loss')
plt.legend()
plt.grid(True)
plt.show()
