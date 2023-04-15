import torch

pt = torch.load('best.pt')

for k, v in pt.items():
    print(k)