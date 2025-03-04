# -*- coding: utf-8 -*-
# @时间     : 2025/2/25 22:31
# @作者     : 王凯
# @File    : p4_非线性激活.py
import torch
import torchvision
from torch import nn
from torch.nn import Sigmoid,ReLU
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


input = torch.tensor([[1,-0.5],
                      [-1,3]])
input = torch.reshape(input,(-1,1,2,2))
print(input.shape)

dataset = torchvision.datasets.CIFAR10(root='../data',train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataLoader = DataLoader(dataset, batch_size=4)

writer = SummaryWriter('../../logs/cifar10')

class MINI(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu1 = ReLU(inplace=True)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        x = self.relu1(x)
        return x

mini = MINI()
step = 0
for data in dataLoader:
    img, label = data
    writer.add_images('input', img, step)  # 记录输入图像
    output = mini(img)
    writer.add_images('output', output, step)  # 记录输出图像
    step += 1

writer.close()