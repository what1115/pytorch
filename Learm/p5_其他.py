# -*- coding: utf-8 -*-
# @时间     : 2025/2/28 09:55
# @作者     : 王凯
# @File    : p5_其他.py
import torch
import torch.nn as nn
from torch.nn import Linear
import torchvision
from torch.utils.data import DataLoader

class MINI(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, x):
        x = self.linear1(x)
        return x

dataset = torchvision.datasets.CIFAR10(root='../data',train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataLoader = DataLoader(dataset, batch_size=64)

mini = MINI()


for data in dataLoader:
    images, labels = data
    print(images.shape)
    # output = torch.reshape(images,(1,1,1,-1))
    output = torch.flatten(images)
    print(output.shape)
    output = mini(output)
    print(output.shape)
    break