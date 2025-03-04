# -*- coding: utf-8 -*-
# @时间     : 2025/3/3 14:37
# @作者     : 王凯
# @File    : model.py
import torch
from torch import nn


class MINI(nn.Module):
    def __init__(self):
        super(MINI, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=5,stride=1,padding=2), # padding 填充
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024,64),
            nn.Linear(64,10)
        )
    def forward(self,x):
        x = self.model(x)
        return x

if __name__ == '__main__':
    input = torch.ones(64,3,32,32)
    mini = MINI()
    print(mini)
    output = mini(input)
    print(output.shape)

