# -*- coding: utf-8 -*-
# @时间     : 2025/3/2 16:04
# @作者     : 王凯
# @File    : p10_模型的保存.py
import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=False)
# 保存模型方式一：
torch.save(vgg16, '../models/vgg16.pth')
# 保存模型方式二：
torch.save(vgg16.state_dict(), '../models_dict/vgg16.pth')


# 方式一的缺陷：
class MINI(nn.Module):
    def __init__(self):
        super(MINI, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)

    def forward(self, x):
        x = self.conv1(x)
        return x

mini = MINI()
torch.save(mini, '../models/mini.pth')