# -*- coding: utf-8 -*-
# @时间     : 2025/3/2 15:47
# @作者     : 王凯
# @File    : p9_现有网络模型的使用及修改.py
import torchvision
from torch.nn import Linear

vgg16_False = torchvision.models.vgg16(pretrained=False)
vgg16_True = torchvision.models.vgg16(pretrained=True)
print(vgg16_True)

train_data = torchvision.datasets.CIFAR10(root='../data',train=True,transform=torchvision.transforms.ToTensor(),
                                          download=True)

# 修改网络结构
vgg16_True.add_module("add_linear", Linear(4096, 1000))
vgg16_True.add_linear = Linear(1000, 10)
print("vgg16_True:")
print(vgg16_True)
# 修改最后一层
vgg16_False.classifier[6] = Linear(4096, 10)
print("vgg16_False:")
print(vgg16_False)