# -*- coding: utf-8 -*-
# @时间     : 2025/2/22 23:13
# @作者     : 王凯
# @File    : p1_神经网络框架.py

# 导入 PyTorch 库和神经网络模块
import torch
import torch.nn as nn


# 定义一个名为 MINI 的神经网络类，继承自 nn.Module
class MINI(nn.Module):
    # 定义构造函数，初始化模型的参数和层
    def __init__(self):
        # 调用父类的构造函数，完成初始化
        super().__init__()

    # 定义前向传播函数，描述输入到输出的计算过程

    def forward(self, x):
        # 将输入张量加上 1，作为输出
        output = x + 1
        # 返回计算后的张量
        return output


# 创建 MINI 类的实例，命名为 mini
mini = MINI()

# 创建一个张量，值为 1.0
input = torch.tensor(1.0)

# 打印张量的类型，输出为 <class 'torch.Tensor'>
print(type(input))
# 打印张量的值，输出为 tensor(1.)
print(input)

# 调用模型 mini 的 forward 方法，将输入张量 input 传递进去
x = mini(input)

# 打印输出张量的类型，仍然是 <class 'torch.Tensor'>
print(type(x))
# 打印输出张量的值，结果为 tensor(2.)
print(x)
