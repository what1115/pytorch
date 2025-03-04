# -*- coding: utf-8 -*-
# @时间     : 2025/2/28 10:23
# @作者     : 王凯
# @File    : p6_实战CIFAR10.py
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter

class MINI(nn.Module):

    """
    MINI 神经网络结构===> 图片地址在 ./神经网络的基本骨架/实战_神经网络框架图.png
    """

    """
    原始的神经网路结构：
        def __init__(self):
            super().__init__()
            self.conv1 = Conv2d(in_channels=3,out_channels=32,kernel_size=5,padding=2)
            self.maxPool1 = MaxPool2d(2)
            self.conv2 = Conv2d(in_channels=32,out_channels=32,kernel_size=5,padding=2)
            self.maxPool2 = MaxPool2d(2)
            self.conv3 = Conv2d(in_channels=32,out_channels=64,kernel_size=5,padding=2)
            self.maxPool3 = MaxPool2d(2)
            self.flatten = Flatten()
            self.linear1 = Linear(1024,64)
            self.linear2 = Linear(64,10)
            
    
        def forward(self,x):
            x = self.conv1(x)
            x = self.maxPool1(x)
            x = self.conv2(x)
            x = self.maxPool2(x)
            x = self.conv3(x)
            x = self.maxPool3(x)
            x = self.flatten(x)
            x = self.linear1(x)
            x = self.linear2(x)
            return x
    """

    """     简化后的神经网络结构：            """
    def __init__(self):
        super().__init__()
        self.model = Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
            MaxPool2d(2),
            Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            MaxPool2d(2),
            Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x




if __name__ == '__main__':
    writer = SummaryWriter(log_dir='../../logs/test')
    input = torch.ones((64, 3, 32, 32))
    mini = MINI()

    # print(mini)
    writer.add_graph(mini, input_to_model=input)
    output = mini(input)

    print(output.shape)
    writer.close()