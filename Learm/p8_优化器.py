# -*- coding: utf-8 -*-
# @时间     : 2025/3/2 15:18
# @作者     : 王凯
# @File    : p8_优化器.py
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MaxPool2d
import torchvision
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=torchvision.transforms.ToTensor())
data_loader = DataLoader(dataset, batch_size=4)

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
        from torch.nn import Sequential
        from torch.nn import Conv2d
        from torch.nn import Flatten
        from torch.nn import Linear
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

mini = MINI()
loss = CrossEntropyLoss()
optim = torch.optim.SGD(mini.parameters(),lr=0.01)

for epoch in range(10):
    run_loss = 0.0
    for data in data_loader:
        img, label = data
        output = mini(img)
        loss_value = loss(output, label)
        optim.zero_grad()
        loss_value.backward()
        optim.step()
        run_loss = run_loss + loss_value
    print(f'epoch:{epoch+1}, loss:{run_loss}')