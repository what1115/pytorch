# -*- coding: utf-8 -*-
# @时间     : 2025/3/2 14:27
# @作者     : 王凯
# @File    : p7_损失函数.py
import torch
import torch.nn as nn
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss, MaxPool2d
import torchvision
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=torchvision.transforms.ToTensor())
data_loader = DataLoader(dataset, batch_size=1)

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
loss = nn.CrossEntropyLoss()

for data in data_loader:
    img, label = data
    output = mini(img)
    loss_value = loss(output, label)
    print(loss_value)



input = torch.tensor([1, 2, 3],dtype=torch.float32)
output = torch.tensor([1, 2, 5],dtype=torch.float32)

input = torch.reshape(input,(1,1,1,3))
output = torch.reshape(output,(1,1,1,3))

loss = L1Loss()
# loss = L1Loss(reduction='sum')
result = loss(input,output)
# print(result)

loss_mse = MSELoss()
result_mse = loss_mse(input,output)
# print(result_mse)

x = torch.tensor([0.1, 0.2, 0.3],requires_grad=True)
y = torch.tensor([1])
x = torch.reshape(x,(1,3))
loss_cross = CrossEntropyLoss()
result_cross = loss_cross(x,y)
# print(result_cross)




