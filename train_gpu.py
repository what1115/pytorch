# -*- coding: utf-8 -*-
# @时间     : 2025/3/3 20:20
# @作者     : 王凯
# @File    : train_gpu.py
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from datetime import timedelta
import time
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("训练设备：",device)
print("torch版本：",torch.__version__)

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


def train_model():
    for i in range(epoch):
        for data in tqdm(train_loader,"第 {} 轮训练".format(i + 1)):
            # 优化器清零
            optimizer.zero_grad()
            img, target = data
            img = img.to(device)
            target = target.to(device)
            output = mini(img)
            # 损失函数
            train_loss = loss_cross(output, target)
            train_loss.backward()
            optimizer.step()
        test_loss = 0
        test_acc = 0
        with torch.no_grad():
            for data in test_loader:
                img, target = data
                img = img.to(device)
                target = target.to(device)
                output = mini(img)
                # 损失函数
                loss = loss_cross(output, target)
                test_loss = test_loss + loss.item()
                acc = (output.argmax(1) == target).sum()
                test_acc = test_acc + acc
        loss = test_loss /  test_len
        acc = test_acc / test_len
        print("第 {} 轮训练完成".format(i + 1))
        print(f"测试集的Acc(准确率):{acc* 100:.2f}%")
        print(f"测试集的Loss(损失函数):{loss:.4f}")


train_data = torchvision.datasets.CIFAR10(root='./data',train=True,transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root='./data',train=False,transform=torchvision.transforms.ToTensor(),
                                         download=True)

train_len = len(train_data)
test_len = len(test_data)
print("训练集：{}".format(train_len))
print("测试集：{}".format(test_len))

train_loader = DataLoader(train_data,batch_size=64)
test_loader = DataLoader(test_data,batch_size=64)

mini = MINI()
mini.to(device)
# 损失函数
loss_cross = nn.CrossEntropyLoss()
loss_cross.to(device)
# 优化器
lr = 0.01
optimizer = torch.optim.SGD(mini.parameters(),lr=lr)
#训练轮数
epoch = 10
#训练开始时间
time_1 = time.time()
train_model()
#训练结束时间
time_2 = time.time()
#用时 time
time = time_2 - time_1
str_time = str(timedelta(seconds=time))
print(f"程序运行时间: {str_time}")
