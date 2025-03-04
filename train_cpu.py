# -*- coding: utf-8 -*-
# @时间     : 2025/3/3 20:17
# @作者     : 王凯
# @File    : train_cpu.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
import time

from tqdm import tqdm


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
    #模型准确率
    train_acc = 0
    # 训练次数
    # train_step = 0
    # 测试次数
    # test_step = 0
    # 训练轮树
    for i in range(epoch):
        for data in tqdm(train_loader,"第 {} 轮训练.".format(i + 1)):
            # 优化器清零
            optimizer.zero_grad()
            img, target = data
            output = mini(img)
            # 损失函数
            train_loss = loss_cross(output, target)
            train_loss.backward()
            optimizer.step()
            # train_step += 1
            # writer.add_scalar('train_loss', train_loss, train_step)
            # if train_step % 100 == 0:
            #     print("训练次数：{}，Loss:{}".format(train_step, train_loss.item()))

        test_loss = 0
        test_acc = 0
        with torch.no_grad():
            for data in test_loader:
                img, target = data
                output = mini(img)
                # 损失函数
                loss = loss_cross(output, target)
                test_loss = test_loss + loss.item()
                acc1 = (output.argmax(1) == target).sum()
                test_acc = test_acc + acc1
        # writer.add_scalar('test_loss', test_loss, test_step)
        # writer.add_scalar('test_acc', test_acc / test_len, test_step)
        print()
        acc = test_acc / test_len
        print("测试集的Acc(准确率):{}".format(acc))
        print("测试集的Loss(损失函数):{}".format(test_loss))
        # test_step += 1
        #保存模型
        if acc > train_acc:
            train_acc = acc
            torch.save(mini.state_dict(), './models_dict/mini.pth')

    print()
    print("最后模型准确率：{}".format(train_acc))



train_data = torchvision.datasets.CIFAR10(root='../data',train=True,transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root='../data',train=False,transform=torchvision.transforms.ToTensor(),
                                         download=True)
train_loader = DataLoader(train_data,batch_size=64)
test_loader = DataLoader(train_data,batch_size=64)
train_len = len(train_loader)
test_len = len(test_loader)
print("训练集：{}".format(train_len))
print("测试集：{}".format(test_len))
mini = MINI()
# 损失函数
loss_cross = nn.CrossEntropyLoss()
# 优化器
lr = 0.01
optimizer = optim.SGD(mini.parameters(),lr=lr)
#训练轮数
epoch = 5
# writer = SummaryWriter('../logs/cifar10')
#训练开始时间
time_1 = time.time()
train_model()
#训练结束时间
time_2 = time.time()
#用时 time
time = time_2 - time_1
# 将秒转换为时:分:秒
hours, remainder = divmod(time, 3600)  # 1小时 = 3600秒
minutes, seconds = divmod(remainder, 60)      # 1分钟 = 60秒
# 打印格式化后的时间
print(f"程序运行时间: {int(hours)}时 {int(minutes)}分 {int(seconds)}秒")
# writer.close()
