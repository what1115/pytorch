# -*- coding: utf-8 -*-
# @时间     : 2025/3/2 16:05
# @作者     : 王凯
# @File    : p10_模型的加载.py
import torch
import torchvision
from torch import nn
from torchvision.models import vgg16

# 方式一：加载整个模型
model = torch.load("../models/vgg16.pth")
# print(model)

# 方式二：加载模型的部分参数
vgg16_model = torchvision.models.vgg16(pretrained=False)
# print(vgg16_model)
vgg16_model.load_state_dict(torch.load("../models_dict/vgg16.pth"))
# print(vgg16_model)

#方式一的缺陷：------->  AttributeError: Can't get attribute 'MINI' on <module '__main__' from 'E:\\Code\\pycharm_code\\torch\\神经网络的基本骨架\\p10_模型的加载.py'>
#原因：MINI类定义在p10_模型的保存.py文件中，而p10_模型的加载.py文件是被当做主程序来运行的，所以MINI类无法被其他文件引用。
#解决方法：将MINI类定义在一个单独的文件中，然后在p10_模型的加载.py文件中导入该文件。
class MINI(nn.Module):
    def __init__(self):
        super(MINI, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)

    def forward(self, x):
        x = self.conv1(x)
        return x


mini = torch.load("../models/mini.pth")
print(mini)