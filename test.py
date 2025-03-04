# -*- coding: utf-8 -*-
# @时间     : 2025/3/4 09:24
# @作者     : 王凯
# @File    : test.py
import torch
import torchvision
from PIL import Image
from matplotlib.font_manager import weight_dict
from torch import nn
from torchvision.datasets import CIFAR10

from model import MINI

img_path = "../Dataset//test//cat//cat.10000.jpg"
img = Image.open(img_path)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                               torchvision.transforms.ToTensor()])
image = transform(img)
print(image.shape)
image = torch.reshape(image,(1,3,32,32))

mini = MINI()
mini.load_state_dict(torch.load("./models_dict/mini.pth",weights_only=True))
mini.eval()
with torch.no_grad():
    output = mini(image)
    # print(output)
    max_values, max_indices = torch.max(output, dim=1)  # dim=1 表示按行操作
    # print("按行最大值:", max_values)
    print("按行最大值的索引:", max_indices)
    _10 = ["飞机","汽车","鸟类","猫","鹿","狗","蛙类","马","船和卡车"]
    print(_10[max_indices])