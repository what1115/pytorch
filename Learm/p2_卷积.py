# -*- coding: utf-8 -*-
# @时间     : 2025/2/23 19:13
# @作者     : 王凯
# @File    : p2_卷积.py

import torch  # 导入PyTorch库，提供深度学习所需的张量计算和自动求导功能
import torchvision  # 导入Torchvision库，提供常用数据集、模型架构和图像变换工具
import torch.nn as nn  # 导入神经网络模块，包含各种网络层的定义
from PIL import Image  # 导入PIL库的Image模块，用于图像的读取、处理和保存
from torch.utils.data import DataLoader  # 导入数据加载器，用于批量加载数据
from torchvision import transforms  # 导入图像变换工具，用于数据预处理和增强
from torch.utils.tensorboard import SummaryWriter  # 导入TensorBoard工具，用于可视化训练过程


# 定义一个名为MINI的神经网络模型，继承自nn.Module（PyTorch所有神经网络的基类）
class MINI(nn.Module):
    def __init__(self):
        super().__init__()  # 调用父类的初始化方法，确保正确初始化nn.Module
        # 定义一个2D卷积层：
        # - in_channels=1: MNIST图像是灰度图，只有1个通道
        # - out_channels=3: 输出3个特征图（相当于3个不同的卷积滤波器）
        # - kernel_size=3: 使用3×3的卷积核
        # - stride=1: 卷积核每次移动1个像素
        # - padding=0: 不进行边缘填充，会导致输出图像尺寸减小
        self.Conv2d = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=0)
        """
        nn.Conv2d参数详解:
        1-->    in_channels: int,  # 输入特征图的通道数
        2-->    out_channels: int,  # 输出特征图的通道数
        3-->    kernel_size: Union[int, Tuple[int, int]],  # 卷积核大小，可以是单个整数或(高,宽)元组
        4-->    stride: Union[int, Tuple[int, int]] = 1,  # 卷积步幅，控制滤波器移动的距离
        5-->    padding: Union[int, Tuple[int, int]] = 0,  # 输入周围填充的大小，用于控制输出尺寸
        6    dilation: Union[int, Tuple[int, int]] = 1,  # 卷积核元素之间的间距，用于空洞卷积
        7    groups: int = 1,  # 控制输入和输出之间的连接，用于分组卷积
        8    bias: bool = True,  # 是否添加可学习的偏置
        9    padding_mode: str = 'zeros'  # 填充模式，默认用0填充
            )
        """

    # 定义前向传播方法，在每次调用模型时执行
    def forward(self, x):
        x = self.Conv2d(x)  # 将输入x通过卷积层处理
        return x  # 返回卷积后的输出特征图


# 下载并加载MNIST数据集:
# - train=False: 使用测试集而非训练集
# - download=True: 如果本地没有数据集则下载
# - transform=transforms.ToTensor(): 将PIL图像转换为张量，并将像素值归一化到[0,1]
data = torchvision.datasets.MNIST("../../data", train=False, download=True, transform=transforms.ToTensor())

# 创建数据加载器，用于批量处理数据
# - batch_size=4: 每批加载4张图像
dataloader = DataLoader(data, batch_size=4)

# 创建TensorBoard的写入器，用于可视化结果
# 日志将保存在../logs/MNIST目录下
writer = SummaryWriter("../../logs/MNIST")

# 实例化MINI模型
mini = MINI()
print(mini)  # 打印模型结构，显示层次和参数信息
step = 0  # 初始化步数计数器，用于TensorBoard记录

# 遍历数据加载器中的批次
for data in dataloader:
    img, label = data  # 获取一批图像和对应的标签（注意：代码中写成了loader，应为label）
    output = mini(img)  # 将图像输入模型，进行前向传播，获取输出特征图

    # 打印形状信息，帮助理解卷积操作对图像尺寸的影响
    print(img.shape)  # 输入形状应为[4, 1, 28, 28]：4张图像，1个通道，28×28像素
    print(output.shape)  # 输出形状应为[4, 3, 26, 26]：4张图像，3个通道，26×26像素
    # 尺寸减小是因为使用3×3卷积核且无填充(padding=0)

    # 将原始图像和卷积后的输出添加到TensorBoard，便于可视化比较
    writer.add_images("img", img, step)  # 记录原始图像
    writer.add_images("output", output, step)  # 记录卷积后的特征图
    step += 1  # 步数计数器加1，用于TensorBoard中区分不同批次

writer.close()  # 关闭TensorBoard写入器，确保所有数据都被保存