# -*- coding: utf-8 -*-
# @时间     : 2025/2/23 15:08
# @作者     : 王凯
# @File    : p3_池化.py
import torch
import torch.nn as nn

input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]],dtype=torch.float32)
input = torch.reshape(input,(-1,1,5,5))
print('input.shape:')
print(input.shape)

class MINI(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxPool1 = nn.MaxPool2d(kernel_size=3, ceil_mode=False)
        """-------参数----------
                nn.MaxPool2d(
                    kernel_size: 池化窗口的大小，用于在该窗口内取最大值。
                    stride: 窗口的步幅。默认值为 `kernel_size`。
                    padding: 在两侧隐式添加的负无穷大填充。
                    dilation: 控制窗口中元素间距的参数。
                    return_indices: 如果为 `True`，将返回最大值的索引，与输出一起返回。这对于后续使用 `torch.nn.MaxUnpool2d` 很有用。
                    ceil_mode: 当为 `True` 时，计算输出形状时将使用 `ceil` 而不是 `floor`。
                )
        """

    def forward(self,x):
        x = self.maxPool1(x)
        return x

mini = MINI()
output = mini(input)

print("output:")
print(output)

