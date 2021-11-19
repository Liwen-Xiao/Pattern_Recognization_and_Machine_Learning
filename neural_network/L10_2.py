import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
import random

sys.path.append("..")
import d2lzh_pytorch as d2l

print(torch.__version__)

from sklearn import datasets
dataset = datasets.load_iris()
data = dataset['data']
iris_type = dataset['target']
print(data)
print(iris_type)

input = torch.FloatTensor(dataset['data'])
print(input)
label = torch.LongTensor(dataset['target'])
print(label)

import torch.nn.functional as Fun


# 定义BP神经网络
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = Fun.relu(self.hidden(x))
        x = self.out(x)
        return x

class net_4541(nn.Module):
    '''
    LeNet网络的类
    '''

    def __init__(self):
        super(net_4541, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 5),
            nn.Sigmoid(),
            nn.Linear(5, 4),
            nn.Sigmoid(),
            nn.Linear(4, 4)
        )

    def forward(self, x):
        output = self.fc(x)
        return output

net = net_4541()
optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
# SGD:随机梯度下降法
loss_func = torch.nn.CrossEntropyLoss()
# 设定损失函数

for i in range(1000):
    out = net(input)
    print(out)
    loss = loss_func(out, label)
    # 输出与label对比
    optimizer.zero_grad()
    # 初始化
    loss.backward()
    optimizer.step()

out = net(input)
# out是一个计算矩阵
prediction = torch.max(out, 1)[1]
pred_y = prediction.numpy()
# 预测y输出数列
target_y = label.data.numpy()
# 实际y输出数据
print(pred_y)
print(target_y)
