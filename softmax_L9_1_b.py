'''
batch_size=1
'''

import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
import time
import matplotlib.pyplot as plt

sys.path.append("..")
import d2lzh_pytorch as d2l
from sklearn import datasets
import random
import pandas as pd


class softmax(nn.Module):
    '''
    LeNet网络的类
    '''

    def __init__(self):
        super(softmax, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 3),
        )

    def forward(self, img):
        output = self.fc(img)
        return output


def initialize(net):
    '''
    网络实例参数（权重、bias）初始化
    :param net: 网络的实例
    :return:
    '''
    for name, param in net.named_parameters():
        if 'weight' in name:
            init.normal_(param, mean = 0, std = 0.1)
        if 'bias' in name:
            init.constant_(param, val = 0)
    print('initialize done')


def get_train_and_test_item(X, Y, total_num, train_num):
    L_train = random.sample(range(1, total_num + 1), train_num)
    X_train_list = []
    Y_train_list = []
    X_test_list = []
    Y_test_list = []
    for i in range(1, total_num + 1):
        if i in L_train:
            X_train_list.append(X[i - 1])
            Y_train_list.append(Y[i - 1])
        else:
            X_test_list.append(X[i - 1])
            Y_test_list.append(Y[i - 1])
    X_train_tensor = torch.FloatTensor(X_train_list)
    Y_train_tensor = torch.LongTensor(Y_train_list)
    X_test_tensor = torch.FloatTensor(X_test_list)
    Y_test_tensor = torch.LongTensor(Y_test_list)
    return [X_train_tensor, Y_train_tensor, X_test_tensor, Y_test_tensor]


def draw(loss_list, train_acc_list, test_acc_list):
    '''
    画出损失、训练样本集正确率、测试样本集正确率随epcoh的变化
    :param loss_list:
    :param train_acc_list:
    :param test_acc_list:
    :return:
    '''
    # plot中参数的含义分别是横轴值，纵轴值，颜色，透明度和标签
    x = [i + 1 for i in range(10)]
    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.plot(np.array(train_acc_list), color = 'blue', label = 'train_acc')

    plt.plot(np.array(test_acc_list),color = 'red', label = 'test_acc')

    # 显示标签，如果不加这句，即使加了label='一些数字'的参数，最终还是不会显示标签
    plt.legend(loc = "upper right")
    plt.xlabel('epoch_num')
    plt.ylabel('accuracy')

    plt.figure(1)
    plt.subplot(1, 2, 2)
    plt.plot(loss_list, color = 'blue', label = 'loss')

    # 显示标签，如果不加这句，即使加了label='一些数字'的参数，最终还是不会显示标签
    plt.legend(loc = "upper right")
    plt.xlabel('epoch_num')
    plt.ylabel('accuracy')

    plt.show()


def evaluate_accuracy_my_for_array(X_test_tensor, Y_test_tensor, net):
    acc_sum, n = 0.0, 0
    for i in range(60):
        y_hat = net(X_test_tensor[i])
        acc_sum += ((y_hat).argmax(dim = 0) == Y_test_tensor[
            i]).float().sum().cpu().item()  # 这里dim改为0才是对的，不能是1（调了好久才发现）
        n += 1
    return acc_sum / (n)


def train_ch5_my_for_array(net, X_train_tensor, Y_train_tensor, X_test_tensor, Y_test_tensor, batch_size, optimizer, device, num_epochs):
    loss = torch.nn.CrossEntropyLoss()
    loss_list = []
    train_acc_list = []
    test_acc_list = []
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()

        y_hat = net(X_train_tensor)  # 在神经网络中得到输出值
        l = loss(y_hat, Y_train_tensor)  # 得到损失值
        optimizer.zero_grad()  # 置零梯度
        l.backward()  # 得到反传梯度值
        optimizer.step()  # 用随机梯度下降法训练参数
        train_l_sum += l.cpu().item()  # 得到损失之和
        train_acc_sum += (y_hat.argmax(dim = 1) == Y_train_tensor).float().sum().cpu().item()  # 得到模型在训练数据中的准确度
        n += Y_train_tensor.shape[0]
        batch_count += 1
        test_acc = evaluate_accuracy_my_for_array(X_test_tensor, Y_test_tensor, net)
        loss_list.extend([train_l_sum / n])
        train_acc_list.extend([train_acc_sum / n])
        test_acc_list.extend([test_acc])
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
    return [loss_list, train_acc_list, test_acc_list]


iris = datasets.load_iris()

X = iris.data
Y = iris.target

total_num = X.shape[0]
train_num = 90

num_epochs = 5
batch_size = 1

time_start = time.time()
net = softmax()  # 实例化网络
initialize(net)  # 权重、bias初始化
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备选择

[X_train_tensor, Y_train_tensor, X_test_tensor, Y_test_tensor] = get_train_and_test_item(X, Y, total_num, train_num)
print(net(X_train_tensor[0]))
print(Y_test_tensor[0])

test_acc = evaluate_accuracy_my_for_array(X_test_tensor, Y_test_tensor, net)
print('the initial accuracy is' + str(test_acc))

lr, num_epochs = 0.01, 1000  # 学习率和训练次数的指定
optimizer = torch.optim.Adam(net.parameters(), lr = lr)  # 梯度下降的优化方法的选择

[loss_list, train_acc_list, test_acc_list] = train_ch5_my_for_array(net, X_train_tensor, Y_train_tensor, X_test_tensor,
                                                                    Y_test_tensor, batch_size, optimizer,
                                                                    'cpu',
                                                                    num_epochs)  # 训练+得到各次训练的损失值，准确度
time_end = time.time()
print('the usage of time is ' + str(time_end - time_start))
draw(loss_list, train_acc_list, test_acc_list)
