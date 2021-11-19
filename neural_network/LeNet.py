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


class LeNet(nn.Module):
    '''
    LeNet网络的类
    '''

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5),  # in_channels, out_channels, kernel_size
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),  # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
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


def train_ch5_my(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    '''
    网络参数训练函数，返回各次epoch训练结束时loss, 在训练样本中的正确率，在测试样本中的正确率
    :param net: 网络实例
    :param train_iter: 训练样本数据集
    :param test_iter: 测试样本数据集
    :param batch_size: 批次大小
    :param optimizer: 梯度下降优化方法
    :param device: 在cpu还是gpu上训练
    :param num_epochs: 训练多少次
    :return:
    '''
    # 将网络实例加载到指定设备上
    net = net.to(device)
    print("training on ", device)

    loss = torch.nn.CrossEntropyLoss()  # 定义损失函数

    # 各次epoch训练结束时loss, 在训练样本中的正确率，在测试样本中的正确率的list
    loss_list = []
    train_acc_list = []
    test_acc_list = []

    # 开始训练
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()  # 参数初始化
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)  # 在神经网络中得到输出值
            l = loss(y_hat, y)  # 得到损失值
            optimizer.zero_grad()  # 置零梯度
            l.backward()  # 得到反传梯度值
            optimizer.step()  # 用随机梯度下降法训练参数
            train_l_sum += l.cpu().item()  # 得到损失之和
            train_acc_sum += (y_hat.argmax(dim = 1) == y).sum().cpu().item()  # 得到模型在训练数据中的准确度
            n += y.shape[0]
            batch_count += 1
        test_acc = d2l.evaluate_accuracy(test_iter, net)  # 得到当前网络参数在测试数据集上的结果

        # 各次epoch训练结束时loss, 在训练样本中的正确率，在测试样本中的正确率的list
        loss_list.extend([train_l_sum / n])
        train_acc_list.extend([train_acc_sum / n])
        test_acc_list.extend([test_acc])
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
    return [loss_list, train_acc_list, test_acc_list]


def random_test(random_num, net, random_test_iter):
    '''
    从测试数据里随机选出一些数据来进行测试
    :param random_num: 随机选的样本的数目
    :param net: 网络
    :param test_iter: 测试数据集
    :return:
    '''

    for i in range(10):
        random_t = random.sample(list(random_test_iter), 1)  # 随机抽取一个有10个样本的batch
        test_acc_sum, n, i = 0.0, 0, 0  # 参数初始化

        # 计算正确率
        for X, y in random_t:
            y_hat = net(X)
            test_acc_sum += (y_hat.argmax(dim = 1) == y).sum().item()
            n += y.shape[0]
            i += 1
        print('the random accuracy is ' + str(test_acc_sum / n))


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
    plt.plot(np.array(train_acc_list), 'o-', color = 'blue', alpha = 0.8, label = 'train_acc')

    plt.plot(np.array(test_acc_list), 'o-', color = 'red', alpha = 0.8, label = 'test_acc')

    # 显示标签，如果不加这句，即使加了label='一些数字'的参数，最终还是不会显示标签
    plt.legend(loc = "upper right")
    plt.xlabel('epoch_num')
    plt.ylabel('accuracy')

    plt.figure(1)
    plt.subplot(1, 2, 2)
    plt.plot(loss_list, 'o-', color = 'blue', alpha = 0.8, label = 'loss')

    # 显示标签，如果不加这句，即使加了label='一些数字'的参数，最终还是不会显示标签
    plt.legend(loc = "upper right")
    plt.xlabel('epoch_num')
    plt.ylabel('accuracy')

    plt.show()


from collections import OrderedDict

time_start = time.time()
net = LeNet()  # 实例化网络
initialize(net)  # 权重、bias初始化
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备选择

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)  # 读取数据集
print(type(train_iter))
print(train_iter)

# 输入输出的个数
num_inputs = 784
num_outputs = 10

# 初始化参数后的正确率
test_acc = d2l.evaluate_accuracy(test_iter, net)
print('the initial accuracy is' + str(test_acc))

lr, num_epochs = 0.01, 10  # 学习率和训练次数的指定
optimizer = torch.optim.Adam(net.parameters(), lr = lr)  # 梯度下降的优化方法的选择

[loss_list, train_acc_list, test_acc_list] = train_ch5_my(net, train_iter, test_iter, batch_size, optimizer, 'cpu',
                                                          num_epochs)  # 训练+得到各次训练的损失值，准确度
time_end = time.time()
print('the usage of time is ' + str(time_end - time_start))

# 随机找10个样本来进行测试
random_train_iter, random_test_iter = d2l.load_data_fashion_mnist(10)
random_test(10, net, random_test_iter)

# 画图
draw(loss_list, train_acc_list, test_acc_list)

