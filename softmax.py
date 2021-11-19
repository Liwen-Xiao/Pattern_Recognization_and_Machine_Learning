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


def softmax(X):
    '''
    softmax 函数，由S得到y_
    :param X:
    :return:
    '''
    X_exp = X.exp()
    partition = X_exp.sum(dim = 1, keepdim = True)
    return X_exp / partition  # 这里应用了广播机制


def net(X):
    '''
    softmax 网络
    :param X: 图像数据集的图像像素
    :return:
    '''
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)


def cross_entropy(y_hat, y):
    '''
    交叉熵函数，得到每个样本对应的交叉熵
    :param y_hat:
    :param y:
    :return:
    '''
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))


def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params = None, lr = None, optimizer = None):
    '''
    训练神经网络的函数
    :param net:什么神经网络
    :param train_iter:训练数据集
    :param test_iter:测试数据集
    :param loss:定义用什么损失函数
    :param num_epochs:epoch的个数
    :param batch_size:每个batch的大小
    :param params:W权重参数
    :param lr:学习率
    :param optimizer:梯度下降优化算法
    :return:
    '''
    loss_list = []
    train_acc_list = []
    test_acc_list = []
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                optimizer.step()  # “softmax回归的简洁实现”一节将用到

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim = 1) == y).sum().item()
            n += y.shape[0]

        test_acc = d2l.evaluate_accuracy(test_iter, net)

        loss_list.extend([train_l_sum / n])
        train_acc_list.extend([train_acc_sum / n])
        test_acc_list.extend([test_acc])
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))
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
        random_t = random.sample(list(random_test_iter), 1)
        test_acc_sum, n, i = 0.0, 0, 0
        for X, y in random_t:
            y_hat = net(X)
            test_acc_sum += (y_hat.argmax(dim = 1) == y).sum().item()
            n += y.shape[0]
            i+=1
        print('the random accuracy is '+str(test_acc_sum / n))


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


batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
random_train_iter, random_test_iter = d2l.load_data_fashion_mnist(10)


num_epochs, lr = 10, 0.1

num_inputs = 784
num_outputs = 10

W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype = torch.float)
b = torch.zeros(num_outputs, dtype = torch.float)

test_acc = d2l.evaluate_accuracy(test_iter, net)
print('the initial accuracy is' + str(test_acc))

W.requires_grad_(requires_grad = True)
b.requires_grad_(requires_grad = True)

[loss_list, train_acc_list, test_acc_list] = train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs,
                                                       batch_size, [W, b], lr)
random_test(10, net, random_test_iter)

draw(loss_list, train_acc_list, test_acc_list)
