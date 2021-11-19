import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
import time
import matplotlib.pyplot as plt

sys.path.append("..")
import d2lzh_pytorch as d2l

print(torch.__version__)

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10


# class LinearNet(nn.Module):
#     def __init__(self, num_inputs, num_outputs):
#         super(LinearNet, self).__init__()
#         self.linear = nn.Linear(num_inputs, num_outputs)
#     def forward(self, x): # x shape: (batch, 1, 28, 28)
#         y = self.linear(x.view(x.shape[0], -1))
#         return y

# net = LinearNet(num_inputs, num_outputs)

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):  # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


def train_ch5_my(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    loss_list = []
    train_acc_list = []
    test_acc_list = []
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
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
        test_acc = d2l.evaluate_accuracy(test_iter, net)
        loss_list.extend([train_l_sum / n])
        train_acc_list.extend([train_acc_sum / n])
        test_acc_list.extend([test_acc])
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
    return [loss_list, train_acc_list, test_acc_list]


def draw(loss_list, train_acc_list, test_acc_list):
    # plot中参数的含义分别是横轴值，纵轴值，颜色，透明度和标签
    x = [1, 2, 3, 4, 5]
    plt.figure(1)
    plt.plot(np.array(train_acc_list), 'o-', color = 'blue', alpha = 0.8, label = 'train_acc')


    plt.plot(np.array(test_acc_list),'o-', color = 'red', alpha = 0.8, label = 'test_acc')


    # 显示标签，如果不加这句，即使加了label='一些数字'的参数，最终还是不会显示标签
    plt.legend(loc = "upper right")
    plt.xlabel('epoch_num')
    plt.ylabel('accuracy')

    plt.show()

    plt.figure(2)
    plt.plot(loss_list, 'o-', color = 'blue', alpha = 0.8, label = 'loss')


    # 显示标签，如果不加这句，即使加了label='一些数字'的参数，最终还是不会显示标签
    plt.legend(loc = "upper right")
    plt.xlabel('epoch_num')
    plt.ylabel('accuracy')

    plt.show()


from collections import OrderedDict

time_start = time.time()
net = nn.Sequential(
    # FlattenLayer(),
    # nn.Linear(num_inputs, num_outputs)
    OrderedDict([
        ('flatten', FlattenLayer()),
        ('linear', nn.Linear(num_inputs, num_outputs))])
)

init.normal_(net.linear.weight, mean = 0, std = 0.01)  # 初始化权重，使之为均值为0，标准差为0.01的正态分布
init.constant_(net.linear.bias, val = 0)
test_acc = d2l.evaluate_accuracy(test_iter, net)
print('the initial accuracy is' + str(test_acc))

# loss = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr = 0.1)

num_epochs = 5

[loss_list, train_acc_list, test_acc_list] = train_ch5_my(net, train_iter, test_iter, batch_size, optimizer, 'cpu',
                                                          num_epochs)  # 训练+得到各次训练的损失值，准确度
time_end = time.time()
print('the usage of time is ' + str(time_end - time_start))

draw(loss_list, train_acc_list, test_acc_list)


