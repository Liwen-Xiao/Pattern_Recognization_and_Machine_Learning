import random
import time

import numpy as np

import matplotlib.pyplot as plt

import math


def get_posotion():
    '''
    :param
    source: 数据的来源是什么
    'specific': 生成给定的数据
    'random': 生成随机的数据
    :return:
'''

    a_train = [{'x1': 121.5255, 'x2': 38.95223, 'bias': 1.0, 'y': 1.0, 'y_': 0},
               {'x1': 119.48458, 'x2': 39.83507, 'bias': 1.0, 'y': 1.0, 'y_': 0},
               {'x1': 117.30983, 'x2': 39.71755, 'bias': 1.0, 'y': 1.0, 'y_': 0},
               {'x1': 121.26757, 'x2': 37.49794, 'bias': 1.0, 'y': 1.0, 'y_': 0},
               {'x1': 120.85737, 'x2': 32.00986, 'bias': 1.0, 'y': 1.0, 'y_': 0},
               {'x1': 121.48941, 'x2': 31.40527, 'bias': 1.0, 'y': 1.0, 'y_': 0},
               {'x1': 119.27345, 'x2': 26.04769, 'bias': 1.0, 'y': 1.0, 'y_': 0},
               {'x1': 113.27324, 'x2': 23.15792, 'bias': 1.0, 'y': 1.0, 'y_': 0},
               {'x1': 119.43396, 'x2': 32.13188, 'bias': 1.0, 'y': 1.0, 'y_': 0},
               {'x1': 109.1175, 'x2': 21.47525, 'bias': 1.0, 'y': 1.0, 'y_': 0}
               ]
    b_train = [{'x1': 139.39, 'x2': 35.27, 'bias': 1.0, 'y': -1.0, 'y_': 0},
               {'x1': 135.1, 'x2': 34.41, 'bias': 1.0, 'y': -1.0, 'y_': 0},
               {'x1': 130.25, 'x2': 32.5, 'bias': 1.0, 'y': -1.0, 'y_': 0},
               {'x1': 132.27, 'x2': 34.24, 'bias': 1.0, 'y': -1.0, 'y_': 0},
               {'x1': 130, 'x2': 33, 'bias': 1.0, 'y': -1.0, 'y_': 0},
               {'x1': 132, 'x2': 34, 'bias': 1.0, 'y': -1.0, 'y_': 0},
               {'x1': 129.87, 'x2': 32.75, 'bias': 1.0, 'y': -1.0, 'y_': 0},
               {'x1': 136.55, 'x2': 35.1, 'bias': 1.0, 'y': -1.0, 'y_': 0},
               {'x1': 135.3, 'x2': 35, 'bias': 1.0, 'y': -1.0, 'y_': 0},
               {'x1': 130.495, 'x2': 30.38, 'bias': 1.0, 'y': -1.0, 'y_': 0}]

    return [a_train, b_train]


class SVM():
    def __init__(self, method):
        self.w = np.array([0, 0, 0])  # w的初始化
        [self.a_train, self.b_train] = get_posotion()
        self.train_list = self.a_train.copy() + self.b_train.copy()  # 将训练数据合成一个list
        self.train(method)  # 训练
        self.draw_train()  # 画图
        plt.show()

    def train(self, method):
        '''
        训练模型
        :param method: 选择训练数据的方法
        'regression':用梯度下降的迭代的方法来做
        :return:
        '''
        if method == 'regression':
            for i in range(1000):  # 最多迭代的次数
                delta = np.array([0, 0, 0])  # 初始化梯度向量
                k = 0  # 用来记录迭代的个数
                for t in self.train_list:  # 对于训练数据中的每一个样本
                    # print(t['y'] * (
                    # self.w[0] * t['x1'] + self.w[1] * t['x2'] + self.w[2] * t['bias']))  # 把每一个样本与w相乘的值输出出来
                    if t['y'] * (
                            self.w[0] * t['x1'] + self.w[1] * t['x2'] + self.w[2] * t['bias']) < 1:  # 当不满足损失函数情况的时候进行迭代
                        delta = delta - t['y'] * np.array([t['x1'], t['x2'], t['bias']])  # 计算梯度
                        k = k + 1  # 记录需要带入计算梯度的样本的个数

                if k == 0:  # 当每个样本都符合条件之后，退出训练
                    print('the train accuracy is 1.0')
                    return
                else:  # 当有样本不符合条件的时候，进行w的迭代
                    print('需要带入计算的样本数为 ' + str(k))
                    delta = delta / k
                    self.w = self.w - delta
                    print('新的w的值为 ' + str(self.w))
                    print('这是第 ' + str(i) + ' 次迭代')
                    print()
                if i == 999:
                    print('the train accuracy is ' + str((len(self.train_list) - k) / len(self.train_list)))

    def draw_train(self):
        '''
        画出图形
        :return:
        '''
        for a in self.a_train:
            plt.scatter(a['x1'], a['x2'], c = 'red', s = 1, label = 'a')
        for b in self.b_train:
            plt.scatter(b['x1'], b['x2'], c = 'blue', s = 1, label = 'b')
        plt.scatter(123, 28, c = 'purple', s = 10, label = 'b')

        plt.plot([122.5, 127.5], [0,
                              40],
                 c = 'green')

        plt.xlabel("x1", fontdict = {'size': 16})
        plt.ylabel("x2", fontdict = {'size': 16})


demo = SVM('regression')
