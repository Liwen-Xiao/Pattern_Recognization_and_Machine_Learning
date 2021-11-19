import random
import time

import numpy as np

import matplotlib.pyplot as plt

import math

'''
def get_train(source):
    
    训练数据的生成
    :param source: 数据的来源是什么
    'specific':生成给定的数据
    'random':生成随机的数据
    :return: 无
    
    if source == 'specific':  # 当训练数据是给定的情况
        a_train = [{'x1': 1.0, 'x2': 1.0, 'bias': 1.0, 'y': 1, 'y_': 0},
                   {'x1': 2.0, 'x2': 2.0, 'bias': 1.0, 'y': 1, 'y_': 0},
                   {'x1': 2.0, 'x2': 0.0, 'bias': 1.0, 'y': 1, 'y_': 0}]
        b_train = [{'x1': 0.0, 'x2': 0.0, 'bias': 1.0, 'y': -1, 'y_': 0},
                   {'x1': 1.0, 'x2': 0.0, 'bias': 1.0, 'y': -1, 'y_': 0},
                   {'x1': 0.0, 'x2': 1.0, 'bias': 1.0, 'y': -1, 'y_': 0}]

    return [a_train, b_train]
'''


def create_points(each_train_num, each_test_num):
    '''
    生成训练和测试用的正态分布点
    :return:
    '''

    a_train = []
    b_train = []
    a_test = []
    b_test = []

    for i in range(0, each_train_num):
        # a组训练样本初始化
        a_train.append({})
        a_train[i]['x1'] = np.random.normal(loc = -3.0, scale = 1.0)
        a_train[i]['x2'] = np.random.normal(loc = 0.0, scale = 1.0)
        a_train[i]['bias'] = 1
        a_train[i]['y'] = 1
        a_train[i]['y_'] = 0

        # b组训练样本初始化
        b_train.append({})
        b_train[i]['x1'] = np.random.normal(loc = 0.0, scale = 1.0)
        b_train[i]['x2'] = np.random.normal(loc = 3.0, scale = 1.0)
        b_train[i]['bias'] = 1
        b_train[i]['y'] = -1
        b_train[i]['y_'] = 0

    for i in range(0, each_test_num):
        # a组测试样本初始化
        a_test.append({})
        a_test[i]['x1'] = np.random.normal(loc = -3.0, scale = 1.0)
        a_test[i]['x2'] = np.random.normal(loc = 0.0, scale = 1.0)
        a_test[i]['bias'] = 1
        a_test[i]['y'] = 1
        a_test[i]['y_'] = 0

        # b组测试样本初始化
        b_test.append({})
        b_test[i]['x1'] = np.random.normal(loc = 0.0, scale = 1.0)
        b_test[i]['x2'] = np.random.normal(loc = 3.0, scale = 1.0)
        b_test[i]['bias'] = 1
        b_test[i]['y'] = -1
        b_test[i]['y_'] = 0

    return [a_train, b_train, a_test, b_test]


class SVM():
    def __init__(self, method):
        self.each_train_num = 160
        self.each_test_num = 40
        self.w = np.array([0, 0, 0])  # w的初始化
        [self.a_train, self.b_train, self.a_test, self.b_test] = create_points(self.each_train_num,
                                                                               self.each_test_num)  # 获得训练数据
        self.train_list = self.a_train.copy() + self.b_train.copy()  # 将训练数据合成一个list
        self.train(method)  # 训练
        self.draw_train()  # 画图
        self.test(self.a_test, self.b_test)
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
                k = 0  # 用来记录迭代的次数
                for t in self.train_list:  # 对于训练数据中的每一个样本
                    #print(t['y'] * (
                            #self.w[0] * t['x1'] + self.w[1] * t['x2'] + self.w[2] * t['bias']))  # 把每一个样本与w相乘的值输出出来
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
                if i==999:
                    print('the train accuracy is '+str((len(self.train_list)-k)/len(self.train_list)))



    def draw_train(self):
        '''
        画出图形
        :return:
        '''
        for a in self.a_train:
            plt.scatter(a['x1'], a['x2'], c = 'red', s = 1, label = 'a')
        for b in self.b_train:
            plt.scatter(b['x1'], b['x2'], c = 'blue', s = 1, label = 'b')

        plt.plot([-5, 5], [-(self.w[0] * (-5) + self.w[2]) / self.w[1],
                           -(self.w[0] * 5 + self.w[2]) / self.w[1]],
                 c = 'green')

        plt.xlabel("x1", fontdict = {'size': 16})
        plt.ylabel("x2", fontdict = {'size': 16})

    def test(self, a_test, b_test):
        self.test_accu = len(a_test) + len(b_test)
        for i in range(0, len(a_test)):
            t = a_test[i]['x1'] * self.w[0] + a_test[i]['x2'] * self.w[1] + a_test[i]['bias'] * \
                self.w[2]
            if t > 0:
                a_test[i]['y_'] = 1
            elif t < 0:
                a_test[i]['y_'] = -1
            else:
                a_test[i]['y_'] = 0

            if a_test[i]['y'] != a_test[i]['y_']:
                self.test_accu = self.test_accu - 1

        for i in range(0, len(b_test)):
            t = b_test[i]['x1'] * self.w[0] + b_test[i]['x2'] * self.w[1] + b_test[i]['bias'] * \
                self.w[2]
            if t > 0:
                b_test[i]['y_'] = 1
            elif t < 0:
                b_test[i]['y_'] = -1
            else:
                b_test[i]['y_'] = 0

            if b_test[i]['y'] != b_test[i]['y_']:
                self.test_accu = self.test_accu - 1
        print('\nThe test accuracy is ' + str(self.test_accu / (len(a_test) + len(b_test))))
        self.draw_test(a_test, b_test)



    def draw_test(self, a_test, b_test):
        for a in a_test:
            plt.scatter(a['x1'], a['x2'], c = 'red', s = 30, label = 'a', marker = '+')
        for b in b_test:
            plt.scatter(b['x1'], b['x2'], c = 'blue', s = 30, label = 'b', marker = '+')

        plt.xlabel("x1", fontdict = {'size': 16})
        plt.ylabel("x2", fontdict = {'size': 16})
        self.end_time = time.time()
        #print('The usage of the time is ' + str(self.end_time - self.start_time) + 's')
        #plt.show()


demo = SVM('regression')
