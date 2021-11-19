import random
import time

import numpy as np

from cvxopt import solvers, matrix

import matplotlib.pyplot as plt

import math

'''
matrix输入是输入列向量,下标从0开始
qp问题只能用matrix来做，而不能用mat
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
        a_train[i]['x1'] = np.random.normal(loc = -5.0, scale = 1.0)
        a_train[i]['x2'] = np.random.normal(loc = 0.0, scale = 1.0)
        a_train[i]['bias'] = 1
        a_train[i]['y'] = 1
        a_train[i]['y_'] = 0

        # b组训练样本初始化
        b_train.append({})
        b_train[i]['x1'] = np.random.normal(loc = 0.0, scale = 1.0)
        b_train[i]['x2'] = np.random.normal(loc = 5.0, scale = 1.0)
        b_train[i]['bias'] = 1
        b_train[i]['y'] = -1
        b_train[i]['y_'] = 0

    for i in range(0, each_test_num):
        # a组测试样本初始化
        a_test.append({})
        a_test[i]['x1'] = np.random.normal(loc = -5.0, scale = 1.0)
        a_test[i]['x2'] = np.random.normal(loc = 0.0, scale = 1.0)
        a_test[i]['bias'] = 1
        a_test[i]['y'] = 1
        a_test[i]['y_'] = 0

        # b组测试样本初始化
        b_test.append({})
        b_test[i]['x1'] = np.random.normal(loc = 0.0, scale = 1.0)
        b_test[i]['x2'] = np.random.normal(loc = 5.0, scale = 1.0)
        b_test[i]['bias'] = 1
        b_test[i]['y'] = -1
        b_test[i]['y_'] = 0

    return [a_train, b_train, a_test, b_test]


class Kernel_SVM():
    def __init__(self):
        self.each_train_num = 160
        self.each_test_num = 40
        self.w = matrix(np.array([0.0, 0.0, 0.0, 0.0, 0.0]))  # w的初始化
        self.b = 0
        [self.a_train, self.b_train, self.a_test, self.b_test] = create_points(self.each_train_num,
                                                                               self.each_test_num)  # 获得训练数据
        self.train_list = self.a_train.copy() + self.b_train.copy()  # 将训练数据合成一个list

        self.train()  # 训练
        self.get_train_acc(self.a_train, self.b_train)
        # self.draw()  # 画图
        self.test(self.a_test, self.b_test)
        plt.show()

    def Kernel(self, x1_dic, x2_dic):
        x1 = matrix([1, x1_dic['x1'], x1_dic['x2']])
        x2 = matrix([1, x2_dic['x1'], x2_dic['x2']])
        return (x1.T * x2 + (x1.T * x2) * (x1.T * x2))[0, 0]

    def get_z(self, x):
        return matrix([x['x1'], x['x2'], x['x1'] * x['x1'], x['x1'] * x['x2'], x['x2'] * x['x2']])

    def draw(self, bias, alpha, w):
        '''
        画出图形
        :return:

        '''
        '''
        for a in self.a_train:
            plt.scatter(a['x1'], a['x2'], c = 'red', s = 1, label = 'a')
        for b in self.b_train:
            plt.scatter(b['x1'], b['x2'], c = 'blue', s = 1, label = 'b')
        '''
        # k = -self.w[0, 0] / self.w[1, 0]
        # b = -bias[0, 0] / self.w[1, 0]
        for i in range(len(self.train_list)):
            if self.train_list[i]['y'] == 1:
                if alpha[i] > 1e-6:
                    plt.scatter(self.train_list[i]['x1'], self.train_list[i]['x2'], c = 'red', s = 30, label = 'a')
                    # b_a = self.train_list[i]['x2'] - k * self.train_list[i]['x1']
                else:
                    plt.scatter(self.train_list[i]['x1'], self.train_list[i]['x2'], c = 'red', s = 1, label = 'a')
            else:
                if alpha[i] > 1e-6:
                    plt.scatter(self.train_list[i]['x1'], self.train_list[i]['x2'], c = 'blue', s = 30, label = 'b')
                    # b_b = self.train_list[i]['x2'] - k * self.train_list[i]['x1']
                else:
                    plt.scatter(self.train_list[i]['x1'], self.train_list[i]['x2'], c = 'blue', s = 1, label = 'b')
        plt.xlabel("x1", fontdict = {'size': 16})
        plt.ylabel("x2", fontdict = {'size': 16})

        for a in self.a_test:
            plt.scatter(a['x1'], a['x2'], c = 'red', s = 10, label = 'a', marker = '+')
        for b in self.b_test:
            plt.scatter(b['x1'], b['x2'], c = 'blue', s = 10, label = 'b', marker = '+')

        # 设置x和y的坐标范围
        x = np.arange(-20, 20, 0.01)
        y = np.arange(-20, 20, 0.01)
        # 转化为网格
        x, y = np.meshgrid(x, y)
        z = bias + x * w[0, 0] + y * w[1, 0] + np.power(x, 2) * w[2, 0] + x * y * w[3, 0] + np.power(y, 2) * w[4, 0]
        plt.contour(x, y, z, 0)

        # plt.plot([-5, 5], [k * (-5) + b, k * 5 + b], c = 'green', linewidth = 3.0)
        # plt.plot([-5, 5], [k * (-5) + b_a, k * 5 + b_a], c = 'green', linewidth = 1.0)
        # plt.plot([-5, 5], [k * (-5) + b_b, k * 5 + b_b], c = 'green', linewidth = 1.0)

    def train(self):
        N = len(self.a_train) + len(self.b_train)  # 计算样本个数
        Q = np.mat(np.arange(N))
        for i in range(N):
            q = []
            for j in range(N):
                '''
                q.append(self.train_list[i]['y'] * self.train_list[j]['y'] * (
                        self.train_list[i]['x1'] * self.train_list[j]['x1'] + self.train_list[i]['x2'] *
                        self.train_list[j]['x2']))  # 计算出Q每一行的每一个元素
                '''
                q.append(self.train_list[i]['y'] * self.train_list[j]['y'] * self.Kernel(
                    self.train_list[i], self.train_list[j]))  # 计算出Q每一行的每一个元素
            Q = np.r_[Q, np.mat(q)]
        Q_array = np.array(Q)
        print(Q_array)
        # np.delete(Q_array, 0, 0)  # 删除第一行,mat不能被操作，只有array可以被操作
        Q = matrix(Q_array[1:][:])
        print('Q')
        print(Q)
        print()

        # 计算行向量p,p是一个1*n的向量
        p = []
        for i in range(N):
            p.append(-1.0)
        p = matrix(p)
        print('p')
        print(p)
        print()

        # 得到A矩阵，A是N*N的单位阵
        A = np.zeros((N, N))
        A_array = np.array(A)
        for i in range(N):
            A_array[i][i] = -1.0
        A = matrix(A_array)
        print('A')
        print(A)
        print()

        c = matrix(np.zeros((1, N)).T)
        print('c')
        print(c)
        print()

        # 生成r矩阵
        r_list = []
        for t in self.train_list:
            r_list.append(float(t['y']))
        r = matrix(r_list).T
        print('r')
        print(r)
        print()

        v = matrix(0.0)
        print('v')
        print(v)
        print()

        solvers.options['show_progress'] = False
        sol = solvers.qp(Q, p, A, c, r, v)
        print(sol['x'])
        print(sol['y'])
        alpha = sol['x']
        w = matrix([0.0, 0.0, 0.0, 0.0, 0.0])
        # print(w)
        # w = matrix([0.0, 0.0])
        for i in range(N):
            if alpha[i, 0] > 1e-6:
                w = w + alpha[i] * self.train_list[i]['y'] * self.get_z(self.train_list[i])
        self.w = w
        for i in range(N):
            if alpha[i, 0] > 1e-6:
                b = self.train_list[i]['y'] - self.w.T * self.get_z(self.train_list[i])
                break

        # print(alpha)
        print(w)
        print(w[3, 0])
        print(b)
        self.draw(b, alpha, w)

        # print(alpha)

    def get_train_acc(self, a_train, b_train):
        self.train_accu = len(a_train) + len(b_train)
        for i in range(0, len(a_train)):
            t=(self.w.T*self.get_z(a_train[i]))[0,0]+self.b
            #t = a_train[i]['x1'] * self.w[0, 0] + a_train[i]['x2'] * self.w[1, 0] + a_train[i]['bias'] * self.b

            if t > 0:
                a_train[i]['y_'] = 1
            elif t < 0:
                a_train[i]['y_'] = -1
            else:
                a_train[i]['y_'] = 0

            if a_train[i]['y'] != a_train[i]['y_']:
                self.train_accu = self.train_accu - 1

        for i in range(0, len(b_train)):
            t=(self.w.T*self.get_z(b_train[i]))[0,0]+self.b
            if t > 0:
                b_train[i]['y_'] = 1
            elif t < 0:
                b_train[i]['y_'] = -1
            else:
                b_train[i]['y_'] = 0

            if b_train[i]['y'] != b_train[i]['y_']:
                self.train_accu = self.train_accu - 1
        print('\nThe train accuracy is ' + str(self.train_accu / self.each_train_num / 2))

    def test(self, a_test, b_test):
        self.test_accu = len(a_test) + len(b_test)
        for i in range(0, len(a_test)):
            t=(self.w.T*self.get_z(a_test[i]))[0,0]+self.b

            if t > 0:
                a_test[i]['y_'] = 1
            elif t < 0:
                a_test[i]['y_'] = -1
            else:
                a_test[i]['y_'] = 0

            if a_test[i]['y'] != a_test[i]['y_']:
                self.test_accu = self.test_accu - 1

        for i in range(0, len(b_test)):
            t=(self.w.T*self.get_z(b_test[i]))[0,0]+self.b
            if t > 0:
                b_test[i]['y_'] = 1
            elif t < 0:
                b_test[i]['y_'] = -1
            else:
                b_test[i]['y_'] = 0

            if b_test[i]['y'] != b_test[i]['y_']:
                self.test_accu = self.test_accu - 1
        print('\nThe test accuracy is ' + str(self.test_accu / self.each_test_num / 2))


demo = Kernel_SVM()
