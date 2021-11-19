import random
import time

import numpy as np

import matplotlib.pyplot as plt

import math


class other_LR():
    def __init__(self, method):
        '''
        调用相应方法实现梯度下降
        :param method:
        method 传入 ‘LR’ 则调用梯度下降法
        method 传入 ‘ramdom LR’ 则调用随机梯度下降法
        method 传入 ‘adgrad’ 则调用adgrad法
        method 传入 'RMS' 则调用RMS法
        method 传入 ‘Momentum’ 则调用动量法
        method 传入 ‘Adam' 则调用Adam法
        '''
        self.yita = 0.4
        self.times = 50
        self.x = -4
        self.x_list = [-4]
        self.draw_fx()
        self.yipuxiluo = 0.000001
        self.grad_f_list = []
        self.delta = 0

        if method == 'LR':
            self.LR()
            self.draw()

        if method == 'random LR':
            self.random_LR()
            self.draw()

        elif method == 'adgrad':
            self.adgrad()
            self.draw()

        elif method == 'RMS':
            self.RMS()
            self.draw()

        elif method == 'Momentum':
            self.Momentum()
            self.draw()
        elif method == 'Adam':
            self.Adam()
            self.draw()
        plt.show()

    def f(self, x):
        return math.cos(math.pi / 4 * x) * x

    def grad_f(self, x):
        return math.cos(math.pi / 4 * x) - x * math.sin(math.pi / 4 * x) * math.pi / 4

    def LR(self):
        for i in range(self.times):
            self.x = self.x - self.yita * self.grad_f(self.x)
            self.x_list.append(self.x)

    def random_LR(self):
        for i in range(self.times):
            self.x = self.x - self.yita * self.grad_f(self.x + random.uniform(-0.5, 0.5))
            self.x_list.append(self.x)

    def adgrad(self):
        for i in range(self.times):
            self.grad_f_list.append(self.grad_f(self.x))
            self.delta = (1 / (i + 1) * sum(np.array(self.grad_f_list) ** 2)) ** (1 / 2) + self.yipuxiluo
            self.x = self.x - self.yita / self.delta * self.grad_f(self.x)
            self.x_list.append(self.x)

    def RMS(self):
        for i in range(self.times):
            alpha = 0.9
            self.grad_f_list.append(self.grad_f(self.x))
            if len(self.grad_f_list) == 1:
                self.delta = ((self.grad_f_list[0]) ** 2) ** (1 / 2)
            else:
                self.delta = (alpha * self.delta ** 2 + (1 - alpha) * (self.grad_f_list[-1] ** 2)) ** (1 / 2)
            self.x = self.x - self.yita / self.delta * self.grad_f(self.x)
            self.x_list.append(self.x)

    def Momentum(self):
        m = 0
        for i in range(self.times):
            namda = 0.9
            self.grad_f_list.append(self.grad_f(self.x))
            m = namda * m - self.yita * self.grad_f_list[-1]
            self.x = self.x + m
            self.x_list.append(self.x)

    def Adam(self):
        m = 0
        v = 0
        for i in range(self.times):

            beta1 = 0.99
            beta2 = 0.999
            self.grad_f_list.append(self.grad_f(self.x))
            m = beta1 * m + (1 - beta1) * self.grad_f_list[-1]
            v = beta2 * v + (1 - beta2) * self.grad_f_list[-1] ** 2
            self.x = self.x - self.yita * m / (1 - beta1) / ((v / (1 - beta2)) ** (1 / 2) + self.yipuxiluo)
            self.x_list.append(self.x)

    def draw(self):
        i = 1
        for x in self.x_list:
            plt.scatter(x, self.f(x), c = 'red', s = 20)
            plt.xlabel("x", fontdict = {'size': 16})
            plt.ylabel("f(x)", fontdict = {'size': 16})
            plt.annotate(i, (x, self.f(x)))
            i += 1

    def draw_fx(self):
        for i in np.arange(-5, 8, 0.01):
            plt.scatter(i, self.f(i), c = 'black', s = 1)


demo = other_LR('Momentum')
