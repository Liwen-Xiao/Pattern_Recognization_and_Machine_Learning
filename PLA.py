import random
import time

import numpy as np

import matplotlib.pyplot as plt


class PLA:
    def __init__(self):
        self.num = -1  # 最佳迭代次数
        self.w = [1, 1, 1]
        self.best_w = self.w.copy()
        self.best_accu = 0
        self.train_times = 10000 #迭代次数上限
        self.accu = 0
        self.test_accu = 0
        self.start_time=0
        self.end_time=0
        self.start_time=time.time()

    def renew_w(self, a_train, b_train):
        '''
        更新分类面
        :param a_train:
        :param b_train:
        :return:
        '''
        for a in a_train:
            if a['x1'] * self.w[0] + a['x2'] * self.w[1] + a['bias'] * self.w[2] <= 0:
                self.w[0] = a['x1'] * a['y'] + self.w[0]
                self.w[1] = a['x2'] * a['y'] + self.w[1]
                self.w[2] = a['bias'] * a['y'] + self.w[2]
                return
        for b in b_train:
            if b['x1'] * self.w[0] + b['x2'] * self.w[1] + b['bias'] * self.w[2] >= 0:
                self.w[0] = b['x1'] * b['y'] + self.w[0]
                self.w[1] = b['x2'] * b['y'] + self.w[1]
                self.w[2] = b['bias'] * b['y'] + self.w[2]
                return

    def judge(self, a_train, b_train):
        '''
        判断正确率
        :param a_train:
        :param b_train:
        :return:
        '''
        self.accu = len(a_train) + len(b_train)
        for i in range(0, len(a_train)):
            t = a_train[i]['x1'] * self.w[0] + a_train[i]['x2'] * self.w[1] + a_train[i]['bias'] * self.w[2]
            if t > 0:
                a_train[i]['y_'] = 1
            elif t < 0:
                a_train[i]['y_'] = -1
            else:
                a_train[i]['y_'] = 0

            if a_train[i]['y'] != a_train[i]['y_']:
                self.accu = self.accu - 1

        for i in range(0, len(b_train)):
            t = b_train[i]['x1'] * self.w[0] + b_train[i]['x2'] * self.w[1] + b_train[i]['bias'] * self.w[2]
            if t > 0:
                b_train[i]['y_'] = 1
            elif t < 0:
                b_train[i]['y_'] = -1
            else:
                b_train[i]['y_'] = 0

            if b_train[i]['y'] != b_train[i]['y_']:
                self.accu = self.accu - 1

    def draw_train(self):
        '''
        画出训练结果和训练样本
        :return:
        '''

        for a in a_train:
            plt.scatter(a['x1'], a['x2'], c = 'red', s = 1, label = 'a')
        for b in b_train:
            plt.scatter(b['x1'], b['x2'], c = 'blue', s = 1, label = 'b')

        plt.plot([-5, 5], [-(self.best_w[0] * (-5) + self.best_w[2]) / self.best_w[1],
                           -(self.best_w[0] * 5 + self.best_w[2]) / self.best_w[1]],
                 c = 'green')

        plt.xlabel("x1", fontdict = {'size': 16})
        plt.ylabel("x2", fontdict = {'size': 16})

    def train(self, a_train, b_train):
        '''
        训练分类面
        :param a_train:
        :param b_train:
        :return:
        '''
        for i in range(0, self.train_times):
            self.renew_w(a_train, b_train)
            self.judge(a_train, b_train)
            if self.best_accu < self.accu:
                self.best_accu = self.accu
                self.best_w = self.w.copy()
                self.num = i + 1
            print('It is the ' + str(i + 1) + 'th renew')
            print('The recent w is ' + str(self.w))
            print(self.best_w)
            print('The recent accuracy is ' + str(self.accu/320))
            print(' ')
            if self.best_accu == len(a_train) + len(b_train):
                break

        print('The best w is ' + str(self.best_w))
        print('The best accuracy is ' + str(self.best_accu/320))
        print('It is the ' + str(self.num) + 'th renew')
        self.draw_train()

    def test(self, a_test, b_test):
        '''
        在测试样本上测试分类面的正确率
        :param a_test:
        :param b_test:
        :return:
        '''
        self.test_accu = len(a_test) + len(b_test)
        for i in range(0, len(a_test)):
            t = a_test[i]['x1'] * self.best_w[0] + a_test[i]['x2'] * self.best_w[1] + a_test[i]['bias'] * \
                self.best_w[2]
            if t > 0:
                a_test[i]['y_'] = 1
            elif t < 0:
                a_test[i]['y_'] = -1
            else:
                a_test[i]['y_'] = 0

            if a_test[i]['y'] != a_test[i]['y_']:
                self.test_accu = self.test_accu - 1

        for i in range(0, len(b_test)):
            t = b_test[i]['x1'] * self.best_w[0] + b_test[i]['x2'] * self.best_w[1] + b_test[i]['bias'] * \
                self.best_w[2]
            if t > 0:
                b_test[i]['y_'] = 1
            elif t < 0:
                b_test[i]['y_'] = -1
            else:
                b_test[i]['y_'] = 0

            if b_test[i]['y'] != b_test[i]['y_']:
                self.test_accu = self.test_accu - 1
        print('\nThe test accuracy is ' + str(self.test_accu/80))
        self.draw_test(a_test, b_test)

    def draw_test(self, a_test, b_test):
        '''
        画出测似乎样本和测试结果
        :param a_test:
        :param b_test:
        :return:
        '''
        for a in a_test:
            plt.scatter(a['x1'], a['x2'], c = 'red', s = 30, label = 'a', marker = '+')
        for b in b_test:
            plt.scatter(b['x1'], b['x2'], c = 'blue', s = 30, label = 'b', marker = '+')

        plt.xlabel("x1", fontdict = {'size': 16})
        plt.ylabel("x2", fontdict = {'size': 16})
        self.end_time=time.time()
        print('The usage of the time is '+str(self.end_time-self.start_time)+'s')
        plt.show()


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
        a_train[i]['x1'] = np.random.normal(loc = -1.0, scale = 1.0)
        a_train[i]['x2'] = np.random.normal(loc = 0.0, scale = 1.0)
        a_train[i]['bias'] = 1
        a_train[i]['y'] = 1
        a_train[i]['y_'] = 0

        # b组训练样本初始化
        b_train.append({})
        b_train[i]['x1'] = np.random.normal(loc = 0.0, scale = 1.0)
        b_train[i]['x2'] = np.random.normal(loc = 1.0, scale = 1.0)
        b_train[i]['bias'] = 1
        b_train[i]['y'] = -1
        b_train[i]['y_'] = 0

    for i in range(0, each_test_num):
        # a组测试样本初始化
        a_test.append({})
        a_test[i]['x1'] = np.random.normal(loc = -1.0, scale = 1.0)
        a_test[i]['x2'] = np.random.normal(loc = 0.0, scale = 1.0)
        a_test[i]['bias'] = 1
        a_test[i]['y'] = 1
        a_test[i]['y_'] = 0

        # b组测试样本初始化
        b_test.append({})
        b_test[i]['x1'] = np.random.normal(loc = 0.0, scale = 1.0)
        b_test[i]['x2'] = np.random.normal(loc = 1.0, scale = 1.0)
        b_test[i]['bias'] = 1
        b_test[i]['y'] = -1
        b_test[i]['y_'] = 0

    return [a_train, b_train, a_test, b_test]


# 初始化

each_train_num = 160
each_test_num = 40
[a_train, b_train, a_test, b_test] = create_points(each_train_num, each_test_num)
demo = PLA()
demo.train(a_train, b_train)
demo.test(a_test, b_test)

