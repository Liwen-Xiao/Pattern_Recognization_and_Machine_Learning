import random
import time

import numpy as np

import matplotlib.pyplot as plt


class LR():
    def __init__(self, a_train, b_train, a_test, b_test):
        self.start_time = 0
        self.end_time = 0
        self.start_time = time.time()

        self.X_mat = np.mat([0, 0, 0])
        for a in a_train:
            self.X_mat = np.r_[self.X_mat, np.mat([a['x1'], a['x2'], a['bias']])]
        for b in b_train:
            self.X_mat = np.r_[self.X_mat, np.mat([b['x1'], b['x2'], b['bias']])]
        np.delete(self.X_mat, 0, axis = 0)
        print(self.X_mat)

        self.Y_mat = np.mat([0])
        for a in a_train:
            self.Y_mat = np.r_[self.Y_mat, np.mat(a['y'])]
        for b in b_train:
            self.Y_mat = np.r_[self.Y_mat, np.mat(b['y'])]
        np.delete(self.Y_mat, 0, axis = 0)

        self.X_T__X = self.X_mat.T * self.X_mat
        self.X_generalized_inverse = self.X_T__X.I * self.X_mat.T

        self.best_w_mat = self.X_generalized_inverse * self.Y_mat
        self.best_w = self.best_w_mat.tolist()

        self.accu_train = 0
        self.accu_test = 0

        print(self.best_w)
        self.judge(a_train, b_train)
        self.draw_train(a_train, b_train)
        self.test(a_test, b_test)

    def judge(self, a_train, b_train):
        self.accu_train = len(a_train) + len(b_train)
        for i in range(0, len(a_train)):
            t = a_train[i]['x1'] * self.best_w[0][0] + a_train[i]['x2'] * self.best_w[1][0] + a_train[i]['bias'] * \
                self.best_w[2][0]
            if t > 0:
                a_train[i]['y_'] = 1
            elif t < 0:
                a_train[i]['y_'] = -1
            else:
                a_train[i]['y_'] = 0

            if a_train[i]['y'] != a_train[i]['y_']:
                self.accu_train = self.accu_train - 1

        for i in range(0, len(b_train)):
            t = b_train[i]['x1'] * self.best_w[0][0] + b_train[i]['x2'] * self.best_w[1][0] + b_train[i]['bias'] * \
                self.best_w[2][0]
            if t > 0:
                b_train[i]['y_'] = 1
            elif t < 0:
                b_train[i]['y_'] = -1
            else:
                b_train[i]['y_'] = 0

            if b_train[i]['y'] != b_train[i]['y_']:
                self.accu_train = self.accu_train - 1

    def draw_train(self, a_train, b_train):

        for a in a_train:
            plt.scatter(a['x1'], a['x2'], c = 'red', s = 1, label = 'a')
        for b in b_train:
            plt.scatter(b['x1'], b['x2'], c = 'blue', s = 1, label = 'b')

        plt.plot([-5, 5], [-(self.best_w[0][0] * (-5) + self.best_w[2][0]) / self.best_w[1][0],
                           -(self.best_w[0][0] * 5 + self.best_w[2][0]) / self.best_w[1][0]],
                 c = 'green')

        plt.xlabel("x1", fontdict = {'size': 16})
        plt.ylabel("x2", fontdict = {'size': 16})
        print('The best accuracy in the training is ' + str(self.accu_train/each_train_num/2))
        #plt.show()

    def test(self, a_test, b_test):
        self.accu_test = len(a_test) + len(b_test)
        for i in range(0, len(a_test)):
            t = a_test[i]['x1'] * self.best_w[0][0] + a_test[i]['x2'] * self.best_w[1][0] + a_test[i]['bias'] * \
                self.best_w[2][0]
            if t > 0:
                a_test[i]['y_'] = 1
            elif t < 0:
                a_test[i]['y_'] = -1
            else:
                a_test[i]['y_'] = 0

            if a_test[i]['y'] != a_test[i]['y_']:
                self.accu_test = self.accu_test - 1

        for i in range(0, len(b_test)):
            t = b_test[i]['x1'] * self.best_w[0][0] + b_test[i]['x2'] * self.best_w[1][0] + b_test[i]['bias'] * \
                self.best_w[2][0]
            if t > 0:
                b_test[i]['y_'] = 1
            elif t < 0:
                b_test[i]['y_'] = -1
            else:
                b_test[i]['y_'] = 0

            if b_test[i]['y'] != b_test[i]['y_']:
                self.accu_test = self.accu_test - 1
        print('\nThe test accuracy is ' + str(self.accu_test/each_test_num/2))
        self.draw_test(a_test, b_test)

    def draw_test(self, a_test, b_test):
        for a in a_test:
            plt.scatter(a['x1'], a['x2'], c = 'red', s = 30, label = 'a', marker = '+')
        for b in b_test:
            plt.scatter(b['x1'], b['x2'], c = 'blue', s = 30, label = 'b', marker = '+')

        plt.xlabel("x1", fontdict = {'size': 16})
        plt.ylabel("x2", fontdict = {'size': 16})
        self.end_time = time.time()
        print('The usage of the time is ' + str(self.end_time - self.start_time) + 's')
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
        a_train[i]['x1'] = np.random.normal(loc = -2.0, scale = 1.0)
        a_train[i]['x2'] = np.random.normal(loc = 0.0, scale = 1.0)
        a_train[i]['bias'] = 1
        a_train[i]['y'] = 1
        a_train[i]['y_'] = 0

        # b组训练样本初始化
        b_train.append({})
        b_train[i]['x1'] = np.random.normal(loc = 0.0, scale = 1.0)
        b_train[i]['x2'] = np.random.normal(loc = 2.0, scale = 1.0)
        b_train[i]['bias'] = 1
        b_train[i]['y'] = -1
        b_train[i]['y_'] = 0

    for i in range(0, each_test_num):
        # a组测试样本初始化
        a_test.append({})
        a_test[i]['x1'] = np.random.normal(loc = -2.0, scale = 1.0)
        a_test[i]['x2'] = np.random.normal(loc = 0.0, scale = 1.0)
        a_test[i]['bias'] = 1
        a_test[i]['y'] = 1
        a_test[i]['y_'] = 0

        # b组测试样本初始化
        b_test.append({})
        b_test[i]['x1'] = np.random.normal(loc = 0.0, scale = 1.0)
        b_test[i]['x2'] = np.random.normal(loc = 2.0, scale = 1.0)
        b_test[i]['bias'] = 1
        b_test[i]['y'] = -1
        b_test[i]['y_'] = 0

    return [a_train, b_train, a_test, b_test]


each_train_num = 160
each_test_num = 40
[a_train, b_train, a_test, b_test] = create_points(each_train_num, each_test_num)
demo = LR(a_train, b_train, a_test, b_test)
