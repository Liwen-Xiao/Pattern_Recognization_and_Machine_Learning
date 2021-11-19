import random

import numpy as np

import matplotlib.pyplot as plt


class Pocket:
    def __init__(self):
        self.num = -1  # 最佳迭代次数
        self.w = [0, 0, 0]
        self.wrong_list = {}
        self.wrong_list['a'] = [i for i in range(0, 100)]
        self.wrong_list['b'] = [i for i in range(0, 100)]
        self.best_w = self.w.copy()
        self.best_accu = 0
        self.train_times = 1000
        self.accu = 0

    def renew_w(self, a_train, b_train):
        a_or_b = random.random()
        if a_or_b < 0.5:
            if self.wrong_list['a']:
                choice = random.choice(self.wrong_list['a'])
                self.w[0] = self.w[0] + a_train[choice]['x1'] * a_train[choice]['y']
                self.w[1] = self.w[1] + a_train[choice]['x2'] * a_train[choice]['y']
                self.w[2] = self.w[2] + a_train[choice]['bias'] * a_train[choice]['y']
            else:
                choice = random.choice(self.wrong_list['b'])
                self.w[0] = self.w[0] + b_train[choice]['x1'] * b_train[choice]['y']
                self.w[1] = self.w[1] + b_train[choice]['x2'] * b_train[choice]['y']
                self.w[2] = self.w[2] + b_train[choice]['bias'] * b_train[choice]['y']
        else:
            if self.wrong_list['b']:
                choice = random.choice(self.wrong_list['b'])
                self.w[0] = self.w[0] + b_train[choice]['x1'] * b_train[choice]['y']
                self.w[1] = self.w[1] + b_train[choice]['x2'] * b_train[choice]['y']
                self.w[2] = self.w[2] + b_train[choice]['bias'] * b_train[choice]['y']
            else:
                choice = random.choice(self.wrong_list['a'])
                self.w[0] = self.w[0] + a_train[choice]['x1'] * a_train[choice]['y']
                self.w[1] = self.w[1] + a_train[choice]['x2'] * a_train[choice]['y']
                self.w[2] = self.w[2] + a_train[choice]['bias'] * a_train[choice]['y']

    def judge(self, a_train, b_train):
        self.accu = len(a_train) + len(b_train)
        self.wrong_list = {}
        self.wrong_list['a'] = []
        self.wrong_list['b'] = []
        for i in range(0, len(a_train)):
            t = a_train[i]['x1'] * self.w[0] + a_train[i]['x2'] * self.w[1] + a_train[i]['bias'] * self.w[2]
            if t > 0:
                a_train[i]['y_'] = 1
            elif t < 0:
                a_train[i]['y_'] = -1
            else:
                a_train[i]['y_'] = 0

            if a_train[i]['y'] != a_train[i]['y_']:
                self.wrong_list['a'].append(i)
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
                self.wrong_list['b'].append(i)
                self.accu = self.accu - 1

    def draw(self):

        for a in a_train:
            plt.scatter(a['x1'], a['x2'], c = 'red', s = 1, label = 'a')
        for b in b_train:
            plt.scatter(b['x1'], b['x2'], c = 'blue', s = 1, label = 'b')

        plt.plot([-3, 5], [-(self.best_w[0] * (-3) + self.best_w[2]) / self.best_w[1],
                           -(self.best_w[0] * 5 + self.best_w[2]) / self.best_w[1]],
                 c = 'green')

        plt.xlabel("x1", fontdict = {'size': 16})
        plt.ylabel("x2", fontdict = {'size': 16})
        plt.show()

    def train(self, a_train, b_train):
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
            print('The recent accuracy is ' + str(self.accu))
            print(' ')
            if self.best_accu == len(a_train) + len(b_train):
                break

        print('The best w is ' + str(self.best_w))
        print('The best accuracy is ' + str(self.best_accu))
        print('It is the ' + str(self.num) + 'th renew')
        self.draw()


def create_points():
    '''
    生成训练和测试用的正态分布点
    :return:
    '''

    a_train = []
    b_train = []
    a_test = []
    b_test = []

    for i in range(0, 100):
        # a组训练样本初始化
        a_train.append({})
        a_train[i]['x1'] = np.random.normal(loc = 0.0, scale = 1.0)
        a_train[i]['x2'] = np.random.normal(loc = 2.0, scale = 1.0)
        a_train[i]['bias'] = 1
        a_train[i]['y'] = 1
        a_train[i]['y_'] = 0

        # b组训练样本初始化
        b_train.append({})
        b_train[i]['x1'] = np.random.normal(loc = 2.0, scale = 1.0)
        b_train[i]['x2'] = np.random.normal(loc = 0.0, scale = 1.0)
        b_train[i]['bias'] = 1
        b_train[i]['y'] = -1
        b_train[i]['y_'] = 0

    for i in range(0, 20):
        # a组测试样本初始化
        a_test.append({})
        a_test[i]['x1'] = np.random.normal(loc = 0.0, scale = 1.0)
        a_test[i]['x2'] = np.random.normal(loc = 10.0, scale = 1.0)
        a_test[i]['bias'] = 1
        a_test[i]['y'] = 1
        a_test[i]['y_'] = 0

        # b组测试样本初始化
        b_test.append({})
        b_test[i]['x1'] = np.random.normal(loc = 1.0, scale = 1.0)
        b_test[i]['x2'] = np.random.normal(loc = 0.0, scale = 1.0)
        b_test[i]['bias'] = 1
        b_test[i]['y'] = -1
        b_test[i]['y_'] = 0

    return [a_train, b_train, a_test, b_test]


[a_train, b_train, a_test, b_test] = create_points()
demo = Pocket()
demo.train(a_train, b_train)
