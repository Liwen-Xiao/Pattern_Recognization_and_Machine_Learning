import random
import time

import numpy as np

import matplotlib.pyplot as plt


class LR():
    def __init__(self, a_train, b_train, a_test, b_test, method):
        self.start_time = 0
        self.end_time = 0
        self.start_time = time.time()
        self.accu_train = 0
        self.accu_test = 0

        if method == 'Linear Regression':
            self.X_mat = np.mat([0, 0, 0])  # 矩阵初始化
            for a in a_train:
                self.X_mat = np.r_[self.X_mat, np.mat([a['x1'], a['x2'], a['bias']])]  # 行扩展
            for b in b_train:
                self.X_mat = np.r_[self.X_mat, np.mat([b['x1'], b['x2'], b['bias']])]  # 行扩展
            np.delete(self.X_mat, 0,  0)  # 删除第一行
            print(self.X_mat)  # 输出矩阵

            self.Y_mat = np.mat([0])  # 矩阵初始化
            for a in a_train:
                self.Y_mat = np.r_[self.Y_mat, np.mat(a['y'])]  # 行扩展
            for b in b_train:
                self.Y_mat = np.r_[self.Y_mat, np.mat(b['y'])]  # 行扩展
            np.delete(self.Y_mat, 0,  0)  # 输出矩阵

            self.X_T__X = self.X_mat.T * self.X_mat
            self.X_generalized_inverse = self.X_T__X.I * self.X_mat.T  # 得到广义逆

            self.best_w_mat = self.X_generalized_inverse * self.Y_mat
            self.best_w = self.best_w_mat.tolist()
            self.best_w[0] = self.best_w[0][0]
            self.best_w[1] = self.best_w[1][0]
            self.best_w[2] = self.best_w[2][0]

            print('The X_generalized_inverse is ' + str(self.X_generalized_inverse))

            print('The best w is ' + str(self.best_w))
            self.judge(a_train, b_train)

        if method == 'gradient descent':
            self.epoch = 1000  # 整个样本集遍历多少次
            self.batch = 400  # batch的大小，因为我们的训练样本集大小为320，所以batch大小应该是320的因子
            self.yita = 0.01  # 学习步长
            self.loss_list = []
            self.w = self.initialize_w(a_train, b_train)
            self.best_w = self.w.copy()
            self.gradient_descent_train(a_train, b_train)
            print('the final w is ' + str(self.best_w))
            self.judge(a_train, b_train)

        self.draw_train(a_train, b_train, method)
        # self.test(a_test, b_test)

    def initialize_w(self, a_train, b_train):
        '''
        随机生成101个w，取其中损失函数最小的作为初始化的w
        :param a_train:
        :param b_train:
        :return:
        '''
        w = [0, 0, 0]
        w[0] = random.uniform(-100, 100)
        w[1] = random.uniform(-100, 100)
        w[2] = random.uniform(-100, 100)
        train = a_train.copy() + b_train.copy()
        best_initialize_Lin = 0
        for t in train:
            best_initialize_Lin += (w[0] * t['x1'] + w[1] * t['x2'] + w[2] * t['bias'] - t['y']) ** 2
        best_initialize_Lin = best_initialize_Lin / len(train)

        best_initialize_w = w.copy()
        for i in range(100):
            w[0] = random.uniform(-100, 100)
            w[1] = random.uniform(-100, 100)
            w[2] = random.uniform(-100, 100)
            initialize_Lin = 0
            for t in train:
                initialize_Lin = initialize_Lin + (w[0] * t['x1'] + w[1] * t['x2'] + w[2] * t['bias'] - t['y']) ** 2
            initialize_Lin = initialize_Lin / len(train)
            if initialize_Lin < best_initialize_Lin:
                best_initialize_Lin = initialize_Lin
                best_initialize_w = w.copy()
            print('now w ' + str(w))
            print('best w ' + str(best_initialize_w))
            print(' ')
        return best_initialize_w

    def gradient_descent_train(self, a_train, b_train):
        train = a_train.copy() + b_train.copy()
        for i in range(self.epoch):  # 将整个数据集遍历多少次
            random.shuffle(train)
            loss = 0
            for j in range(int(each_train_num / self.batch)):  # 在每一次数据集的遍历中，对每一个batch进行循环
                grad_Lin = [0, 0, 0]
                for k in range(self.batch):  # 处理每个batch中的样本，即迭代一次w
                    grad_Lin[0] += (self.w[0] * train[j * self.batch + k]['x1'] + self.w[1] * train[j * self.batch + k][
                        'x2'] + self.w[2] * train[j * self.batch + k]['bias'] - train[j * self.batch + k]['y']) * \
                                   train[j * self.batch + k]['x1']

                    grad_Lin[1] += (self.w[0] * train[j * self.batch + k]['x1'] + self.w[1] * train[j * self.batch + k][
                        'x2'] + self.w[2] * train[j * self.batch + k]['bias'] - train[j * self.batch + k]['y']) * \
                                   train[j * self.batch + k]['x2']

                    grad_Lin[2] += (self.w[0] * train[j * self.batch + k]['x1'] + self.w[1] * train[j * self.batch + k][
                        'x2'] + self.w[2] * train[j * self.batch + k]['bias'] - train[j * self.batch + k]['y']) * \
                                   train[j * self.batch + k]['bias']
                grad_Lin[0] = grad_Lin[0] * 2 / self.batch
                grad_Lin[1] = grad_Lin[1] * 2 / self.batch
                grad_Lin[2] = grad_Lin[2] * 2 / self.batch

                self.w[0] = self.w[0] - self.yita * grad_Lin[0]
                self.w[1] = self.w[1] - self.yita * grad_Lin[1]
                self.w[2] = self.w[2] - self.yita * grad_Lin[2]
            for t in train:
                loss += (self.w[0] * t['x1'] + self.w[1] * t['x2'] + self.w[2] * t['bias'] - t['y']) ** 2

            self.loss_list.append(loss)
        self.best_w = self.w.copy()

    def judge(self, a_train, b_train):
        '''
        用于判断训练集中哪些样本分类正确和分类错误，并且计算总的正确率
        :param a_train:
        :param b_train:
        :return:
        '''
        self.accu_train = len(a_train) + len(b_train)
        for i in range(0, len(a_train)):
            t = a_train[i]['x1'] * self.best_w[0] + a_train[i]['x2'] * self.best_w[1] + a_train[i]['bias'] * \
                self.best_w[2]
            if t > 0:
                a_train[i]['y_'] = 1
            elif t < 0:
                a_train[i]['y_'] = -1
            else:
                a_train[i]['y_'] = 0

            if a_train[i]['y'] != a_train[i]['y_']:
                self.accu_train = self.accu_train - 1

        for i in range(0, len(b_train)):
            t = b_train[i]['x1'] * self.best_w[0] + b_train[i]['x2'] * self.best_w[1] + b_train[i]['bias'] * \
                self.best_w[2]
            if t > 0:
                b_train[i]['y_'] = 1
            elif t < 0:
                b_train[i]['y_'] = -1
            else:
                b_train[i]['y_'] = 0

            if b_train[i]['y'] != b_train[i]['y_']:
                self.accu_train = self.accu_train - 1

    def draw_train(self, a_train, b_train, method):
        '''
        将训练样本在2维图中画出来，并且把最佳分类面画出来，输出最佳分类面在训练数据集中的最佳正确率
        :param a_train:
        :param b_train:
        :return:
        '''
        if method == 'gradient descent':
            plt.figure()
            plt.plot(range(self.epoch), self.loss_list)
            plt.xlabel("epoch", fontdict = {'size': 16})
            plt.ylabel("Lin", fontdict = {'size': 16})

        plt.figure()
        for a in a_train:
            plt.scatter(a['x1'], a['x2'], c = 'red', s = 1, label = 'a')
        for b in b_train:
            plt.scatter(b['x1'], b['x2'], c = 'blue', s = 1, label = 'b')

        plt.plot([-5, 5], [-(self.best_w[0] * (-5) + self.best_w[2]) / self.best_w[1],
                           -(self.best_w[0] * 5 + self.best_w[2]) / self.best_w[1]],
                 c = 'green')

        plt.xlabel("x1", fontdict = {'size': 16})
        plt.ylabel("x2", fontdict = {'size': 16})
        print('The best accuracy in the training is ' + str(self.accu_train / each_train_num / 2))
        plt.show()

    def test(self, a_test, b_test):
        '''
        检测训练出来的最佳分类面在测试数据集上的正确率，并找出哪些样本分类正确，哪些样本分类错误，最后调用画图函数在二维图中画出测试样本的位置
        :param a_test:
        :param b_test:
        :return:
        '''
        self.accu_test = len(a_test) + len(b_test)
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
                self.accu_test = self.accu_test - 1

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
                self.accu_test = self.accu_test - 1
        print('\nThe test accuracy is ' + str(self.accu_test / each_test_num / 2))
        self.draw_test(a_test, b_test)

    def draw_test(self, a_test, b_test):
        '''
        画出测试样本在二维图中的位置
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
        a_train[i]['x1'] = np.random.normal(loc = 1.0, scale = 1.0)
        a_train[i]['x2'] = np.random.normal(loc = 0.0, scale = 1.0)
        a_train[i]['bias'] = 1
        a_train[i]['y'] = 1
        a_train[i]['y_'] = 0

        # b组训练样本初始化
        b_train.append({})
        b_train[i]['x1'] = np.random.normal(loc = 0.0, scale = 1.0)
        b_train[i]['x2'] = np.random.normal(loc = -1.0, scale = 1.0)
        b_train[i]['bias'] = 1
        b_train[i]['y'] = -1
        b_train[i]['y_'] = 0

    for i in range(0, each_test_num):
        # a组测试样本初始化
        a_test.append({})
        a_test[i]['x1'] = np.random.normal(loc = 1.0, scale = 1.0)
        a_test[i]['x2'] = np.random.normal(loc = 0.0, scale = 1.0)
        a_test[i]['bias'] = 1
        a_test[i]['y'] = 1
        a_test[i]['y_'] = 0

        # b组测试样本初始化
        b_test.append({})
        b_test[i]['x1'] = np.random.normal(loc = 0.0, scale = 1.0)
        b_test[i]['x2'] = np.random.normal(loc = -1.0, scale = 1.0)
        b_test[i]['bias'] = 1
        b_test[i]['y'] = -1
        b_test[i]['y_'] = 0

    return [a_train, b_train, a_test, b_test]


each_train_num = 200
each_test_num = 40
[a_train, b_train, a_test, b_test] = create_points(each_train_num, each_test_num)
# demo = LR(a_train, b_train, a_test, b_test, 'gradient descent')
a_train = [{'x1': 0.2, 'x2': 0.7, 'bias': 1, 'y': 1, 'y_': 0},
           {'x1': 0.3, 'x2': 0.3, 'bias': 1, 'y': 1, 'y_': 0},
           {'x1': 0.4, 'x2': 0.5, 'bias': 1, 'y': 1, 'y_': 0},
           {'x1': 0.6, 'x2': 0.5, 'bias': 1, 'y': 1, 'y_': 0},
           {'x1': 0.1, 'x2': 0.4, 'bias': 1, 'y': 1, 'y_': 0},
           ]
b_train = [{'x1': 0.4, 'x2': 0.6, 'bias': 1, 'y': -1, 'y_': 0},
           {'x1': 0.6, 'x2': 0.2, 'bias': 1, 'y': -1, 'y_': 0},
           {'x1': 0.7, 'x2': 0.4, 'bias': 1, 'y': -1, 'y_': 0},
           {'x1': 0.8, 'x2': 0.6, 'bias': 1, 'y': -1, 'y_': 0},
           {'x1': 0.7, 'x2': 0.5, 'bias': 1, 'y': -1, 'y_': 0},
           ]

demo = LR(a_train, b_train, a_test, b_test, 'Linear Regression')
