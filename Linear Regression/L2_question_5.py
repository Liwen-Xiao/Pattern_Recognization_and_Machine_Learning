import random

import numpy as np

import matplotlib.pyplot as plt


def create_points():
    '''
    生成训练和测试用的正态分布点
    :return:测试数据的列表
    '''

    a_train = [{'x1': 0.2, 'x2': 0.7, 'bias': 1, 'y': 1, 'y_': 0},
               {'x1': 0.3, 'x2': 0.3, 'bias': 1, 'y': 1, 'y_': 0},
               {'x1': 0.4, 'x2': 0.5, 'bias': 1, 'y': 1, 'y_': 0},
               {'x1': 0.6, 'x2': 0.5, 'bias': 1, 'y': 1, 'y_': 0},
               {'x1': 0.1, 'x2': 0.4, 'bias': 1, 'y': 1, 'y_': 0}]
    b_train = [{'x1': 0.4, 'x2': 0.6, 'bias': 1, 'y': -1, 'y_': 0},
               {'x1': 0.6, 'x2': 0.2, 'bias': 1, 'y': -1, 'y_': 0},
               {'x1': 0.7, 'x2': 0.4, 'bias': 1, 'y': -1, 'y_': 0},
               {'x1': 0.8, 'x2': 0.6, 'bias': 1, 'y': -1, 'y_': 0},
               {'x1': 0.7, 'x2': 0.5, 'bias': 1, 'y': -1, 'y_': 0}]

    return [a_train, b_train]


def judge(w, a_train, b_train):
    '''

    :param w:前一个迭代的w
    :param a_train:训练数据列表
    :param b_train:训练数据列表
    :return:
    '''
    accu = 10
    wrong_list = {}
    wrong_list['a'] = []
    wrong_list['b'] = []
    for i in range(0, 5):
        t = a_train[i]['x1'] * w[0] + a_train[i]['x2'] * w[1] + a_train[i]['bias'] * w[2]
        if t > 0:
            a_train[i]['y_'] = 1
        elif t < 0:
            a_train[i]['y_'] = -1
        else:
            a_train[i]['y_'] = 0

        if a_train[i]['y'] != a_train[i]['y_']:
            wrong_list['a'].append(i)
            accu = accu - 1

    for i in range(0, 5):
        t = b_train[i]['x1'] * w[0] + b_train[i]['x2'] * w[1] + b_train[i]['bias'] * w[2]
        if t > 0:
            b_train[i]['y_'] = 1
        elif t < 0:
            b_train[i]['y_'] = -1
        else:
            b_train[i]['y_'] = 0

        if b_train[i]['y'] != b_train[i]['y_']:
            wrong_list['b'].append(i)
            accu = accu - 1

    return [accu, wrong_list, a_train, b_train]


def renew_w(w, wrong_list, a_train, b_train):
    '''

    :param w:前一个迭代w
    :param wrong_list:错误分类列表
    :return:新迭代的w
    '''
    a_or_b = random.random()
    if a_or_b < 0.5:
        if wrong_list['a']:
            choice = random.choice(wrong_list['a'])
            w[0] = w[0] + a_train[choice]['x1'] * a_train[choice]['y']
            w[1] = w[1] + a_train[choice]['x2'] * a_train[choice]['y']
            w[2] = w[2] + a_train[choice]['bias'] * a_train[choice]['y']
        else:
            choice = random.choice(wrong_list['b'])
            w[0] = w[0] + b_train[choice]['x1'] * b_train[choice]['y']
            w[1] = w[1] + b_train[choice]['x2'] * b_train[choice]['y']
            w[2] = w[2] + b_train[choice]['bias'] * b_train[choice]['y']
    else:
        if wrong_list['b']:
            choice = random.choice(wrong_list['b'])
            w[0] = w[0] + b_train[choice]['x1'] * b_train[choice]['y']
            w[1] = w[1] + b_train[choice]['x2'] * b_train[choice]['y']
            w[2] = w[2] + b_train[choice]['bias'] * b_train[choice]['y']
        else:
            choice = random.choice(wrong_list['a'])
            w[0] = w[0] + a_train[choice]['x1'] * a_train[choice]['y']
            w[1] = w[1] + a_train[choice]['x2'] * a_train[choice]['y']
            w[2] = w[2] + a_train[choice]['bias'] * a_train[choice]['y']

    return w


# 初始化
[a_train, b_train] = create_points() #训练样本初始化
num = -1        #最佳迭代次数计数初始化
w = [0, 0, 0]   #w初始化

#错误分类样本初始化
wrong_list = {}
wrong_list['a'] = [i for i in range(0, 5)]
wrong_list['b'] = [i for i in range(0, 5)]

#最佳w、正确率初始化
best_w = [0, 0, 0]
best_accu = 0

for i in range(0, 20):
    w = renew_w(w, wrong_list, a_train, b_train)                        #w更新
    [accu, wrong_list, a_train, b_train] = judge(w, a_train, b_train)   #判断哪些点错误分类
    if best_accu < accu:        #记录最佳迭代
        best_w = w.copy()
        best_accu = accu
        num = i + 1

    #输出每次迭代结果
    print('It is the ' + str(i + 1) + 'th renew')
    print('The recent w is ' + str(w))
    print('The best_w is ' + str(best_w))
    print('The best accu is ' + str(best_accu))
    print('The recent accuracy is ' + str(accu))
    print(' ')
    if best_accu == 10:
        break

print('The best w is ' + str(best_w))
print('The best accuracy is ' + str(best_accu))
print('It is the ' + str(num) + 'th renew')

# 可视化
for a in a_train:
    plt.scatter(a['x1'], a['x2'], c = 'red', s = 1, label = 'a')
for b in b_train:
    plt.scatter(b['x1'], b['x2'], c = 'blue', s = 1, label = 'b')

plt.plot([0, 1], [-(best_w[2]) / best_w[1], -(best_w[0] * 1 + best_w[2]) / best_w[1]], c = 'green')

plt.xlabel("x1", fontdict = {'size': 16})
plt.ylabel("x2", fontdict = {'size': 16})
plt.show()
