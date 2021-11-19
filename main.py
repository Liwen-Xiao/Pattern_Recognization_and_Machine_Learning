import random

import numpy as np

import matplotlib.pyplot as plt


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


def judge(w, a_train, b_train):
    '''

    :param w:
    :param a_train:
    :param b_train:
    :return:
    '''
    accu = 200
    wrong_list = {}
    wrong_list['a'] = []
    wrong_list['b'] = []
    for i in range(0, 100):
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

    for i in range(0, 100):
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

    return [w, accu, wrong_list, a_train, b_train]


def renew_w(w, wrong_list, a_train, b_train):
    '''

    :param w:
    :param wrong_list:
    :return:
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
[a_train, b_train, a_test, b_test] = create_points()
num = -1
w = [0, 0, 0]
wrong_list = {}
wrong_list['a'] = [i for i in range(0, 100)]
wrong_list['b'] = [i for i in range(0, 100)]
best_w = w
best_accu = 0

for i in range(0, 10000):
    w = renew_w(w, wrong_list, a_train, b_train)
    [w, accu, wrong_list, a_train, b_train] = judge(w, a_train, b_train)
    if best_accu < accu:
        best_accu = accu
        best_w = w.copy()
        num = i + 1
    print('It is the ' + str(i + 1) + 'th renew')
    print('The recent w is ' + str(w))
    print(w)
    print('The recent accuracy is ' + str(accu))
    print(' ')
    if best_accu == 200:
        break

print('The best w is ' + str(best_w))
print('The best accuracy is ' + str(best_accu))
print('It is the ' + str(num) + 'th renew')

for a in a_train:
    plt.scatter(a['x1'], a['x2'], c = 'red', s = 1, label = 'a')
for b in b_train:
    plt.scatter(b['x1'], b['x2'], c = 'blue', s = 1, label = 'b')


plt.plot([-3, 5], [-(best_w[0]*(-3)+best_w[2])/best_w[1], -(best_w[0]*5+best_w[2])/best_w[1]], c = 'green')

plt.xlabel("x1", fontdict = {'size': 16})
plt.ylabel("x2", fontdict = {'size': 16})
plt.show()
