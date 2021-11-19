import random
import time

import numpy as np

import matplotlib.pyplot as plt


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


def get_mv(a_train, b_train):
    mv_a = np.array([0.0, 0.0])
    mv_b = np.array([0.0, 0.0])
    for a in a_train:
        mv_a += np.array([a['x1'], a['x2']])
    for b in b_train:
        mv_b += np.array([b['x1'], b['x2']])
    mv_a = mv_a / len(a_train)
    mv_b = mv_b / len(b_train)
    return [np.mat(mv_a).T, np.mat(mv_b).T]


def get_w_and_s(a_train, b_train, mv_a, mv_b):
    segma_a = np.mat([[0.0, 0.0],
                      [0.0, 0.0]])
    segma_b = np.mat([[0.0, 0.0],
                      [0.0, 0.0]])
    sw = np.mat([[0.0, 0.0],
                 [0.0, 0.0]])
    for a in a_train:
        segma_a += (np.mat([a['x1'], a['x2']]).T - mv_a) * (np.mat([a['x1'], a['x2']]).T - mv_a).T
    for b in b_train:
        segma_b += (np.mat([b['x1'], b['x2']]).T - mv_b) * (np.mat([b['x1'], b['x2']]).T - mv_b).T
    sw = segma_a + segma_b
    print(segma_a)
    sw_inverse = sw.I
    w = sw_inverse * (mv_a - mv_b)
    s = w.T * (mv_a + mv_b)
    print("the s= "+str(s[0,0]))
    return [w, s]


def test(a_test, b_test, w, s):
    acc = 0
    for a in a_test:
        if w.T * np.mat([a['x1'], a['x2']]).T > s:
            a['y_'] = 1
            acc += 1
        else:
            a['y_'] = -1
    for b in b_test:
        if w.T * np.mat([b['x1'], b['x2']]).T > s:
            a['y_'] = 1
        else:
            a['y_'] = -1
            acc += 1
    acc /= (len(a_test) + len(b_test))
    print('test_acc= ' + str(acc))


def get_train_acc():
    acc = 0
    for a in a_train:
        if w.T * np.mat([a['x1'], a['x2']]).T > s:
            a['y_'] = 1
            acc += 1
        else:
            a['y_'] = -1
    for b in b_train:
        if w.T * np.mat([b['x1'], b['x2']]).T > s:
            a['y_'] = 1
        else:
            a['y_'] = -1
            acc += 1
    acc /= (len(a_train) + len(b_train))
    print('train_acc= ' + str(acc))


def draw(w, s):
    for a in a_train:
        plt.scatter(a['x1'], a['x2'], c = 'red', s = 1, label = 'a')
    for b in b_train:
        plt.scatter(b['x1'], b['x2'], c = 'blue', s = 1, label = 'b')
    for a in a_test:
        plt.scatter(a['x1'], a['x2'], c = 'red', s = 20, label = 'a', marker = '+')
    for b in b_test:
        plt.scatter(b['x1'], b['x2'], c = 'blue', s = 20, label = 'b', marker = '+')
    # plt.plot([-5, 5], [-(w[0,0] * (-5) + s[0,0]) / w[1,0], -(w[0,0] * 5 + s[0,0]) / w[1,0]], c = 'green')
    plt.plot([-5, 5], [-(w[0, 0] * 5 + s[0, 0]) / w[1, 0], -(w[0, 0] * (-5) + s[0, 0]) / w[1, 0]],
             c = 'green')  # 取mat中的某个元素m[i,j]
    plt.plot([-5, 5], [-(w[0, 0] * (-5) + s[0, 0]) / w[1, 0], -(w[0, 0] * 5 + s[0, 0]) / w[1, 0]], c = 'pink')

    plt.xlabel("x1", fontdict = {'size': 16})
    plt.ylabel("x2", fontdict = {'size': 16})
    plt.show()


each_train_num = 160
each_test_num = 40

[a_train, b_train, a_test, b_test] = create_points(each_train_num, each_test_num)  # 得到训练、测试数据
[mv_a, mv_b] = get_mv(a_train, b_train)  # 得到mv
[w, s] = get_w_and_s(a_train, b_train, mv_a, mv_b)  # 得到w和s
get_train_acc()
test(a_test, b_test, w, s)  # 在测试数据上测试正确率
draw(w, s)  # 画图
