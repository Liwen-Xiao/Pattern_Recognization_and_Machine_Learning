'''
在iris样本集上测试OAO算法
'''
import numpy as np
import sys
sys.path.append("..")
from sklearn import datasets
import random


def PLA(a_train, b_train):
    '''
    通过PLA训练模型
    :param a_train:
    :param b_train:
    :return:
    '''
    w = np.mat([1.0, 1.0, 1.0, 1.0, 1.0]).T
    epoch = 1000
    for i in range(epoch):
        acc = a_train.shape[0] + b_train.shape[0]
        for j in range(a_train.shape[0]):
            if (w.T * a_train[j].T)[0, 0] <= 0:
                w = w + a_train[j].T
                acc -= 1
            elif (w.T * b_train[j].T)[0, 0] >= 0:
                w = w - b_train[j].T
                acc -= 1
        if acc == a_train.shape[0] + b_train.shape[0]:
            print('the final accuracy is 1.0')
            return w
    print('the final accuracy is ' + str(acc / (a_train.shape[0] + b_train.shape[0])))
    return w


def OAO_train():
    '''
    得到每两类之间的分类面
    :return:
    '''
    w12 = PLA(flower_1_data_train, flower_2_data_train)
    w13 = PLA(flower_1_data_train, flower_3_data_train)
    w23 = PLA(flower_2_data_train, flower_3_data_train)

    return [w12, w13, w23]


def OAO_test(flower_1_data_test, flower_2_data_test, flower_3_data_test):
    '''
    在测试样本上面检测分类面的正确率
    :param flower_1_data_test:
    :param flower_2_data_test:
    :param flower_3_data_test:
    :return:
    '''
    acc_1 = 0
    for i in range(20):
        type = [0, 0, 0]
        if (w12.T * flower_1_data_test[i].T)[0, 0] > 0:
            type[0] += 1
        else:
            type[1] += 1

        if (w13.T * flower_1_data_test[i].T)[0, 0] > 0:
            type[0] += 1
        else:
            type[2] += 1

        if (w23.T * flower_1_data_test[i].T)[0, 0] > 0:
            type[1] += 1
        else:
            type[2] += 1

        if type[0] > type[1] and type[0] > type[2]:
            acc_1 += 1
    print('the accuracy in the type_1 is ' + str(acc_1 / 20.0))

    acc_2 = 0
    for i in range(20):
        type = [0, 0, 0]
        if (w12.T * flower_2_data_test[i].T)[0, 0] > 0:
            type[0] += 1
        else:
            type[1] += 1

        if (w13.T * flower_2_data_test[i].T)[0, 0] > 0:
            type[0] += 1
        else:
            type[2] += 1

        if (w23.T * flower_2_data_test[i].T)[0, 0] > 0:
            type[1] += 1
        else:
            type[2] += 1

        if type[1] > type[0] and type[1] > type[2]:
            acc_2 += 1
    print('the accuracy in the type_2 is ' + str(acc_2 / 20.0))

    acc_3 = 0
    for i in range(20):
        type = [0, 0, 0]
        if (w12.T * flower_3_data_test[i].T)[0, 0] > 0:
            type[0] += 1
        else:
            type[1] += 1

        if (w13.T * flower_3_data_test[i].T)[0, 0] > 0:
            type[0] += 1
        else:
            type[2] += 1

        if (w23.T * flower_3_data_test[i].T)[0, 0] > 0:
            type[1] += 1
        else:
            type[2] += 1

        if type[2] > type[1] and type[2] > type[0]:
            acc_3 += 1
    print('the accuracy in the type_3 is ' + str(acc_3 / 20.0))


iris = datasets.load_iris()
'''
print('data')  # array(150*4)
print(iris.data)

print(type(iris.data))
print(iris.data.shape)
print()

print('feature_names')  # list
print(iris.feature_names)
print(type(iris.feature_names))
# print(iris.feature_names.shape)
print()

print('target')  # array(150*1)
print(iris.target)
print(type(iris.target))

print(iris.target.shape)
print()

print('target_names')  # array(3*1)
print(iris.target_names)
print(type(iris.target_names))
print(iris.target_names.shape)
print()
'''
# 将iris样本集的data转成矩阵形式，并且第一列全为1
iris.data = np.mat(iris.data)
a = []
for i in range(150):
    a.append(1)
a = np.mat(a).T
iris.data = np.c_[a, iris.data]
print(iris.data)

# 得到三类样本的训练数据集和测试数据集
flower_1_data = iris.data[0:50, :]
random.shuffle(flower_1_data)
flower_1_data_train = flower_1_data[0:30, :]
flower_1_data_test = flower_1_data[30:50, :]

flower_2_data = iris.data[50:100, :]
random.shuffle(flower_2_data)
flower_2_data_train = flower_2_data[0:30, :]
flower_2_data_test = flower_2_data[30:50, :]

flower_3_data = iris.data[100:150, :]
random.shuffle(flower_3_data)
flower_3_data_train = flower_3_data[0:30, :]
flower_3_data_test = flower_3_data[30:50, :]

# print(len(flower_3_data_train))

[w12, w13, w23] = OAO_train()  # 训练得到三个分类面
OAO_test(flower_1_data_test, flower_2_data_test, flower_3_data_test)  # 测试分类面的正确率
