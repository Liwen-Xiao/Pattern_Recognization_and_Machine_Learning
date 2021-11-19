import numpy as np

from cvxopt import solvers, matrix

import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
import time
import matplotlib.pyplot as plt

x=matrix([1,1,1,1,1])
print(x)
#y=matrix([2,2,2,2,2])
#print(np.c_[1,x.T])

'''
x_axis_data = [float(i/10) for i in range (51)]
print(x_axis_data)
y_axis_data=[0,0.11,0.48,1.21,1.46,1.75,2.04,2.32,2.6,2.88,3.17,3.47,3.76,4.03,4.32,4.63,4.94,5.24,5.56,5.84,6,6.32,6.64,6.96,7.28,7.57,7.9,8.23,8.45,8.49,8.51,8.52,8.52,8.53,8.53,8.53,8.53,8.53,8.53,8.53,8.53,8.54,8.54,8.54,8.54,8.54,8.54,8.54,8.54,8.54,8.54]


# plot中参数的含义分别是横轴值，纵轴值，颜色，透明度和标签
plt.plot(x_axis_data, y_axis_data, 'o-', color='#4169E1', alpha=0.8)

#for x, y in zip(x_axis_data, y_axis_data):
 #   plt.text(x, y+0.3, '%.0f' % y, ha='center', va='bottom', fontsize=10.5)


# 显示标签，如果不加这句，即使加了label='一些数字'的参数，最终还是不会显示标签
plt.legend(loc="upper right")
plt.xlabel('X(mm)')
plt.ylabel('电压(V)')
plt.show()


x = [1, 2, 3, 4, 5]
train_acc_list = [1, 2, 3, 4, 5]
plt.plot(x, train_acc_list, 'o-', color = 'blue', alpha = 0.8, label = 'train_acc')
for x, y in zip(x, train_acc_list):
    plt.text(x, y, '%.0f' % y, ha = 'center', va = 'bottom', fontsize = 10.5)
plt.show()
'''
