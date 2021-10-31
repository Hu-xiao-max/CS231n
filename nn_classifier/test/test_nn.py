from __future__ import print_function
import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt

Xtr, Ytr, Xte, Yte = load_CIFAR10('data/cifar-10-batches-py')  # a magic function we provide
# flatten out all images to be one-dimensional
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)  # 拉成一维向量的训练集Xtr_rows becomes 50000 x 3072
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)  # 拉成一维向量的验证集Xte_rows becomes 10000 x 3072

num1 = Xtr_rows.shape[0]
num2 = Xte_rows.shape[0]
# print(num)
Ypred = np.zeros(num1, dtype=Ytr.dtype)


for i in range(num1):
    # find the nearest training image to the i'th test image
    # using the L1 distance (sum of absolute value differences)
    distances = np.sum(np.abs(Xtr_rows - Xte_rows[i, :]), axis=1)
    # 把train中的50000张图片依次取出跟test中的第i个进行比较算出l1距离
    min_index = np.argmin(distances)  # get the index with smallest distance
    print(min_index)
    # 找到distances中最小元素的位置
    Ypred = np.zeros(num1, dtype=Ytr.dtype)
    Ypred[i] = Ytr[min_index]
    print(Ypred[i])
    # predict the label of the nearest example找到了距离最小的图片即认为为一类的图片
print('accuracy: %f' % (np.mean(Ypred == Yte)))
'''-----------------------------------------------------------------------------------'''


