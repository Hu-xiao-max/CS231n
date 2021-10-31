from __future__ import print_function
import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt

Xtr, Ytr, Xte, Yte = load_CIFAR10('data/cifar-10-batches-py')  # a magic function we provide
# flatten out all images to be one-dimensional
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)  # 拉成一维向量的训练集Xtr_rows becomes 50000 x 3072
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)  # 拉成一维向量的验证集Xte_rows becomes 10000 x 3072

# print(Ytr.shape,Yte.shape)


Xval_rows = Xtr_rows[:1000, :]  # 取1000张图片
# print(Xval_rows.shape)
Yval = Ytr[:1000]  # 取出前1000
# print(Yval.shape)
Xtr_rows = Xtr_rows[1000:, :]  # keep last 49,000 for train
# print(Xtr_rows.shape)#Xtr_rows：(49000, 3072)
Ytr = Ytr[1000:]  # 取出之后的49000个数据
# print(Ytr.shape)


num = Xtr_rows.shape[0]
# print(num)
k = [1, 3, 5, 10, 20, 50, 100]
Ypred = np.zeros(num, dtype=Ytr.dtype)
for t in k:
    for i in range(num):
        # find the nearest training image to the i'th test image
        # using the L1 distance (sum of absolute value differences)
        distances = np.sum(np.abs(Xtr_rows - Xte_rows[i, :]), axis=1)
        # 把train中的50000张图片依次取出跟test中的第i个进行比较算出l1距离
        # print(distances.shape)
        idx = np.argpartition(distances, t)
        # print(idx.shape)
        knn = idx[:t]
        # print(knn)
        knn_1 = np.zeros(t)
        knn_2 = Ytr[knn]
        # print(knn_2)
        vote = np.argmax(np.bincount(knn_2))
        Ypred[i] = Ytr[vote]
        print(Ypred)
print('accuracy: %f' % (np.mean(Ypred == Yte)))
