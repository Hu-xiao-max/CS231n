from __future__ import print_function
import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt

Xtr, Ytr, Xte, Yte = load_CIFAR10('data/cifar-10-batches-py')  # a magic function we provide

"""
  unvectorized version. Compute the multiclass svm loss for a single example (x,y)
  - x is a column vector representing an image (e.g. 3073 x 1 in CIFAR-10)
    with an appended bias dimension in the 3073-rd position (i.e. bias trick)
  - y is an integer giving index of correct class (e.g. between 0 and 9 in CIFAR-10)
  - W is the weight matrix (e.g. 10 x 3073 in CIFAR-10)
  """

b = np.ones(50000)
x_train_process = np.c_[Xtr.reshape(Xtr.shape[0], 32 * 32 * 3), b]#做齐次变换加上一列[0,0,0...0]
y_train = Ytr
W = np.random.rand(10, 3073)#随机一个权重矩阵
for i in range(x_train_process.shape[0] - 1):
    x_train = x_train_process[i + 1]#依次取出x_train的每一行即为每一张图片
    delta = 1.0  # see notes about delta later in this section设置的安全边界
    scores = W.dot(x_train)  # scores becomes of size 10 x 1, the scores for each class
    for i in range(x_train_process.shape[0] - 1):
        num = y_train[i]#当前图片的label
        correct_class_score = scores[num]  # 正确类别的分因为有10个种类，label是什么，就是当前正确类别的索引
        D = W.shape[0]  # number of classes, e.g. 10,W有几行就有几个种类
        loss_i = 0.0
        for j in range(D):  # iterate over all wrong classes
            if j == num:  # skip for the true class to only loop over incorrect classes

                continue
                # accumulate loss for the i-th example
            loss_i += max(0, scores[j] - correct_class_score + delta)#svm函数
            print(loss_i)
