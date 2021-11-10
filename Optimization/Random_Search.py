from __future__ import print_function
import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
from cs231n.classifiers import *


import time
start=time.time()
Xtr, Ytr, Xte, Yte = load_CIFAR10('data/cifar-10-batches-py')  # a magic function we provide
b = np.ones(50000)
x_train_process = np.c_[Xtr.reshape(Xtr.shape[0], 32 * 32 * 3), b]
y_train = Ytr
#print(x_train_process.shape,y_train.shape)
#print(x_train_process.shape[0] - 1)
bestloss = float("inf")  # 正无穷
c=0
Loss=[]
for num in range(1000):
    W = np.random.randn(10, 3073) *0.001  # generate random parameters
    c+=1
    for i in range(x_train_process.shape[0]):
        x_train = x_train_process[i]  # 依次取出x_train的每一行即为每一张图片
        delta = 1.0  # see notes about delta later in this section设置的安全边界
        scores = W.dot(x_train)  # scores becomes of size 10 x 1, the scores for each class

        num = y_train[i]  # 当前图片的label
        correct_class_score = scores[num]  # 正确类别的分因为有10个种类，label是什么，就是当前正确类别的索引
        D = W.shape[0]  # number of classes, e.g. 10,W有几行就有几个种类

        #print(correct_class_score)
        for j in range(D):  # iterate over all wrong classes
            if j == num:  # skip for the true class to only loop over incorrect classes

                continue
                    # accumulate loss for the i-th example
            loss_i = max(0, scores[j] - correct_class_score + delta)  # svm函数
            #print(loss_i)
            Loss.append(loss_i)
        Loss_every=sum(Loss)/len(Loss)
        #print(Loss_every)
        if Loss_every < bestloss:  # keep track of the best solution
            bestloss = Loss_every
            bestW = W

            #print('in attempt %d the loss was %f, best %f' % (c, Loss_every, bestloss))
        else:

            #print('in attempt %d the loss was %f, best %f' % (c, Loss_every, bestloss))
            continue
    print('in attempt %d the loss was %f, best %f' % (c, Loss_every, bestloss))




'''以下为测试准确率'''
scores = bestW.dot(Xte) # 10 x 10000, the class scores for all test examples
# 找到在每列中评分值最大的索引（即预测的分类）
Yte_predict = np.argmax(scores, axis = 0)
# 以及计算准确率
np.mean(Yte_predict == Yte)#Yte_predict跟Yte相等的概率
# 示例返回 0.1555
end=time.time()
print(start-end)
