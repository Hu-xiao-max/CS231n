
from __future__ import print_function
import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt


Xtr, Ytr, Xte, Yte = load_CIFAR10('data/cifar-10-batches-py') # a magic function we provide
# flatten out all images to be one-dimensional
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3) # 拉成一维向量的训练集Xtr_rows becomes 50000 x 3072
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3) # 拉成一维向量的验证集Xte_rows becomes 10000 x 3072

#print(Xtr_rows,np.shape(Xtr_rows))#测试看数据用
#print(Xte_rows,np.shape(Xte_rows))#测试看数据用
#print(Ytr,Yte,np.shape(Ytr),np.shape(Yte))
'''Ytr(50000,),Yte(10000,)分别是训练集和测试集的种类label'''


class NearestNeighbor(object):
  def __init__(self):
    pass

  def train(self, X, y):
    """ X is N x D where each row is an example. Y is 1-dimension of size N """
    # the nearest neighbor classifier simply remembers all the training data
    self.Xtr = X
    self.ytr = y

  def predict(self, X):
    """ X is N x D where each row is an example we wish to predict label for """
    num_test = X.shape[0]#x的行数也就是图片的数量，因为每张图片被拉成一维向量了
    # lets make sure that the output type matches the input type
    Ypred = np.zeros(num_test, dtype = self.ytr.dtype)
    #Ypred是列数为图片数量的零向量，数据类型跟ytr一样

    # loop over all test rows
    for i in range(num_test):
      # find the nearest training image to the i'th test image
      # using the L1 distance (sum of absolute value differences)
      distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
      #对每行求所有元素求差并累加，所以distances的大小为图片数量(即行数）*图片数量
      min_index = np.argmin(distances) # get the index with smallest distance
      #找到distances中最小元素的位置
      Ypred[i] = self.ytr[min_index]
      #predict the label of the nearest example找到了距离最小的图片即认为为一类的图片

    return Ypred

nn = NearestNeighbor() # create a Nearest Neighbor classifier class
nn.train(Xtr_rows, Ytr) # train the classifier on the training images and labels
Yte_predict = nn.predict(Xte_rows) # predict labels on the test images
# and now print the classification accuracy, which is the average number
# of examples that are correctly predicted (i.e. label matches)
print('accuracy: %f' % (np.mean(Yte_predict == Yte)))

#distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis = 1))
#l2距离只要把上面那段替换进去就可以



