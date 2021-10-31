import numpy as np

a=np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9]])
b = a[:5]
c = a[1:]
print(b,c)
#idx = np.argpartition(a, 3)
A = np.array([1, 7, 9, 2, 0.1, 17, 17, 1.5])
k = 4

idx = np.argpartition(A, k)
print(idx)

#print(a[:3,:])
e=A[0,3]