import numpy as np
'''
A= np.array([[1,1,1,1,1,1,1,1,1,1,1],
                [-5,-4,-3,-2,-1,0,1,2,3,4,5],
            ]).transpose()
b= np.array([2,7,9,12,13,14,14,13,10,8,4])

AA = np.dot(A.transpose(),A)
print(AA)
AB = np.dot(A.transpose(),b)
print(AB)
x = np.dot(np.linalg.inv(AA),AB)
print(x)

print(np.dot(b-np.dot(A,x),(b-np.dot(A,x).transpose())))
#[25, 16, 9, 4, 1, 0, 1, 4, 9, 16, 25]


A = np.array([[1,3,1,-4],
             [-1,-3,1,0],
             [2,6,2,-8]])

AA = np.dot(A.transpose(),A)
print(AA)

print(np.linalg.matrix_rank(A))
'''

import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
x = np.random.rand(10000)
t = np.arange(len(x))
# plt.plot(t, x, 'g.', label=u'均匀分布')  # 散点图
plt.hist(x, 1000, color='m', alpha=0.6, label=u'均匀分布', normed=True)
plt.legend(loc='upper right')
plt.grid(True, ls=':')
plt.show()






