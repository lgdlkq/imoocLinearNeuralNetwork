#!usr/bin/env python3
# coding=utf-8

__author__ = 'lgd'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.colors import ListedColormap


# 线性神经元模型
class Perceptron(object):
    """
    eta:学习率
    n_iter:权重向量的训练次数
    w_:神经分叉权重向量
    errors_:用于记录神经元判断出错次数
    """

    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """
        输入训练数据，培训神经元
        X：shape(n_samples,n_features)#n_samples表示有多少样本向量，n_features表示分叉数量
            eg:
               X:[[1,2,3],[4,5,6]]
               n_samples=2   n_feayures=3
        y:[1,-1]  #表示X:[[1,2,3],[4,5,6]]中的[1,2,3]属于1分类,[4,5,6]属于-1分类
        :param X:输入样本向量
        :param y:对应样本的正确分类
        """
        """
        初始化权重向量为0
        加一是算法中的W0，也就是步调函数阈值
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):  # 更新权重
                """
                update=η*(y-y')
                """
                update = self.eta * (target - self.predict(xi))
                """
                xi是一个向量
                update*xi等价于：▽w(i)=X[i]*update
                """
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
                self.errors_.append(errors)

    def net_input(self, x):  # 对所有的电信号进行加权求和
        """
        z=W0*1+W1*X1+...Wn*Xn
        """
        return np.dot(x, self.w_[1:]) + self.w_[0]

    def predict(self, x):
        return np.where(self.net_input(x) >= 0.0, 1, -1)


# 数据
file = 'iris.data.csv'  # 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(file, header=None)
# print(df.head(10))

'''
pandas中数据截取方式，loc,iloc,ix对比

1.loc:通过行索引获取行数据

2.iloc:通过行号获取行数据

3.ix:混合方式，既可以通过行号，又可以通过行索引获取行数据
'''
y = df.iloc[0:100, 4].values  # 取出前100行的第四列的内容
# print(y)
y = np.where(y == 'Iris-setosa', -1, 1)
# print(y)
X = df.iloc[0:100, [0, 2]].values  # 取出前100行第一和第三列
# print(X)

font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)
# 可视化数据图像
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')

plt.xlabel('花瓣的长度', fontproperties=font_set)
plt.ylabel('花茎的长度', fontproperties=font_set)
plt.legend(loc='upper left')
plt.show()

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)

def plot_decision_regions(X,y,classifier,resolution=0.02):
    markers=('s','x','o','v')
    colors=('red','blue','lightgreen','gray','cyan')
    cmap=ListedColormap(colors[:len(np.unique(y))])
    x1_min,x1_max=X[:,0].min()-1,X[:,0].max()
    x2_min,x2_max=X[:,1].min()-1,X[:,1].max()

    # print(x1_min,x1_max)
    # print(x2_min,x2_max)

    xx1,xx2=np.meshgrid(np.arange(x1_min,x1_max,resolution),
                        np.arange(x2_min,x2_max,resolution))
    # print(np.arange(x1_min,x1_max,resolution).shape)
    # print(np.arange(x2_min, x2_max, resolution).shape)
    # print(np.arange(x1_min,x1_max,resolution))
    # print(np.arange(x2_min, x2_max, resolution))
    # print(xx1.shape)
    # print(xx1)
    # print(xx2.shape)
    # print(xx2)

    z=classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)#ravel()函数是把向量还原成原来的单维数组
    # print(xx1.ravel())
    # print(xx2.ravel())
    # print(z)
    z=z.reshape(xx1.shape)
    plt.contourf(xx1,xx2,z,alpha=0.4,cmap=cmap)#给数据画分类线
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())
    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0],y=X[y==cl,1],alpha=0.8,c=cmap(idx),marker=markers[idx],label=cl)

plot_decision_regions(X,y,classifier=ppn)
plt.xlabel('花瓣的长度', fontproperties=font_set)
plt.ylabel('花茎的长度', fontproperties=font_set)
plt.legend(loc='upper left')
plt.show()

print(len(ppn.errors_))
plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker='o')
plt.xlabel('Iterations')
plt.ylabel('Missclassifications')
plt.show()