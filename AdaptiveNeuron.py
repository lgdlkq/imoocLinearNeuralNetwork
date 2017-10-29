#!usr/bin/env python3
# coding=utf-8

__author__ = 'lgd'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.font_manager import FontProperties


# 适应性线性神经元模型（自动改进）
class AdalineGD(object):
    """
    eta:float 学习效率，介于0到1之间
    n_iter:int 对训练数据进行学习改进的次数
    w_:一维向量 存储权重数值
    error_:存储每次迭代改进时网络对数据进行判断错误的次数
    """

    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """
            X：shape(n_samples,n_features)#n_samples表示X中有多少样本向量，n_features为含义4一个数据的一维向量，用于表示一条训练项目
                eg:
                    X:[[1,2,3],[4,5,6]]
                    n_samples=2   n_feayures=3
            y:[1,-1]  #一维向量，表示X:[[1,2,3],[4,5,6]]中的[1,2,3]属于1分类,[4,5,6]属于-1分类
            :param X:输入样本向量
            :param y:对应样本的正确分类（用于存储每一条训练项目对应的正确分类）
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []  # 成本向量

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)


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

def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max()
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max()

    # print(x1_min,x1_max)
    # print(x2_min,x2_max)

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    # print(np.arange(x1_min,x1_max,resolution).shape)
    # print(np.arange(x2_min, x2_max, resolution).shape)
    # print(np.arange(x1_min,x1_max,resolution))
    # print(np.arange(x2_min, x2_max, resolution))
    # print(xx1.shape)
    # print(xx1)
    # print(xx2.shape)
    # print(xx2)

    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)  # ravel()函数是把向量还原成原来的单维数组
    # print(xx1.ravel())
    # print(xx2.ravel())
    # print(z)
    z = z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, z, alpha=0.4, cmap=cmap)  # 给数据画分类线
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)


ada = AdalineGD(eta=0.0001, n_iter=50)
ada.fit(X, y)
plot_decision_regions(X, y, classifier=ada)

font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)
plt.title('Adaline-Gradient descent')
plt.xlabel('花瓣的长度', fontproperties=font_set)
plt.ylabel('花茎的长度', fontproperties=font_set)
plt.legend(loc='upper left')
plt.show()

plt.plot(range(1,len(ada.cost_)+1),ada.cost_,marker='o')
plt.title('Adaline - Learning rate 0.0001')
plt.xlabel('Epochs')
plt.ylabel('sum-squard-error')
plt.show()
