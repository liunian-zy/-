# -*- coding: utf-8 -*-
# 作者：李军
# 时间：2020年月日
# 功能：
# https://blog.csdn.net/Eastmount/article/details/78692227?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522160319543919726892460530%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=160319543919726892460530&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_v2~rank_v28-2-78692227.pc_first_rank_v2_rank_v28&utm_term=%E9%B8%A2%E5%B0%BE%E8%8A%B1&spm=1018.2118.3001.4187

# 一. 鸢尾花数据集介绍
# 数据集共包含4个特征变量、1个类别变量，共有150个样本。
# 类别变量分别对应鸢尾花的三个亚属，分别是
# 山鸢尾 (Iris-setosa)、变色鸢尾(Iris-versicolor)
# 和维吉尼亚鸢尾(Iris-virginica)。
# 通过sklearn.datasets扩展包中的load_iris()函数导入鸢尾花数据集，
# 该Iris中有两个属性，分别是：iris.data和iris.target。
# data里是一个矩阵，每一列代表了萼片或花瓣的长宽，一共4列，
# 每一列代表某个被测量的鸢尾植物，一共采样了150条记录。代码如下：

import matplotlib.pyplot as plt
# 导入数据集iris  
from sklearn.datasets import load_iris

# 载入数据集  
iris = load_iris()
# 输出数据集  
print(iris.data)

# target是一个数组，存储了data中每条记录属于哪一类鸢尾植物，
# 数组长度是150，数组元素的值因为共有3类鸢尾植物，所以不同值只有3个。种类：
#     Iris Setosa（山鸢尾）
#     Iris Versicolour（杂色鸢尾）
#     Iris Virginica（维吉尼亚鸢尾）

# 输出真实标签  
print(iris.target)
print(len(iris.target))
# 150个样本 每个样本4个特征  
print(iris.data.shape)

# 二. 可视化分析鸢尾花
# 数据可视化可以更好地了解数据，主要调用Pandas扩展包进行绘图操作。
# 首先绘制直方图，直观的表现花瓣、花萼的长和宽特征的数量，
# 纵坐标表示汇总的数量，横坐标表示对应的长度。
import pandas

# 导入数据集iris  
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)  # 读取csv数据
print(dataset.describe())
# 直方图 histograms
plt.figure(1)
dataset.hist()
# plt.show()
# 散点图
plt.figure(2)
dataset.plot(x='sepal-length', y='sepal-width', kind='scatter')
# plt.show()
# 核密度估计,通过四条曲线反映四个特征的变化情况
plt.figure(3)
dataset.plot(kind='kde')
# plt.show()
# 箱形图的纵坐标（y轴）的刻度是不同的，有明显的区分，
# 因此可以看到，各变量表示的属性是有区分的
plt.figure(4)
dataset.plot(kind='box', subplots=True, layout=(2, 2),
             sharex=False, sharey=False)
# plt.show()
# quit()
# 接下来调用radviz()函数、andrews_curves()函数和parallel_coordinates()
# 函数绘制图形这里选择petal-length特征
from pandas.plotting import radviz

plt.figure(5)
radviz(dataset, 'class')
# plt.show()
# quit()
from pandas.plotting import andrews_curves

plt.figure(6)
andrews_curves(dataset, 'class')
# plt.show()
from pandas.plotting import parallel_coordinates

plt.figure(7)
parallel_coordinates(dataset, 'class')
# plt.show()
# 散点图矩阵，这有助于发现变量之间的结构化关系，散点图代表了两变量的相关程度
# 如果呈现出沿着对角线分布的趋势，说明它们的相关性较高
from pandas.plotting import scatter_matrix

plt.figure(8)
scatter_matrix(dataset, alpha=0.2, figsize=(6, 6), diagonal='kde')
plt.show()

# 三. 线性回归分析鸢尾花
# 该部分主要采用线性回归算法对鸢尾花的特征数据进行分析，
# 预测花瓣长度、花瓣宽度、花萼长度、花萼宽度四个特征之间的线性关系。
from sklearn.datasets import load_iris

hua = load_iris()
# 获取花瓣的长和宽
x = [n[0] for n in hua.data]
y = [n[1] for n in hua.data]
import numpy as np  # 转换成数组

x = np.array(x).reshape(len(x), 1)
y = np.array(y).reshape(len(y), 1)

from sklearn.linear_model import LinearRegression

clf = LinearRegression()
clf.fit(x, y)
pre = clf.predict(x)

# 第三步 画图
import matplotlib.pyplot as plt

plt.scatter(x, y, s=100)
plt.plot(x, pre, "r-", linewidth=4)
for idx, m in enumerate(x):
    plt.plot([m, m], [y[idx], pre[idx]], 'g-')
plt.show()

print(u"系数", clf.coef_)
print(u"截距", clf.intercept_)
print(np.mean(y - pre) ** 2)
# 系数 [[-0.05726823]]
# 截距 [ 3.38863738]
# 1.91991214088e-31

# 假设现在存在一个花萼长度为5.0的花，需要预测其花萼宽[3.10229621]。度，
# 则使用该已经训练好的线性回归模型进行预测，其结果应为
print(clf.predict([[5.0]]))
# [[ 3.10229621]]

# 四. 决策树分析鸢尾花
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
# 训练集  
train_data = np.concatenate((iris.data[0:40, :], iris.data[50:90, :], iris.data[100:140, :]), axis=0)
train_target = np.concatenate((iris.target[0:40], iris.target[50:90], iris.target[100:140]), axis=0)
# 测试集  
test_data = np.concatenate((iris.data[40:50, :], iris.data[90:100, :], iris.data[140:150, :]), axis=0)
test_target = np.concatenate((iris.target[40:50], iris.target[90:100], iris.target[140:150]), axis=0)

# 训练
clf = DecisionTreeClassifier()
clf.fit(train_data, train_target)
predict_target = clf.predict(test_data)
print(predict_target)

# 预测结果与真实结果比对  
print(sum(predict_target == test_target))
# 输出准确率 召回率 F值  
from sklearn import metrics

print(metrics.classification_report(test_target, predict_target))
print(metrics.confusion_matrix(test_target, predict_target))
X = test_data
L1 = [n[0] for n in X]
print(L1)
L2 = [n[1] for n in X]
print(L2)
import numpy as np
import matplotlib.pyplot as plt

plt.scatter(L1, L2, c=predict_target, marker='x')  # cmap=plt.cm.Paired  
plt.title("DecisionTreeClassifier")
plt.show()

# 五. Kmeans聚类分析鸢尾花
# -*- coding: utf-8 -*-
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

iris = load_iris()
clf = KMeans()
clf.fit(iris.data, iris.target)
print(clf)
predicted = clf.predict(iris.data)

# 获取花卉两列数据集
X = iris.data
L1 = [x[0] for x in X]
print(L1)
L2 = [x[1] for x in X]
print(L2)
import numpy as np
import matplotlib.pyplot as plt

plt.scatter(L1, L2, c=predicted, marker='s', s=200, cmap=plt.cm.Paired)
plt.title("DTC")
plt.show()
