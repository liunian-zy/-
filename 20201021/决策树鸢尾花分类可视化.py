# -*- coding: utf-8 -*-
# 作者：李军
# 时间：2020年月日
# 功能：
import graphviz
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn import datasets
import pandas as pd

# 原始数据集
iris = datasets.load_iris()
print(iris)
print(100 * "+")

# 经过pandas处理后的数据集
feature = pd.DataFrame(data=iris.data, columns=iris.feature_names)
target = pd.DataFrame(data=list(map(lambda item: iris.target_names[item],
                                    iris.target)), columns={'target_names'})
iris_datasets = pd.concat([feature, target], axis=1)
print(iris_datasets)
print(100 * "+")

feature_train, feature_test, target_train, target_test = \
    train_test_split(feature, target, test_size=0.3)
# 利用决策树分类器构建分类模型
model = DecisionTreeClassifier()
model.fit(feature_train, target_train)
pred = model.predict(feature_test)
print(pred)
print(100 * "+")

# 对比真实值数据
true = target_test.values.flatten()
print(true)


print(classification_report(true, pred))


decision_tree_graph = export_graphviz(model, out_file=None, feature_names=iris.feature_names,
                                      class_names=iris.target_names, filled=True, node_ids=True)
DTgraph = graphviz.Source(decision_tree_graph)
DTgraph.render(filename='Decision Tree', directory='', view=True)
# DTgraph.view()
