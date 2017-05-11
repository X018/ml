# -*- coding: utf-8 -*-
# perceptron.py

"""
Created by jin.xia on May 10 2017

@author: jin.xia
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append("../perceptron")
import perceptron


data_frame = pd.read_csv('input/iris.data', header=None)
print(data_frame.tail())
training_num = 100


# ----------------------------------------- 多维数据 -----------------------------------------
print('----------------------------------------- four features -----------------------------------------')
dataset = data_frame.iloc[0:training_num, [0,1,2,3,4]].values
X = dataset[:, :4]
Y = dataset[:, 4]
Y = np.where(Y == 'Iris-setosa', -1, 1)
per = perceptron.perceptron('dual')
per.learning(X, Y)
X_new = np.array([[3,3,3,4],[2,4,5,6],[3,5,5,1]])
predictions = per.predict(X_new)
print(predictions)



# ----------------------------------------- 二维数据画图 -----------------------------------------
print('----------------------------------------- two features -----------------------------------------')
dataset = data_frame.iloc[0:training_num, [0, 2, 4]].values
X = dataset[:, :2]
Y = dataset[:, 2]
Y = np.where(Y == 'Iris-setosa', -1, 1)

per = perceptron.perceptron('dual')
per.learning(X, Y)
# per.plt_scatter_line(X, Y)

X_new = np.array([[3,4],[5,6],[5,1]])
predictions = per.predict(X_new)
print(predictions)
# 绘制带预测点的数据图
axis_mat = np.append(X[:,0], X_new[:,0])
minx, maxx = per.get_bounds(axis_mat)
per.plt_scatter(X, Y)
per.plt_line(minx, maxx)
per.plt_scatter(X_new, predictions, alpha=1, size=50)
per.plt_show()


