# -*- coding: utf-8 -*-
# perceptron.py

"""
Created by jin.xia on May 10 2017

@author: jin.xia
"""
import sys
sys.path.append("..")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import perceptron.perceptron as perceptron
import perceptron.perceptron_dual as perceptron_dual
import perceptron.perceptron_plt as per_plt


print('------------------------------------------------------------')
data_frame = pd.read_csv('input/iris.data', header=None)
print(data_frame.tail())

training_num = 100
dataset = data_frame.iloc[0:training_num, [0, 2, 4]].values
X = dataset[:, :2]
Y = dataset[:, 2]

Y = np.where(Y == 'Iris-setosa', -1, 1)
print(X)
print(Y)


# print('------------------------------------------------------------')
# X0 = np.array([X[i] for i in range(training_num) if Y[i]==-1])
# X1 = np.array([X[i] for i in range(training_num) if Y[i]==1])
# fig = plt.figure()
# fig.set(alpha=0.2)
# plt.scatter(X0[:,0], X0[:,1], marker='x', s=25,
# 	color='blue', alpha=0.4, label='versicolor')
# plt.scatter(X1[:,0], X1[:,1], marker='o', s=25,
# 	color='red', alpha=0.4, label='setosa')
# plt.show()



print('------------------------------------------------------------')
per = perceptron.perceptron()
per.learning(X, Y)
per_plt.figure()
per_plt.plt_scatter(X, Y)
per_plt.plt_sep_line(X, per.w, per.b)
per_plt.show()





# per_dual = perceptron_dual.perceptron_dual()
# per_dual.learning(X, Y)
# per_plt.plt_sep_line(X, per.w, per.b)
# per_plt.plt_show()