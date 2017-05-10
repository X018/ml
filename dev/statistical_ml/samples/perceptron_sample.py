# -*- coding: utf-8 -*-
# perceptron.py

"""
Created by jin.xia on May 10 2017

@author: jin.xia
"""


import sys
sys.path.append("../perceptron")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import perceptron


print('------------------------------------------------------------feature')
data_frame = pd.read_csv('input/iris.data', header=None)
print(data_frame.tail())

training_num = 100
dataset = data_frame.iloc[0:training_num, [0, 2, 4]].values
X = dataset[:, :2]
Y = dataset[:, 2]
Y = np.where(Y == 'Iris-setosa', -1, 1)


print('------------------------------------------------------------learning')
per = perceptron.learning(X, Y, type='primal')



print('------------------------------------------------------------predict')
X_new = np.array([[3,4],[5,6],[5,1]])
predictions = per.predict(X_new)
print(predictions)

perceptron.plt_scatter_line(X, Y, per.w, per.b, X_new, predictions)

