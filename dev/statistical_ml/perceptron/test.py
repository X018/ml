# -*- coding: utf-8 -*-
# perceptron.py

"""
Created by jin.xia on May 10 2017

@author: jin.xia
"""


import numpy as np
import perceptron


def generate_dataset():
	arr = [[3,3,1], [4,3,1], [1,1,-1]]
	dataset = np.array(arr)
	X = dataset[:,:-1]
	Y = dataset[:,-1]
	return X, Y


X, Y = generate_dataset()
print(X)
print(Y)


# per = perceptron.learning(X, Y, type='primal')
per = perceptron.learning(X, Y, type='dual')
perceptron.plt_scatter_line(X, Y, per.w, per.b)