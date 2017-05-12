# -*- coding: utf-8 -*-
# test.py

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


per = perceptron.perceptron('primal')
# per = perceptron.perceptron('dual')
per.learning(X, Y)
per.plt_scatter_line(X, Y)