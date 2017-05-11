# -*- coding: utf-8 -*-
# perceptron.py

"""
Created by jin.xia on May 09 2017

@author: jin.xia
"""


import numpy as np
import naivebayes as bayes


def generate_dataset():
	arr = [[3,3,1], [4,3,1], [1,1,-1], [2,3,2]]
	dataset = np.array(arr)
	X = dataset[:,:-1]
	Y = dataset[:,-1]
	return X, Y


X, Y = generate_dataset()
nb = bayes.naivebayes()
nb.learning(X, Y)



