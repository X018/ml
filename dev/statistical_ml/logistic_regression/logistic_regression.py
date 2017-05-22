# -*- coding: utf-8 -*-
# logistic_regression.py

"""
Created by jin.xia on May 09 2017

@author: jin.xia
"""


import numpy as np
import matplotlib.pyplot as plt


class logistic_regression():
	"""docstring for logistic_regression"""
	def __init__(self, arg):
		super(logistic_regression, self).__init__()
		self.arg = arg
		

	def sigmod(self, x):
		return 1 / (1 + np.exp(-x))


	def learning(training_X, training_Y):
		self.training_X = training_X
		self.training_Y = training_Y
		self.training_num = self.training_X.shape[0]
		self.feature_num = self.training_X.shape[1]

		w = np.ones(feature_num)
		while True:
			x = np.dot(training_X, w)
			y = sigmod(x)
			diff = training_Y - y