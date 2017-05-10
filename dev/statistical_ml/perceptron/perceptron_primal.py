# -*- coding: utf-8 -*-
# perceptron.py

"""
Created by jin.xia on May 09 2017

@author: jin.xia
"""


import os
import operator
import numpy as np


# 
class perceptron_primal:
	def __init__(self, learning_rate=1):
		self.learning_rate = learning_rate
		self.w = []
		self.b = 0


	def learning(self, training_X, training_Y):
		shape = training_X.shape
		training_num = shape[0]
		feature_num = shape[1]
		self.w = [0] * feature_num
		self.b = 0
		
		is_error = True
		while is_error:
			print('[percetron] try w b : ', self.w, self.b)
			for i in range(training_num):
				x, y = training_X[i], training_Y[i]
				if self.is_classify_error(x, y):
					self.update_w_b(x, y)
					break
				elif i == training_num - 1:
					is_error = False


	def is_classify_error(self, x, y):
		fx = self.calculate_training_fx(x)
		loss = fx * y
		return loss <= 0


	def calculate_training_fx(self, x):
		fx = 0
		feature_num = len(x)
		for i in range(feature_num):
			fx += self.w[i] * x[i]
		fx += self.b
		return fx


	def update_w_b(self, x, y):
		feature_num = len(x)
		for i in range(feature_num):
			self.w[i] += self.learning_rate * x[i] * y
		self.b += self.learning_rate * y


	def predict(self, X):
		num = X.shape[0]
		return [self.calculate_fx(x) for x in X]


	def calculate_fx(self, x):
		return self.sign(np.dot(self.w, x) + self.b)


	def sign(self, x):
		if x >=0:
			return 1
		return -1


