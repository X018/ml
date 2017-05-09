# -*- coding: utf-8 -*-
# perceptron.py

"""
Created by jin.xia on May 09 2017

@author: jin.xia
"""


import os
import operator
import numpy as np
import matplotlib.pyplot as plt


# 
class perceptron:
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
		fx = self.calculate_fx(x)
		loss = fx * y
		return loss <= 0


	def calculate_fx(self, x):
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


	def plt_learning_plane(self, training_X, training_Y, line_plt=True):
		plt.figure()
		training_num = training_X.shape[0]
		for i in range(training_num):
			marker = (training_Y[i] == 1 and 'o') or 'x'
			plt.scatter(training_X[i][0], training_X[i][1], s=50, marker=marker)

		if line_plt:
			axis_matrix = training_X[:,0]
			x = np.linspace(np.min(axis_matrix), np.max(axis_matrix), 100)
			y = -(self.w[0] * x + self.b) / self.w[1]
			plt.plot(x, y)

		plt.show()




