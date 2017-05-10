# -*- coding: utf-8 -*-
# perceptron_dual.py

"""
Created by jin.xia on May 09 2017

@author: jin.xia
"""


import os
import operator
import numpy as np


class perceptron_dual():
	"""docstring for perceptron_dual"""
	def __init__(self, learning_rate=1):
		self.learning_rate = learning_rate
		self.w = []
		self.α = []
		self.b = 0


	def learning(self, training_X, training_Y):
		gram_matrix = self.calculate_gram_mat(training_X)
		training_num = training_X.shape[0]
		self.α = np.zeros(training_num)
		self.b = 0

		is_error = True
		while is_error:
			for idx in range(training_num):
				if self.is_classify_error(training_X, training_Y, gram_matrix, idx):
					self.calculate_w(training_X, training_Y)
					self.update_α_b(idx, training_Y[idx])
					break
				elif idx == training_num - 1:
					self.calculate_w(training_X, training_Y)
					is_error = False


	def calculate_gram_mat(self, training_X):
		# return np.matmul(training_X, training_X.T)
		row_num = col_num = training_X.shape[0]
		gram_matrix = np.zeros((row_num, col_num))
		for row in range(row_num):
			for col in range(col_num):
				gram_matrix[row][col] = np.dot(training_X[row], training_X[col])
		return gram_matrix


	def is_classify_error(self, training_X, training_Y, gram_matrix, idx):
		fx = self.calculate_fx(training_X, training_Y, gram_matrix, idx)
		# yi(∑j=1Nαjyjxj⋅xi+b)≤0
		loss = fx * training_Y[idx]
		return loss <= 0


	def calculate_fx(self, training_X, training_Y, gram_matrix, idx):
		# ∑j=1Nαjyjxj⋅xi+b
		fx = 0
		num = training_Y.shape[0]
		for i in range(num):
			# fx += self.α[i] * training_Y[i] * np.dot(training_X[i], training_X[idx])
			fx += self.α[i] * training_Y[i] * gram_matrix[i][idx]
		fx += self.b
		return fx


	def calculate_w(self, training_X, training_Y):
		shape = training_X.shape
		training_num = shape[0]
		feature_num = shape[1]
		self.w = np.zeros(feature_num)
		for i in range(training_num):
			self.w += self.α[i] * training_Y[i] * training_X[i]
		print('[percetron_dual] try w b : ', self.w, self.b)


	def update_α_b(self, idx, y):
		self.α[idx] += self.learning_rate
		self.b += self.learning_rate * y


	# def plt_learning_plane(self, training_X, training_Y, line_plt=True):
	# 	plt.figure()
	# 	training_num = training_X.shape[0]
	# 	for i in range(training_num):
	# 		marker = (training_Y[i] == 1 and 'o') or 'x'
	# 		plt.scatter(training_X[i][0], training_X[i][1], s=50, marker=marker)

	# 	if line_plt:
	# 		axis_matrix = training_X[:,0]
	# 		x = np.linspace(np.min(axis_matrix), np.max(axis_matrix), 100)
	# 		y = -(self.w[0] * x + self.b) / self.w[1]
	# 		plt.plot(x, y)

	# 	plt.show()

