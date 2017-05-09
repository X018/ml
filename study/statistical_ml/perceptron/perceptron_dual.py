import os
import operator
import numpy as np


class perceptron_dual():
	"""docstring for perceptron_dual"""
	def __init__(self, learning_reate=1):
		self.learning_reate = learning_reate
		self.a = []
		self.b = 0


	def classify(self, training_X, training_Y):
		shape = training_X.shape
		training_num = shape[0]
		# feature_num = shape[1]
		self.a = [0] * training_num
		self.b = 0

		gram_matrix = self.calculate_gram_matrix(training_X)
		print(gram_matrix)

		is_error = True
		while is_error:
			for i in range(training_num):
				if self.is_classify_error(gram_matrix, training_Y, i):
					self.calculate_w_b(training_X, training_Y)
					self.update_w_b(i, training_Y[i])
					break
				elif i == training_num - 1:
					self.calculate_w_b(training_X, training_Y)
					is_error = False
				

	def calculate_gram_matrix(self, matrix):
		row_num = col_num = matrix.shape[0]
		gram_matrix = np.zeros((row_num, col_num))
		for row in range(row_num):
			for col in range(col_num):
				gram_matrix[row][col] = np.dot(matrix[row], matrix[col])
		return gram_matrix


	def is_classify_error(self, gram_matrix, Y, idx):
		loss = self.calculate_loss(gram_matrix, Y, idx)
		return loss <= 0


	def calculate_loss(self, gram_matrix, Y, idx):
		loss = 0
		num = len(Y)
		gram_x = gram_matrix[idx]
		for i in range(num):
			loss += self.a[i] * Y[i] * gram_x[i]
		loss = (loss + self.b) * Y[idx]
		return loss


	def calculate_w_b(self, X, Y):
		feature_num = X.shape[1]
		label_num = len(Y)
		w = [0] * feature_num
		h = 0
		for i in range(label_num):
			h += self.a[i] * Y[i]
			w += self.a[i] * Y[i] * X[i]
		print(w, h)


	def update_w_b(self, i, y):
		self.a[i] += 1
		self.b += y
