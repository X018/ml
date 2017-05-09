import os
import operator
import numpy as nu


# 
class perceptron:
	def __init__(self, learning_rate=1):
		self.learning_rate = learning_rate
		self.w = []
		self.b = 0


	def classify(self, training_X, training_Y):
		shape = training_X.shape
		training_num = shape[0]
		feature_num = shape[1]
		self.w = [0] * feature_num
		self.b = 0
		
		is_error = True
		while is_error:
			print(self.w, self.b)
			for i in range(training_num):
				x, y = training_X[i], training_Y[i]
				if self.is_classify_error(x, y):
					print(self.w, self.b)
					self.update_w_b(x, y)
					break
				elif i == training_num - 1:
					is_error = False


	def is_classify_error(self, x, y):
		loss = self.calculate_loss(x, y)
		return loss <= 0


	def calculate_loss(self, x, y):
		loss = 0
		feature_num = len(x)
		for i in range(feature_num):
			loss += self.w[i] * x[i]
		loss += self.b
		loss *= y
		return loss


	def update_w_b(self, x, y):
		feature_num = len(x)
		for i in range(feature_num):
			self.w[i] += self.learning_rate * x[i] * y
		self.b += self.learning_rate * y