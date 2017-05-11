# -*- coding: utf-8 -*-
# perceptron.py

"""
Created by jin.xia on May 09 2017

@author: jin.xia
"""


import numpy as np
import matplotlib.pyplot as plt

import perceptron_dual as per_dual
import perceptron_primal as per_primal


class perceptron():
	"""docstring for perceptron"""
	def __init__(self, type='dual', learning_rate=1):
		super(perceptron, self).__init__()
		self.learning_rate = learning_rate
		self.type = type
		self.w = []
		self.b = 0
		self.per = per_dual.perceptron_dual(self.learning_rate)
		if self.type == 'primal':
			self.per = per_primal.perceptron_primal(self.learning_rate)


	def learning(self, X, Y):
		self.w, self.b = self.per.learning(X, Y)
		return self.w, self.b


	def predict(self, X):
		num = X.shape[0]
		return [self.calculate_fx(x) for x in X]


	def calculate_fx(self, x):
		return self.sign(np.dot(self.w, x) + self.b)


	def sign(self, x):
		if x >=0:
			return 1
		return -1


	# ----------------------------------------- 二维数据画图 -----------------------------------------
	def plt_scatter_line(self, X, Y):
		minx, maxx = self.get_bounds(X[:,0])
		self.plt_line(minx, maxx)
		self.plt_scatter(X, Y)
		self.plt_show()


	def plt_scatter(self, X, Y, alpha=0.4, size=25):
		num = X.shape[0]
		X0 = np.array([X[i] for i in range(num) if Y[i]==-1])
		X1 = np.array([X[i] for i in range(num) if Y[i]==1])
		if X0.shape[0] >= 1:
			plt.scatter(X0[:,0], X0[:,1], marker='x', s=size, color='red', alpha=alpha, label='x0')
		if X1.shape[0] >= 1:
			plt.scatter(X1[:,0], X1[:,1], marker='o', s=size, color='blue', alpha=alpha, label='x1')


	def plt_line(self, minx, maxx):
		x = np.linspace(minx, maxx, 100)
		y = self.expression(self.w, self.b, x)
		plt.plot(x, y)


	def plt_show(self):
		plt.show()


	def expression(self, w, b, x):
		y = -(w[0] * x + b) / w[1]
		return y


	def get_bounds(self, matrix):
		maxx = np.max(matrix)
		minx =np.min(matrix)
		return minx, maxx



