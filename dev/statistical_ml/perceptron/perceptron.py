# -*- coding: utf-8 -*-
# perceptron_plt.py

"""
Created by jin.xia on May 09 2017

@author: jin.xia
"""


import numpy as np
import matplotlib.pyplot as plt

import perceptron_dual as per_dual
import perceptron_primal as per_primal


def learning(X, Y, learning_rate=1, type='dual'):
	per = per_dual.perceptron_dual(learning_rate)
	if type == 'primal':
		per = per_primal.perceptron_primal(learning_rate)
	per.learning(X, Y)
	return per


def plt_scatter_line(X, Y, w, b, X_new, predictions, alpha=0.2):
	fig = plt.figure()
	fig.set(alpha=alpha)
	plt_scatter(X, Y)
	plt_sep_line(X, w, b)
	plt_scatter(X_new, predictions, True)
	plt.show()


def plt_scatter(X, Y, is_new=False):
	# deal X, Y for plot
	num = X.shape[0]
	X0 = np.array([X[i] for i in range(num) if Y[i]==-1])
	X1 = np.array([X[i] for i in range(num) if Y[i]==1])
	size = (is_new and  60) or 25
	alpha = (is_new and 1) or 0.4
	if X0.shape[0] >= 1:
		plt.scatter(X0[:,0], X0[:,1], marker='x', s=size,
			color='blue', alpha=alpha, label='x0')
	if X1.shape[0] >= 1:
		plt.scatter(X1[:,0], X1[:,1], marker='o', s=size,
			color='red', alpha=alpha, label='x1')


def plt_sep_line(X, w, b):
	axis_matrix = X[:,0]
	x = np.linspace(np.min(axis_matrix), np.max(axis_matrix), 100)
	y = -(w[0] * x + b) / w[1]
	plt.plot(x, y)


