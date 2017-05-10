# -*- coding: utf-8 -*-
# perceptron_plt.py

"""
Created by jin.xia on May 09 2017

@author: jin.xia
"""


import numpy as np
import matplotlib.pyplot as plt


def figure(alpha=0.2):
	fig = plt.figure()
	fig.set(alpha=alpha)
	

def plt_scatter(X, Y, line_plt=True):
	# deal X, Y for plot
	num = X.shape[0]
	X0 = np.array([X[i] for i in range(num) if Y[i]==-1])
	X1 = np.array([X[i] for i in range(num) if Y[i]==1])
	plt.scatter(X0[:,0], X0[:,1], marker='x', s=25,
		color='blue', alpha=0.4, label='x0')
	plt.scatter(X1[:,0], X1[:,1], marker='o', s=25,
		color='red', alpha=0.4, label='x1')


def plt_sep_line(X, w, b):
	axis_matrix = X[:,0]
	x = np.linspace(np.min(axis_matrix), np.max(axis_matrix), 100)
	y = -(w[0] * x + b) / w[1]
	plt.plot(x, y)

def show():
	plt.show()