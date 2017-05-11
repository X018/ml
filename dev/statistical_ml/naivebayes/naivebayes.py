# -*- coding: utf-8 -*-
# perceptron.py

"""
Created by jin.xia on May 09 2017

@author: jin.xia
"""


import numpy as np
import matplotlib.pyplot as plt


class naivebayes():
	"""docstring for naivebayes"""
	def __init__(self):
		super(naivebayes, self).__init__()
		self.feature_dict = {}
		self.label_dict = {}


	def learning(self, training_X, training_Y):
		self.feature_dict = self.generate_feature_dict(training_X)
		self.label_dict = self.generate_label_dict(training_Y)
		print(self.feature_dict)

		shape = training_X.shape
		training_num = shape[0]
		feature_num = shape[1]


		# list_Y = training_Y.tolist()
		# label_dict = {label: list_Y.count(label) for label in list_Y}
		# label_key_arr = list(label_dict.keys())
		# label_num = len(label_dict)
		# py = {label: num / training_num for (label, num) in label_dict.items()}


		# pNum = np.zeros([label_num, feature_num])
		# pDenom = np.ones(label_num)

		# for i in range(training_num):
		# 	for j in range(label_num):
		# 		key = label_key_arr[j]
		# 		if training_Y[i] == label_dict[key]:
		# 			pNum[j] += training_X[i]
		# 			pDenom[j] += np.sum(training_X[i])
		# 			print(training_X[i], np.sum(training_X[i]))

		# print(pNum)
		# print(pDenom)
		# print([pNum[k] / pDenom[k] for k in range(label_num)])
		# P(x|yi)


	def generate_label_dict(self, training_Y):
		label_num_dict = {}
		training_num = training_Y.shape[0]
		for i in range(training_num):
			label = training_Y[i]
			label_num_dict.setdefault(label, 0)
			label_num_dict[label] += 1
		return {label: (num, num / training_num) for (label, num) in label_num_dict.items()}


	def generate_feature_dict(self, training_X):
		feature_dict = {}
		shape = training_X.shape
		training_num = shape[0]
		feature_num = shape[1]
		for i in range(feature_num):
			feature_dict.setdefault(i, {})
			for j in range(training_num):
				xij = training_X[j][i]
				feature_dict[i].setdefault(xij, 0)
				feature_dict[i][xij] += 1
		return feature_dict
		


	# p(X=x.i|Y=y)(feature p)
	def get_features_for_label(self, training_X, training_Y, label):
		shape = training_X.shape
		fps = np.zeros(shape[1])
		for i in range(shape[0]):
			if training_Y[i] == y:
				fps[i] += training_X[i]
				# pNum[j] += training_X[i]
				# pDenom[j] += np.sum(training_X[i])
				# print(training_X[i], np.sum(training_X[i]))
