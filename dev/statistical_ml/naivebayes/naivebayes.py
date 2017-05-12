# -*- coding: utf-8 -*-
# perceptron.py

"""
Created by jin.xia on May 09 2017

@author: jin.xia
"""


import numpy as np
import matplotlib.pyplot as plt


class naivebayes():
	"""
	xi->X[i]
	xj->xi[j]
	xij->X[i][j]
	yk->Y[k]
	cln->clabels[n]
	"""
	def __init__(self):
		super(naivebayes, self).__init__()
		self.training_X = np.array([])
		self.training_Y = np.array([])
		self.feature_dict = {}
		self.label_dict = {}
		self.lf_dict = {}


	def learning(self, training_X, training_Y):
		self.training_X = training_X
		self.training_Y = training_Y
		self.feature_num = self.training_X.shape[1]
		self.training_num = self.training_X.shape[0]

		self.clabel_dict = self.geneate_clabel_dict()
		print(self.clabel_dict)

		self.feature_arr = self.generate_feature_arr()
		print(self.feature_arr)

		self.lf_dict = {}
		for (clabel, idxs) in self.clabel_dict.items():
			self.lf_dict[clabel] = self.generate_cfjs_for_clabel(clabel)
		print(self.lf_dict)


	def geneate_clabel_dict(self):
		clabel_dict = {}
		for i in range(self.training_num):
			clabel = self.training_Y[i]
			clabel_dict.setdefault(clabel, [])
			clabel_dict[clabel].append(i)
		return clabel_dict


	def generate_feature_arr(self):
		feature_arr = []
		for j in range(self.feature_num):
			fx_dict = {}
			feature_arr.append(fx_dict)
			for i in range(self.training_num):
				feature = self.training_X[i][j]
				fx_dict.setdefault(feature, [])
				fx_dict[feature].append(i)
		return feature_arr


	def generate_cfjs_for_clabel(self, clabel):
		cfjs = []
		ctis = self.get_tis_for_clabel(clabel)
		for j in range(self.feature_num):
			ftis = {}
			cfjs.append(ftis)
			fj_dict = self.feature_arr[j]
			for (feature, fftis) in fj_dict.items():
				ftis[feature] = [ti for ti in fftis if ti in ctis]
		return cfjs


	def get_p_clabel(self, clabel):
		clabel_num = len(self.get_tis_for_clabel(clabel))
		return clabel_num / self.training_num


	# P(X=x|Y=clabel) 独立同分布
	# 条件为类型取clabel时x的概率分布
	def get_px_for_clabel(self, x, clabel):
		px = 1
		clabel_num = len(self.get_tis_for_clabel(clabel))
		for j in range(len(x)):
			tis = self.get_tis_with_cfj(clabel, x[j], j)
			px *= (len(tis) + 1) / (clabel_num + 5)
			# px *= len(tis) / clabel_num
		return px


	# P(Xj=feature|Y=cabel) 指定特征值数量
	# 条件为类型取clabel时，第j维特征值取feature的training_X的i索引列表(training_idxs)
	def get_tis_with_cfj(self, clabel, feature, j):
		featrue_arr = self.lf_dict[clabel]
		feature_n = featrue_arr[j]
		return feature_n[feature]


	def get_tis_for_clabel(self, clabel):
		return self.clabel_dict[clabel]


	# def get_fn_for_j():
	# 	pass






	def generate_label_dict(self, training_Y):
		label_num_dict = {}
		training_num = training_Y.shape[0]
		for i in range(training_num):
			label = training_Y[i]
			label_num_dict.setdefault(label, 0)
			label_num_dict[label] += 1
		return {label: (num, num / training_num) for (label, num) in label_num_dict.items()}


	# def generate_feature_dict(self, training_X):
	# 	feature_dict = {}
	# 	shape = training_X.shape
	# 	training_num = shape[0]
	# 	feature_num = shape[1]
	# 	for i in range(feature_num):
	# 		feature_dict.setdefault(i, {})
	# 		for j in range(training_num):
	# 			xij = training_X[j][i]
	# 			feature_dict[i].setdefault(xij, 0)
	# 			feature_dict[i][xij] += 1
	# 	return feature_dict
		


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