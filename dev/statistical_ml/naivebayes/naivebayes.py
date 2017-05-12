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
	ti->training index(i)
	fj->feature index(j)
	cm->class lable index(m)
	fjset->featurej set
	features_set->feature set [feature_num]=featurej set
	fjsetn->fjset index(n)

	xi->X[i]
	xj->xi[j]
	xij->X[i][j]
	yk->Y[k]
	cln->clabels[n]
	"""
	def __init__(self, λ=1):
		super(naivebayes, self).__init__()
		self.λ = λ
		# self.training_X = np.array([])
		# self.training_Y = np.array([])
		# self.feature_dict = {}
		# self.label_dict = {}
		# self.lf_dict = {}


	def learning(self, training_X, training_Y):
		self.training_X = training_X
		self.training_Y = training_Y
		self.feature_num = self.training_X.shape[1]
		self.training_num = self.training_X.shape[0]

		self.clabel_tis_dict = self.generate_clabel_tis_dict()
		print(self.clabel_tis_dict)

		self.fallset_tis_arr = self.generate_fallset_tis_arr()
		print(self.fallset_tis_arr)

		self.call_fall_tis_dict = self.generate_call_fall_tis_dict()
		# for (clabel, idxs) in self.clabel_tis_dict.items():
		# 	self.lf_dict[clabel] = self.generate_cfjs_for_clabel(clabel)
		# print(self.lf_dict)


	# P(Y=clabel)的training的idx索引字典(training_idx_dict)
	def generate_clabel_tis_dict(self):
		clabel_tis_dict = {}
		for i in range(self.training_num):
			ck = self.training_Y[i]
			clabel_tis_dict.setdefault(ck, [])
			clabel_tis = clabel_tis_dict[ck]
			clabel_tis.append(i)
		return clabel_tis_dict


	# P(X=xjm)的training的idx索引列表(training_idxs)
	def generate_fjset_tis_dict(self, j):
		fjset_tis_dict = {}
		for i in range(self.training_num):
			feature = self.training_X[i][j]
			fjset_tis_dict.setdefault(feature, [])
			fjset_tis = fjset_tis_dict[feature]
			fjset_tis.append(i)
		return fjset_tis_dict


	def generate_fallset_tis_arr(self):
		fallset_tis_arr = []
		for j in range(self.feature_num):
			fallset_tis_arr.append(self.generate_fjset_tis_dict(j))
		return fallset_tis_arr


	def generate_ck_fj_tis_dict(self, ck, j):
		ck_fj_tis_dict = {}
		ck_tis = self.get_tis_for_ck(ck)
		fjset_tis_dict = self.get_fjset_tis_dict(j)
		for (feature, feature_tis) in fjset_tis_dict.items():
			ck_fj_tis_dict[feature] = [ti for ti in ck_tis if ti in feature_tis]
		return ck_fj_tis_dict


	def generate_ck_fall_tis_arr(self, ck):
		ck_fall_tis_arr = []
		for j in range(self.feature_num):
			ck_fall_tis_arr.append(self.generate_ck_fj_tis_dict(ck, j))
		return ck_fall_tis_arr


	def generate_call_fall_tis_dict(self):
		call_fall_tis_dict = {}
		for (ck, tis) in self.clabel_tis_dict.items():
			call_fall_tis_dict[ck] = self.generate_ck_fall_tis_arr(ck)
		return call_fall_tis_dict



	def get_tis_for_ck(self, ck):
		return self.clabel_tis_dict[ck]


	def get_fjset_tis_dict(self, j):
		return self.fallset_tis_arr[j]


	# P(Xj=feature|Y=cabel) 指定特征值数量
	# 条件为类型取clabel时，第j维特征值取feature的training_X的i索引列表(training_idxs)
	def get_ck_fj_tis(self, ck, feature, j):
		ck_fall_tis_arr = self.call_fall_tis_dict[ck]
		ck_fj_tis_dict = ck_fall_tis_arr[j]
		return ck_fj_tis_dict[feature]


	def get_p_ck(self, ck):
		ck_tis = self.get_tis_for_ck(ck)
		total_num = self.training_num
		ck_num = len(ck_tis)
		return ck_num / total_num


	# P(X=x|Y=clabel) 独立同分布
	# 条件为类型取clabel时x的概率分布
	def get_p_ck_x(self, ck, x):
		p = 1
		ck_num = len(self.get_tis_for_ck(ck))
		for j in range(len(x)):
			fjset_num = len(self.get_fjset_tis_dict(j))
			total_num = ck_num + self.λ * fjset_num
			tis = self.get_ck_fj_tis(ck, x[j], j)
			feature_num = len(tis) + self.λ
			p *= feature_num / total_num
		return p


	def get_p_call_x(self, x):
		p_dict = {}
		for (ck, tis) in self.clabel_tis_dict.items():
			p = self.get_p_ck_x(ck, x)
			p_dict[ck] = p
		return p_dict


	def get_ck_with_maxp(self, x):
		p_ck, p_max = 0, 0
		p_dict = self.get_p_call_x(x)
		for (ck, p) in p_dict.items():
			if p > p_max:
				p_max = p
				p_ck = ck
		return p_ck, p_max


	def predict(self, X):
		return [self.get_ck_with_maxp(x) for x in X]

