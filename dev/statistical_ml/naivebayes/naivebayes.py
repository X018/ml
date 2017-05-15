# -*- coding: utf-8 -*-
# naivebayes.py

"""
Created by jin.xia on May 09 2017

@author: jin.xia
"""


import numpy as np
import matplotlib.pyplot as plt


class naivebayes():
	"""
	X:	输入空间 []
	Xi(x):	多个输入变量的第i个
	XJ:	X的第J个特征空间[XJ1, XJ2, ..., XJN] 特殊可取值类型数量为N
	tis_XJ:	输出空间 {XJ1:[t1, t3, ...], ..., XJN:[t1, t3, ...])}
	xj:	x的第j个特征

	K:		Y的可能取值数量
	Y:		输出空间 [c1, c2, ..., ck]
	tis_Y:	输出空间 {c1:[t1, t3, ...], ..., ck:[t1, t3, ...])}

	training_X->
	training_Y->
	"""
	def __init__(self, λ=1):
		super(naivebayes, self).__init__()
		self.λ = λ
		# 输出空间 
		self.K = 0
		self.Y = []
		self.tis_Y = {}
		self.K = len(self.Y)
		self.X = []
		self.XY = {}

		self.training_X = []
		self.training_Y = []
		self.feature_num = 0
		self.training_num = 0


	def learning(self, training_X, training_Y):
		self.training_X = training_X
		self.training_Y = training_Y
		self.feature_num = self.training_X.shape[1]
		self.training_num = self.training_X.shape[0]
		# 输出空间
		self.Y, self.tis_Y = self.generate_Y()
		self.K = len(self.Y)
		print(self.Y)
		# 输入空间
		self.X = self.generate_X()
		# 
		self.XY = self.generate_XY()

		
	def predict(self, X):
		return [self.get_max_pxy_for_Y(x) for x in X]


	# P(Y=clabel)的training的idx索引字典(training_idx_dict)
	def generate_Y(self):
		tis_Y = {}
		for i in range(self.training_num):
			ck = self.training_Y[i]
			tis_Y.setdefault(ck, [])
			tis = tis_Y[ck]
			tis.append(i)
		Y = tis_Y.keys()
		return Y, tis_Y


	def generate_X(self):
		X = []
		for j in range(self.feature_num):
			XJ, tis_XJ = self.generate_XJ(j)
			X.append(tis_XJ)
		return X


	# P(X=xjm)的training的idx索引列表(training_idxs)
	def generate_XJ(self, j):
		tis_XJ = {}
		for i in range(self.training_num):
			feature = self.training_X[i][j]
			tis_XJ.setdefault(feature, [])
			tis = tis_XJ[feature]
			tis.append(i)
		XJ = tis_XJ.keys()
		return XJ, tis_XJ


	def generate_XY(self):
		YX = {}
		for y in self.Y:
			YX[y] = self.generate_Xy(y)
		return YX


	def generate_Xy(self, y):
		Xy = []
		for j in range(self.feature_num):
			Xy.append(self.generateXJy(j, y))
		return Xy


	def generateXJy(self, j, y):
		tis_XJy = {}
		tis_y = self.get_tis_y(y)
		tis_XJ = self.get_tisXJ(j)
		for (feature, tis_XJn) in tis_XJ.items():
			tis_XJy[feature] = [ti for ti in tis_y if ti in tis_XJn]
		return tis_XJy

	############################################################################################### 
	def get_tis_y(self, y):
		return self.tis_Y[y]


	def get_tisXJ(self, j):
		return self.X[j]

	# P(Xj=feature|Y=cabel) 指定特征值数量
	# 条件为类型取clabel时，第j维特征值取feature的training_X的i索引列表(training_idxs)
	def get_tis_XJy_feature(self, y, j, feature):
		Xy = self.XY[y]
		XJy = Xy[j]
		return XJy[feature]


	# P(Y=y)
	def get_py(self, y):
		numberator = self.training_num + self.λ * self.K
		fractions = len(self.get_tis_y(y)) + self.λ
		return fractions / numberator


	# P(X=x|Y=y) 独立同分布
	# 条件为类型取y时x的概率分布
	def get_px_y(self, y, x):
		px = 1
		num_y = len(self.get_tis_y(y))
		for j in range(self.feature_num):
			num_XJ = len(self.get_tisXJ(j))
			tis = self.get_tis_XJy_feature(y, j, x[j])
			numberator = num_y + self.λ * num_XJ
			fractions = len(tis) + self.λ
			px *= fractions / numberator
		return px


	def get_pxy(self, y, x):
		px_y = self.get_px_y(y, x)
		py = self.get_py(y)
		return px_y * py


	def get_pxy_dict_for_Y(self, x):
		return {y:self.get_pxy(y, x) for y in self.Y}


	def get_max_pxy_for_Y(self, x):
		pxy_dict = self.get_pxy_dict_for_Y(x)
		return max(pxy_dict.items(), key=lambda a: a[1])[0]



	############################################################################################### 
	def plt_pxy_for_Y(self, X):
		for x in X:
			pxy_dict = self.get_pxy_dict_for_Y(x)
			print(pxy_dict)
			pltx = np.array([y for (y, p) in pxy_dict.items()])
			plty = np.array([p for (y, p) in pxy_dict.items()])
			plt.scatter(pltx, plty)
		plt.show()


