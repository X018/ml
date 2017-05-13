# -*- coding: utf-8 -*-
# test.py

"""
Created by jin.xia on May 13 2017

@author: jin.xia
"""


import numpy as np
import pandas as pd
import naivebayes as bayes


def generate_dataset():
	arr = [ [1, 'S', -1],
			[1, 'M', -1],
			[1, 'M', 1],
			[1, 'S', 1],
			[1, 'S', -1],
			[2, 'S', -1],
			[2, 'M', -1],
			[2, 'M', 1],
			[2, 'L', 1],
			[2, 'L', 1],
			[3, 'L', 1],
			[3, 'M', 1],
			[3, 'M', 1],
			[3, 'L', 1],
			[3, 'L', -1]]
	dataframe = pd.DataFrame(arr)
	dataset = dataframe.iloc[:,:].values
	X = dataset[:, :-1]
	Y = dataset[:, -1]
	return X, Y


X, Y = generate_dataset()
nb = bayes.naivebayes(0)
nb.learning(X, Y)
print(nb.predict([[2,'S']]))



