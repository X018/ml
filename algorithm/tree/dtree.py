# -*- coding:utf-8 -*-
# Author:jin.xia
# Date:2017.06.06


import time
import math

ID3 = 1
C45 = 2


# 创建决策树
def create(dataset, feat_keys=None, feat_choose_func=None):
	# 标记列表
	labels = [inst[-1] for inst in dataset]
	# 所有类别相同，返回单结点
	label0 = labels[0]
	if labels.count(label0) == len(labels):
		return label0
	# 属性种类数量为空时，返回实例类数量最大的类
	feat_num = len(dataset[0]) - 1
	if feat_num == 0:
		return majority_count(class_list)
	# 补全参数
	if feat_keys is None:
		feat_keys = [idx for idx in range(feat_num)]
	if feat_choose_func is None:
		feat_choose_func = choose_best_feat_by_c45
	# 特征选择
	best_feat_idx = feat_choose_func(dataset)
	best_feat_key = feat_keys[best_feat_idx]
	# del(feat_keys[best_feat_idx])
	tree = {best_feat_key:{}}
	# 子结点
	feat_vals = [inst[best_feat_idx] for inst in dataset]
	feat_set = set(feat_vals)
	for feat_val in feat_set:
		sub_labels = feat_keys[:best_feat_idx]
		sub_labels.extend(feat_keys[best_feat_idx+1:])
		sub_dataset = split_dataset(dataset, best_feat_idx, feat_val)
		tree[best_feat_key][feat_val] = create(sub_dataset, sub_labels, feat_choose_func)
	return tree


#使用决策树执行分类 
def classify(tree, test_x, feat_keys=None): 
	tree_keys = list(tree.keys())
	node_key = tree_keys[0]
	node_dict = tree[node_key]
	#index方法查找当前列表中第一个匹配node_key变量的元素的索引
	feat_idx = feat_keys.index(node_key)
	for key in node_dict.keys():
		if test_x[feat_idx] == key:
			if type(node_dict[key]).__name__ == 'dict':
				label = classify(node_dict[key], test_x, feat_keys)
			else:
				label = node_dict[key]
	return label


####################################################################################################
# 指出实例类数量最大的类
def majority_count(labes):
	label_counts = {}
	for label in labes:
		label_counts.setdefault(label, 0)
		label_counts[label] += 1
	return max(label_counts)


# ID3算法选择最佳属性(信息增益g(D,A))
def choose_best_feat_by_id3(dataset):
	best_feat_idx = -1
	best_info_gain = 0
	# 最后一列是分类
	feat_num = len(dataset[0]) - 1
	entropy = calc_entropy(dataset)
	for idx in range(feat_num):
		info_gain = calc_info_gain(dataset, idx, entropy)
		if info_gain > best_info_gain:
			best_info_gain = info_gain
			best_feat_idx = idx
	return best_feat_idx



# C4.5算法选择最佳属性(信息增益比gr(D,A))
def choose_best_feat_by_c45(dataset):
	best_feat_idx = -1
	best_info_gain_ratio = 0
	# 最后一列是分类
	feat_num = len(dataset[0]) - 1
	entropy = calc_entropy(dataset)
	for idx in range(feat_num):
		info_gain_ratio = calc_info_gain_ratio(dataset, idx, entropy)
		if info_gain_ratio > best_info_gain_ratio:
			best_info_gain_ratio = info_gain_ratio
			best_feat_idx = idx
	return best_feat_idx


# 计算数据集的信息熵H(p)
def calc_entropy(dataset, unit=2):
	entropy = 0
	label_counts = {}
	inst_num = len(dataset)
	for inst in dataset:
		label = inst[-1]
		label_counts.setdefault(label, 0)
		label_counts[label] += 1

	for (label, count) in label_counts.items():
		probility = count / inst_num
		entropy -= probility * math.log(probility, unit)

	return entropy


# 计算i属性条件(信息)熵H(D,A)
def calc_cond_entropy(dataset, i, feat_set):
	cond_entropy = 0
	inst_num = len(dataset)
	for feat_val in feat_set:
		sub_dataset = split_dataset(dataset, i, feat_val)
		probility = len(sub_dataset) / inst_num
		cond_entropy += probility * calc_entropy(sub_dataset)
	return cond_entropy


# 计算信息增益g(D, A)
def calc_info_gain(dataset, idx, entropy=None):
	if entropy is None:
		entropy = calc_entropy(dataset)
	# 第i维特征列表
	feat_vals = [inst[idx] for inst in dataset]
	feat_set = set(feat_vals)
	cond_entropy = calc_cond_entropy(dataset, idx, feat_set)
	#信息增益，就是熵的减少，也就是不确定性的减少
	info_gain = entropy - cond_entropy
	return info_gain


# 计算信息增益比gr(D, A)
def calc_info_gain_ratio(dataset, idx, entropy=None):
	if entropy is None:
		entropy = calc_entropy(dataset)
	info_gain = calc_info_gain(dataset, idx, entropy)
	info_gain_ratio = info_gain / entropy
	return info_gain_ratio


# 分割数据集
def split_dataset(dataset, idx, feat_val):
	result = []
	for inst in dataset:
		if inst[idx] == feat_val:
			inst_new = inst[:idx]
			inst_new.extend(inst[idx+1:])
			result.append(inst_new)
	return result



####################################################################################################
# 
# 测试用例
# 
####################################################################################################
def test(feat_choose_func=None):
	dataset, feat_keys = create_test_dataset()
	tree = create(dataset, feat_keys, feat_choose_func)
	# TreePlotter.createPlot(tree)
	print(tree)
	print(classify(tree, ['senior','high','no','excellent'], feat_keys))


# 创建测试数据集
def create_test_dataset():
	dataset = [ ['youth','high','no','fair','no'],
				['youth','high','no','excellent','no'],
				['middle_aged','high','no','fair','yes'],
				['senior','medium','no','fair','yes'],
				['senior','low','yes','fair','yes'],
				['senior','low','yes','excellent','no'],
				['middle_aged','low','yes','excellent','yes'],
				['youth','medium','no','fair','no'],
				['youth','low','yes','fair','yes'],
				['senior','medium','yes','fair','yes'],
				['youth','medium','yes','excellent','yes'],
				['middle_aged','medium','no','excellent','yes'],
				['middle_aged','high','yes','fair','yes'],
				['senior','medium','no','excellent','no']]
	feat_keys = ['age','income','student','credit_rating']
	return dataset, feat_keys