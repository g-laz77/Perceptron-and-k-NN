import pandas as pd
import numpy as np
import sys
import math

train = sys.argv[1]
#test = sys.argv[2]

mapp = list()
for i in range(10):
	mapp.append(i)

# Calculate the entropy for a split dataset
def entropy_cost(groups, classes):
	n_instances = sum([len(group) for group in groups])
	cost = 0.0
	for group in groups:
		size = len(group)
		if size == 0:
			continue
		score = 0.0
		for class_val in classes:
			# for row in group:
			# 	print(row['left'])
			p = [row[6] for row in group].count(class_val)
			#print(p,class_val,group)
			if p != 0:
				score -= p/size * math.log(p/size,2)
		cost += score
	return cost

# Split a dataset based on an attribute and an attribute value
def train_split(index, value, dic, dataset, typ):
	left, right = list(), list()
	for row in dataset:
		if typ == "continuous":
			if row[index] < value:
				left.append(row)
			else:
				right.append(row)
		elif typ == "discrete":
			# print("yo ",dic[row[index]],value)
			if dic[row[index]] < value:
				left.append(row)
			else:
				right.append(row)
		else:
			if row[index] == value:
				left.append(row)
			else:
				right.append(row)
	return left, right

# Select the best split point for a dataset
def best_split(dataset):
	classes = list(set(row[-4] for row in dataset))
	best_index = 0
	best_value = 0
	least_entropy = 10000
	best_groups = None
	for index in range(len(dataset[0])):
		print("---------------------------",index,"---------------------------")
		if index == 6:
			continue
		print(dataset)
		data_val = dataset[0:,index]
		distinct = list(set(dt for dt in data_val))
		dict_dis = dict()
		counter = 0
		for i in distinct:
			dict_dis.update({i : counter})
			counter += 1
		if len(distinct) == 2:
			for i in range(len(distinct)):
				groups = train_split(index, distinct[i], distinct, dataset, "binary")
				entropy = entropy_cost(groups, classes)
				print(entropy,index,distinct[i])
				if entropy < least_entropy:
					best_index = index
					best_value = distinct[i]
					least_entropy = entropy
					best_groups = groups

		elif len(distinct) == 3 or len(distinct) == 10:   # Non-continuous data
			for i in range(len(distinct)):
				groups = train_split(index, dict_dis[distinct[i]], dict_dis, dataset, "discrete")
				entropy = entropy_cost(groups, classes)
				print(entropy,index,distinct[i],dict_dis[distinct[i]])				
				if entropy < least_entropy:
					best_index = index
					best_value = distinct[i]
					least_entropy = entropy
					best_groups = groups

		else:													#continuous data
			mx = np.amax(data_val)
			mn = np.amin(data_val)
			for val in range(100):
				groups = train_split(index, float((mx-mn)*val)/100,distinct, dataset,"continuous")
				entropy = entropy_cost(groups, classes)
				print(entropy,index,float((mx-mn)*val)/100)				
				if entropy < least_entropy:
					best_index = index
					best_value = float((mx-mn)*0.01*val)
					least_entropy = entropy
					best_groups = groups
		print("Least Entropy:",least_entropy,index,best_value)

	return {'index':best_index, 'value':best_value, 'groups':best_groups}

# Create a terminal node value
def node_terminal_eval(group):
	classes = [row[-4] for row in group]
	return max(set(classes), key=classes.count)

# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
	left, right = node['groups']
	del(node['groups'])
    #to prevent overfitting
	if depth >= max_depth:
		node['l'] = node_terminal_eval(left) 
		node['r'] = node_terminal_eval(right)
		print("depth")
		return
	#to prevent overfitting
	if not left or not right:
		node['l'] = node['r'] = node_terminal_eval(left + right)
		print("Nothing")
		return

	if len(left) <= min_size:
		node['l'] = node_terminal_eval(left)
		print("left_size")
	else:
		print("split_left")
		print(left)
		node['l'] = best_split(left)
		split(node['l'], max_depth, min_size, depth+1)

	if len(right) <= min_size:
		print("right_size")
		node['r'] = node_terminal_eval(right)
	else:
		print("split_right")
		node['r'] = best_split(right)
		split(node['r'], max_depth, min_size, depth+1)
 
# Build a decision tree
def build_tree(train, max_depth, min_size):
	root = best_split(train)
	split(root, max_depth, min_size, 1)
	return root

datafile = pd.read_csv(train)  #("~/Documents/smai/assignment-1/dummy/datasets/q3/train.csv")
train_data = datafile.iloc[1:,0:].values
train_class = datafile.iloc[1:,6]
num_rows, num_cols = train_data.shape[:]

k = build_tree(train_data,5,50)
# while(1):
print(k)
# 	k = k['r']
