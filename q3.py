import pandas as pd
import numpy as np

# Calculate the Gini index for a split dataset
def cost(groups, classes):
	n_instances = sum([len(group) for group in groups])
	cost = 0.0
	for group in groups:
		size = len(group)
		if size == 0:
			continue
		score = 0.0
		for class_val in classes:
			p = [row['left'] for row in group].count(class_val) / size
			score += p * p
		cost += (1.0 - score) * (size / n_instances)
	return cost

# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right

# Select the best split point for a dataset
def best_split(dataset):
	classes = list(set(row[-4] for row in dataset))
	best_index = 10000
    best_value = 10000
    best_score = 10000
    best_groups = None
	for index in range(len(dataset[0])):
        if index == 6:
            continue
		for row in dataset:
			groups = test_split(index, row[index], dataset)
			cost = cost(groups, classes)
			if cost < best_score:
				best_index = index
                best_value = row[index]
                best_score = cost
                best_groups = groups

	return {'index':b_index, 'value':b_value, 'groups':b_groups}

# Create a terminal node value
def node_terminal_eval(group):
	classes = [row[-4] for row in group]
	return max(set(classses), key=classes.count)

# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
	left, right = node['groups']
	# del(node['groups'])
    #to prevent overfitting
    if depth >= max_depth:
		node['l'] = node_terminal_eval(left) 
        node['r'] = node_terminal_eval(right)
		return
    #to prevent overfitting
	if not left or not right:
		node['l'] = node['r'] = node_terminal_eval(left + right)
		return
	
	if len(left) <= min_size:
		node['l'] = node_terminal_eval(left)
	else:
		node['l'] = best_split(left)
		split(node['l'], max_depth, min_size, depth+1)

	if len(right) <= min_size:
		node['r'] = node_terminal_eval(right)
	else:
		node['r'] = best_split(right)
		split(node['r'], max_depth, min_size, depth+1)
 
# Build a decision tree
def build_tree(train, max_depth, min_size):
	root = best_split(train)
	split(root, max_depth, min_size, 1)
	return root

datafile = pd.read_csv("~/Documents/smai/assignment-1/dummy/datasets/q3/decision_tree_train.csv")
train_data = datafile.iloc[1:,0:].values
num_rows, num_cols = train_data.shape[:]

build_tree(train_data,5,10)