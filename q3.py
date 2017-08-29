import pandas as pd
import numpy as np
import sys
import math

train = sys.argv[1]
test = sys.argv[2]

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
			p = [row[6] for row in group].count(class_val)
			if p != 0:

				score -= (float(p)/size * math.log(float(p)/size)) / math.log(2)
		cost += score
	# print "Cost:",cost 
	return cost

# Split a dataset based on an attribute and an attribute value
def train_split(index, value, dic, dataset, typ):
	left, right = [], []
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
	classes = list(set(dataset[i][6] for i in range(len(dataset))))
	best_index = -1
	best_value = 0
	least_entropy = 10000
	best_groups = None
	for index in range(len(dataset[0])):
		#print "---------------------------",index,"---------------------------"
		if index == 6:
			continue
		
		data_val = list(dataset[i][index] for i in range(len(dataset)))
		#print(data_val)
		distinct = list(set(dt for dt in data_val))
		dict_dis = dict()
		counter = 0
		for i in distinct:
			dict_dis.update({i : counter})
			counter += 1
		if isinstance(distinct[0], int) and distinct[0] == 0 or distinct[0] == 1:
			for i in range(len(distinct)):
				groups = train_split(index, distinct[i], distinct, dataset, "binary")
				entropy = entropy_cost(groups, classes)
				#print(entropy,index,distinct[i])
				if entropy <= least_entropy:
					best_index = index
					best_value = distinct[i]
					least_entropy = entropy
					best_groups = groups

		elif isinstance(distinct[0], (str, unicode)):   # Non-continuous data
			for i in range(len(distinct)):
				groups = train_split(index, dict_dis[distinct[i]], dict_dis, dataset, "discrete")
				entropy = entropy_cost(groups, classes)
				#print(entropy,index,distinct[i],dict_dis[distinct[i]])				
				if entropy <= least_entropy:
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
				#print(entropy,index,float((mx-mn)*val)/100)				
				if entropy <= least_entropy:
					best_index = index
					best_value = float((mx-mn)*0.01*val)
					least_entropy = entropy
					best_groups = groups
		#print "Least Entropy:",least_entropy,best_index,best_value

	return {'index':best_index, 'value':best_value, 'groups':best_groups}

# Create a terminal node value
def node_terminal_eval(group):
	classes = [row[-4] for row in group]
	return max(set(classes), key=classes.count)

# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
	left, right = node['groups']
	# print("\n----------------------Right----------------------")
	# for ite in right:
	# 	print(ite[6])
	del(node['groups'])
    #to prevent overfitting
	if depth >= max_depth:
		node['l'] = node_terminal_eval(left) 
		node['r'] = node_terminal_eval(right)
		#print "depth"
		return
	#to prevent overfitting
	if not left or not right:
		node['l'] = node['r'] = node_terminal_eval(left + right)
		#print "Nothing"
		return

	if len(left) <= min_size:
		node['l'] = node_terminal_eval(left)
		#print "left_size" 
	else:
		#print "split_left"
		# print(left)
		node['l'] = best_split(left)
		split(node['l'], max_depth, min_size, depth+1)

	if len(right) <= min_size:
		#print "right_size"
		node['r'] = node_terminal_eval(right)
	else:
		#print "split_right"
		node['r'] = best_split(right)
		#print node['r']
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
data_val = list(train_data[i][8] for i in range(len(train_data)))
distinct = list(set(dt for dt in data_val))
sales = dict()
counter = 0
for i in distinct:
	sales.update({i : counter})
	counter += 1
data_val = list(train_data[i][9] for i in range(len(train_data)))
distinct = list(set(dt for dt in data_val))
salary = dict()
counter = 0
for i in distinct:
	salary.update({i : counter})
	counter += 1

decision_tree = build_tree(train_data,5,50)
# while(1):
#print root
def print_tree(root):
	if isinstance(root,int):
		print(root)
	else:
		print root['index'],root['value']
		print_tree(root['l'])
		print_tree(root['r'])
	
# 	k = k['r']
print_tree(decision_tree)

def check(data_vec,node):
	if isinstance(node,int):
		#print node
		return int(node)
	
	if isinstance(node['value'],str):
		if node['value'] == 'high' or node['value'] == 'medium' or node['value'] == 'low':
			if salary[node['value']] >= salary[data_vec[node['index']]]:
				# print node['value'], data_vec[node['index']]
				return check(data_vec,node['l'])
			elif salary[node['value']] < salary[data_vec[node['index']]]:
				# print node['value'], data_vec[node['index']]		
				return check(data_vec,node['r'])
		else:
			if sales[node['value']] >= sales[data_vec[node['index']]]:
				# print node['value'], data_vec[node['index']]
				return check(data_vec,node['l'])
			elif sales[node['value']] < sales[data_vec[node['index']]]:
				# print node['value'], data_vec[node['index']]		
				return check(data_vec,node['r'])
	
	elif isinstance(node['value'],int) and node['value'] == 0 or node['value'] == 1:
		if node['value'] == 0:
			# print node['value'], data_vec[node['index']]
			return check(data_vec,node['l'])
		else:
			# print node['value'], data_vec[node['index']]
			return check(data_vec,node['r'])
	else:
		if node['value'] >= data_vec[node['index']]:
			# print node['value'], data_vec[node['index']]
			return check(data_vec,node['l'])
		elif node['value'] < data_vec[node['index']]:
			# print node['value'], data_vec[node['index']]		
			return check(data_vec,node['r'])

datafile = pd.read_csv(test)  #("~/Documents/smai/assignment-1/dummy/datasets/q3/train.csv")
test_data = datafile.iloc[0:,0:].values
# test_class = datafile.iloc[1:,6]
num_rows, num_cols = test_data.shape[:]
# print num_rows, num_cols

classified = [-1 for i in range(num_rows)]
# print classified

def test_tree(data, num_rows):
	for i in range(num_rows):
		classified[i] = check(data[i],decision_tree)

test_tree(test_data,num_rows)
for i in range(len(classified)):
	print classified[i]