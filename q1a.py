import pandas as pd
import numpy as np

# TRAINING

datafile = pd.read_csv("~/Documents/smai/assignment-1/dummy/datasets/q1/train.csv")
train_data = datafile.iloc[0:,1:].values
# print(train_data[0])
train_class = datafile.iloc[0:,0].values
num_rows = train_data.shape[0]
train_data = np.c_[np.ones((num_rows,1)), train_data]
num_rows, num_cols = train_data.shape[:]
w = np.zeros((1, num_cols))
print(num_rows,num_cols)

def correct(row, weight_vec):
    prod = np.dot(weight_vec,train_data[row])
    if prod >= float(0):
        if train_class[row] == 1:
            err = 0
        else:
            err = 1
            weight_vec = np.subtract(weight_vec,train_data[row])
    elif prod < float(0):
        if train_class[row] == 0:
            err = 0
        else:
            err = 1
            weight_vec = np.add(weight_vec,train_data[row])
    return weight_vec, err

while(1):
    count = 0
    for i in range(num_rows):
        #print(i)
        w, err = correct(i, w)
        count += err

    print(count)
    if count == 0:
        print("Trained the weight vector")
        break

