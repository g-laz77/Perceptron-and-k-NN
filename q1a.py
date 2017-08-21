import pandas as import pd
import numpy as np

# TRAINING

datafile = pd.read_csv("~/Documents/smai/assignment-1/dummy/datasets/q1/train.csv")
train_data = datafile.iloc[0:,1:]
test_class = datafile.iloc[0:,0]
num_rows = train_data.shape[0]
num_cols = train_data.shape[1]
w = np.zeros(1,num_cols)

def correct(row, weight_vec):
    prod = np.dot(train_data[row],weight_vec)
    if prod > (float)0:
        if test_class[row] == 1:
            err = 0
        else:
            err = 1
            weight_vec = np.add(weight_vec,train_data[row])
    else:
        if test_class[row] == 0:
            err = 0
        else:
            err = 1
            weight_vec = np.subtract(weight_vec,train_data[row])
    return weight_vec, err

while(1):
    count = 0
    for i in range(num_rows):
        w, err = correct(i, w)
        count += err
        
