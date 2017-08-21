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
    err = 0
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

#Testing

def testFile(test_data, rows):
    count = 0
    for i in range(rows):
        prod = np.dot(w,test_data[i])
        if prod >= float(0):
            clas = 1
            if test_class[i] == clas:
                count += 1     
        else:
            clas = 0
            if test_class[i] == clas:
                count += 1
    return count

datafile = pd.read_csv("~/Documents/smai/assignment-1/dummy/datasets/q1/test.csv")
test_data = datafile.iloc[0:,1:].values
num_rows, num_cols = test_data.shape[:]
print(num_rows,num_cols)
# print(train_data[0])
test_class = datafile.iloc[0:,0].values
num_rows = test_data.shape[0]
test_data = np.c_[np.ones((num_rows,1)), test_data]
num_rows, num_cols = test_data.shape[:]
print(num_rows,num_cols)

count = testFile(test_data, num_rows)
print(count/float(num_rows))
