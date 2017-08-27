import pandas as pd
import numpy as np
import sys

# TRAINING
train = sys.argv[1]
test = sys.argv[2]
datafile = pd.read_csv(train)
train_data = datafile.iloc[0:,1:10].values
# print(train_data[0])
train_class = datafile.iloc[0:,10].values

num_rows, num_cols = train_data.shape[:]
for i in range(num_rows):
    train_data[i][5] = int(train_data[i][5])
train_data = np.c_[np.ones((num_rows,1)), train_data]
num_rows, num_cols = train_data.shape[:]
a = np.zeros((1, num_cols))
print(num_rows,num_cols)
b = 50

for i in range(len(train_class)):       # Multiply by -1
    if train_class[i] == 2:
        for j in range(len(train_data[i])):
            train_data[i][j] *= -1

def relax(row, weight_vec):
    # print("ssup")
    prod = np.dot(weight_vec,train_data[row])
    err = 0
    if prod <= float(b):
        modulus = np.dot(train_data[row],train_data[row])
        const = float(b - prod) / modulus
        temp = np.multiply(const,train_data[row])
        weight_vec = np.add(weight_vec,temp)
        err = 1
    return weight_vec, err

for i in range(1000):        #number of epochs [Trial and error]
    #print("ssup")
    count = 0
    for i in range(num_rows):
        #print(i)
        a, err = relax(i, a)
        count += err

    print(count)
    if count == 0:
        print("Trained the weight vector")
        break

#Testing

def testFile(test_data, rows):
    count = 0
    for i in range(rows):
        prod = np.dot(a, test_data[i])
        if prod >= float(b):
            clas = 4
            if test_class[i] == clas:
                count += 1     
        else:
            clas = 2
            if test_class[i] == clas:
                count += 1
    return count

datafile = pd.read_csv(test)
test_data = datafile.iloc[0:,1:10].values
num_rows, num_cols = test_data.shape[:]
print(num_rows,num_cols)
num_rows, num_cols = test_data.shape[:]
for i in range(num_rows):
    test_data[i][5] = int(test_data[i][5])
test_data = np.c_[np.ones((num_rows,1)), test_data]
num_rows, num_cols = test_data.shape[:]
# print(train_data[0])
test_class = datafile.iloc[0:,10].values
num_rows = test_data.shape[0]
#test_data = np.c_[np.ones((num_rows,1)), test_data]
num_rows, num_cols = test_data.shape[:]
print(num_rows,num_cols)

count = testFile(test_data, num_rows)
print(count/float(num_rows))
