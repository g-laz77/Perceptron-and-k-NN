import pandas as pd
import numpy as np
import sys

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
a1 = np.zeros((1, num_cols))
a2 = np.zeros((1, num_cols))

def relax(row, weight_vec, n):
    # print("ssup")
    prod = np.dot(weight_vec,train_data[row])
    err = 0
    if prod <= float(b):
        modulus = np.dot(train_data[row],train_data[row])
        const = float(n * (b - prod)) / modulus
        temp = np.multiply(const,train_data[row])
        weight_vec = np.add(weight_vec,temp)
        err = 1
    return weight_vec, err

def modified(row, weight_vec, n):
    # print("ssup")
    prod = np.dot(weight_vec,train_data[row])
    err = 0
    if prod <= float(b):
        modulus = np.dot(train_data[row],train_data[row])
        const = float(n * (b - prod)) / modulus
        temp = np.multiply(const,train_data[row])
        weight_vec = np.add(weight_vec,temp)
        err = 1
    return weight_vec, err

def part1():
    n = 1
    global a1
    for i in range(5000):        #number of epochs [Trial and error]
        count = 0
        for i in range(num_rows):
            a1, err = relax(i, a1, 1)
            count += err

def part2():
    n = 1
    global a2

    for i in range(500):        #number of epochs [Trial and error]
        count = 0
        for i in range(num_rows):
            a2, err = modified(i, a2, n)
            count += err

        accuracy = float(err/num_rows)
        n = 1 - accuracy

# TRAINING

b = 50
# n = 1
for i in range(len(train_class)):       # Multiply by -1
    if train_class[i] == 2:
        for j in range(len(train_data[i])):
            train_data[i][j] *= -1


part1()
part2()

#Testing

def classify(test_data, rows, weight_vec):

    for i in range(rows):
        prod = np.dot(weight_vec, test_data[i])
        if prod >= float(0):
            clas = 4
            print clas  
        else:
            clas = 2
            print clas

datafile = pd.read_csv(test)
test_data = datafile.iloc[0:,1:10].values
num_rows, num_cols = test_data.shape[:]
#print num_rows,num_cols
num_rows, num_cols = test_data.shape[:]
for i in range(num_rows):
    test_data[i][5] = int(test_data[i][5])
test_data = np.c_[np.ones((num_rows,1)), test_data]
num_rows, num_cols = test_data.shape[:]
#test_class = datafile.iloc[0:,10].values
num_rows = test_data.shape[0]
classify(test_data, num_rows, a1)
classify(test_data, num_rows, a2)

# print count/float(num_rows)
