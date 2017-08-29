import pandas as pd
import numpy as np
import sys

# TRAINING
train = sys.argv[1]
test = sys.argv[2]
datafile = pd.read_csv(train)
train_data = datafile.iloc[0:,1:].values
# print(train_data[0])
train_class = datafile.iloc[0:,0].values
num_rows = train_data.shape[0]
train_data = np.c_[np.ones((num_rows,1)), train_data]
num_rows, num_cols = train_data.shape[:]
w1 = np.zeros((1, num_cols))
w2 = np.zeros((1, num_cols))
w3 = np.zeros((1, num_cols))
w4 = np.zeros((1, num_cols))

def correct2(weight_vec,margin):
    prod = np.dot(train_data,np.transpose(weight_vec))
    err = 0
    w_temp = np.zeros((1, num_cols))

    for i in range(len(prod)):
        if prod[i] >= float(margin):
            if train_class[i] == 1: 
                err += 0
            else:
                err += 1
                w_temp = np.subtract(w_temp,train_data[i])
        elif prod[i] < float(margin):
            if train_class[i] == 0:
                err += 0
            else:
                err += 1
                w_temp = np.add(w_temp,train_data[i])
    weight_vec = np.add(weight_vec,w_temp)
    return weight_vec, err

def correct1(row, weight_vec,margin):
    prod = np.dot(weight_vec,train_data[row])
    err = 0
    if prod >= float(margin):
        if train_class[row] == 1:
            err = 0
        else:
            err = 1
            weight_vec = np.subtract(weight_vec,train_data[row])
    elif prod < float(margin):
        if train_class[row] == 0:
            err = 0
        else:
            err = 1
            weight_vec = np.add(weight_vec,train_data[row])
    return weight_vec, err

def part1():
    global w1
    while(1):
        count = 0
        for i in range(num_rows):

            w1, err = correct1(i, w1, 0)
            count += err
            
        if count == 0:
            break

def part2():
    global w2
    while(1):
        count = 0
        for i in range(num_rows):
            w2, err = correct1(i, w2, 100000)
            count += err
            
        if count == 0:
            break

def part3():
    global w3    
    ite = 0
    while(1):
        count = 0

        temp, err = correct2(w3,0)
        # if ite % 20 == 0:
        w3 = temp
        # print err
        ite += 1
        if err == 0:
            #print "Trained the weight vector"
            break

def part4():
    global w4
    ite = 0
    while(1):
        count = 0

        temp, err = correct2(w4,100000)
        # if ite % 20 == 0:
        w4 = temp
        # print err
        ite += 1
        if err == 0:
            #print "Trained the weight vector"
            break

part1()
part2()
part3()
part4()
#Testing

def testFile(test_data, rows, weight_vec):
    count = 0
    for i in range(rows):
        prod = np.dot(weight_vec,test_data[i])
        if prod >= float(0):
            clas = 1
            # if test_class[i] == clas:
            #     count += 1  
            print clas   
        else:
            clas = 0
            # if test_class[i] == clas:
            #     count += 1
            print clas
    return count

datafile = pd.read_csv(test)
test_data = datafile.iloc[0:,0:].values
num_rows, num_cols = test_data.shape[:]
#print num_rows,num_cols
# print(train_data[0])
#test_class = datafile.iloc[0:,0].values
num_rows = test_data.shape[0]
test_data = np.c_[np.ones((num_rows,1)), test_data]
num_rows, num_cols = test_data.shape[:]
#print num_rows,num_cols

testFile(test_data, num_rows,w1)
testFile(test_data, num_rows,w2)
testFile(test_data, num_rows,w3)
testFile(test_data, num_rows,w4)

#print count/float(num_rows)
