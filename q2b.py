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
#print(num_rows,num_cols)
b = 50
eta = 0.001

for i in range(len(train_class)):       # Multiply by -1
    if train_class[i] == 2:
        for j in range(len(train_data[i])):
            train_data[i][j] *= -1

def descent(weight_vec):            #gradient descent
    summ = np.zeros((1, num_cols))
    err = 0

    # for tr in range(len(train_data)):
    #     prod = np.dot(weight_vec,train_data[tr])
    #     if prod <= float(b):
    #         modulus = np.dot(train_data[tr],train_data[tr])
    #         const = (b - prod) / modulus
    #         summ = np.add(summ,train_data[tr]*const)
    #         err += 1
    # print(train_data)
    prod = np.dot(train_data,np.transpose(weight_vec))
    #print(prod)
    for i in range(len(prod)):
        if prod[i] <= float(b):
            modulus = np.dot(train_data[i],train_data[i])
            const = eta * (b - prod[i]) / modulus
            #print(modulus)
            summ = np.add(summ,train_data[i]*const)
            #print(summ)
            err += 1
    weight_vec = np.add(weight_vec,summ)
    # print(weight_vec)
    return weight_vec, err

for i in range(15000):        #number of epochs [Trial and error]
    count = 0
    a, err= descent(a)
    print(err)
    if err == 0:
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
test_class = datafile.iloc[0:,10].values

test_data = np.c_[np.ones((num_rows,1)), test_data]
num_rows, num_cols = test_data.shape[:]
print(num_rows,num_cols)

count = testFile(test_data, num_rows)
print(count/float(num_rows))
