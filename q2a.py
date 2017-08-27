import pandas as pd
import numpy as np

# TRAINING
# datafile = pd.read_csv("~/Documents/smai/assignment-1/dummy/datasets/q2/train.csv")
# train_data = datafile.iloc[0:,1:9].values
# # print(train_data[0])
# train_class = datafile.iloc[0:,10].values
# #num_rows = train_data.shape[0]
# #train_data = np.c_[np.ones((num_rows,1)), train_data]
datafile = np.genfromtxt("/Users/sphinx/Documents/smai/assignment-1/dummy/datasets/q2/train.csv",delimiter=",")
train_data = datafile[0:,1:10]
train_class = datafile[0:,10]
num_rows, num_cols = train_data.shape[:]
print(train_data[0])

a = np.zeros((1, num_cols))
print(num_rows,num_cols)
b = 35

for i in range(len(train_class)):       # Multiply by -1
    if train_class[i] == 2:
        for j in range(len(train_data[i])):
            train_data[i][j] *= -1

def relax(row, weight_vec):
    # print("ssup")
    prod = np.dot(weight_vec,train_data[row])
    err = 0
    prod = 20
    if prod < float(b):
        modulus = np.dot(train_data[row],train_data[row])
        const = (b - prod) / modulus
        temp = train_data[row]
        for i in range(len(temp)):
            temp[i] *= const
        weight_vec = np.add(weight_vec,temp)
        err = 1

    return weight_vec, err

for i in range(200):        #number of epochs [Trial and error]
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

# def testFile(test_data, rows):
#     count = 0
#     for i in range(rows):
#         prod = np.dot(a, test_data[i])
#         if prod >= float(b):
#             clas = 4
#             if test_class[i] == clas:
#                 count += 1     
#         else:
#             clas = 2
#             if test_class[i] == clas:
#                 count += 1
#     return count

# datafile = pd.read_csv("~/Documents/smai/assignment-1/dummy/datasets/q2/test.csv")
# test_data = datafile.iloc[0:,1:9].values
# num_rows, num_cols = test_data.shape[:]
# print(num_rows,num_cols)
# # print(train_data[0])
# test_class = datafile.iloc[0:,10].values
# num_rows = test_data.shape[0]
# #test_data = np.c_[np.ones((num_rows,1)), test_data]
# num_rows, num_cols = test_data.shape[:]
# print(num_rows,num_cols)

# count = testFile(test_data, num_rows)
# print(count/float(num_rows))
