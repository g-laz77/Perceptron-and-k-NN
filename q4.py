#!/usr/bi

import sys
import os
import numpy as np

"""Feel free to add any extra classes/functions etc as and when needed.
This code is provided purely as a starting point to give you a fair idea
of how to go about implementing machine learning algorithms in general as
a part of the first assignment. Understand the code well"""

classes = ['galsworthy/','galsworthy_2/','mill/','shelley/','thackerey/','thackerey_2/','wordsmith_prose/','cia/','johnfranklinjameson/','diplomaticcorr/']

class FeatureVector(object):
	def __init__(self,vocabsize,numdata):
		self.vocabsize = vocabsize
		self.X =  np.zeros((numdata,self.vocabsize), dtype=np.int)
		self.Y =  np.zeros((numdata,), dtype=np.int)

	def make_featurevector(self, inpu, classid,vec_id):
        for i in inpu:
            words = i.split(" ")
            for word in words:
                self.X[vec_id][vocab[word]] += 1
                self.Y[vec_id] = classid

class KNN(object):
	def __init__(self,trainVec,testVec):
		self.X_train = trainVec.X
		self.Y_train = trainVec.Y
		self.X_test = testVec.X
		self.Y_test = testVec.Y
		self.metric = Metrics('accuracy')

	def classify(self, nn=1):
		for i in range(self.Y_test.shape[0]):
			Y_pred = classes[np.random.randint(0,10)]
			print(Y_pred.strip('/'))

class Metrics(object):
	def __init__(self,metric):
		self.metric = metric

	def score(self):
		if self.metric == 'accuracy':
			return self.accuracy()
		elif self.metric == 'f1':
			return self.f1_score()

	def get_confmatrix(self,y_pred,y_test):
		"""
		Implements a confusion matrix
		"""

	def accuracy(self):
		"""
		Implements the accuracy function
		"""

	def f1_score(self):
		"""
		Implements the f1-score function
		"""

if __name__ == '__main__':
	traindir = "dummy/datasets/q4/train/"
	testdir = "dummy/datasets/q4/test/"
	inputdir = [traindir,testdir]
    vocab = dict()
    trainsz = 0
    testsz = 0
    i = 0
    #vocabulary
    for idir in inputdir:
        for c in classes:
            listing  = os.listdir(idir+c)
            for filename in listing:
                f = open(idir+c+filename,'r')
                if idir == traindir:
                    trainsz += 1
                else:
                    testsz += 1
                if f:
                    text = f.readlines()
                    for tt in text:
                        # tt = tt.split()
                        words = tt.split(" ")
                        for word in words:
                            if vocab.get(word) == None:
                                vocab[word] = i
                                i = i + 1

    print len(vocab),trainsz, testsz

    trainVec = FeatureVector(vocab,trainsz)
    testVec = FeatureVector(vocab,testsz)
    
    for idir in inputdir:
        classid = 0
        vec_id = 0
        for c in classes:
            listing = os.listdir(idir+c)
            for filename in listing:
                f = open(+c+filename,'r')
                inputs = f.readlines()
                if idir == traindir:
                    trainVec.make_featurevector(inputs,classid,vec_id)
                else:
                    testVec.make_featurevector(inputs,classid,vec_id)
                vec_id += 1
                
            classid += 1

    knn = KNN(trainVec,testVec)
    knn.classify() 
	