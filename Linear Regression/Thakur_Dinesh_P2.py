# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 00:57:33 2021

@author: dines
"""

import pandas as pd
import numpy as np

FileName = input("Enter the name of your training file: ")
trainD = pd.read_csv(FileName)
trainD = trainD.dropna()
total_rows = trainD.count
trainD = trainD.iloc[:, 2:-1]
X_td = trainD.iloc[:,:-1]
Y_td = trainD[['median_house_value']].values

A = np.linalg.pinv(np.dot(X_td.T, X_td))
B = np.dot(X_td.T,Y_td)
w = np.dot(A, B)

temp1 = np.dot (X_td, w) - Y_td
Jtrain = (1/len(X_td))*np.dot(temp1.T, temp1)

np.set_printoptions(formatter={'float_kind':'{:f}'.format})

print("weights of the train data: ", w)
print("J value of the train data: ", Jtrain)


FileName = input("Enter the name of your test file: ")
testD = pd.read_csv(FileName)
testD = testD.dropna()
testD = testD.iloc[:, 2:-1]
X_testD = testD.iloc[:,:-1]
Y_testD = testD[['median_house_value']].values

temp2 = np.dot (X_testD, w) - Y_testD
Jtest = (1/len(X_testD))*np.dot(temp2.T, temp2)
print("J value of the test data: ", Jtest)

Y_pre = np.dot(X_testD, w)
Comparision = pd.DataFrame(Y_pre, columns=['Predicted output'])
Comparision['Ground truth'] = Y_testD
print(Comparision)
         