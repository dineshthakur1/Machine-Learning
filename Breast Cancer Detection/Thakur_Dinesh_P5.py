# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 21:39:15 2021

@author: dines
"""

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt

# Loading the data
cancer = load_breast_cancer()

# Building K-Nearest Neighbor Model
Tlabels=["Benign","Malignant"]
x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.25, train_size=0.75,random_state=0, stratify=cancer.target)
init=k=0

# Running loop for finding optimal k with lowest error rate
error_rate = []
for i in range(1, 40, 2):
  knn = KNeighborsClassifier(n_neighbors=i)
  knn.fit(x_train,y_train)
  K_pred = knn.predict(x_test)
  error_rate.append(np.mean(K_pred != y_test))

# Plotting the Error rate vs K-value for the reference
plt.figure(figsize=(10,6))
plt.plot(range(1,40, 2),error_rate,color='black', linestyle='-.', marker='o',markerfacecolor='blue', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
k= 2*error_rate.index(min(error_rate))+1
print("Minimum error:-",min(error_rate),"at K =",k)

# Retraining the KNN classifier for the best value of k
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(x_train, y_train)
K_pred = knn.predict(x_test)
# Printing confusion matrix for K Nearest Neighbor
cm = confusion_matrix(y_test, K_pred, labels=knn.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=Tlabels)
disp.plot()

# Printing Accuracy, Precision, Recall & F1 metrics for K Nearest Neighbor 
Knn_acc=accuracy_score(y_test, K_pred)
print("Accuracy score for KNN is ", Knn_acc)
Knn_precision=precision_score(y_test, K_pred)
print("Precision value for KNN is ", Knn_precision)
Knn_recall=recall_score(y_test, K_pred)
print("Recall value for KNN is ", Knn_recall)
Knn_f1=f1_score(y_test, K_pred)
print("F1 score for KNN is ", Knn_f1)

# Building Logistic Regression Model
log_reg = LogisticRegression(max_iter=10000)
log_reg.fit(x_train, y_train)

LR_pred = log_reg.predict(x_test)
# Printing confusion matrix for Logistic Regression
cm = confusion_matrix(y_test, LR_pred, labels=log_reg.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=Tlabels)
disp.plot()

# Printing Accuracy, Precision, Recall & F1 metrics for Logistic Regression 
LR_acc=accuracy_score(y_test, LR_pred)
print("\nAccuracy score for Logistic Regression is ", LR_acc)
LR_precision=precision_score(y_test, LR_pred)
print("Precision value for Logistic Regression is ", LR_precision)
LR_recall=recall_score(y_test, LR_pred)
print("Recall value for Logistic Regression is ", LR_recall)
LR_f1=f1_score(y_test, LR_pred)
print("F1 score for Logistic Regression is ", LR_f1)

# Building Support Vector Machine model 
svc_model= make_pipeline(StandardScaler(), SVC(gamma='auto'))
svc_model.fit(x_train, y_train)
S_pred= svc_model.predict(x_test)
# Printing confusion matrix for Support Vector Machine
cm = confusion_matrix(y_test, S_pred, labels=log_reg.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=Tlabels)
disp.plot()

# Printing Accuracy, Precision, Recall & F1 metrics for Support Vector Machine
Svm_acc=accuracy_score(y_test, S_pred)
print("\nAccuracy score for Support Vector Machine is ", Svm_acc)
Svm_precision=precision_score(y_test, S_pred)
print("Precision value for Support Vector Machine is ", Svm_precision)
Svm_recall=recall_score(y_test, S_pred)
print("Recall value for Support Vector Machine is ", Svm_recall)
Svm_f1=f1_score(y_test, S_pred)
print("F1 score for Support Vector Machine is ", Svm_f1)

# Building MLP model
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
classf = MLPClassifier(hidden_layer_sizes=(256,128,64,32),activation="relu",solver='lbfgs',max_iter=1000, random_state=1).fit(x_train, y_train)
MLP_pred=classf.predict(x_test)
# Printing confusion matrix for Multilayer Perceptron
cm = confusion_matrix(y_test, MLP_pred, labels=log_reg.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=Tlabels)
disp.plot()

# Printing Accuracy, Precision, Recall & F1 metrics for Multilayer Perceptron
MLP_acc=accuracy_score(y_test, MLP_pred)
print("\nAccuracy score for Multilayer Perceptron is ", MLP_acc)
MLP_precision=precision_score(y_test, MLP_pred)
print("Precision value for Multilayer Perceptron is ", MLP_precision)
MLP_recall=recall_score(y_test, MLP_pred)
print("Recall value for Multilayer Perceptron is ", MLP_recall)
MLP_f1=f1_score(y_test, MLP_pred)
print("F1 score for Multilayer Perceptron is ", MLP_f1)

plt.show()
