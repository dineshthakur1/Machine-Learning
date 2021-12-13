# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 00:57:33 2021

@author: dines
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FileName = input("Enter the name of your data file: ")
DataFile= open(FileName, "r")
# Reading contents of the data file from the user.
aString = DataFile.readline()
d=aString.split("\t")
rows = int(d[0])
columns = int(d[1])

data= np.zeros([rows, columns])

for k in range(rows):
    aString = DataFile.readline()
    t = aString.split("\t")
    t[-1]=t[-1].strip()
    for j in range(columns):    
        data[k,j]= float(t[j])
        
df = pd.DataFrame(data=data, columns=["x", "y"])
df['cluster']=""      


FileName = input("Enter the name of your Centroid file: ")
CFile= open(FileName, "r")
# Reading contents of the centroid data from the user given file.
bString = CFile.readline()
d=bString.split("\t")
rows = int(d[0])
data= np.zeros([rows, 2])

for k in range(rows):
    bString = CFile.readline()
    t = bString.split("\t")
    t[-1]=t[-1].strip()
    for j in range(2):    
        data[k,j]= float(t[j])
        
df1 = pd.DataFrame(data=data, columns=["x", "y"])        
# Spliting the centroids data from dataframe to list for further calculations
c1 = df1.iloc[0]
Centroid1 = c1.values.tolist()
print("The initial centroids are:")
print("Centroid 1:", Centroid1)
c2 = df1.iloc[1]
Centroid2 = c2.values.tolist()
print("Centroid 2:", Centroid2)

# Plotting the data from the initial file(color-coded)
colormap = np.array(['purple', 'orange'])
fig, ax = plt.subplots()
ax.scatter(df['x'], df['y'])
ax.scatter(df1['x'], df1['y'], c=colormap, marker = 'x')


# Defining the function to calculate the new centroids and cost function.
def newCentroid(c1,c2):
    countC1=countC2=xSumC1=ySumC1=xSumC2=ySumC2=0
    Cen1=[0,0]
    Cen2=[0,0]
    # Assigning each data point to the nearest cluster centroid
    for index in df.index:
        distC1 = (((df['x'][index]-c1[0])**2 + (df['y'][index]-c1[1])**2)**0.5)
        distC2 = (((df['x'][index]-c2[0])**2 + (df['y'][index]-c2[1])**2)**0.5)
        if (distC1<distC2):
            df['cluster'][index] = 'cluster1'
            xSumC1+=df['x'][index]
            ySumC1+=df['y'][index]
            countC1+=1
        else:
            df['cluster'][index] = 'cluster2'
            xSumC2+=df['x'][index]
            ySumC2+=df['y'][index]
            countC2+=1
    
    # Defining new centroid by averaging coordinates of the data points in each cluster            
    Cen1[0], Cen1[1] = round(xSumC1 / countC1, 4), round(ySumC1 / countC1, 4)
    Cen2[0], Cen2[1] = round(xSumC2 / countC2, 4), round(ySumC2 / countC2, 4)
    
    df_c1 = df[df['cluster'] == 'cluster1']
    df_c2 = df[df['cluster'] == 'cluster2']
    
    # Calculating the cost value by finding the squared distance between all the data points in each cluster with their centroids
    Jvalue = ((np.sqrt((df_c1['x'] - c1[0]) ** 2 + (df['y'] - c1[1]) ** 2)).sum() + (np.sqrt((df_c2['x'] - c2[0]) ** 2 + (df['y'] - c2[1]) ** 2)).sum())/len(df)
    
    # Printing plot of new clusters with their centroids marked as X
    fig, ax = plt.subplots()
    ax.scatter(df_c1['x'], df_c1['y'], c = 'yellow')
    ax.scatter(df_c2['x'], df_c2['y'], c = 'purple')
    colormap = np.array(['yellow','purple'])
    ax.scatter([Cen1[0], Cen2[0]], [Cen1[1], Cen2[1]], c = colormap, marker = 'x', s = 150)

    return Cen1,Cen2, Jvalue
    

new_c1, new_c2, Error = newCentroid(c1, c2)
# Process of finding the final cluster centroids based on whether the new centroids are different than the previous ones or not
while True:
    if new_c1 == Centroid1 and new_c2 == Centroid2:
        print('Fianl Centroids are:')
        print("Centroid 1:", new_c1)
        print("Centroid 2:", new_c2)
        print('Error is ', Error)
        break
    else:
        Centroid1, Centroid2 = new_c1, new_c2
        new_c1, new_c2, Error = newCentroid(Centroid1, Centroid2)
     