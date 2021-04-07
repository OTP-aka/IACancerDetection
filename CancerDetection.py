#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 15:08:19 2021

@author: edouardbotherel
"""

#Librairies 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

#Chargement Données 
df = pd.read_csv('/Users/edouardbotherel/Desktop/data.csv') 
df.head(7)

 
df.isna().sum()
df = df.dropna(axis=1)

#Compte du nombre de lignes et colonnes 
df.shape

#Compte du nombre de cellules M & B
df['diagnosis'].value_counts()

#Visualisation graphique
sns.countplot(df['diagnosis'],label="Count")
df.dtypes

#Encodage des M & B 
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
df.iloc[:,1]= labelencoder_Y.fit_transform(df.iloc[:,1].values)
print(labelencoder_Y.fit_transform(df.iloc[:,1].values))

sns.pairplot(df, hue="diagnosis")

df.head(5)

#Corrélations
df.corr()

plt.figure(figsize=(20,20))  
sns.heatmap(df.corr(), annot=True, fmt='.0%')

X = df.iloc[:, 2:31].values 
Y = df.iloc[:, 1].values 

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Modèles 
def models(X_train,Y_train):
  
  #Regression Linéaire
  from sklearn.linear_model import LogisticRegression
  log = LogisticRegression(random_state = 0)
  log.fit(X_train, Y_train)
  
  #ClassificateurVoisins
  from sklearn.neighbors import KNeighborsClassifier
  knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
  knn.fit(X_train, Y_train)


  #Gaussienne
  from sklearn.naive_bayes import GaussianNB
  gauss = GaussianNB()
  gauss.fit(X_train, Y_train)

  #Arbre de décision
  from sklearn.tree import DecisionTreeClassifier
  tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
  tree.fit(X_train, Y_train)

  
  #print model accuracy on the training data.
  print('[0]Logistic Regression Training Accuracy:', log.score(X_train, Y_train))
  print('[1]K Nearest Neighbor Training Accuracy:', knn.score(X_train, Y_train))
  print('[4]Gaussian Naive Bayes Training Accuracy:', gauss.score(X_train, Y_train))
  print('[5]Decision Tree Classifier Training Accuracy:', tree.score(X_train, Y_train))
  
  return log, knn, gauss, tree

model = models(X_train,Y_train)


from sklearn.metrics import confusion_matrix
for i in range(len(model)):
  cm = confusion_matrix(Y_test, model[i].predict(X_test))
  
  TN = cm[0][0]
  TP = cm[1][1]
  FN = cm[1][0]
  FP = cm[0][1]
  
  print(cm)
  print('Model[{}] Testing Accuracy = "{}!"'.format(i,  (TP + TN) / (TP + TN + FN + FP)))
  
  
  #Test 

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

for i in range(len(model)):
  print('Model ',i)
  print( classification_report(Y_test, model[i].predict(X_test)) )
  print( accuracy_score(Y_test, model[i].predict(X_test)))
  
pred = model[6].predict(X_test)
print(pred)
print(Y_test)






















