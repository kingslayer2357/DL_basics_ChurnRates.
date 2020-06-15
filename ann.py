# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 17:42:47 2020

@author: kingslayer
"""

#ARTIFICIAL NEURAL NETWORK

#PART-1(data preprocessing)

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset 

dataset=pd.read_csv("Churn_Modelling.csv")

#creating matrix of features and dependant vector

X=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values

#categorical variables
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder1=LabelEncoder()
X[:,1]=labelencoder1.fit_transform(X[:,1])
labelencoder2=LabelEncoder()
X[:,2]=labelencoder2.fit_transform(X[:,2])
onehotencoder=OneHotEncoder(categorical_features=[1])
X=onehotencoder.fit_transform(X).toarray()
X=X[:,1:]

#Splitting into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
X_train=sc_x.fit_transform(X_train)
X_test=sc_x.transform(X_test)


#PART-2 (Creating The ANN)

#importing libraries
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initialising the ANN
classifier=Sequential()


#Adding the input layer and first hidden layer
classifier.add(Dense(output_dim=6,init="uniform",activation="relu",input_dim=11))

#Adding the second hidden layer
classifier.add(Dense(output_dim=6,init="uniform",activation="relu"))

#Adding output layer
classifier.add(Dense(output_dim=1,init="uniform",activation="sigmoid"))


#Compiling the ANN
classifier.compile(optimizer="adam",metrics=["accuracy"],loss="binary_crossentropy")

#Fitting
classifier.fit(X_train,y_train,batch_size=10,nb_epoch=100)

#PART-3 (Prediction)

y_pred=classifier.predict(X_test)
y_pred=(y_pred>0.5)

#Confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)