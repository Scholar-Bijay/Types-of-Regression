# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 02:28:38 2020

@author: Kahar's
"""


#Multiple Linear Regression


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Companies.csv')
#print(dataset.columns)

# One variable endoding
n = pd.get_dummies(dataset , columns=["Location"])

#X =n.loc['Location_Hyderabad','Location_Pune','Major&Minor_Works','R&D_Spend', 'Marketing_Spend']
#X= n.loc['Location_Hyderabad']
X=n.loc[:, ['Location_Hyderabad','Location_Pune','Major&Minor_Works','R&D_Spend', 'Marketing_Spend']] 
Y = n.iloc[:, 4].values

# Avoiding the Dummy Variable Trap


#splitting the dataset as training and testing dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2, random_state=0)


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)








