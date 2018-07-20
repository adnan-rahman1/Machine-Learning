# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 19:57:14 2018

@author: Adnan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


data = pd.read_csv('Data.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1:]


## Remove NaN value from the table
imputer = Imputer()
X.iloc[:, 1:3] = imputer.fit_transform(X.iloc[:, 1:3])


## Solve categorical data problem
le = LabelEncoder()
X.iloc[:, 0:1] = le.fit_transform(X.iloc[:, -3])

## Create dummy varibles to solve categorial data issue
ohe = OneHotEncoder(categorical_features=[0])
X = ohe.fit_transform(X).toarray();

y = le.fit_transform(y.iloc[:,0])



## train and test model using train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Standardized the feature value
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)