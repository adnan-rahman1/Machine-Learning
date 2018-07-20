# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 19:57:14 2018

@author: Adnan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv('Data.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1:]


## Remove NaN value from the table
imputer = Imputer()
X.iloc[:, 1:3] = imputer.fit_transform(X.iloc[:, 1:3])


## Solve categorical data problem
le = LabelEncoder()
X.iloc[:, 0:1] = le.fit_transform(X.iloc[:, 0:1])
