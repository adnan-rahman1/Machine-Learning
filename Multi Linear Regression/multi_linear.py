#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 16:07:49 2018

@author: devilknown
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, 4]

le = LabelEncoder()
X.iloc[:, -1] = le.fit_transform(X.iloc[:, -1])

ohe = OneHotEncoder(categorical_features=[3])
X = ohe.fit_transform(X).toarray();

# remove one dummy variable 
X = X[:, 1:]