# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 16:12:01 2018

@author: Adnan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


data = pd.read_csv('Salary_Data.csv')
X = data.iloc[:, 0:1]
y = data.iloc[:, -1:]

X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=1/3, random_state=0)


sl = LinearRegression()
sl.fit(X_train, y_train)

y_predict = sl.predict(X_test)

plt.scatter(X_test, y_test, color='red')
plt.scatter(X_train, y_train, color='green')
plt.plot(X_train, sl.predict(X_train))
plt.show()
