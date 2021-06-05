import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('./salarios.csv')
a = dataset.head(5)
# print(a)

# print ( dataset.shape)
x = dataset.iloc[:, 0].values
y = dataset.iloc[:, 1].values
# print(x)
# print(y)

X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=0.2, random_state = 0)

print(X_train)