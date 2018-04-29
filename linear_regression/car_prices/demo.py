import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression

le_x = LabelEncoder()
le_y = LabelEncoder()
hot_encoding = OneHotEncoder(categorical_features = [5])
regressor = LinearRegression()

dataset = pd.read_csv("carprice.csv")

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 6].values


x[:,0] = le_x.fit_transform(x[:,0])
x = hot_encoding.fit_transform(x).toarray()

x = x[:, 1:]



xTrain, xTest, yTrain,yTest = train_test_split(x, y, test_size = 0.2, random_state = 0)

regressor.fit(xTrain, yTrain)

yPred = regressor.predict(xTest)

print(yPred)