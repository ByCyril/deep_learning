
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression

labelencoder_x = LabelEncoder()
labelencoder_y = LabelEncoder()
onehotencoder = OneHotEncoder(categorical_features = [3])
regressor = LinearRegression()

dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

x[:,3] = labelencoder_x.fit_transform(x[:,3])
x = onehotencoder.fit_transform(x).toarray()

x = x[:, 1:]

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 0)

regressor.fit(xTrain, yTrain)
ypred = regressor.predict(xTest)

for i in range(0, len(ypred)):
	print(str(yTest[i]) + "     " + str(ypred[i]))










