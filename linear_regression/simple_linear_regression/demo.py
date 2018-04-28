
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

dataset = pd.read_csv("Salary_Data.csv")
x = dataset.iloc[:, :-1].values # Experience
y = dataset.iloc[:, 1].values # Salary

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)

yoe = 20
prediction = regressor.predict(yoe)
print("Years of Experience:" + str(yoe) + " Salary: " + str(prediction))


plt.scatter(x_train, y_train)
plt.plot(x_train, regressor.predict(x_train))



plt.show()

