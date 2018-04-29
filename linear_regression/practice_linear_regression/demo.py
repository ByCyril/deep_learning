
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.linear_model as lm

data = pd.read_csv("data.csv")

x = data.iloc[:, :-1].values
y = data.iloc[:, 1].values

regression = lm.LinearRegression()
regression.fit(x, y)

plt.scatter(x, y)
plt.plot(x, regression.predict(x))

plt.show()
