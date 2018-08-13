# Cyril Garcia
# Intro to Deep Learning
# April 13, 2017


import pandas as pd 
from sklearn import linear_model
import matplotlib.pyplot as plt 

# Read data
dataframe = pd.read_fwf('brain_body.txt')
x_values = dataframe[['Brain']]
y_values = dataframe[['Body']]

print(x_values)
print(y_values)
# Train model on data
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values) # Get the line of best fit and plug the x and y variables

# Visualize the results
plt.scatter(x_values, y_values)
plt.plot(x_values, body_reg.predict(x_values))

# Show the results
plt.show()