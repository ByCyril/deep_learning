
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# step 1 - load data
dataframe = pd.read_csv('data.csv')

# Removes these columns we dont need
dataframe = dataframe.drop(['index', 'price', 'sq_price'], axis=1)

# Gets the rows we want
dataframe = dataframe[0:10]


# Step 2 - add labels
dataframe.loc[:,('y1')] = [1,1,1,0,0,1,0,1,1,1]
dataframe.loc[:,('y2')] = dataframe['y1'] == 0
dataframe.loc[:,('y2')] = dataframe['y2'].astype(int)


# Step 3 - prepare data for tensorflow (tensors)
# convert featurs to input tensors
inputx = dataframe.loc[:, ['area', 'bathrooms']].as_matrix()

# converts labels to input tensors
inputy = dataframe.loc[:, ['y1', 'y2']].as_matrix() 


# Step 4 - write out hyperparameters
learning_rate = 0.000001 # how fast to reach convergence
training_epochs = 2000 # training 2,000 times
display_steps = 50
n_samples = inputy.size


# Step 5 - Create our computation graph/neural network

# feature input tenors, none means any number of examples
# placeholders are gateways for data int our computetion graph
x = tf.placeholder(tf.float32, [None, 2])

# create weights
# 2x2 float matrix that will continue to update during training process
w = tf.Variable(tf.zeros([2,2]))

# add biases
b = tf.Variable(tf.zeros([2]))

y_values = tf.add(tf.matmul(x,w),b)

# apply the softmax/sigmoid to value created above. 
y = tf.nn.softmax(y_values)

# feed in a matrix of labels
y_ = tf.placeholder(tf.float32, [None, 2])


# step 6 - perform training
# create cost function, mean squared error
cost = tf.reduce_sum(tf.pow(y_ - y, 2)) / (2 * n_samples)


# gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


for i in range(training_epochs):
	sess.run(optimizer, feed_dict={x: inputx, y_: inputy})

	if (i) % display_steps == 0:
		cc = sess.run(cost, feed_dict={x: inputx, y_: inputy})
		print "Training step:", '%04d' % (i), "cost=", "{:.9f}".format(cc) #, \"W=", sess.run(W), "b=", sess.run(b)

print "Optimization Finished!"
training_cost = sess.run(cost, feed_dict={x: inputx, y_: inputy})
print "Training cost=", training_cost, "W=", sess.run(w), "b=", sess.run(b), '\n'


sess.run(y, feed_dict={x:inputx})





