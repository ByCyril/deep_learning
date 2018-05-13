
from numpy import *

# inputs = [1, 0, 1, 1]

# weights = [[1, 4, 5, 2], [3, 5, 1, 4], [4, 2, 4, 1], [1, 2, 1, 1]]

train_inputs = array([[1,0,-1,-1,-1,0,1,1,1], [1,0,-1,1,-1,0,1,-1,1], [1,0,0,0,0,0,1,0,1], [1,1,1,0,-1,0,1,0,1], [1,0,0,1,1,0,-1,-1,1]])
synaptic_weights = 2 * random.random((9,9)) - 1
train_outputs = array([[1,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0]])

def _tanh(x):
	return tanh(x)

def tanhPrime(x):
	return 1 - (tanh(x))**2

def prediction(inputs):
	return _tanh(dot(inputs, synaptic_weights))

def train(inputs, outputs, iteration):
	global synaptic_weights
	output = prediction(inputs)
	error = outputs - output
	adj = dot(inputs.T, error * tanhPrime(output))
	synaptic_weights += adj

train(train_inputs, train_outputs, 600000)
print(prediction(array([[1,0,0,0,0,0,1,0,1]])))

# for i in xrange(60000):
# results = dot(inputs, synaptic_weights.T)

# error = train_outputs - results

# adj = 0.0001 * dot(inputs.T, error)
# synaptic_weights += adj

# print(train_inputs)
# print(results)
# print(error)
# print(adj)
# print(synaptic_weights)	


