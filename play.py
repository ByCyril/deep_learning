
from numpy import *

# inputs = [1, 0, 1, 1]

# weights = [[1, 4, 5, 2], [3, 5, 1, 4], [4, 2, 4, 1], [1, 2, 1, 1]]

# train_inputs = array([[1,0,-1,-1,-1,0,1,1,1], [1,0,-1,1,-1,0,1,-1,1], [1,0,0,0,0,0,1,0,1], [1,1,1,0,-1,0,1,0,1], [1,0,0,1,1,0,-1,-1,1]])
# synaptic_weights = 2 * random.random((9,9)) - 1
# print(synaptic_weights)
# train_outputs = array([[1,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0]])

train_inputs = array([[1,0,0,0], [1,0,1,1], [1,1,1,1]])
synaptic_weights = 2 * random.random((4,4)) - 1
print(synaptic_weights)
train_outputs = array([[0,1,1,1], [0,1,0,0], [0,0,0,0]])



def _tanh(x):
	# return (exp(x) - exp(-x)) / (exp(x) + exp(-x))
	return 1/(1 + exp(-x))

def tanhPrime(x):
	# return 1 - ((exp(x) - exp(-x))**2 / (exp(x) + exp(-x))**2)
	return x * (1-x)

def prediction(inputs):
	return _tanh(dot(inputs, synaptic_weights))

def train(inputs, outputs, iteration):

	for i in range(iteration):
		global synaptic_weights
		output = prediction(inputs)
		error = outputs - output
		adj = dot(inputs.T, error * tanhPrime(output))
		synaptic_weights += adj


train(train_inputs, train_outputs, 600000)

print(synaptic_weights)
print(prediction(array([[1,0,0,0]])))


