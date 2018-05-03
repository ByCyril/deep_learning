
from numpy import *

test_inputs = array([[1,-1,1,-1,1,0,1,0,1],[1,-1,1,1,-1,0,1,0,1],[1,1,1,-1,-1,0,1,0,-1],[1,-1,1,1,-1,0,1,0,-1],[1,0,0,0,-1,0,1,0,-1],[-1,1,-1,0,-1,0,1,0,-1]])
test_outputs = array([[1,1,-1,1,0,0]]).T

random.seed(1)
weights = 2 * random.random((9,1)) - 1

def tanh(x):
	return (exp(x) - exp(-x))/(exp(x) + exp(-x))

def tanhPrime(x):
	return 1 - ((exp(x) - exp(-x)**2)/(exp(x) + exp(-x)**2))

def predict(inputs):
	return tanh(dot(inputs, weights))

def train(t_inputs, t_outputs, iteration):
	global weights
	for i in xrange(iteration):
		output = predict(t_inputs)
		error = t_outputs - output
		adj = dot(t_inputs.T, error * tanhPrime(error))
		weights += adj


train(test_inputs, test_outputs, 10000)

results = predict(array([[1,1,1,-1,-1,0,1,0,-1]]))

print(results)