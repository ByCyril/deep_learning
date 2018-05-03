
from numpy import *


def signmoid(x):
	return 1/(1+exp(-x))

def sigmoidPrime(x):
	return x * (1-x)

def debug(title, val):
	print(str(title) + str(val))

training_inputs = array([[1,0,0,0,0,0,-1,0,1],[1,0,1,0,0,0,-1,0,0],[1,0,0,0,1,0,-1,0,0],[1,1,0,0,0,0,-1,0,0]])
training_outputs = array([[0,0,0,0,1,0,0,0,0],[0,1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1],[0,0,1,0,0,0,0,0,0]])

weights = 2 * random.random((9,9)) - 1

# inputSum = dot(training_inputs[1], weights)


# print(weights)

for h in xrange(4):
	for j in xrange(9):
		for i in xrange(10000):
			inputSum = dot(training_inputs[h], weights[j])

			output = signmoid(inputSum)

			error = training_outputs.T[j] - output

			adj = dot(training_inputs.T, error * sigmoidPrime(output))

			weights[j] += adj



y = dot([1,0,0,0,1,0,-1,0,0], weights[1])
print(signmoid(y))


# print(weights)






