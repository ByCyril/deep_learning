
from numpy import *


training_inputs = array([[1,2,3,4],[1,1,2,3],[1,1,2,1],[2,3,1,1]])
training_outputs = array([[3,2,3,1]]).T

class NeuralNetwork():

	def __init__(self):
		random.seed(1)
		self.weights = 2 * random.random((4,1)) - 1

	def predict(self, inputs):
		return dot(inputs, self.weights)

	def train(self, inputs, outputs, iteration):

		for i in xrange(iteration):
			output = self.predict(inputs)
			error = outputs - output
			adj = 0.01 * dot(inputs.T, error)

			self.weights += adj


nets = NeuralNetwork()
print(nets.weights)
nets.train(training_inputs, training_outputs, 60000)

print(nets.weights)


# test_inputs = raw_input(": ").split(',')
# arr = [int(x) for x in test_inputs]

# r = nets.predict(array(arr))
# print(r)
# while test_inputs != "":

# 	r = nets.predict(array(arr))
# 	print(r)

# 	test_inputs = raw_input(": ").split(',')
# 	arr = [int(x) for x in test_inputs]






