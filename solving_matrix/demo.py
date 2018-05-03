# Cyril Garcia
# May 3, 2018
# Solving the system of a matrix using neural networks

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
nets.train(training_inputs, training_outputs, 60000)

print(nets.weights)


