from numpy import *

def sigmoid(x, deriv=False):

	if deriv == True:
		return x * (1 - x)

	return 1/(1 + exp(-x))

input_data = array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
output_data = array([[0,1,1,0]]).T

print(output_data)

random.seed(1)

syn0 = 2 * random.random((3,4)) - 1
syn1 = 2 * random.random((4,1)) - 1

for j in xrange(60000):

	l0 = input_data
	l1 = sigmoid(dot(l0, syn0))
	l2 = sigmoid(dot(l1, syn1))

	l2_error = output_data - l2

	if (j % 10000) == 0:
		print("Error: " + str(mean(abs(l2_error))))

	l2_delta = l2_error * sigmoid(l2, deriv=True)
	l1_error = dot(l2_delta,syn1.T)
	l1_delta = l1_error * sigmoid(l1, deriv=True)


	syn1 = dot(l1.T, l2_delta)
	syn0 = dot(l0.T,l1_delta)

print(l2)