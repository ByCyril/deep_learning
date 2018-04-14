# Cyril Garcia
# Intro to Deep learning
# How to do Linear Regression using Gradient Descent
# April 13, 2018
# https://www.youtube.com/watch?v=XdM6ER7zTLk&index=2&list=PL2-dafEMk2A7YdKv4XfKpfbTH5z6rEEj3

from numpy import *

def step_gradient(b_current, m_current, points, learning_rate):
	b_gradient = 0
	m_gradient = 0
	N = float(len(points))

	for i in range(0, len(points)):
		x = points[i, 0]
		y = points[i, 1]
		b_gradient += -(2/N) * (y-((m_current * x) + b_current))
		m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))

	new_b = b_current - (learning_rate * b_gradient)
	new_m = m_current - (learning_rate * m_gradient)

	return [new_b, new_m]

def compute_error_for_given_points(b, m, points):
	totalError = 0

	for i in range(0, len(points)):
		x = points[i, 0]
		y = points[i, 1]
		totalError += (y - (m * x + b)) **2

	return totalError / float(len(points))


def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
	b = starting_b
	m = starting_m

	for i in range(num_iterations):
		b, m = step_gradient(b, m, array(points), learning_rate)

	return [b, m]

def run():
	# get the data
	points = genfromtxt('data.csv', delimiter=',')
	# hyperparameters
	learning_rate = 0.0001

	# y = mx+b
	initial_b = 0
	initial_m = 0
	num_iterations = 1000
	[b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)


	print("After 1000 iterations b = " + str(b) + ", m = " + str(m) + ", error = " + str(compute_error_for_given_points(b, m, points)))
	print("Optimal best fit value: y = (" + str(m) +")x + " + str(b))


if __name__ == '__main__':
	run()





