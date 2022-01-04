import numpy as np

def relu(x):
	return np.maximum(x, 0)

def relu_derivative(relu_x):
	return 1 * (x > 0)

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoid_derivative(sigmoid_x):
	return sigmoid_x * (1 - sigmoid_x)

def softmax(x):
  return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

def tanh(x):
	return (np.exp(x)-np.exp(-x)) / (np.exp(x)+np.exp(-x))

def tanh_derivative(tanh_x):
	return 1 - tanh_x**2

activations = {
	'relu': relu,
	'sigmoid': sigmoid,
	'softmax': softmax,
	'tanh': tanh
}

activation_derivatives = {
	'relu': relu_derivative,
	'sigmoid': sigmoid_derivative,
	'tanh': tanh_derivative
}