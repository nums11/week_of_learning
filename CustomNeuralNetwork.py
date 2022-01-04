import numpy as np
from tabulate import tabulate
from tqdm import tqdm
from activations import activations, activation_derivatives
from loss_functions import loss_functions
from metrics import metrics
from helpers import *
from layers import *

class CustomNeuralNetwork(object):
	def __init__(self, loss_func):
		self.layers = []
		self.num_input_features = None
		self.loss_func = loss_functions[loss_func]

	def addInputLayer(self, input_shape):
		num_dimensions = len(input_shape)
		if num_dimensions == 1: # tabular data
			self.num_input_features = input_shape[0]
		elif num_dimensions == 3: # images
			self.input_shape = input_shape
		else:
			raise Exception("Input shape must be either 1 or 3 dimensional. Got shape", input_shape)

	def add(self, layer):
		if isinstance(layer, Dense):
			layer.initWeightMatrix(self.num_input_features, self.layers)
		self.layers.append(layer)
	
	def summary(self):
		print("Custom Neural Network")
		if self.num_input_features == None or len(self.layers) == 0:
			print("No layers")
		else:
			table = []
			for i, layer in enumerate(self.layers):
				layer_type = None
				if isinstance(layer, Dense):
					layer_type = "Dense"
					num_params = layer.W.shape[0] * layer.W.shape[1] + layer.B.shape[0]
					output_shape = (None, layer.num_units)
				else:
					layer_type = "Conv2D"
					num_params = layer.num_filters * layer.filter_size * layer.filter_size + layer.num_filters
					output_shape = (None, layer.filter_size, layer.filter_size, layer.num_filters)

				layer_activation = None
				if layer.activation_func == activations["relu"]:
					layer_activation = "ReLU"
				elif layer.activation_func == activations["sigmoid"]:
					layer_activation = "Sigmoid"
				elif layer.activation_func == activations["softmax"]:
					layer_activation = "Softmax"

				table.append([layer_type, output_shape, num_params, layer_activation])
			print(tabulate(table, headers=['Layer Type', 'Output Shape', '# Params', 'Activation'], tablefmt='pretty'))

	def fit(self, X, Y, alpha, epochs):
		self.X = X
		if self.loss_func == loss_functions["sparse_categorical_cross_entropy"]:
			Y = oneHot(Y)
		self.Y = Y
		self.alpha = alpha
		self.m = len(self.X)

		self.losses = []
		for epoch in tqdm(range(epochs)):
			self.forwardProp()
			self.backProp()
			self.updateWeights()
		return self.losses

	def forwardProp(self, custom_X=None):
		X = self.X if custom_X is None else custom_X
		for i, layer in enumerate(self.layers):
			if i == 0:
				A_prev_layer = X
			else:
				A_prev_layer = self.layers[i-1].A

			if isinstance(layer, Dense):
				layer.Z = np.dot(A_prev_layer, layer.W.T) + layer.B.T
				layer.A =  layer.activation_func(layer.Z)
				assert(layer.Z.shape == (self.m, layer.num_units))
				assert(layer.Z.shape == layer.A.shape)
			elif isinstance(layer, Flatten):
				layer.A = flatten(A_prev_layer)
				print("layer.A\n", layer.A, layer.A.shape)
			else:
				# Convolutional layer
				layer.Z = convolve(A_prev_layer, layer)
				layer.A = layer.activation_func(layer.Z)

		predictions = self.layers[-1].A
		if custom_X is None:
			self.losses.append(self.loss_func(predictions, self.Y))
		return predictions

	def backProp(self):
		# skip over the flatten layers
		for i, layer in reversed(list(enumerate(self.layers))):
			if i == len(self.layers) - 1:
				layer.dZ = layer.A - self.Y
			else:
				layer.dZ = \
					np.dot(self.layers[i+1].dZ, self.layers[i+1].W) * layer.activation_derivative(layer.A)

			if i == 0:
				layer.dW = (1/2) * np.dot(layer.dZ.T, self.X)
			else:
				layer.dW = (1/2) * np.dot(layer.dZ.T, self.layers[i-1].A)

			layer.dB = (1/2) * np.sum(layer.dZ.T, axis=1, keepdims=True)
			assert(layer.dZ.shape == layer.Z.shape)
			assert(layer.dW.shape == layer.W.shape)
			assert(layer.dB.shape == layer.B.shape)

	def updateWeights(self):
		for layer in self.layers:
			layer.W = layer.W - self.alpha * layer.dW
			layer.B = layer.B - self.alpha * layer.dB

	def predict(self, X):
		self.m = len(X)
		return self.forwardProp(custom_X=X)

	def evaluate(self, X, Y, metric):
		predictions = self.predict(X)
		return metrics[metric](predictions, Y)

	def printWeightsDebug(self):
		for i, layer in enumerate(self.layers):
			print("Layer " + str(i) + " weights")
			print(layer.W)