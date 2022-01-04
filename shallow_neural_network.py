import numpy as np
import matplotlib.pyplot as plt
from planar_data_utils import plot_decision_boundary, plot_decision_boundary_custom, sigmoid, load_planar_dataset, load_extra_datasets
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.python.keras.layers import Input, Dense
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time

class Layer(object):
	def __init__(self, num_units, num_units_in_prev_layer):
		self.num_units = num_units
		self.W = np.random.randn(self.num_units, num_units_in_prev_layer) * 0.01
		self.B = np.random.randn(self.num_units, 1) * 0.01
		self.Z = np.zeros((self.num_units, 1))
		self.A = np.zeros((self.num_units, 1))

class ShallowNeuralNetwork(object):
	def __init__(self, num_hidden_units):
		self.num_hidden_units = num_hidden_units
		print("Initialized Shallow Neural Network")

	def fit(self, X, Y, alpha, epochs):
		self.X = X
		self.Y = Y
		self.m = self.X.shape[1]
		self.learning_rate = alpha
		self.hidden_layer = Layer(self.num_hidden_units, self.X.shape[0])
		self.output_layer = Layer(1, self.num_hidden_units)
		self.losses = []
		for epoch in tqdm(range(epochs)):
			self.forwardProp()
			self.backProp()
		# self.plotCostOverTime()

	def forwardProp(self):
		self.hidden_layer.Z = \
			np.dot(self.hidden_layer.W, self.X) + self.hidden_layer.B
		self.hidden_layer.A = sigmoid(self.hidden_layer.Z)
		self.output_layer.Z = \
			np.dot(self.output_layer.W, self.hidden_layer.A) + self.output_layer.B
		self.output_layer.A = sigmoid(self.output_layer.Z)
		self.losses.append(self.getBinaryCrossEntropyLoss(self.output_layer.A, self.Y))

	def backProp(self):
		dZ_output_layer = self.output_layer.A - self.Y
		dW_output_layer = (1/self.m) * np.dot(dZ_output_layer, self.hidden_layer.A.T)
		db_output_layer = (1/self.m) * np.sum(dZ_output_layer, axis=1, keepdims=True)
		dZ_hidden_layer = \
			np.dot(self.output_layer.W.T, dZ_output_layer) * (self.hidden_layer.A * (1 - self.hidden_layer.A))
		dW_hidden_layer = (1/self.m) * np.dot(dZ_hidden_layer, self.X.T)
		db_hidden_layer = (1/self.m) * np.sum(dZ_hidden_layer, axis=1, keepdims=True)
		self.hidden_layer.W = self.hidden_layer.W - self.learning_rate * dW_hidden_layer
		self.hidden_layer.B = self.hidden_layer.B - self.learning_rate * db_hidden_layer
		self.output_layer.W = self.output_layer.W - self.learning_rate * dW_output_layer
		self.output_layer.B = self.output_layer.B - self.learning_rate * db_output_layer

	def getBinaryCrossEntropyLoss(self, predictions, Y):
		cost = (-1 / self.m) * np.sum(Y * np.log(predictions) + (1 - Y) * np.log(1 - predictions))
		return cost

	def plotCostOverTime(self):
		plt.plot(self.losses)
		plt.show()

	def predict(self, X):
		hidden_layer_Z = np.dot(self.hidden_layer.W, X) + self.hidden_layer.B
		hidden_layer_A = sigmoid(hidden_layer_Z)
		output_layer_Z = np.dot(self.output_layer.W, hidden_layer_A) + self.output_layer.B
		probabilities = sigmoid(output_layer_Z)
		predictions = np.rint(probabilities)
		return predictions

	def evaluate(self, X, Y):
		predictions = self.predict(X).flatten()
		return getAccuracy(predictions, Y.flatten())

def getAccuracy(y_predictions, y_test):
	num_correct = np.array([True if i == j else False for i,j in zip(y_predictions, y_test)]).sum()
	return num_correct / len(y_test)


def main():
	# X, Y = load_planar_dataset()
	noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()
	X, Y = noisy_moons
	print(X.shape, Y.shape)
	shallow_nn = ShallowNeuralNetwork(4)
	shallow_nn.fit(X.T, Y.T, 0.1, 10000)
	accuracy = shallow_nn.evaluate(X.T, Y.T)
	print("Accuracy", accuracy)
	plot_decision_boundary_custom(lambda x: shallow_nn.predict(x), X.T, Y.T)

def trainTF():
	X, Y = load_planar_dataset()
	model = Sequential()
	model.add(Input(shape=(2,)))
	model.add(Dense(4, activation='sigmoid'))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
		loss=tf.keras.losses.BinaryCrossentropy(),
		metrics=[tf.keras.metrics.BinaryAccuracy()]
	)
	training_history = model.fit(X, Y, epochs=1000)
	plt.plot(training_history.history["loss"])
	plt.show()
	# print(model.evaluate(X, Y, return_dict=True)['binary_accuracy'])
	# plot_decision_boundary(lambda x: model.predict(x), X.T, Y.T)

def trainPyTorch():
	torch.set_default_dtype(torch.float64)
	X, Y = load_planar_dataset()
	X_tensor = torch.from_numpy(X).to(torch.float64)
	Y_tensor = torch.from_numpy(Y).to(torch.float64)

	dataset = TensorDataset(X_tensor, Y_tensor)
	training_data_loader = DataLoader(dataset, batch_size=X.shape[0], num_workers=2)

	model = nn.Sequential(
		nn.Linear(2, 4),
		nn.Sigmoid(),
		nn.Linear(4,1),
		nn.Sigmoid()
	)

	lossFunc = nn.BCELoss()
	optimizer = optim.Adam(model.parameters(), lr=0.01)
	losses = []
	for epoch in tqdm(range(2000)):
		for i, data in enumerate(training_data_loader, 0):
			inputs, labels = data
			optimizer.zero_grad()

			predictions = model(inputs)
			loss = lossFunc(predictions, labels)
			# Computes the gradients (derivatives)
			loss.backward()
			# Backpropogates
			optimizer.step()
			losses.append(loss.item())

	plt.plot(losses)
	plt.show()

	probabilities = model(X_tensor)
	predictions = torch.round(probabilities)
	num_correct = (predictions == Y_tensor).float().sum()
	accuracy = 100 * num_correct / len(dataset)
	print(accuracy)

main()
# trainTF()
# trainPyTorch()
