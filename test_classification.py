import sys
sys.path.append('../')
from CustomNeuralNetwork import CustomNeuralNetwork
from layers import Dense as CustomDense
from planar_data_utils import *
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.python.keras.layers import Input, Dense
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import copy

# Get sparse to work
def testCustomModel():
	planar = load_planar_dataset()
	noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()
	X, Y = gaussian_quantiles
	# print(Y)
	# Y_one_hot = np.squeeze(np.eye(6)[Y.reshape(-1)])
	# print(Y_one_hot)

	nn = CustomNeuralNetwork("binary_cross_entropy")
	nn.addInputLayer((2,))
	nn.add(CustomDense(4, "sigmoid"))
	nn.add(CustomDense(1, "sigmoid"))

	# print(Y, Y.shape)
	Y = Y.reshape(-1,1)
	# Y = np.squeeze(np.eye(6)[Y.reshape(-1)])

	print("New net --------------------------------------")
	loss = nn.fit(X, Y, 0.1, 1000)
	plt.plot(loss)
	plt.show()
	print("Accuracy", nn.evaluate(X, Y, 'binary_accuracy'))

def testTFModel():
	planar = load_planar_dataset()
	noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()
	X, Y = blobs
	# Y = Y.reshape(-1,1)
	Y = tf.one_hot(Y, 6)

	# model = Sequential()
	# model.add(Input(shape=(2,)))
	# model.add(Dense(4, activation='sigmoid'))
	# model.add(Dense(1, activation='sigmoid'))
	# model.compile(
	# 	optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
	# 	loss=tf.keras.losses.BinaryCrossentropy(),
	# 	metrics=[tf.keras.metrics.BinaryAccuracy()]
	# )
	# training_history = model.fit(X, Y, epochs=1000)
	# plt.plot(training_history.history["loss"])
	# print(model.evaluate(X, Y, return_dict=True)['binary_accuracy'])

	# Everything appears to be equal but one of them isn't training.

	model = Sequential()
	model.add(Input(shape=(2,)))
	model.add(Dense(4, activation='sigmoid'))
	model.add(Dense(6, activation='softmax'))
	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
		loss=tf.keras.losses.CategoricalCrossentropy(),
		metrics=[tf.keras.metrics.CategoricalAccuracy()]
	)
	# model.summary()
	training_history = model.fit(X, Y, epochs=1000)
	plt.plot(training_history.history["loss"])
	plt.show()
	print(model.evaluate(X, Y, return_dict=True)['categorical_accuracy'])
	# plot_decision_boundary(lambda x: model.predict(x), X.T, Y.T)

def testPytorchModel():
	torch.set_default_dtype(torch.float64)

	# X, Y = load_planar_dataset()
	noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()
	X, Y = blobs
	# Y = np.array([Y]).T
	# print(Y)

	X_tensor = torch.from_numpy(X).to(torch.float64)
	Y_tensor = nn.functional.one_hot(torch.from_numpy(Y)).to(torch.float64)

	dataset = TensorDataset(X_tensor, Y_tensor)
	training_data_loader = DataLoader(dataset, batch_size=X.shape[0], num_workers=2)

	model = nn.Sequential(
		nn.Linear(2, 4),
		nn.Sigmoid(),
		nn.Linear(4, 6),
		nn.Softmax()
	)

	lossFunc = nn.BCELoss()
	optimizer = optim.Adam(model.parameters(), lr=0.01)
	losses = []
	for epoch in tqdm(range(1000)):
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

	probabilities = model(X_tensor).detach().numpy()
	# Binary Accuracy
	# predictions = torch.round(probabilities)
	# num_correct = (predictions == Y_tensor).float().sum()
	# accuracy = 100 * num_correct / len(dataset)
	# print(accuracy)

	# 
	# probabilities = np.argmax(probabilities, axis=1)
	# num_correct = np.array([True if i == j else False for i,j in zip(probabilities, Y)]).sum()
	# accuracy = num_correct / len(Y)
	# print("accuracy", accuracy)

testCustomModel()
# testTFModel()
# testPytorchModel()