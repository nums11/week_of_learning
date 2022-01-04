import sys
sys.path.append('../')
from ml_projects.custom_neural_network.CustomNeuralNetwork import CustomNeuralNetwork
from ml_projects.custom_neural_network.layers import Dense as CustomDense
from ml_projects.custom_neural_network.layers import Conv2D as CustomConv2D
from ml_projects.custom_neural_network.layers import Flatten as CustomFlatten
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.python.keras.layers import Input, Dense, Conv2D, Flatten, AveragePooling2D, Activation
from tensorflow.keras.datasets import mnist
import time

def testCustomModel():
	(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
	num_classes = 10
	# test_mat = np.array([
	# 	[3,5,1,4,6],
	# 	[2,9,7,8,1],
	# 	[0,4,1,3,1],
	# 	[5,6,0,4,2],
	# 	[7,8,9,5,3]
	# ])

	# mat_2 = np.array([
	# 	[
	# 		[[10,11],
	# 		 [12,13]],
	# 		[[14,15],
	# 		 [16,17]]
	# 	]
	# ])

	# print("mat_1\n", mat_1, mat_1.shape)
	# print("mat_2\n", mat_2, mat_2.shape)
	# mult = mat_1 * mat_2
	# print("mult\n", mult, mult.shape)
	# print(np.matmul(mat_1, mat_2).shape)

	# How to grab all the slices from the array at once.
	# How to grab 2 at once?
	# I can easily store all of the traversals in a list
	# Would it be faster to simply grab the image slices in the double for loop
	# and do all the other work outside of the for loop? I think so.
	# It's currently doing the convolution step at every slice as opposed to
	# Gather all the slices and doing the convolution step at once.
	# First just removing the list comprehension from the inner for loop
	# then trying to vectorize the 2d list comprehension into a 1-d list comprehension

	# Figure out how to vectorized multiply 1 slice across all the samples by the weight matrix
	# I have a hunch that if I reshape W properly, I can do a matrix mulitplication to get the
	# output that I'm looking for


	nn = CustomNeuralNetwork("sparse_categorical_cross_entropy")
	nn.addInputLayer((28,28,1))
	nn.add(CustomConv2D(1, 3, "tanh"))
	nn.add(CustomFlatten())
	# # # nn.add(CustomDense(4, "sigmoid"))
	# # # nn.add(CustomDense(6, "softmax"))
	# # # nn.summary()
	nn.fit(X_train, Y_train, 0.01, 1)

def displayDataPoint(index):
	plt.imshow(X_train[index], cmap=plt.get_cmap('gray'))
	plt.show()

def testTF():
	(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
	num_classes = 10

	X_train = X_train.reshape(-1, 28, 28, 1)
	X_train = tf.cast(X_train, tf.float64)
	X_test = X_test.reshape(-1, 28, 28, 1)
	X_test = tf.cast(X_test, tf.float64)

	model = Sequential()
	model.add(Conv2D(1, 3, activation='relu'))
	model.add(Flatten())
	model.add(Dense(num_classes, activation='softmax'))
	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
		loss=tf.keras.losses.SparseCategoricalCrossentropy(),
		metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
	)
	history = model.fit(X_train, Y_train, epochs=5, validation_data=(X_test, Y_test))
	# model.summary()
	fig, axs = plt.subplots(2, 1, figsize=(15,15))
	axs[0].plot(history.history['loss'])
	axs[0].plot(history.history['val_loss'])
	axs[0].title.set_text('Training Loss vs Validation Loss')
	axs[0].legend(['Train', 'Val'])
	axs[1].plot(history.history['sparse_categorical_accuracy'])
	axs[1].plot(history.history['val_sparse_categorical_accuracy'])
	axs[1].title.set_text('Training Accuracy vs Validation Accuracy')
	axs[1].legend(['Train', 'Val'])
	plt.show()
	print(model.evaluate(X_test, Y_test, return_dict=True)['sparse_categorical_accuracy'])

def testTFLeNet1(x_train, y_train, x_test, y_test):
	# Pad images to be 32 x 32 as per original LeNet
	X_train = tf.pad(x_train, [[0, 0], [2,2], [2,2]]) / 255
	X_test = tf.pad(x_test, [[0, 0], [2,2], [2,2]]) / 255
	X_train = tf.cast(tf.expand_dims(X_train, axis=3, name=None), tf.float64)
	X_test = tf.cast(tf.expand_dims(X_test, axis=3, name=None), tf.float64)

	model = Sequential()
	model.add(Input(shape=(32,32,1)))
	model.add(Conv2D(6, 5, activation='tanh'))
	model.add(AveragePooling2D(2))
	model.add(Activation('sigmoid'))
	model.add(Conv2D(16, 5, activation='tanh'))
	model.add(AveragePooling2D(2))
	model.add(Activation('sigmoid'))
	model.add(Conv2D(120, 5, activation='tanh'))
	model.add(Flatten())
	model.add(Dense(84, activation='tanh'))
	model.add(Dense(10, activation='softmax'))
	model.compile(
		optimizer=tf.keras.optimizers.Adam(),
		loss=tf.keras.losses.SparseCategoricalCrossentropy(),
		metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
	)
	model.summary()
	# history = model.fit(X_train, Y_train, epochs=5, validation_data=(X_test, Y_test))
	# fig, axs = plt.subplots(2, 1, figsize=(15,15))
	# axs[0].plot(history.history['loss'])
	# axs[0].plot(history.history['val_loss'])
	# axs[0].title.set_text('Training Loss vs Validation Loss')
	# axs[0].legend(['Train', 'Val'])
	# axs[1].plot(history.history['sparse_categorical_accuracy'])
	# axs[1].plot(history.history['val_sparse_categorical_accuracy'])
	# axs[1].title.set_text('Training Accuracy vs Validation Accuracy')
	# axs[1].legend(['Train', 'Val'])
	# plt.show()
	# print(model.evaluate(X_test, Y_test, return_dict=True)['sparse_categorical_accuracy'])

# testTF()
# testTFLeNet1(X_train, Y_train, X_test, Y_test)
testCustomModel()
