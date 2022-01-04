import numpy as np
from ml_projects.custom_neural_network.helpers import oneHot

def BinaryAccuracy(predictions, labels):
	predictions = np.rint(predictions.flatten())
	labels = labels.flatten()
	num_correct = np.array([True if i == j else False for i,j in zip(predictions, labels)]).sum()
	return num_correct / len(labels)

def CategoricalAccuracy(predictions, labels):
	predictions = np.argmax(predictions, axis=1).flatten()
	labels = np.argmax(labels, axis=1).flatten()
	return BinaryAccuracy(predictions, labels)

def SparseCategoricalAccuracy(predictions, labels):
	return CategoricalAccuracy(predictions, oneHot(labels))

metrics = {
	'binary_accuracy': BinaryAccuracy,
	'categorical_accuracy': CategoricalAccuracy,
	'sparse_categorical_accuracy': SparseCategoricalAccuracy
}