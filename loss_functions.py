import numpy as np

def binaryCrossEntroypy(predictions, Y):
	predictions = predictions.flatten()
	Y = Y.flatten()
	m = predictions.shape[0]
	epsilon = 1e-5    
	return (-1 / m) * np.sum(Y * np.log(predictions + epsilon) + (1 - Y) * np.log(1 - predictions + epsilon))

def CategoricalCrossEntropy(predictions, Y):
	m = predictions.shape[0]
	epsilon = 1e-5
	return (-1/m) * np.sum(Y * np.log(predictions + epsilon))

def SparseCategoricalCrossEntropy(predictions, Y):
	return CategoricalCrossEntropy(predictions, Y)

loss_functions = {
	'binary_cross_entropy': binaryCrossEntroypy,
	'categorical_cross_entropy': CategoricalCrossEntropy,
	'sparse_categorical_cross_entropy': SparseCategoricalCrossEntropy
}