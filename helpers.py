import numpy as np
import time

def oneHot(arr):
	num_classes = len(set(arr))
	return np.squeeze(np.eye(num_classes)[arr.reshape(-1)])

def convolve(samples, layer):
	image = np.array([
		[1,2,3,4,5],
		[6,7,8,9,10],
		[11,12,13,14,15],
		[16,17,18,19,20],
		[21,22,23,24,25],
	])
	image2 = np.array(image*2)
	image3 = np.array(image*3)
	samples = np.array([image, image2, image3])
	# print(samples, samples.shape)


	m = len(samples)
	print("m",m)
	n = samples[0].shape[0]
	f = layer.filter_size
	output_shape = n - f + 1
	outputs = np.empty([m, layer.num_filters, output_shape, output_shape])
	for row in range(output_shape):
		for col in range(output_shape):
			print("row", row, "col", col)
			# One step of the convolution vectorized across filters and samples
			# Grab the current fxf slice across all samples
			image_slices = samples[:, row:row+f, col:col+f]
			assert(image_slices.shape == (m, f, f))
			# Element-wise multiply the fxf slices across samples with all of the filters
			total_mult = np.array([slice * layer.W for slice in image_slices])
			assert(total_mult.shape == (m, layer.num_filters, f, f))
			# Sum the result to complete this step of the convolution across all filters and samples
			inner_sum = np.sum(np.sum(total_mult, axis=2), axis=2)
			assert(inner_sum.shape == (m, layer.num_filters))

			outputs[:,:,row,col] = inner_sum

	Z = outputs + layer.B
	assert(Z.shape == (m, layer.num_filters, output_shape, output_shape))

	return Z

def flatten(matrix):
	return matrix.reshape(matrix.shape[0], -1)