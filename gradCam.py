#!/usr/bin/env python2.7



import tables, dicom, pandas, numpy, sys, cv2, timeit
from keras.models import load_model
import keras.backend as K

from dsbTests import arrayFile, tsvFile
from generators import getImage

from dsbTests import getImageCubes, makeBatches


def normalize(x): # utility function to normalize a tensor by its L2 norm
	return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def grad_cam(image, gradient_function): 

	output, grads_val = gradient_function([image,0])

	#print output.shape, grads_val.shape

	output, grads_val = output[0, :], grads_val[0, :, :, :]
	print 'output', output.shape
	print 'grads_val', grads_val.shape

	weights = numpy.mean(grads_val, axis = (0, 1, 2))
	cam = numpy.ones(output.shape[0 : 2], dtype = numpy.float32)

	print 'weights_shape', weights.shape
	print 'cam shape', cam.shape

	for i, w in enumerate(weights):
		cam += w * output[:, :, i]

	_, height, width, _ = image.shape
	cam = cv2.resize(cam, (width, height))

	return cam




def buildGradientFunction(model):


	# loss_layer = 'predictions'
	loss_layer = model.get_layer('nodule')
	#loss_layer = model.layers[7]
	loss = K.sum(loss_layer.output)
	layer = model.get_layer("pool2")
	conv_output = layer.output
	grads = normalize(K.gradients(loss, conv_output)[0])

	gradient_function = K.function(
		[model.layers[0].input, K.learning_phase()],
		[conv_output, grads]
	)


	return gradient_function




#imagesDF = pandas.read_csv(DATADIR + 'resampledImages.tsv', sep='\t')

#DB = tables.open_file(DATADIR + 'resampled.h5', mode='r')
#imageArray = DB.root.resampled

if __name__ == '__main__':
	import sys
	from keras.models import load_model

	modelFile = sys.argv[1]
	model = load_model(modelFile)
	_, x,y,z, channels = model.input_shape
	cubeSize = x

	DB = tables.open_file(arrayFile, mode='r')
	array = DB.root.resampled


	DF = pandas.read_csv(tsvFile, sep='\t')
	DF = DF.sample(frac=1)
	DF = DF[DF.cancer != -1]  # remove images from the submission set

	DF = DF.head(1)



	model.compile('sgd', 'binary_crossentropy')
	gradient_function = buildGradientFunction(model)




	testY = []
	y_score = []

	for index, row in DF.iterrows():
		cancer = row['cancer']
		image, imgNum = getImage(array, row)
		print image.shape


		cubes, indexPos = getImageCubes(image, cubeSize)

		for cube in cubes:
			cam = grad_cam([cube], gradient_function)
			print cam.shape

		c













