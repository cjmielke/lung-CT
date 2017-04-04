#!/usr/bin/env python2.7



import keras.backend as K
import numpy
from scipy.stats import describe

from convertToNifti import vol2Nifti
from utils import getImage, getImageCubes, ImageArray


def normalize(x): # utility function to normalize a tensor by its L2 norm
	return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def buildGradientFunction(model, output='nodule', poolLayer = 'pool2'):

	loss_layer = model.get_layer(output)
	loss = K.sum(loss_layer.output)
	layer = model.get_layer(poolLayer)
	conv_output = layer.output
	grads = normalize(K.gradients(loss, conv_output)[0])

	gradient_function = K.function(
		[model.layers[0].input, K.learning_phase()],
		[conv_output, grads]
	)

	return gradient_function



def grad_cam(image, gradient_function): 

	output, grads_val = gradient_function([image,0])
	output, grads_val = output[0, :], grads_val[0, :, :, :]

	weights = numpy.mean(grads_val, axis = (0, 1, 2))
	cam = numpy.ones(output.shape[0 : 3], dtype = numpy.float32)

	for i, w in enumerate(weights):
		cam += w * output[:, :, :, i]

	#z, x, y, _ = image[0].shape
	#cam = cv2.resize(cam, (width, height))

	return cam







#imagesDF = pandas.read_csv(DATADIR + 'resampledImages.tsv', sep='\t')

#DB = tables.open_file(DATADIR + 'resampled.h5', mode='r')
#imageArray = DB.root.resampled

if __name__ == '__main__':
	import sys
	from keras.models import load_model

	DATADIR = '/data/datasets/lung/resampled_order1/'
	tsvFile = DATADIR + 'resampledImages.tsv'
	arrayFile = DATADIR + 'resampled.h5'
	arrayFile = DATADIR + 'segmented.h5'

	modelFile = sys.argv[1]
	model = load_model(modelFile)
	_, x,y,z, channels = model.input_shape
	cubeSize = x


	imgSrc = ImageArray(arrayFile, tsvFile=tsvFile)
	DF, array = imgSrc.DF, imgSrc.array

	#DF = DF.sample(frac=1)
	DF = DF[DF.cancer != -1]  # remove images from the submission set

	DF = DF[DF.cancer == 1]  # remove images from the submission set

	DF = DF.head(1)



	model.compile('sgd', 'binary_crossentropy')
	gradient_function = buildGradientFunction(model)




	testY, y_score = [], []

	for index, row in DF.iterrows():
		print row
		cancer = row['cancer']
		image, imgNum = getImage(array, row)
		imgShape = numpy.asarray(image.shape)

		print 'image', describe(image.flatten())

		from scipy.ndimage import zoom
		smallImg = zoom(image, 0.25)
		print smallImg.shape

		numpy.save('segImg.npy', smallImg)


		cubes, indexPos = getImageCubes(image, cubeSize, filterBackground=True, prep=True)
		for c in cubes: assert c.max() <= 499.0

		res = []
		print 'number of cubes returned: ', len(cubes)
		for cube, pos in zip(cubes,indexPos):
			cam = grad_cam([cube], gradient_function)
			cube = numpy.expand_dims(cube, axis=0)
			predictions = model.predict([cube])
			noduleScore, diam, decodedImg = predictions
			cam = cam*noduleScore
			res.append((cam, pos))
			#break
			#print cam.shape


		firstCam, _ = res[0]
		camShape = numpy.asarray(firstCam.shape)

		nChunks = imgShape / cubeSize
		outSize = nChunks*camShape
		print outSize
		bigCam = numpy.zeros(outSize)

		cs = camShape[0]
		for cam, pos in res:
			z, y, x = pos*camShape
			#print pos, z, y, x
			#print numpy.asarray(pos)
			#print cam.mean()
			bigCam[z:z+cs, y:y+cs, x:x+cs] = cam 

		zoomDown = 1.0*outSize/imgShape
		print zoomDown
		sImage = zoom(image, zoomDown)


		print bigCam.shape
		print bigCam.min(), bigCam.mean(), bigCam.max()
		numpy.save('cam.npy', bigCam)
		numpy.save('simage.npy', sImage)


		zoomUp = imgShape/outSize
		zoomUp=0.3*zoomUp
		bigCam = zoom(bigCam, zoomUp)
		print image.shape, bigCam.shape

		numpy.save('bigcam.npy', bigCam)
		numpy.save('image.npy', image)


		bigCam = bigCam.astype('float32')
		print bigCam.dtype
		affine = numpy.eye(4)
		s = zoomUp
		vol2Nifti(bigCam, 'cam.nii.gz', affine=affine)













