#!/usr/bin/env python2.7
import argparse
from Queue import Queue
from threading import Thread

import keras
import keras.backend as K
import numpy
import pandas
import tables
from keras.layers import Lambda
from scipy.stats import describe
from scipy.ndimage import zoom
from tqdm import tqdm

from convertToNifti import vol2Nifti
from extractNonzeroCubes import SparseImageSource
from utils import getImage, getImageCubes, ImageArray, prepCube, forceImageIntoShape, boundingBox
import tensorflow as tf
from tensorflow.python.framework import ops
from skimage.transform import resize

CAM_SHAPE = (150, 150, 150)





###### Guided backprop stuff

def register_gradient():
	if "GuidedBackProp" not in ops._gradient_registry._registry:
		@ops.RegisterGradient("GuidedBackProp")
		def _GuidedBackProp(op, grad):
			dtype = op.inputs[0].dtype
			return grad * tf.cast(grad > 0., dtype) * tf.cast(op.inputs[0] > 0., dtype)


def modify_backprop(model, name, modelPath):
	g = tf.get_default_graph()
	with g.gradient_override_map({'Relu': name}):

		# get layers that have an activation
		layer_dict = [layer for layer in model.layers[1:] if hasattr(layer, 'activation')]

		# replace relu activation
		for layer in layer_dict:
			if layer.activation == keras.activations.relu:
				layer.activation = tf.nn.relu

		# re-instanciate a new model
		#new_model = VGG16(weights='imagenet')
		new_model = load_model(modelPath)
	return new_model


#def compile_saliency_function(model, activation_layer='block5_conv3'):
def compile_saliency_function(model, activation_layer='block2_conv1'):
	input_img = model.input

	#layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
	#layer_output = layer_dict[activation_layer].output

	#layer_output = model.get_layer(activation_layer).output

	# need to get last conv layer BEFORE deconv layers kick in
	convLayer = None
	for layer in model.layers:
		if 'up_sampling' in layer.name: break
		if 'conv3d' in layer.name: convLayer = layer


	layer_output = convLayer.output


	max_output = K.max(layer_output, axis=3)
	saliency = K.gradients(K.sum(max_output), input_img)[0]
	return K.function([input_img, K.learning_phase()], [saliency])




###### Grad cam stuff

def normalize(x): # utility function to normalize a tensor by its L2 norm
	return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


######## gradcam functions for softmax, adapted from https://github.com/jacobgil/keras-grad-cam

def target_category_loss(x, category_index, nb_classes):
	return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
	return input_shape


def buildSoftmaxGradientFunction(model, category_index, outLayerName='diam', poolLayerName ='pool2'):

	outLayer = model.get_layer(outLayerName)
	_, nb_classes = outLayer.output_shape

	target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
	out = Lambda(target_layer, output_shape=target_category_loss_output_shape, name='new')(outLayer.output)

	loss = K.sum(out)

	poolLayer = model.get_layer(poolLayerName)
	conv_output = poolLayer.output

	grads = normalize(K.gradients(loss, conv_output)[0])
	gradient_function = K.function(
		[model.layers[0].input, K.learning_phase()], 
		[conv_output, grads]
	)

	gradShape = poolLayer.output_shape[1:4]			# (None, 8, 8, 8, 64) - want to reject instances and channels shapes

	return gradient_function, gradShape[0]



######## gradcam function for single output neurons, adapted from https://github.com/jacobgil/keras-grad-cam


def buildGradientFunction(model, output='nodule', poolLayerName ='pool2'):
	loss = K.sum(model.get_layer(output).output)

	poolLayer = model.get_layer(poolLayerName)
	conv_output = poolLayer.output
	grads = normalize(K.gradients(loss, conv_output)[0])

	gradient_function = K.function(
		[model.layers[0].input, K.learning_phase()],
		[conv_output, grads]
	)
	gradShape = poolLayer.output_shape[1:4]			# (None, 8, 8, 8, 64) - want to reject instances and channels shapes

	return gradient_function, gradShape[0]			# assuming cube ...



######

def batchImages(iterable, n=200):
	l = len(iterable)
	for i in range(0, l, n): yield iterable[i: min(i + n, l)]


def gradCamsFromList(images, gradient_function): 
	cams = []

	for imgBatch in batchImages(images, n=500):

		outputs, gradsVals = gradient_function([imgBatch,0])

		for output, grads in zip(outputs, gradsVals):

			weights = numpy.mean(grads, axis = (0, 1, 2))
			cam = numpy.ones(output.shape[0 : 3], dtype = numpy.float32)

			for i, w in enumerate(weights): cam += w * output[:, :, :, i]

			cams.append(cam)

	return cams

def saliencyMapsFromList(cubes, saliency_fn): 
	maps = []
	for cubeBatch in batchImages(cubes, n=300):
		saliencyMaps = saliency_fn([cubeBatch,0])
		maps.extend(saliencyMaps)
		return saliencyMaps

	return maps


11111111111111111111111111






def makeCamImageFromCubesFaster(cubes, positions, gradFuncs, cubeCamSize, saliencyFn=None, storeSourceImg=False):

	nChannels = len(gradFuncs)
	#if storeSourceImg: nChannels += 1


	try: nChunks = numpy.asarray(positions).max(axis=0)		# get maximum position in each dimension
	except:
		if storeSourceImg: nChannels += 1
		return numpy.zeros( (120,120,120,nChannels) )	# theres a segmented image thats all zero!

	preppedCubes = [prepCube(cube, augment=False) for cube in cubes]


	camImageSize = (nChunks+1) * cubeCamSize
	D, W, H = camImageSize
	camImageSize = (D, W, H, nChannels)

	bigCam = numpy.zeros(camImageSize)
	print 'number of cubes returned: ', len(preppedCubes)

	cs = cubeCamSize

	for gradNum, gradient_function in enumerate(gradFuncs):

		camCubes = gradCamsFromList(preppedCubes, gradient_function)

		for cam, pos in zip(camCubes, positions):
			z, y, x = numpy.asarray(pos) * cubeCamSize
			bigCam[z:z + cs, y:y + cs, x:x + cs, gradNum] = cam

	if saliencyFn:
		bigMap = numpy.zeros((D, W, H))
		saliencyMaps = saliencyMapsFromList(preppedCubes, saliencyFn)

		for sMap, pos in zip(saliencyMaps[0], positions):
			print sMap.shape, sMap.min(), sMap.mean(), sMap.max()
			resizedMap = resize(sMap[:,:,:,0], (cs, cs, cs), preserve_range=True, mode='constant', cval=0)
			print resizedMap.mean()
			z, y, x = numpy.asarray(pos) * cubeCamSize
			bigMap[z:z + cs, y:y + cs, x:x + cs] = resizedMap

		for channel in xrange(nChannels):
			bigCam[:,:,:,channel] = bigCam[:,:,:,channel] * bigMap 

		print 'bigMam', bigMap.min(), bigMap.mean(), bigMap.max()


	if storeSourceImg:
		scam = numpy.zeros((D, W, H, 1)) - 2000
		for cube, pos in zip(cubes, positions):

			sImg = resize(cube, (cs, cs, cs), preserve_range=True, mode='constant', cval=-2000)
			z, y, x = numpy.asarray(pos) * cubeCamSize
			scam[z:z + cs, y:y + cs, x:x + cs, 0] = sImg

		bigCam = numpy.concatenate([bigCam, scam], axis=3)


	print 'bigCam', bigCam.mean()

	return bigCam









def makeCamImgFromImage(image, cubeSize):
	cubes, indexPos = getImageCubes(image, cubeSize, filterBackground=True, prep=False, paddImage=True)
	camImage = makeCamImageFromCubes(cubes, indexPos)
	return camImage



def makeResizedSourceImage(camImg, sourceImage):

	camImageShape = numpy.asarray(camImg.shape)
	zoomDown = 1.0 * camImageShape / numpy.asarray(sourceImage.shape)
	#zoomDown = 0.25
	print zoomDown

	sImage = zoom(sourceImage, zoomDown)

	print sImage.shape, camImg.shape
	paddedCamImg = numpy.zeros(sImage.shape)
	paddedCamImg[:camImageShape[0], :camImageShape[1], :camImageShape[2]] = camImg
	print sImage.shape, paddedCamImg.shape

	paddedCamImg[paddedCamImg < 2.5] = 0.0
	# sImage[sImage<-700] = -700.0

	return paddedCamImg, sImage




def testAndVisualize(DF, array, model, modelFile):

	DF = DF[DF.cancer==1]
	imgRow = DF.iloc[0]


	sparseImages = SparseImageSource('/data/datasets/lung/resampled_order1/segmentedNonzero.h5')
	sDF = sparseImages.DF
	sRow = sDF[sDF.imgNum==imgRow.imgNum].iloc[0]

	cubes, positions = sparseImages.getCubesAndPositions(sRow, posType='pos')


	########    For models trained that predict binary nodule and linear diameter


	#noduleGrad, cubeCamSize  = buildGradientFunction(model)
	#diamGrad, cubeCamSize  = buildGradientFunction(model, output='diam')
	#gradientFunctions = [noduleGrad, diamGrad]



	########    For models trained with softmax nodule+diameter classifier


	smallNoduleGrad, cubeCamSize = buildSoftmaxGradientFunction(model, 1)
	bigNoduleGrad, cubeCamSize = buildSoftmaxGradientFunction(model, 2)
	gradientFunctions = [smallNoduleGrad, bigNoduleGrad]

	# guided backprop functions
	register_gradient()
	guided_model = modify_backprop(model, 'GuidedBackProp', modelFile)
	saliency_fn = compile_saliency_function(guided_model)



	######### Do it and save stuff

	cam = makeCamImageFromCubesFaster(cubes, positions, gradientFunctions, cubeCamSize, storeSourceImg=True)
	#cam = makeCamImageFromCubesFaster(cubes, positions, gradientFunctions, cubeCamSize, saliencyFn=saliency_fn, storeSourceImg=True)

	vol2Nifti(cam[:,:,:,0], 'scam1.nii.gz')
	vol2Nifti(cam[:,:,:,1], 'scam2.nii.gz')
	vol2Nifti(cam[:,:,:,2], 'simg.nii.gz')


	#numpy.save('cam.npy', bigCam)
	#numpy.save('simage.npy', resizedSourceImage)


	crash


	###### Guided backprop tests


	register_gradient()
	guided_model = modify_backprop(model, 'GuidedBackProp', modelFile)
	saliency_fn = compile_saliency_function(guided_model)


	cam = cam[:,:,:,1]

	heatmap = cam / numpy.max(cam) 


	preppedCubes = [prepCube(cube, augment=False) for cube in cubes]

	saliencyCubes = saliency_fn([preppedCubes[0:50], 0])	# works!

	sc = saliencyCubes[0]
	print sc.shape, sc.min(), sc.mean(), sc.max()

	crash

	#inputs = [[cube, 0] for cube in preppedCubes]

	for cube, pos in zip(preppedCubes, positions):

		saliency = saliency_fn([cube, 0])
		print saliency.shape
		gradcam = saliency[0] * heatmap[..., numpy.newaxis]
		print gradcam.min(), gradcam.mean(), gradcam.max()



	'''



	image = sparseImages.getImageFromSparse(imgRow)

	# camImage = makeCamImageFromCubes(cubes, positions, gradientFunctions, cubeCamSize, storeSourceImg=storeSourceImg)
	bigCam = makeCamImageFromCubesFaster(cubes, positions, gradientFunctions, cubeCamSize, storeSourceImg=False)

	vol2Nifti(bigCam[:,:,:,0], 'cam1.nii.gz')
	vol2Nifti(bigCam[:,:,:,1], 'cam2.nii.gz')

	sImg = resize(image, bigCam[:,:,:,0].shape)

	vol2Nifti(sImg, 'simg.nii.gz')

	# now the cam image and source image are roughly aligned to eachother
	paddedCamImg, resizedSourceImage = makeResizedSourceImage(bigCam[:,:,:,1], image)

	vol2Nifti(resizedSourceImage, 'psimg.nii.gz')
	vol2Nifti(paddedCamImg, 'pcimg.nii.gz')



	'''





def buildTrainingSet(DF, inArray, outArray, storeSourceImg=True):


	#DF = DF.head(573).tail(10)

	sparseImages = SparseImageSource(inArray)
	outTsv = outArray.replace('.h5', '.tsv')



	# single output neuron cams
	noduleGrad, cubeCamSize  = buildGradientFunction(model)
	#diamGrad, cubeCamSize  = buildGradientFunction(model, output='diam')
	gradientFunctions = [noduleGrad]



	# softmax output models
	#noduleGrad, cubeCamSize  = buildSoftmaxGradientFunction(model, 0)




	nChannels = len(gradientFunctions)
	if storeSourceImg: nChannels += 1

	CAM_SHAPE = (160, 160, 160, nChannels)

	DBo = tables.open_file(outArray, mode='w')
	filters = tables.Filters(complevel=1, complib='blosc:snappy')      # 7.7sec / 1.2 GB   (14 sec 1015MB if precision is reduced)           140s 3.7GB
	#filters = None
	cams = DBo.create_earray(DBo.root, 'cams', atom=tables.Float32Atom(shape=CAM_SHAPE), shape=(0,), expectedrows=len(DF), filters=filters)

	camImageDF = pandas.DataFrame()
	#camImages = []

	#DF = DF.head(20)


	def cubeDecompressor(Q):
		for index, row in tqdm(DF.iterrows(), total=len(DF)):
			# print row
			row = row.copy(deep=True)
			cancer = row['cancer']
			cubes, positions = sparseImages.getCubesAndPositions(row, posType='pos')
			Q.put((row, cubes, positions))
		Q.put(None)


	Q = Queue(maxsize=4)

	t = Thread(target=cubeDecompressor, args=(Q,))
	t.daemon = True
	t.start()

	while True:
		res = Q.get()
		if res is None:
			Q.task_done()
			break
		row, cubes, positions = res

		#camImage = makeCamImageFromCubes(cubes, positions, gradientFunctions, cubeCamSize, storeSourceImg=storeSourceImg)
		camImage = makeCamImageFromCubesFaster(cubes, positions, gradientFunctions, cubeCamSize, storeSourceImg=storeSourceImg)

		if camImage.mean() == 0: print 'THIS IMAGE IS BAD ========================'

		print camImage.shape


		print 'nodule image    mean %s    min %s    max %s : ' % (
			camImage[:,:,:,0].mean(), camImage[:,:,:,0].min(), camImage[:,:,:,0].max())

		print 'diam image      mean %s    min %s    max %s : ' % (
			camImage[:,:,:,1].mean(), camImage[:,:,:,1].min(), camImage[:,:,:,1].max())

		#print 'source image      mean %s    min %s    max %s : ' % (
		#	camImage[:,:,:,2].mean(), camImage[:,:,:,2].min(), camImage[:,:,:,2].max())

		cam = forceImageIntoShape(camImage, CAM_SHAPE)
		#cam = resize(camImage, CAM_SHAPE)

		#crop = boundingBox(camImage, channel=0)
		print 'Cam shape: ', cam.shape

		cams.append([cam])
		camImageDF = camImageDF.append(row)
		#camImages.append(row)
		camImageDF.to_csv(outTsv, sep='\t')



if __name__ == '__main__':
	import sys
	from keras.models import load_model, Sequential

	DATADIR = '/data/datasets/lung/resampled_order1/'
	tsvFile = DATADIR + 'resampledImages.tsv'
	#arrayFile = DATADIR + 'resampled.h5'
	#arrayFile = DATADIR + 'segmented.h5'


	modelFile = sys.argv[1]
	model = load_model(modelFile)
	_, x,y,z, channels = model.input_shape
	cubeSize = x


	#imgSrc = ImageArray(arrayFile, tsvFile=tsvFile)
	#DF = imgSrc.DF

	DF = pandas.read_csv(tsvFile, sep='\t')


	#testAndVisualize(DF, array, model, modelFile)
	#crash

	#model.compile('sgd', 'binary_crossentropy')

	#DF = DF[DF.imgNum==17]
	#DF = DF.head(5)


	#outArray = '/ssd/cams/cam3/cams.h5'
	outArray = '/ssd/foo.h5'
	#inArray = '/data/datasets/lung/resampled_order1/segmentedNonzero.h5'
	inArray = '/ssd/lung/resampled_order1/segmentedNonzeroLessStrict.h5'

	buildTrainingSet(DF, inArray, outArray)


















