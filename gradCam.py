#!/usr/bin/env python2.7
import argparse

import keras.backend as K
import numpy
import pandas
import tables
from scipy.stats import describe
from scipy.ndimage import zoom
from tqdm import tqdm

from convertToNifti import vol2Nifti
from extractNonzeroCubes import SparseImageSource
from utils import getImage, getImageCubes, ImageArray, prepCube, forceImageIntoShape, boundingBox

from skimage.transform import resize

CAM_SHAPE = (150, 150, 150)


def normalize(x): # utility function to normalize a tensor by its L2 norm
	return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def buildGradientFunction(model, output='nodule', poolLayer = 'pool2'):
	loss = K.sum(model.get_layer(output).output)

	outLayer = model.get_layer(poolLayer)
	conv_output = outLayer.output
	grads = normalize(K.gradients(loss, conv_output)[0])

	gradient_function = K.function(
		[model.layers[0].input, K.learning_phase()],
		[conv_output, grads]
	)
	gradShape = outLayer.output_shape[1:4]			# (None, 8, 8, 8, 64) - want to reject instances and channels shapes

	return gradient_function, gradShape[0]			# assuming cube ...



def grad_cam(imageListWithSingleImage, gradient_function): 

	output, grads_val = gradient_function([imageListWithSingleImage, 0])
	output, grads_val = output[0, :], grads_val[0, :, :, :]

	weights = numpy.mean(grads_val, axis = (0, 1, 2))
	cam = numpy.ones(output.shape[0 : 3], dtype = numpy.float32)

	for i, w in enumerate(weights): cam += w * output[:, :, :, i]

	return cam





def gradCamsFromList(images, gradient_function): 

	cams = []

	def batchImages(iterable, n=200):
		l = len(iterable)
		for i in range(0, l, n): yield iterable[i: min(i + n, l)]


	for imgBatch in batchImages(images, n=500):

		outputs, gradsVals = gradient_function([imgBatch,0])

		for output, grads in zip(outputs, gradsVals):

			weights = numpy.mean(grads, axis = (0, 1, 2))
			cam = numpy.ones(output.shape[0 : 3], dtype = numpy.float32)

			for i, w in enumerate(weights): cam += w * output[:, :, :, i]

			cams.append(cam)

	return cams







def makeCamImageFromCubes(cubes, indexPos, gradientFunctions, cubeCamSize, storeSourceImg=False):

	try: nChunks = numpy.asarray(indexPos).max(axis=0)		# get maximum position in each dimension
	except: nChunks = numpy.asarray([1,1,1])			# one image failed segmentation and has no associated cubes :(
													# this should just return a blank cam image


	camImageSize = (nChunks+1) * cubeCamSize
	x, y, z = camImageSize
	camImageSize = (x, y, z, len(gradientFunctions))

	bigCam = numpy.zeros(camImageSize)
	print 'number of cubes returned: ', len(cubes)

	for cube, pos in zip(cubes, indexPos):
		cube = prepCube(cube, augment=False)

		#cube = numpy.expand_dims(cube, axis=0)
		#predictions = model.predict([cube])
		#noduleScore, diam, decodedImg = predictions
		#print 'score =========', noduleScore
		#cam = cam * noduleScore

		cs = cubeCamSize
		for gradNum, gradient_function in enumerate(gradientFunctions):

			cam = grad_cam([cube], gradient_function)

			z, y, x = numpy.asarray(pos) * cubeCamSize
			#print pos, z, y, x
			#print 'cam mean: ', cam.mean(axis=3), cam.shape
			bigCam[z:z + cs, y:y + cs, x:x + cs, gradNum] = cam

		if storeSourceImg:
			sImg = resize(cube, (cs, cs, cs))
			bigCam[z:z + cs, y:y + cs, x:x + cs, -1] = sImg


	return bigCam






def makeCamImageFromCubesFaster(cubes, positions, gradFuncs, cubeCamSize, storeSourceImg=False):

	nChannels = len(gradFuncs)
	if storeSourceImg: nChannels += 1


	try: nChunks = numpy.asarray(positions).max(axis=0)		# get maximum position in each dimension
	except: return numpy.zeros( (120,120,120,nChannels) )	# theres a segmented image thats all zero!

	cubes = [prepCube(cube, augment=False) for cube in cubes]


	camImageSize = (nChunks+1) * cubeCamSize
	x, y, z = camImageSize
	camImageSize = (x, y, z, nChannels)

	bigCam = numpy.zeros(camImageSize)
	print 'number of cubes returned: ', len(cubes)

	cs = cubeCamSize

	for gradNum, gradient_function in enumerate(gradFuncs):

		camCubes = gradCamsFromList(cubes, gradient_function)

		for cam, pos in zip(camCubes, positions):
			z, y, x = numpy.asarray(pos) * cubeCamSize
			bigCam[z:z + cs, y:y + cs, x:x + cs, gradNum] = cam


	if storeSourceImg:

		for cube, pos in zip(cubes, positions):

			sImg = resize(cube[:,:,:,0], (cs, cs, cs), preserve_range=True)
			z, y, x = numpy.asarray(pos) * cubeCamSize
			bigCam[z:z + cs, y:y + cs, x:x + cs, -1] = sImg


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

	return paddedCamImg




def testAndVisualize(DF, array):

	gradient_function, cubeCamSize  = buildGradientFunction(model)

	DF = DF[DF.cancer==1].head(1)
	row = DF.iloc[0]

	image, imgNum = getImage(array, row)
	imgShape = numpy.asarray(image.shape)

	bigCam = makeCamImgFromImage(image, cubeSize)

	# now the cam image and source image are roughly aligned to eachother
	paddedCamImg, resizedSourceImage = makeResizedSourceImage(bigCam, image)

	affine = numpy.eye(4)
	vol2Nifti(paddedCamImg, 'cam.nii.gz', affine=affine)
	vol2Nifti(resizedSourceImage, 'simg.nii.gz', affine=affine)

	numpy.save('cam.npy', bigCam)
	numpy.save('simage.npy', resizedSourceImage)

	'''
	zoomUp = imgShape / camImageShape
	zoomUp = 0.3 * zoomUp
	bigCam = zoom(bigCam, zoomUp)
	print image.shape, bigCam.shape

	numpy.save('bigcam.npy', bigCam)
	numpy.save('image.npy', image)

	bigCam = bigCam.astype('float32')
	print bigCam.dtype

	s = zoomUp
	'''




def buildTrainingSet(DF, inArray, outArray, storeSourceImg=True):
	sparseImages = SparseImageSource(inArray)
	outTsv = outArray.replace('.h5', '.tsv')

	noduleGrad, cubeCamSize  = buildGradientFunction(model)
	diamGrad, cubeCamSize  = buildGradientFunction(model, output='diam')
	gradientFunctions = [noduleGrad, diamGrad]

	nChannels = len(gradientFunctions)
	if storeSourceImg: nChannels += 1

	CAM_SHAPE = (140, 140, 140, nChannels)

	DBo = tables.open_file(outArray, mode='w')
	filters = tables.Filters(complevel=3, complib='blosc:snappy')      # 7.7sec / 1.2 GB   (14 sec 1015MB if precision is reduced)           140s 3.7GB
	#filters = None
	cams = DBo.create_earray(DBo.root, 'cams', atom=tables.Int16Atom(shape=CAM_SHAPE), shape=(0,), expectedrows=len(DF), filters=filters)

	camImageDF = pandas.DataFrame()



	for index, row in tqdm(DF.iterrows(), total=len(DF)):
		#print row
		cancer = row['cancer']

		# slow
		#image, imgNum = getImage(segImages, row)
		#camImage = makeCamImgFromImage(image, cubeSize)
		# faster
		#image = sparseImages.getImageFromSparse(row)
		#camImage = makeCamImgFromImage(image, cubeSize)

		# should be fastest
		cubes, positions = sparseImages.getCubesAndPositions(row, posType='pos')
		#camImage = makeCamImageFromCubes(cubes, positions, gradientFunctions, cubeCamSize, storeSourceImg=storeSourceImg)
		camImage = makeCamImageFromCubesFaster(cubes, positions, gradientFunctions, cubeCamSize, storeSourceImg=storeSourceImg)

		#print 'CAM IMAGE SHAPE %s    mean %s   max %s     ==========' % (camImage.shape, camImage.mean(), camImage.max())


		if camImage.mean() == 0: print 'THIS IMAGE IS BAD ========================'




		'''
		if storeSourceImg:
			sourceImg = sparseImages.getImageFromSparse(row)
			x, y, z, c = camImage.shape
			rImage = resize(sourceImg, (x, y, z))
			rImage = numpy.expand_dims(rImage, axis=3)
			print camImage.shape, rImage.shape
			camImage = numpy.concatenate((camImage, rImage), axis=3)
		'''

		print camImage.shape




		print 'nodule image    mean %s    min %s    max %s : ' % (
			camImage[:,:,:,0].mean(), camImage[:,:,:,0].min(), camImage[:,:,:,0].max())

		print 'diam image      mean %s    min %s    max %s : ' % (
			camImage[:,:,:,1].mean(), camImage[:,:,:,1].min(), camImage[:,:,:,1].max())

		print 'source image      mean %s    min %s    max %s : ' % (
			camImage[:,:,:,2].mean(), camImage[:,:,:,2].min(), camImage[:,:,:,2].max())





		# here, the source image has been resized to fit the size of the CAM image
		# the cam images will have slightly different sizes depending on the source images,
		# since peoples lungs are different size

		# next, we either put this image into a biggish box by padding, or we resize

		cam = forceImageIntoShape(camImage, CAM_SHAPE)
		#cam = resize(camImage, CAM_SHAPE)


		#crop = boundingBox(camImage, channel=0)
		print cam.shape


		cams.append([cam])
		camImageDF = camImageDF.append(row)
		camImageDF.to_csv(outTsv, sep='\t')



if __name__ == '__main__':
	import sys
	from keras.models import load_model


	DATADIR = '/data/datasets/lung/resampled_order1/'
	tsvFile = DATADIR + 'resampledImages.tsv'
	#arrayFile = DATADIR + 'resampled.h5'
	arrayFile = DATADIR + 'segmented.h5'

	modelFile = sys.argv[1]
	model = load_model(modelFile)
	_, x,y,z, channels = model.input_shape
	cubeSize = x


	imgSrc = ImageArray(arrayFile, tsvFile=tsvFile)
	DF, array = imgSrc.DF, imgSrc.array



	#testAndVisualize(DF, array)
	#crash

	#model.compile('sgd', 'binary_crossentropy')

	#DF = DF[DF.imgNum==17]
	#DF = DF.head(5)


	outArray = '/ssd/camsB.h5'
	inArray = '/data/datasets/lung/resampled_order1/segmentedNonzero.h5'

	buildTrainingSet(DF, inArray, outArray)


















