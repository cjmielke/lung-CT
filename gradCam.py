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
from utils import getImage, getImageCubes, ImageArray, prepCube, forceImageIntoShape



CAM_SHAPE = (150, 150, 150)


def normalize(x): # utility function to normalize a tensor by its L2 norm
	return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def buildGradientFunction(model, output='nodule', poolLayer = 'pool2'):

	loss_layer = model.get_layer(output)
	loss = K.sum(loss_layer.output)
	outLayer = model.get_layer(poolLayer)
	conv_output = outLayer.output
	grads = normalize(K.gradients(loss, conv_output)[0])

	gradient_function = K.function(
		[model.layers[0].input, K.learning_phase()],
		[conv_output, grads]
	)
	gradShape = outLayer.output_shape[1:4]			# (None, 8, 8, 8, 64) - want to reject instances and channels shapes

	return gradient_function, gradShape[0]			# assuming cube ...



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




def makeCamImageFromCubes(cubes, indexPos):

	try: nChunks = numpy.asarray(indexPos).max(axis=0)		# get maximum position in each dimension
	except: nChunks = numpy.asarray([1,1,1])			# one image failed segmentation and has no associated cubes :(
													# this should just return a blank cam image

	camImageSize = (nChunks+1) * cubeCamSize



	bigCam = numpy.zeros(camImageSize)
	print 'number of cubes returned: ', len(cubes)

	for cube, pos in zip(cubes, indexPos):
		cube = prepCube(cube, augment=False)
		cam = grad_cam([cube], gradient_function)

		#cube = numpy.expand_dims(cube, axis=0)
		#predictions = model.predict([cube])
		#noduleScore, diam, decodedImg = predictions
		#print 'score =========', noduleScore
		#cam = cam * noduleScore

		z, y, x = numpy.asarray(pos) * cubeCamSize
		cs = cubeCamSize
		#print pos, z, y, x
		#print 'cam mean: ', cam.mean()
		bigCam[z:z + cs, y:y + cs, x:x + cs] = cam

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




def buildTrainingSet(DF, segImages):

	sparseImages = SparseImageSource('/data/datasets/lung/resampled_order1/segmentedNonzero.h5')

	outArray = '/ssd/camsB.h5'
	outTsv = outArray.replace('.h5', '.tsv')

	camImageDF = pandas.DataFrame()



	DBo = tables.open_file(outArray, mode='w')
	filters = tables.Filters(complevel=6, complib='blosc:snappy')      # 7.7sec / 1.2 GB   (14 sec 1015MB if precision is reduced)           140s 3.7GB
	#filters = None
	cams = DBo.create_earray(DBo.root, 'cams', atom=tables.Int16Atom(shape=CAM_SHAPE), shape=(0,), expectedrows=len(DF), filters=filters)




	for index, row in tqdm(DF.iterrows(), total=len(DF)):
		print row
		cancer = row['cancer']

		# slow
		#image, imgNum = getImage(segImages, row)
		#camImage = makeCamImgFromImage(image, cubeSize)

		# faster
		#image = sparseImages.getImageFromSparse(row)
		#camImage = makeCamImgFromImage(image, cubeSize)

		# should be fastest
		cubes, positions = sparseImages.getCubesAndPositions(row, posType='pos')
		camImage = makeCamImageFromCubes(cubes, positions)


		print 'CAM IMAGE SHAPE %s    mean %s   max %s     ==========', camImage.shape, camImage.mean(), camImage.max()

		if camImage.mean() == 0: print 'THIS IMAGE IS BAD ========================'


		cam = forceImageIntoShape(camImage, CAM_SHAPE)

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

	gradient_function, cubeCamSize  = buildGradientFunction(model)


	#model.compile('sgd', 'binary_crossentropy')

	#DF = DF[DF.imgNum==17]
	#DF = DF.head(5)

	buildTrainingSet(DF, array)















