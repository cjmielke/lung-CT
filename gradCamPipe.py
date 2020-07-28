#!/usr/bin/env python2.7
import argparse



import mpipe
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









def buildTrainingSet(DF, inArray, outArray, storeSourceImg=True):

	outTsv = outArray.replace('.h5', '.tsv')

	modelFile = sys.argv[1]


	#nChannels = len(gradientFunctions)
	nChannels = 1
	if storeSourceImg: nChannels += 1

	CAM_SHAPE = (140, 140, 140, nChannels)



	class cubeSource(mpipe.OrderedWorker):
		def doInit(self):
			self.sparseImages = SparseImageSource(inArray)

		def doTask(self, row):
			cubes, positions = self.sparseImages.getCubesAndPositions(row, posType='pos')
			self.putResult((row, cubes, positions))
			print 'returning image'

	class camImgMaker(mpipe.OrderedWorker):
		def doInit(self):
			from keras.models import load_model, Sequential

			from gradCam import register_gradient, modify_backprop, compile_saliency_function, normalize
			from gradCam import target_category_loss, target_category_loss_output_shape, buildGradientFunction
			from gradCam import buildSoftmaxGradientFunction

			from gradCam import batchImages, gradCamsFromList, saliencyMapsFromList, makeCamImageFromCubesFaster
			from gradCam import makeCamImgFromImage, makeResizedSourceImage

			self.model = load_model(modelFile)

			# single output neuron cams
			noduleGrad, cubeCamSize = buildGradientFunction(self.model)
			# diamGrad, cubeCamSize  = buildGradientFunction(model, output='diam')

			self.gradientFunctions = [noduleGrad]
			nChannels = len(self.gradientFunctions)
			global nChannels
			self.cubeCamSize = cubeCamSize

			# softmax output models
			# noduleGrad, cubeCamSize  = buildSoftmaxGradientFunction(model, 0)


		def doTask(self, task):
			row, cubes, positions = task
			from gradCam import makeCamImageFromCubesFaster
			camImage = makeCamImageFromCubesFaster(cubes, positions, self.gradientFunctions, self.cubeCamSize, storeSourceImg=storeSourceImg)


			print 'returning cubes and positions'
			return row, camImage

	class storeCams(mpipe.OrderedWorker):
		def doInit(self):
			DBo = tables.open_file(outArray, mode='w')
			filters = tables.Filters(complevel=1, complib='blosc:snappy')  # 7.7sec / 1.2 GB   (14 sec 1015MB if precision is reduced)           140s 3.7GB
			# filters = None
			self.cams = DBo.create_earray(DBo.root, 'cams', atom=tables.Float32Atom(shape=CAM_SHAPE), shape=(0,),
									 expectedrows=len(DF), filters=filters)

			self.camImageDF = pandas.DataFrame()

		def doTask(self, task):
			row, camImage = task

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
			print cam.shape

			self.cams.append([cam])
			self.camImageDF = self.camImageDF.append(row)
			self.camImageDF.to_csv(outTsv, sep='\t')



	print 'starting workers'
	stage1 = mpipe.Stage(cubeSource, 1, )
	stage2 = mpipe.Stage(camImgMaker, 1)
	stage3 = mpipe.Stage(storeCams, 1, disable_result=True)
	stage1.link(stage2.link(stage3))

	pipe = mpipe.Pipeline(stage1)



	for index, row in tqdm(DF.iterrows(), total=len(DF)):
		print 'putting row', index
		pipe.put(row)

	#for row, camImage in pipe.results():





if __name__ == '__main__':
	import sys


	DATADIR = '/data/datasets/lung/resampled_order1/'
	tsvFile = DATADIR + 'resampledImages.tsv'
	#arrayFile = DATADIR + 'resampled.h5'
	#arrayFile = DATADIR + 'segmented.h5'


	#modelFile = sys.argv[1]
	#model = load_model(modelFile)
	#_, x,y,z, channels = model.input_shape
	#cubeSize = x


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


















