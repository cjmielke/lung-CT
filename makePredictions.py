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


	arrayFile = '/ssd/cams.h5'
	imgSrc = ImageArray(arrayFile)
	DF, array = imgSrc.DF, imgSrc.array

	modelFile = sys.argv[1]
	model = load_model(modelFile)
	_, x,y,z, channels = model.input_shape
	camSize = x





	DF, array = imgSrc.DF, imgSrc.array

	DF = DF[DF.cancer==-1]
	print len(DF)
















