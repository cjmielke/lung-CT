#!/usr/bin/env python2.7


import tables

import cv2
import numpy
import pandas
from joblib import Parallel, delayed
from tqdm import trange

from dsbTests import tsvFile

print(cv2.__version__)


def segmentSlice(slice):

	sliceC = slice.copy()

	# threshold HU > -300
	sliceC[sliceC > -300] = 255
	sliceC[sliceC < -300] = 0
	sliceC = numpy.uint8(sliceC)

	# find surrounding torso from the threshold and make a mask
	im2, contours, _ = cv2.findContours(sliceC, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	#print contours
	largest_contour = max(contours, key=cv2.contourArea)
	mask = numpy.zeros(sliceC.shape, numpy.uint8)
	cv2.fillPoly(mask, [largest_contour], 255)

	# apply mask to threshold image to remove outside. this is our new mask
	sliceC = ~sliceC
	sliceC[(mask == 0)] = 0  # <-- Larger than threshold value

	# apply closing to the mask
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
	sliceC = cv2.morphologyEx(sliceC, cv2.MORPH_OPEN, kernel)  # <- to remove speckles...
	sliceC = cv2.morphologyEx(sliceC, cv2.MORPH_DILATE, kernel)
	sliceC = cv2.morphologyEx(sliceC, cv2.MORPH_DILATE, kernel)
	sliceC = cv2.morphologyEx(sliceC, cv2.MORPH_CLOSE, kernel)
	sliceC = cv2.morphologyEx(sliceC, cv2.MORPH_CLOSE, kernel)
	sliceC = cv2.morphologyEx(sliceC, cv2.MORPH_ERODE, kernel)
	sliceC = cv2.morphologyEx(sliceC, cv2.MORPH_ERODE, kernel)

	# apply mask to image
	img2 = slice.copy()
	img2[(sliceC == 0)] = -2000  # <-- Larger than threshold value

	# closing
	# sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
	# largest_contour = max(contours, key=cv2.contourArea)
	# rgb = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
	# aaa = np.concatenate( sorted_contours[1:3] )
	# cv2.drawContours(rgb, [cv2.convexHull(aaa)], -1, (0,255,0), 3)

	return img2


def segmentVol(vol):

	#newVol = []
	#for slice in vol:
	#	segSlice = segmentSlice(slice)
	#	newVol.append(segSlice)


	newVol = Parallel(n_jobs=7, backend='threading')(delayed(segmentSlice)(slice) for slice in vol)


	newVol = numpy.asarray(newVol)
	return newVol



#imagesDF = pandas.read_csv(DATADIR + 'resampledImages.tsv', sep='\t')

#DB = tables.open_file(DATADIR + 'resampled.h5', mode='r')
#imageArray = DB.root.resampled



# For testing
def testSegmentation():

	DATADIR = '/data/datasets/luna/resampled_order1/'
	tsvFile = DATADIR + 'nodules.tsv'
	arrayFile = DATADIR + 'resampled.h5'

	DB = tables.open_file(arrayFile, mode='r')
	array = DB.root.resampled


	DF = pandas.read_csv(tsvFile, sep='\t')
	row = DF.iloc[200]
	#DF = DF[DF.imgNum==2]		# we've got a specific nodule in mind

	image = array[int(row['imgNum'])]

	image = segmentVol(image)

	voxelC = row[['voxelZ', 'voxelY', 'voxelX']].as_matrix()
	from extractCubes import extractCubeAtLocation
	cube = extractCubeAtLocation(image, voxelC, 40)

	numpy.save('cube.npy', cube)



def segmentCollection(arrayFile, arrayOut):

	DBi = tables.open_file(arrayFile, mode='r')
	array = DBi.root.resampled
	DFi = pandas.read_csv(tsvFile, sep='\t')


	IMG_SHAPE = array[0].shape
	nVols = len(array)



	DBo = tables.open_file(arrayOut, mode='w')
	filters = tables.Filters(complevel=1, complib='blosc:snappy')      # 7.7sec / 1.2 GB   (14 sec 1015MB if precision is reduced)           140s 3.7GB
	images = DBo.create_carray(DBo.root, 'resampled', atom=tables.Int16Atom(shape=IMG_SHAPE), shape=(nVols,), filters=filters)


	print nVols
	for index in trange(nVols):
		vol = array[index]
		seg = segmentVol(vol)
		print 'seg: ', seg.min(), seg.mean(), seg.max()
		#images.append([seg])
		images[index] = seg





if __name__ == '__main__':




	#testSegmentation()
	#sys.exit(0)



	DATADIR = '/data/datasets/lung/resampled_order1/'
	arrayFile = DATADIR + 'resampled.h5'
	arrayOut = DATADIR + 'segmented.h5'

	segmentCollection(arrayFile, arrayOut)

	# luna, not lung! nearly the same but different dataset
	DATADIR = '/data/datasets/luna/resampled_order1/'
	arrayFile = DATADIR + 'resampled.h5'
	arrayOut = DATADIR + 'segmented.h5'

	segmentCollection(arrayFile, arrayOut)


