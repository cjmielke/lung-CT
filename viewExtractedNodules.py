#!/usr/bin/env python2.7

import os
import sys
from cv2 import imshow, waitKey

import numpy
import pandas
import tables


from utils import normalizeRange
from viz import grey2color, plot_nodule, mayaPlot
import time
from mayavi import mlab
from scipy.stats import mode

import itertools

#dataset_file = sys.argv[1]
#df = pandas.read_csv(dataset_file, sep='\t')
#path, file = os.path.split(dataset_file)
#arrFile = os.path.join(path, 'resampled.h5')

arrFile = sys.argv[1]





from skimage.morphology import watershed
from skimage import morphology
from skimage.feature import peak_local_max
from scipy import ndimage



def segmentNodule(image, threshold=-500):

	cube = numpy.copy(image)

	# apply make anything that looks like lung, the background
	cube[cube<threshold] = 0
	#cube += -threshold		# sets background to 0

	distance = ndimage.distance_transform_edt(cube)
	#local_maxi = peak_local_max(distance, indices=False, footprint=numpy.ones((3, 3, 3)), labels=cube)
	local_maxi = peak_local_max(distance, indices=False, footprint=numpy.ones((6, 6, 6)))
	#local_maxi = peak_local_max(distance, indices=False, min_distance=8, labels=(cube>0)*1)
	markers = morphology.label(local_maxi)
	labels_ws = watershed(-distance, markers, mask=cube)


	# determine dominant label # in core of this box
	cubeSize = cube.shape[0]
	r=cubeSize/2
	core = labels_ws[r-2:r+2, r-2:r+2, r-2:r+2]
	M = mode(core.flatten())
	dominantLabel = M.mode[0]
	print dominantLabel

	mask = (labels_ws==dominantLabel)
	masked = numpy.zeros(image.shape)

	print mask.sum(), cubeSize**3

	masked[mask] = image[mask]


	if masked[0,0,0] != 0.0 or masked[-1,0,0]!=0.0:
		print 'bad segmentation'
		masked*=0
		labels_ws*=0


	return labels_ws, masked









#arrFile = os.path.join(path, 'resampledCarray.h5')
#arrFile = os.path.join('/data/datasets/luna/resampled.h5')

DB = tables.open_file(arrFile, mode='r')
cubes = DB.root.cubes

cubeSize = cubes[0].shape[0]
print cubeSize



#from generators import prepVGGimages

bigArrSize = 3*cubeSize

arr = numpy.zeros((bigArrSize,bigArrSize,bigArrSize))

#nChunks = (3,3,3)
nChunks = numpy.asarray(arr.shape)/cubeSize
print nChunks


for index, pos in enumerate(itertools.product(*map(xrange, nChunks))):
	print pos

	pos = numpy.asarray(pos)
	pos *= cubeSize
	z, y, x = pos

	print pos

	cube = cubes[index]
	labels, seg = segmentNodule(cube, threshold=-600)

	arr[z:z+cubeSize, y: y+cubeSize, x: x+cubeSize] = labels
	#arr[z:z+cubeSize, y: y+cubeSize, x: x+cubeSize] = seg
	#arr[z:z + cubeSize, y: y + cubeSize, x: x + cubeSize] = cube







def viewNodule(arr):

	src = mlab.pipeline.scalar_field(arr)
	mlab.pipeline.iso_surface(src, name='bone', contours=[500, 1200], opacity=1.0, color=(1, 1, 1))
	mlab.pipeline.iso_surface(src, name='lung', contours=[-700, -600], opacity=0.3)
	# mlab.pipeline.iso_surface(src, name='fat', contours=[-100, -50], opacity=0.3, color=(1,0.5,0))
	# mlab.pipeline.iso_surface(src, name='other', contours=[50, 300], opacity=0.7, color=(0,1.0,0))
	mlab.show(stop=False)
	mlab.show_pipeline(engine=None, rich_view=True)


# response = raw_input("hit enter to continue or ctrl-c to quit")
# time.sleep(3)
# mlab.pipeline.iso_surface(src, contours=[s.max()-0.1*s.ptp(), ],)


def viewImgMaya(arr):
	src = mlab.pipeline.scalar_field(arr)
	#mlab.pipeline.iso_surface(src, name='bone', contours=[500, 1200], opacity=1.0, color=(1, 1, 1))
	#mlab.pipeline.iso_surface(src, name='lung', contours=[-700, -600], opacity=0.3)
	# mlab.pipeline.iso_surface(src, name='fat', contours=[-100, -50], opacity=0.3, color=(1,0.5,0))
	# mlab.pipeline.iso_surface(src, name='other', contours=[50, 300], opacity=0.7, color=(0,1.0,0))
	mlab.show(stop=False)
	mlab.show_pipeline(engine=None, rich_view=True)


# response = raw_input("hit enter to continue or ctrl-c to quit")
# time.sleep(3)
# mlab.pipeline.iso_surface(src, contours=[s.max()-0.1*s.ptp(), ],)




viewNodule(arr)




