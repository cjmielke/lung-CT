import gc
import glob
import itertools
import os
import shutil

import tables
import threading

import numpy
import numpy as np
import pandas

from keras import backend as K
from numpy.random import random
from scipy import ndimage
from scipy.ndimage import rotate, affine_transform


class LeafLock():
	def __init__(self, leaf):
		self.leaf = leaf
		self.lock = threading.Lock()
		self.nrows = leaf.nrows

	def __getitem__(self, item):
		#print 'locking'
		with self.lock:
			r = self.leaf[item]
			#print 'unlocking'
			return r







def boundingBox(img, channel=None):
	imgMin = img.min()
	img = img - imgMin
	assert img.min() == 0.0, 'Image min should be %s but is %s' % (imgMin, img.min())

	if channel:
		rows = np.any(img[:,:,:,channel], axis=1)
		cols = np.any(img[:,:,:,channel], axis=0)
		depth = np.any(img[:,:,:,channel], axis=2)
	else:
		rows = np.any(img, axis=1)
		cols = np.any(img, axis=0)
		depth = np.any(img, axis=2)



	rmin, rmax = np.where(rows)[0][[0, -1]]
	cmin, cmax = np.where(cols)[0][[0, -1]]
	dmin, dmax = np.where(depth)[0][[0, -1]]

	#return rmin, rmax, cmin, cmax

	img = img + imgMin
	assert img.min() == imgMin

	crop = img[rmin: rmax, cmin: cmax, dmin: dmax]


	print 'original/cropped sizes : %s / %s ' % (img.shape, crop.shape)
	return crop








def make_mask(center,diam,z,width,height,spacing,origin):
	'''
Center : centers of circles px -- list of coordinates x,y,z
diam : diameters of circles px -- diameter
widthXheight : pixel dim of image
spacing = mm/px conversion rate np array x,y,z
origin = x,y,z mm np.array
z = z position of slice in world coordinates mm
	'''
	mask = np.zeros([height,width]) # 0's everywhere except nodule swapping x,y to match img
	#convert to nodule space from world coordinates

	# Defining the voxel range in which the nodule falls
	v_center = (center-origin)/spacing
	v_diam = int(diam/spacing[0]+5)
	v_xmin = np.max([0,int(v_center[0]-v_diam)-5])
	v_xmax = np.min([width-1,int(v_center[0]+v_diam)+5])
	v_ymin = np.max([0,int(v_center[1]-v_diam)-5]) 
	v_ymax = np.min([height-1,int(v_center[1]+v_diam)+5])

	v_xrange = range(v_xmin,v_xmax+1)
	v_yrange = range(v_ymin,v_ymax+1)

	# Convert back to world coordinates for distance calculation
	x_data = [x*spacing[0]+origin[0] for x in range(width)]
	y_data = [x*spacing[1]+origin[1] for x in range(height)]

	# Fill in 1 within sphere around nodule
	for v_x in v_xrange:
		for v_y in v_yrange:
			p_x = spacing[0]*v_x + origin[0]
			p_y = spacing[1]*v_y + origin[1]
			if np.linalg.norm(center-np.array([p_x,p_y,z]))<=diam:
				mask[int((p_y-origin[1])/spacing[1]),int((p_x-origin[0])/spacing[0])] = 1.0
	return(mask)


def matrix2int16(matrix):
	''' 
	matrix must be a numpy array NXN
	Returns uint16 version
	'''
	m_min= np.min(matrix)
	m_max= np.max(matrix)
	matrix = matrix-m_min
	return(np.array(np.rint( (matrix-m_min)/float(m_max-m_min) * 65535.0),dtype=np.uint16))




'''
This function is used to convert the world coordinates to voxel coordinates using 
the origin and spacing of the ct_scan
'''
def world2voxel(world_coordinates, origin, spacing):
	stretched_voxel_coordinates = np.absolute(world_coordinates - origin)
	voxel_coordinates = stretched_voxel_coordinates / spacing
	return voxel_coordinates

'''
This function is used to convert the voxel coordinates to world coordinates using 
the origin and spacing of the ct_scan.
'''
def voxel2world(voxel_coordinates, origin, spacing):
	stretched_voxel_coordinates = voxel_coordinates * spacing
	world_coordinates = stretched_voxel_coordinates + origin
	return world_coordinates







def normalizeStd(arr):
	norm1 = arr / numpy.linalg.norm(arr)
	return norm1


# normalize methods found in the preprocessing tutorial
def normalizeRange(image, MIN_BOUND=-1000.0, MAX_BOUND=400.0):
	image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
	image[image>1] = 1.
	image[image<0] = 0.
	return image



def zero_center(image):
	PIXEL_MEAN = 0.25
	image = image - PIXEL_MEAN
	return image





def findNodules(dataframe, x, y, z, cubeSize):
	"""
	Find nodules in a dataframe based on voxel coordinates
	"""
	nodulesInCube = dataframe[
		(dataframe.voxelZ > z) & (dataframe.voxelZ < z + cubeSize) &
		(dataframe.voxelY > y) & (dataframe.voxelY < y + cubeSize) &
		(dataframe.voxelX > x) & (dataframe.voxelX < x + cubeSize)]
	return nodulesInCube


def resample(image, spacing, new_spacing=[3,3,3], order=1):
	resize_factor = spacing / new_spacing
	new_real_shape = image.shape * resize_factor
	new_shape = np.round(new_real_shape)
	real_resize_factor = new_shape / image.shape
	new_spacing = spacing / real_resize_factor

	image = ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest', order=order)

	return image, new_spacing


class ImageArray():

	def __init__(self, arrayFile, tsvFile=None, leafName ='resampled', threadSafe=False):
		"""
		Utility function to open pytables arrays containing images and their corresponding pandas dataframes
		:param arrayFile: path to a pytables array - this function will find a tsv file with the same name
		:param leafName: the leaf name in the pytables root
		:return: (pytables_array, pandas_dataframe) 
		"""


		path, f = os.path.split(arrayFile)
		#print path
		fileName, ext = os.path.splitext(f)
		assert ext=='.h5'
		#print path, fileName

		if not tsvFile: tsvFile = os.path.join(path, fileName) + '.tsv'
		self.DF = pandas.read_csv(tsvFile, sep='\t')

		self.DB = tables.open_file(arrayFile, mode='r')
		self.array = self.DB.root.__getattr__(leafName)

		if threadSafe: self.array = LeafLock(self.array)

		assert len(self.DF) == len(self.array), 'DataFrame has %s rows but array has %s images' % (len(self.DF), len(self.array))

		self.nImages = len(self.DF)



def getImage(imageArray, row, convertType=True):
	imageNum = row['imgNum']
	image = imageArray[int(imageNum)]

	if convertType:
		if image.dtype != K.floatx(): image = image.astype(K.floatx())

	Z, Y, X = row[['shapeZ', 'shapeY', 'shapeX']].as_matrix()
	#print int(Z), Y, X
	image = image[:int(Z), :int(Y), :int(X)]

	#print image.min(), image.mean(), image.max()
	#print normImg.min(), normImg.mean(), normImg.max()
	gc.collect()
	return image, imageNum











def getImageCubes(image, cubeSize, filterBackground=True, prep=None, paddImage=False):

	assert prep is not None, 'You must explicitly tell me if I should prep cubes for the model, or if these are RAW'



	# loop over the image, extracting cubes and applying model
	dim = numpy.asarray(image.shape)
	print 'dimension of image: ', dim

	nChunks = dim / cubeSize
	# print 'Number of chunks in each direction: ', nChunks

	'''
	if paddImage:
		gap = dim - nChunks*cubeSize
		pad = [(0,p+1) for p in gap]
		image = numpy.pad(image, pad, mode='constant')
		dim = numpy.asarray(image.shape)
		nChunks = dim / cubeSize		# recompute
	'''

	if paddImage:
		gap = dim - nChunks*cubeSize
		newSize = dim + gap + 1
		newImage = numpy.zeros(newSize)
		newImage[:dim[0], :dim[1], :dim[2]] = image
		image = newImage
		dim = numpy.asarray(image.shape)
		nChunks = dim / cubeSize		# recompute


	print 'dimension of image: ', image.shape



	positions = [p for p in itertools.product(*map(xrange, nChunks))]
	print len(positions)

	#if filterBackground: image[image<-1000] = -1000			# dont let weird background values bias things


	cubes = []
	indexPosL = []
	for pos in positions:
		indexPos = numpy.asarray(pos)
		realPos = indexPos*cubeSize
		z, y, x = realPos

		cube = image[z:z + cubeSize, y:y + cubeSize, x:x + cubeSize]
		assert cube.shape == (cubeSize, cubeSize, cubeSize)

		if filterBackground:
			if cube.mean() <= -1000: continue

		# apply same normalization as in training
		if prep: cube = prepCube(cube, augment=False)

		cubes.append(cube)
		indexPosL.append(indexPos)

	print 'Rejected %d cubes ' % (len(positions) - len(cubes))
	#assert len(cubes), 'Damn, no cubes. Image stats: %s %s %s ' % (image.min(), image.mean(), image.max())
	#assert len(indexPosL)
	return cubes, indexPosL






def augmentCube(cube):

	affine = numpy.eye(3) + 0.9*(random((3,3))-0.5)
	#print affine

	#return ebuc
	cube = rotate(cube, 360*random(), axes=(0,1), reshape=False)
	cube = rotate(cube, 360*random(), axes=(1,2), reshape=False)

	cube = affine_transform(cube, affine)

	return cube


def prepCube(cube, augment=True, cubeSize=None):
	#cubeSize = cube.shape[0]
	#cube = normalizeStd(cube)
	cube = normalizeRange(cube,MAX_BOUND=500.0)
	#cube = normalizeRange(cube,MAX_BOUND=1000.0)
	size = cube.shape[0]

	if cubeSize and cubeSize!=size:
		p = size-cubeSize
		p = p/2
		cube = cube[p:p+cubeSize, p:p+cubeSize, p:p+cubeSize]
		assert cube.shape == (cubeSize, cubeSize, cubeSize)

	if augment: cube = augmentCube(cube)

	cube = numpy.expand_dims(cube, axis=3)		# insert dummy channel

	return cube


def convertColsToInt(DF, columns):
	for col in columns:
		DF[col] = DF[col].astype('int')
		return DF


def forceImageIntoShape(image, DESIRED_SHAPE):

	resized = np.zeros(DESIRED_SHAPE, dtype='int16')

	xCopy = min(DESIRED_SHAPE[0], image.shape[0])
	yCopy = min(DESIRED_SHAPE[1], image.shape[1])
	zCopy = min(DESIRED_SHAPE[2], image.shape[2])

	if len(image.shape)==3:
		resized[:xCopy, :yCopy, :zCopy] = image[:xCopy, :yCopy, :zCopy]
	elif len(image.shape)==4:
		resized[:xCopy, :yCopy, :zCopy, :] = image[:xCopy, :yCopy, :zCopy, :]

	return resized


def prepOutDir(OUTDIR, file):

	if not os.path.exists(OUTDIR): os.makedirs(OUTDIR)
	codeDir = OUTDIR + 'snap'
	if not os.path.exists(codeDir): os.makedirs(codeDir)

	thisDir = os.path.dirname(os.path.realpath(__file__))
	print thisDir
	for f in glob.glob(thisDir+ '/*.py'):
		print f
		shutil.copy(f, codeDir)



