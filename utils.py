import gc
import itertools
import os
import threading

import numpy
import pandas
import tables
from keras import backend as K
from scipy import ndimage

from generators import prepCube


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




def threaded(fn):
	def wrapper(*args, **kwargs):
		thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
		thread.setDaemon(True)
		thread.start()
		return thread
	return wrapper





def boundingBox(img):
	imgMin = img.min()
	img = img - imgMin
	assert img.min() == 0.0, 'Image min should be %s but is %s' % (imgMin, img.min())

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


## Plot volume in 2D

import numpy as np

# Plot one example
#img_crop = np.load('25.npy')
#plot_nodule(img_crop)



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



PIXEL_MEAN = 0.25

def zero_center(image):
	image = image - PIXEL_MEAN
	return image





def findNodules(dataframe, x, y, z, cubeSize):
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

	def __init__(self, arrayFile, tsvFile=None, leafName ='resampled'):
		'''
		Utility function to open pytables arrays containing images and their corresponding pandas dataframes
		:param arrayFile: path to a pytables array - this function will find a tsv file with the same name
		:param leafName: the leaf name in the pytables root
		:return: (pytables_array, pandas_dataframe) 
		'''

		path, f = os.path.split(arrayFile)
		print path
		fileName, ext = os.path.splitext(f)
		assert ext=='.h5'
		print path, fileName

		if not tsvFile: tsvFile = os.path.join(path, fileName) + '.tsv'
		self.DF = pandas.read_csv(tsvFile, sep='\t')

		self.DB = tables.open_file(arrayFile, mode='r')
		self.array = self.DB.root.__getattr__(leafName)

		assert len(self.DF) == len(self.array)

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


def getImageCubes(image, cubeSize, filterBackground=True, expandChannelDim=True):

	# loop over the image, extracting cubes and applying model
	dim = numpy.asarray(image.shape)
	print 'dimension of image: ', dim

	nChunks = dim / cubeSize
	# print 'Number of chunks in each direction: ', nChunks

	positions = [p for p in itertools.product(*map(xrange, nChunks))]

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
		cube = prepCube(cube, augment=False)

		if expandChannelDim: cube = numpy.expand_dims(cube, axis=3)

		cubes.append(cube)
		indexPosL.append(indexPos)

	print 'Rejected %d cubes ' % (len(positions) - len(cubes))
	#assert len(cubes), 'Damn, no cubes. Image stats: %s %s %s ' % (image.min(), image.mean(), image.max())
	#assert len(indexPosL)
	return cubes, indexPosL