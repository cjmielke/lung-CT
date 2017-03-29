import threading

import numpy
from scipy import ndimage


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

MIN_BOUND = -1000.0
MAX_BOUND = 400.0

def normalizeRange(image):
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