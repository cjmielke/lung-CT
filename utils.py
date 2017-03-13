import numpy as np


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