
# Settings
import itertools

import pandas

from utils import make_mask

LUNA_DATA_PATH = "/data/datasets/luna/"
luna_subset_path = LUNA_DATA_PATH + "subset0/"

output_path = "/data/datasets/luna/preproc/"


RESAMPLED_IMG_SHAPE = (750, 750, 750)


import SimpleITK as sitk
import numpy as np
from scipy import ndimage
from glob import glob
import os
import pandas as pd

from tqdm import tqdm


#file_list = glob(luna_subset_path+"*.mhd")
file_list = glob(LUNA_DATA_PATH + "subset?/*.mhd")


import tables
tables.set_blosc_max_threads(4)
filters = tables.Filters(complevel=1, complib='blosc:lz4')      # 7.7sec / 1.2 GB   (14 sec 1015MB if precision is reduced)           140s 3.7GB

DB = tables.open_file('/data/datasets/luna/resampled.h5', mode='w', filters=None)
images = DB.create_earray(DB.root, 'resampled', atom=tables.Int16Atom(shape=RESAMPLED_IMG_SHAPE), shape=(0,), expectedrows=len(file_list), filters=filters)


# Helper function to get rows in data frame associated  with each file
def get_filename(file_list, case):
	for f in file_list:
		if case in f:
			return(f)

# The locations of the nodules
noduleDF = pd.read_csv(LUNA_DATA_PATH + "annotations.csv")
noduleDF["file"] = noduleDF["seriesuid"].map(lambda file_name: get_filename(file_list, file_name))
#noduleDF = noduleDF.dropna()


candidatesDF = pd.read_csv(LUNA_DATA_PATH + "candidates.csv")
candidatesDF["file"] = candidatesDF["seriesuid"].map(lambda file_name: get_filename(file_list, file_name))
#candidatesDF = candidatesDF.dropna()



#crash


def resample(image, spacing, new_spacing=[3,3,3], order=1):
	resize_factor = spacing / new_spacing
	new_real_shape = image.shape * resize_factor
	new_shape = np.round(new_real_shape)
	real_resize_factor = new_shape / image.shape
	new_spacing = spacing / real_resize_factor

	image = ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest', order=order)

	return image, new_spacing




# this dataframe just stores seriesUIDs .... index is auto-incrementing and follows pytables index
#imageDF = pandas.DataFrame(columns=['seriesuid'])
imageDF = pandas.DataFrame(columns=noduleDF.columns)


for fcount, img_file in enumerate(tqdm(file_list)):

	mini_df = noduleDF[noduleDF["file"] == img_file]			#get all nodules associate with file

	itk_img = sitk.ReadImage(img_file)
	img_array = sitk.GetArrayFromImage(itk_img)			# indexes are z,y,x (notice the ordering)
	num_z, height, width = img_array.shape				# heightXwidth constitute the transverse plane
	print 'image shape: ', img_array.shape
	origin = np.array(itk_img.GetOrigin())				# x,y,z  Origin in world coordinates (mm)
	spacing = np.array(itk_img.GetSpacing())			# spacing of voxels in world coor. (mm)
	#print 'image spacing: ', spacing
	spacing = np.asarray((spacing[2], spacing[0], spacing[1]))
	print 'image spacing: ', spacing

	# resample
	resampled, spacing = resample(img_array, spacing, new_spacing=[0.6, 0.6, 0.6])
	print 'resampled array: ', resampled.shape, spacing
	#resampled = img_array


	# TODO, store resampled array into pytables
	# TODO, also store slices in pytables that are just resampled along 2D plane ... IE, for 2D ML without depth interpolation


	# store mapping between pytables index and seriesuid

	mini_df['pytablesIndex'] = fcount
	mini_df['resampledX'] = spacing[0]
	mini_df['resampledY'] = spacing[1]
	mini_df['resampledZ'] = spacing[2]

	imageDF = imageDF.append(mini_df)
	imageDF.to_csv('/data/datasets/luna/resampledNodules.tsv', sep='\t')
	resized = np.zeros(RESAMPLED_IMG_SHAPE, dtype='int16')

	xCopy = min(RESAMPLED_IMG_SHAPE[0], resampled.shape[0])
	yCopy = min(RESAMPLED_IMG_SHAPE[1], resampled.shape[1])
	zCopy = min(RESAMPLED_IMG_SHAPE[2], resampled.shape[2])
	resized[:xCopy, :yCopy, :zCopy] = resampled[:xCopy, :yCopy, :zCopy]

	images.append([resized])  # get the remaining ones


	# FIXME - all this stuff should be downstream, coming out of pytables ...











def foo():

	# iterate a cube through the image
	CUBE_SIZE = 32
	'''
	C = CUBE_SIZE
	i=j=k=0
	while i<resampled.shape[0]:
		while j < resampled.shape[0]:
			while k < resampled.shape[0]:
				cube = resampled[i:i+C, j:j+C, k:k+C]
				k+=C
			j+=C
		i+=C
	'''

	dim = np.asarray(resampled.shape)
	print 'dimension of image: ', dim

	nChunks = dim/CUBE_SIZE
	print 'Number of chunks in each direction: ', nChunks


	for pos in itertools.product(*map(xrange, nChunks)):
		pos = np.asarray(pos)
		#print pos
		pos*=CUBE_SIZE
		x, y, z = pos
		print x,y,z
		cube = resampled[x:x+CUBE_SIZE, y:y+CUBE_SIZE, z:z+CUBE_SIZE]
		assert cube.shape == (CUBE_SIZE,CUBE_SIZE,CUBE_SIZE)

		# TODO - create lookup method that determines if nodule is present
		# nodulePresent = findNodules(x,y,z,CUBE_SIZE) 



	crash

	# go through all nodes (why just the biggest?)
	for node_idx, cur_row in mini_df.iterrows():
		node_x = cur_row["coordX"]
		node_y = cur_row["coordY"]
		node_z = cur_row["coordZ"]
		diam = cur_row["diameter_mm"]

		# just keep 3 slices
		imgs = np.ndarray([3, height, width], dtype=np.float32)
		masks = np.ndarray([3, height, width], dtype=np.uint8)
		center = np.array([node_x, node_y, node_z])  # nodule center
		v_center = np.rint((center - origin) / spacing)  # nodule center in voxel space (still x,y,z ordering)
		for i, i_z in enumerate(np.arange(int(v_center[2]) - 1,
										  int(v_center[2]) + 2).clip(0,
																	 num_z - 1)):  # clip prevents going out of bounds in Z
			mask = make_mask(center, diam, i_z * spacing[2] + origin[2], width, height, spacing, origin)
			masks[i] = mask
			imgs[i] = img_array[i_z]
		np.save(os.path.join(output_path, "images_%04d_%04d.npy" % (fcount, node_idx)), imgs)
		np.save(os.path.join(output_path, "masks_%04d_%04d.npy" % (fcount, node_idx)), masks)
















#####
#
# Looping over the image files
#
for fcount, img_file in enumerate(tqdm(file_list)):

	mini_df = noduleDF[noduleDF["file"] == img_file] #get all nodules associate with file

	if mini_df.shape[0]>0: # some files may not have a nodule--skipping those 

		# load the data once
		itk_img = sitk.ReadImage(img_file) 
		img_array = sitk.GetArrayFromImage(itk_img) # indexes are z,y,x (notice the ordering)
		num_z, height, width = img_array.shape        #heightXwidth constitute the transverse plane
		origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
		spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
		# go through all nodes (why just the biggest?)
		for node_idx, cur_row in mini_df.iterrows():       
			node_x = cur_row["coordX"]
			node_y = cur_row["coordY"]
			node_z = cur_row["coordZ"]
			diam = cur_row["diameter_mm"]
			# just keep 3 slices
			imgs = np.ndarray([3,height,width],dtype=np.float32)
			masks = np.ndarray([3,height,width],dtype=np.uint8)
			center = np.array([node_x, node_y, node_z])   # nodule center
			v_center = np.rint((center-origin)/spacing)  # nodule center in voxel space (still x,y,z ordering)
			for i, i_z in enumerate(np.arange(int(v_center[2])-1,
							 int(v_center[2])+2).clip(0, num_z-1)): # clip prevents going out of bounds in Z
				mask = make_mask(center, diam, i_z * spacing[2] + origin[2],
								 width, height, spacing, origin)
				masks[i] = mask
				imgs[i] = img_array[i_z]
			np.save(os.path.join(output_path,"images_%04d_%04d.npy" % (fcount, node_idx)),imgs)
			np.save(os.path.join(output_path,"masks_%04d_%04d.npy" % (fcount, node_idx)),masks)





