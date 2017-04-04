
# Settings
import sys
import tables
from glob import glob
from random import shuffle

import SimpleITK as sitk
import numpy as np
import pandas
import pandas as pd
from tqdm import tqdm

from utils import resample
from utils import world2voxel

LUNA_DATA_PATH = "/data/datasets/luna/"
luna_subset_path = LUNA_DATA_PATH + "subset0/"



RESAMPLED_IMG_SHAPE = (750, 750, 750)






#file_list = glob(luna_subset_path+"*.mhd")
file_list = glob(LUNA_DATA_PATH + "subset?/*.mhd")
shuffle(file_list)


# Helper function to get rows in data frame associated  with each file
def get_filename(file_list, case):
	for f in file_list:
		if case in f:
			return(f)

# The locations of the nodules
noduleDF = pd.read_csv(LUNA_DATA_PATH + "annotations.csv")
noduleDF["file"] = noduleDF["seriesuid"].map(lambda uid: get_filename(file_list, uid))
del noduleDF['seriesuid']

candidatesDF = pd.read_csv(LUNA_DATA_PATH + "candidates.csv")
candidatesDF["file"] = candidatesDF["seriesuid"].map(lambda uid: get_filename(file_list, uid))
del candidatesDF['seriesuid']




#crash








tables.set_blosc_max_threads(4)
#filters = tables.Filters(complevel=5, complib='blosc:lz4')      # 7.7sec / 1.2 GB   (14 sec 1015MB if precision is reduced)           140s 3.7GB
#filters = tables.Filters(complevel=5, complib='lzo')
filters = tables.Filters(complevel=5, complib='blosc:snappy')

DB = tables.open_file('/data/datasets/luna/resampled.h5', mode='w', filters=None)
#images = DB.create_earray(DB.root, 'resampled', atom=tables.Int16Atom(shape=RESAMPLED_IMG_SHAPE), shape=(0,), expectedrows=len(file_list), filters=filters)

# with a carray!
images = DB.create_carray(DB.root, 'resampled', atom=tables.Int16Atom(shape=RESAMPLED_IMG_SHAPE), shape=(len(file_list),), filters=filters)
#images = DB.create_carray(DB.root, 'resampled', atom=tables.Int16Atom(shape=RESAMPLED_IMG_SHAPE), shape=(len(file_list),), filters=filters, chunkshape=(1,))

# this dataframe just stores seriesUIDs .... index is auto-incrementing and follows pytables index
imageDF = pandas.DataFrame()



resampledNodulesDF = pandas.DataFrame()
resampledCandidatesDF = pandas.DataFrame()

#file_list = file_list[0:10]

for imgNum, img_file in enumerate(tqdm(file_list)):


	itk_img = sitk.ReadImage(img_file)
	img_array = sitk.GetArrayFromImage(itk_img)			# indexes are z,y,x (notice the ordering)
	num_z, height, width = img_array.shape				# heightXwidth constitute the transverse plane
	print 'image shape: ', img_array.shape
	origin = np.array(itk_img.GetOrigin())[::-1]			# z,y,x  Origin in world coordinates (mm)
	spacing = np.array(itk_img.GetSpacing())[::-1]			# z,y,x spacing of voxels in world coor. (mm)
	#print 'image spacing: ', spacing
	#spacing = np.asarray((spacing[2], spacing[0], spacing[1]))
	print 'image spacing: ', spacing


	resampled, newSpacing = resample(img_array, spacing, new_spacing=[0.6, 0.6, 0.6])
	print 'resampled array: ', resampled.shape, newSpacing


	s = pandas.Series({
		#'file': img_file,
		'spacingZ': newSpacing[0],
		'spacingY': newSpacing[1],
		'spacingX': newSpacing[2],
		'shapeZ': int(resampled.shape[0]),
		'shapeY': int(resampled.shape[1]),
		'shapeX': int(resampled.shape[2]),
		'originZ': origin[0],
		'originY': origin[1],
		'originX': origin[2]
	})
	s.name = imgNum
	imageDF = imageDF.append(s)

	imageDF.shapeZ = imageDF.shapeZ.astype('int')
	imageDF.shapeY = imageDF.shapeY.astype('int')
	imageDF.shapeX = imageDF.shapeX.astype('int')

	imageDF.to_csv('/data/datasets/luna/resampledImages.tsv', sep='\t', index_label='imgNum')




	# get all nodules associate with file
	mini_df = noduleDF[noduleDF["file"] == img_file]
	mini_df['imgNum'] = int(imgNum)
	del mini_df['file']
	for index, row in mini_df.iterrows():
		nodulePos = row[['coordZ','coordY','coordX']].as_matrix()
		voxelC = world2voxel(nodulePos, origin, newSpacing)
		row['voxelZ'] = voxelC[0]
		row['voxelY'] = voxelC[1]
		row['voxelX'] = voxelC[2]
		resampledNodulesDF = resampledNodulesDF.append(row, ignore_index=True)

		'''
		print 'voxel center: ', nodulePos, voxelC
		p = 12
		crop = resampled[
			int(voxelC[0] - p) : int(voxelC[0] + p),
			int(voxelC[1] - p) : int(voxelC[1] + p),
			int(voxelC[2] - p) : int(voxelC[2] + p)
			]
		print crop.shape
		plot_nodule(crop)
		'''

	if len(resampledNodulesDF):
		resampledNodulesDF.imgNum = resampledNodulesDF.imgNum.astype('int')
		resampledNodulesDF.to_csv('/data/datasets/luna/resampledNodules.tsv', sep='\t', index=False)


	mini_df = candidatesDF[candidatesDF["file"] == img_file]
	mini_df['imgNum'] = imgNum
	del mini_df['file']
	for index, row in mini_df.iterrows():
		nodulePos = row[['coordX','coordY','coordZ']].as_matrix()
		voxelC = world2voxel(nodulePos, origin, newSpacing)
		row['voxelX'] = voxelC[0]
		row['voxelY'] = voxelC[1]
		row['voxelZ'] = voxelC[2]

		resampledCandidatesDF = resampledCandidatesDF.append(row, ignore_index=True)

	resampledCandidatesDF.imgNum = resampledCandidatesDF.imgNum.astype('int')
	resampledCandidatesDF.cancer = resampledCandidatesDF.cancer.astype('int')
	resampledCandidatesDF.to_csv('/data/datasets/luna/resampledCandidates.tsv', sep='\t', index=False)



	resized = np.zeros(RESAMPLED_IMG_SHAPE, dtype='int16')
	xCopy = min(RESAMPLED_IMG_SHAPE[0], resampled.shape[0])
	yCopy = min(RESAMPLED_IMG_SHAPE[1], resampled.shape[1])
	zCopy = min(RESAMPLED_IMG_SHAPE[2], resampled.shape[2])
	resized[:xCopy, :yCopy, :zCopy] = resampled[:xCopy, :yCopy, :zCopy]


	#images.append([resized])  # get the remaining ones

	# carray
	images[imgNum] = resized



	# FIXME - all this stuff should be downstream, coming out of pytables ...




DB.close()






sys.exit(0)









#####
#
# Looping over the image files
#
for imgNum, img_file in enumerate(tqdm(file_list)):

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
			np.save(os.path.join(output_path,"images_%04d_%04d.npy" % (imgNum, node_idx)), imgs)
			np.save(os.path.join(output_path,"masks_%04d_%04d.npy" % (imgNum, node_idx)), masks)





