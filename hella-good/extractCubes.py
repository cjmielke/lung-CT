#!/usr/bin/env python

'''
Extract cubes from dataset, creating a new pytables library to pull from
'''


#np.random.seed(31337)
import numpy
import tables
tables.set_blosc_max_threads(4)
from keras import backend as K
K.set_floatx('float32')
K.set_image_dim_ordering('tf')

DATADIR = '/data/datasets/luna/resampled_order1/'
SSD_DATADIR = '/ssd/luna/resampled_order1/'

#DATASET = DATADIR+'resampled.h5'
#DATASET = DATADIR+'resampledCarray.h5'
DATASET = SSD_DATADIR + 'resampled.h5'
#DATASET = '/data/datasets/luna/resampled.h5'


import pandas

from utils import LeafLock

from generators import getImage
from tqdm import tqdm




import argparse

parser = argparse.ArgumentParser(description='Extract cubes from dataset')
#parser.add_argument('-dir', dest='dir')
parser.add_argument('-cubeSize', dest='cubeSize', default=32, type=int)

args = parser.parse_args()
print args


imageDF = pandas.read_csv(DATADIR + 'resampledImages.tsv', sep='\t')
#imageDF.shapeX = imageDF.shapeX.astype('int')
#imageDF.shapeY = imageDF.shapeY.astype('int')
#imageDF.shapeZ = imageDF.shapeZ.astype('int')


noduleDF = pandas.read_csv(DATADIR+'resampledNodules.tsv', sep='\t').merge(imageDF, on='imgNum')

candidatesDF = pandas.read_csv(DATADIR+'resampledCandidates.tsv', sep='\t').merge(imageDF, on='imgNum')


DF = candidatesDF

cubeSize = args.cubeSize
CUBE_SHAPE = (cubeSize, cubeSize, cubeSize)
print 'CUBE SHAPE : ', CUBE_SHAPE

DB = tables.open_file(DATASET, mode='r')
images = DB.root.resampled
#images = LeafLock(images)

DBo = tables.open_file(DATADIR+'cubes.h5', mode='w')
filters = tables.Filters(complevel=1, complib='blosc:snappy')      # 7.7sec / 1.2 GB   (14 sec 1015MB if precision is reduced)           140s 3.7GB
#filters = tables.Filters(complevel=0)      # 7.7sec / 1.2 GB   (14 sec 1015MB if precision is reduced)           140s 3.7GB
#cubes = DBo.create_earray(DBo.root, 'cubes', atom=tables.Int16Atom(shape=CUBE_SHAPE), shape=(0,), expectedrows=len(DF), filters=filters)
cubes = DBo.create_carray(DBo.root, 'cubes', atom=tables.Int16Atom(shape=CUBE_SHAPE), shape=(len(DF),), filters=filters)



cubeDF = pandas.DataFrame()




oldImgNum = None
#for index, row in DF.iterrows():
for index, row in tqdm(DF.iterrows(), total=len(DF)):


	if row['imgNum'] != oldImgNum:
		#print ' new image loded =============='
		image, imageNum = getImage(images, row, convertType=False)
		#print image.dtype
		if image.dtype != 'int16': image = image.astype('int16')
		oldImgNum = imageNum

	voxelC = row[['voxelZ', 'voxelY', 'voxelX']].as_matrix()

	p = cubeSize / 2
	for i in [0, 1, 2]: voxelC[i] = numpy.clip(voxelC[i], p, image.shape[i] - p)
	zMin, zMax = int(voxelC[0] - p), int(voxelC[0] + p)
	yMin, yMax = int(voxelC[1] - p), int(voxelC[1] + p)
	xMin, xMax = int(voxelC[2] - p), int(voxelC[2] + p)

	cube = image[zMin: zMax, yMin: yMax, xMin: xMax]
	assert cube.shape == (cubeSize, cubeSize, cubeSize)

	# print cube.min(), cube.mean(), cube.max()

	#if cube.mean() < -3000: continue		# dont store bullshit

	#print cube.shape
	#cubes.append([cube])
	cubes[index] = cube


	row.name = index
	cubeDF = cubeDF.append(row)

	cubeDF.imgNum = cubeDF.imgNum.astype('int')
	cubeDF.shapeX = cubeDF.shapeX.astype('int')
	cubeDF.shapeY = cubeDF.shapeY.astype('int')
	cubeDF.shapeZ = cubeDF.shapeZ.astype('int')

	cubeDF.to_csv(DATADIR+'cubes.tsv', sep='\t', index_label='noduleNum')




DB.close()
DBo.close()







