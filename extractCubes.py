#!/usr/bin/env python

'''
Extract cubes from dataset, creating a new pytables library to pull from
'''

import tables
import pandas
from utils import getImage, convertColsToInt
from tqdm import tqdm
import argparse
import numpy


DATADIR = '/data/datasets/luna/resampled_order1/'
SSD_DATADIR = '/ssd/luna/resampled_order1/'

DATASET = DATADIR+'resampled.h5'
#DATASET = DATADIR+'resampledCarray.h5'
#DATASET = SSD_DATADIR + 'resampled.h5'
#DATASET = '/data/datasets/luna/resampled.h5'

#DATASET = DATADIR + 'segmented.h5'






def extractCubeAtLocation(image, voxelC, cubeSize):
	p = cubeSize / 2
	for i in [0, 1, 2]: voxelC[i] = numpy.clip(voxelC[i], p, image.shape[i] - p)
	zMin, zMax = int(voxelC[0] - p), int(voxelC[0] + p)
	yMin, yMax = int(voxelC[1] - p), int(voxelC[1] + p)
	xMin, xMax = int(voxelC[2] - p), int(voxelC[2] + p)

	cube = image[zMin: zMax, yMin: yMax, xMin: xMax]
	assert cube.shape == (cubeSize, cubeSize, cubeSize)

	return cube







if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Extract cubes from dataset')
	parser.add_argument('-cubeSize', dest='cubeSize', default=32, type=int)
	args = parser.parse_args()


	imageDF = pandas.read_csv(DATADIR + 'resampledImages.tsv', sep='\t')
	noduleDF = pandas.read_csv(DATADIR+'resampledNodules.tsv', sep='\t').merge(imageDF, on='imgNum')
	candidatesDF = pandas.read_csv(DATADIR+'resampledCandidates.tsv', sep='\t').merge(imageDF, on='imgNum')


	DF = candidatesDF
	outArray = DATADIR + 'candidateCubes.h5'
	outDF = DATADIR + 'candidateCubes.tsv'


	cubeSize = args.cubeSize
	CUBE_SHAPE = (cubeSize, cubeSize, cubeSize)
	print 'CUBE SHAPE : ', CUBE_SHAPE

	DB = tables.open_file(DATASET, mode='r')
	images = DB.root.resampled





	DBo = tables.open_file(outArray, mode='w')
	filters = tables.Filters(complevel=6, complib='blosc:snappy')      # 7.7sec / 1.2 GB   (14 sec 1015MB if precision is reduced)           140s 3.7GB
	#filters = None
	cubes = DBo.create_earray(DBo.root, 'cubes', atom=tables.Int16Atom(shape=CUBE_SHAPE), shape=(0,), expectedrows=len(DF), filters=filters)
	#cubes = DBo.create_carray(DBo.root, 'cubes', 
	#						  atom=tables.Int16Atom(shape=CUBE_SHAPE), 
	#						  shape=(len(DF),), filters=filters, chunkshape=(10,))



	cubeDF = pandas.DataFrame()


	oldImgNum = None
	image = None
	L = []
	for index, row in tqdm(DF.iterrows(), total=len(DF)):

		if row['imgNum'] != oldImgNum:
			#print ' new image loded =============='
			del image
			image, imageNum = getImage(images, row, convertType=False)
			#print image.dtype
			if image.dtype != 'int16': image = image.astype('int16')
			oldImgNum = imageNum

		voxelC = row[['voxelZ', 'voxelY', 'voxelX']].as_matrix()

		cube = extractCubeAtLocation(image, voxelC, cubeSize)

		#print cube.shape
		#cubes.append([cube])
		L.append(cube)
		#cubes[index] = cube
		row.name = index
		cubeDF = cubeDF.append(row)

		if len(L) > 10:
			cubes.append(L)
			L=[]
			DBo.flush()


	cubes.append(L)

	cubeDF = convertColsToInt(cubeDF, ['imgNum', 'shapeX', 'shapeY', 'shapeZ'])
	cubeDF.to_csv(outDF, sep='\t', index_label='noduleNum')

	DB.close()
	DBo.close()







