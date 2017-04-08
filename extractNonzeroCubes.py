#!/usr/bin/env python2.7

import tables

import pandas
import numpy
from tqdm import trange

from utils import ImageArray, getImage, getImageCubes, convertColsToInt, prepCube


class SparseImageSource():
	def __init__(self, arrayFile, tsvFile=None):
		sparseImages = ImageArray(arrayFile, tsvFile=tsvFile, leafName='cubes')
		self.DF = sparseImages.DF
		self.array = sparseImages.array
		self.cubeSize = self.array[0].shape[0]

	def getImageCubes(self, imgNum):
		imageCubes = self.DF[self.DF.imgNum==imgNum]
		cubes = []
		for _, cubeRow in imageCubes.iterrows():
			cubeNum = cubeRow['cubeNum']
			cube = self.array[cubeNum]
			cubes.append(cube)
		return cubes

	def getCubesAndPositions(self, row, posType=None):

		imgNum = row['imgNum']

		assert posType is not None, 'You must select a position type ... in cube units or real voxels'

		cubeRows = self.DF[self.DF.imgNum==imgNum]

		cubes, positions = [], []

		for _, r in cubeRows.iterrows():
			#print cubeRow
			if posType=='real': z, y, x = r.realZ, r.realY, r.realX
			elif posType=='pos': z, y, x = r.posZ, r.posY, r.posX

			cubeNum = r['cubeNum']
			cube = self.array[cubeNum]
			pos = (z, y, x)

			positions.append(pos)

			cubes.append(cube)

		return cubes, positions



	def getImageFromSparse(self, row, convertType=True):
		image = numpy.zeros((row.shapeZ, row.shapeY, row.shapeX)) - 2000		# important! Dont forget background is NOT ZERO

		imgNum = row['imgNum']

		cs = self.cubeSize
		#for cube, position in self.getCubesAndPositions(row, posType='real'):
		cubes, positions = self.getCubesAndPositions(row, posType='real')
		for cube, position in zip(cubes, positions):
			z, y, x = position
			image[z:z + cs, y:y + cs, x:x + cs] = cube

		print image.dtype
		#if convertType:
		#	if image.dtype != K.floatx(): image = image.astype(K.floatx())

		return image











def extractNonzero(arrayFile, arrayOut, cubeSize=32):

	tsvOut = arrayOut.replace('.h5','.tsv')

	CUBE_SHAPE = (cubeSize, cubeSize, cubeSize)


	segImages = ImageArray(arrayFile)
	print len(segImages.DF), len(segImages.array)


	DBo = tables.open_file(arrayOut, mode='w')
	filters = tables.Filters(complevel=5, complib='blosc:snappy')      # 7.7sec / 1.2 GB   (14 sec 1015MB if precision is reduced)           140s 3.7GB
	expectedCubesPerImage = 700
	cubesArray = DBo.create_earray(DBo.root, 'cubes', atom=tables.Int16Atom(shape=CUBE_SHAPE), shape=(0,),
							  expectedrows=segImages.nImages * expectedCubesPerImage, filters=filters)


	cubeDF = pandas.DataFrame()

	cubeNum = 0
	for i in trange(segImages.nImages):
		row = segImages.DF.iloc[i]
		image, _ = getImage(segImages.array, row)
		print image.shape


		cubes, positions = getImageCubes(image, cubeSize, prep=False)		# extract raw cubes from image
		print 'Got %d cubes' % len(cubes)

		for cube, pos in zip(cubes, positions):

			cubeShape = cube.shape[0]
			realPos = pos*cubeShape

			s = pandas.Series({
				'imgNum': row['imgNum'],
				'posZ': pos[0],
				'posY': pos[1],
				'posX': pos[2],
				'realZ': realPos[0],
				'realY': realPos[1],
				'realX': realPos[2]
			})
			s.name = cubeNum
			cubeNum += 1
			cubeDF = cubeDF.append(s)
			#print cube.min(), cube.mean(), cube.max()
			cubesArray.append([cube])




	assert len(cubeDF) == len(cubesArray)




	cubeDF = convertColsToInt(cubeDF, cubeDF.columns)

	#cubeDF.imgNum = cubeDF.imgNum.astype('int')
	#cubeDF.posZ = cubeDF.posZ.astype('int')
	#cubeDF.posY = cubeDF.posY.astype('int')
	#cubeDF.posX = cubeDF.posX.astype('int')
	#cubeDF.realZ = cubeDF.realZ.astype('int')
	#cubeDF.realY = cubeDF.realY.astype('int')
	#cubeDF.realX = cubeDF.realX.astype('int')

	cubeDF.to_csv(tsvOut, sep='\t', index_label='cubeNum')

	DBo.close()
	segImages.DB.close()


if __name__ == '__main__':


	#testSegmentation()
	#sys.exit(0)


	DATADIR = '/data/datasets/lung/resampled_order1/'
	arrayFile = DATADIR + 'segmented.h5'
	arrayOut = DATADIR + 'segmentedNonzero.h5'

	extractNonzero(arrayFile, arrayOut)




