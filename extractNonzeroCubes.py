#!/usr/bin/env python2.7
from Queue import Queue
from threading import Thread

import tables
tables.set_blosc_max_threads(6)
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
		assert posType is not None, 'You must select a position type ... in cube units or real voxels'
		imgNum = row['imgNum']
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
	#filters = tables.Filters(complevel=1, complib='blosc:snappy')      # 7.7sec / 1.2 GB   (14 sec 1015MB if precision is reduced)           140s 3.7GB
	filters = tables.Filters(complevel=1, complib='lzo')      # 7.7sec / 1.2 GB   (14 sec 1015MB if precision is reduced)           140s 3.7GB
	#filters = None
	expectedCubesPerImage = 700
	cubesArray = DBo.create_earray(DBo.root, 'cubes', atom=tables.Int16Atom(shape=CUBE_SHAPE), shape=(0,),
							  expectedrows=segImages.nImages * expectedCubesPerImage, filters=filters)


	#cubeDF = pandas.DataFrame()
	cubesList = []
	cubeNum = 0

	def imageReader(Q):
		for i in trange(segImages.nImages):
			row = segImages.DF.iloc[i]
			image, _ = getImage(segImages.array, row)
			#image = numpy.zeros((750, 750, 750))
			Q.put((row,image))
		Q.put(None)


	imageQ = Queue(maxsize=2)

	t = Thread(target=imageReader, args=(imageQ,))
	t.daemon = True
	t.start()

	while True:
		res = imageQ.get()
		if res is None :
			imageQ.task_done()
			break

		row, image = res

		print image.shape

		cubes, positions = getImageCubes(image, cubeSize, prep=False)  # extract raw cubes from image
		print 'Got %d cubes' % len(cubes)

		print 'saving'
		for cube, pos in zip(cubes, positions):
			cubeShape = cube.shape[0]
			realPos = pos * cubeShape

			d = {
				'cubeNum': cubeNum,
				'imgNum': row['imgNum'],
				'posZ': pos[0],
				'posY': pos[1],
				'posX': pos[2],
				'realZ': realPos[0],
				'realY': realPos[1],
				'realX': realPos[2]
			}
			cubeNum += 1
			cubesList.append(d)
			#cubeDF = cubeDF.append(s)
			# print cube.min(), cube.mean(), cube.max()

			#cubesArray.append([cube])

		if len(cubes): cubesArray.append(cubes)

		assert len(cubesArray) == len(cubesList)

		imageQ.task_done()


	cubeDF = pandas.DataFrame(cubesList)
	cubeDF = cubeDF.set_index('cubeNum')

	assert len(cubeDF) == len(cubesArray)




	cubeDF = convertColsToInt(cubeDF, cubeDF.columns)
	cubeDF.to_csv(tsvOut, sep='\t', index_label='cubeNum')

	DBo.close()
	segImages.DB.close()

	imageQ.join()








if __name__ == '__main__':


	#testSegmentation()
	#sys.exit(0)


	#DATADIR = '/data/datasets/lung/resampled_order1/'
	#DATADIR = '/ssd/lung/resampled_order1/'

	DATADIR = '/data/datasets/lung/stage2/resampled_order1/'
	arrayFile = DATADIR + 'segmented.h5'

	#arrayOut = DATADIR + 'segmentedNonzeroLessStrict.h5'
	arrayOut = '/data/datasets/lung/stage2/resampled_order1/segmentedNonzeroLessStrict.h5'

	extractNonzero(arrayFile, arrayOut)




