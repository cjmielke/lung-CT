#!/usr/bin/env python2.7

import pandas
import tables
from tqdm import trange, tqdm

from dsbTests import tsvFile, getImageCubes
from utils import ImageArray, getImage, getImageCubes


def extractNonzero(arrayFile, arrayOut, cubeSize=32):

	tsvOut = arrayOut.replace('.h5','.tsv')

	CUBE_SHAPE = (cubeSize, cubeSize, cubeSize, 1)


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


		cubes, positions = getImageCubes(image, cubeSize, prep=False)
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
			print cube.min(), cube.mean(), cube.max()
			cubesArray.append([cube])




	assert len(cubeDF) == len(cubesArray)


	cubeDF.imgNum = cubeDF.imgNum.astype('int')
	cubeDF.posZ = cubeDF.posZ.astype('int')
	cubeDF.posY = cubeDF.posY.astype('int')
	cubeDF.posX = cubeDF.posX.astype('int')
	cubeDF.realZ = cubeDF.realZ.astype('int')
	cubeDF.realY = cubeDF.realY.astype('int')
	cubeDF.realX = cubeDF.realX.astype('int')

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



