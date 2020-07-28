#!/usr/bin/env python2.7
from Queue import Queue
from skimage.transform import resize
from threading import Thread

import sys
import tables
tables.set_blosc_max_threads(6)
import pandas
import numpy
from tqdm import trange, tqdm

from utils import ImageArray, getImage, getImageCubes, convertColsToInt, prepCube, boundingBox

# testSegmentation()
# sys.exit(0)




inArray = sys.argv[1]

outArray = inArray.replace('.h5','-resized.h5')

inCams = ImageArray(inArray, leafName='cams')

nChannels = inCams.array[0].shape[-1]
RESIZED_SHAPE = (100, 80, 100, nChannels)


DBo = tables.open_file(outArray, mode='w')
filters = tables.Filters(complevel=1, complib='lzo')      # 7.7sec / 1.2 GB   (14 sec 1015MB if precision is reduced)           140s 3.7GB
expectedCubesPerImage = 700
outCams = DBo.create_earray(DBo.root, 'cams', atom=tables.Float32Atom(shape=RESIZED_SHAPE), shape=(0,),
							expectedrows=len(inCams.DF), filters=filters)



for _, row in tqdm(inCams.DF.iterrows()):
	imgNum = int(row['imgNum'])
	cam = inCams.array[imgNum]

	print cam.shape

	#channel = cam.shape[3]-1


	if cam.mean()==0:
		lung = cam
	else:
		i = cam[:, :, :, -1]
		i[i == 0] = -2000
		cam[:, :, :, -1] = i

		lung = boundingBox(cam, channel=-1)

	print cam.shape, lung.shape


	resized = []
	D, W, H, C = RESIZED_SHAPE
	shape = (D, W, H)
	for c in xrange(nChannels):
		cImg = lung[:, :, :, c]
		#print cImg.shape, cImg.dtype
		rc = resize(cImg, shape, preserve_range=True, mode='edge')
		resized.append(rc)


	resized = numpy.stack(resized, axis=3)
	print 'resized', resized.shape

	outCams.append([resized])



