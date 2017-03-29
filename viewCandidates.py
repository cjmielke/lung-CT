#!/usr/bin/env python2.7

import os
import sys
from cv2 import imshow, waitKey

import numpy
import pandas
import tables

from utils import normalizeRange
from viz import grey2color, plot_nodule, mayaPlot
import time
from mayavi import mlab

dataset_file = sys.argv[1]
df = pandas.read_csv(dataset_file, sep='\t')

path, file = os.path.split(dataset_file)
arrFile = os.path.join(path, 'resampled.h5')
arrFile = '/data/datasets/luna/resampled.h5'
DB = tables.open_file(arrFile, mode='r')
images = DB.root.resampled



#from generators import prepVGGimages





for index, row in df.iterrows():
	#data = transpose(data, (1, 2, 0))

	if index<30: continue

	imageNum = row['imgNum']


	image = images[int(imageNum)]
	#image = normalizeRange(image)

	voxelC = row[['voxelZ', 'voxelY', 'voxelX']].as_matrix()

	p = 64
	crop = image[
		   int(voxelC[0] - p): int(voxelC[0] + p),
		   int(voxelC[1] - p): int(voxelC[1] + p),
		   int(voxelC[2] - p): int(voxelC[2] + p)
		   ]

	print crop.min(), crop.mean(), crop.max()

	#plot_nodule(crop)



	print crop.min(), crop.mean(), crop.max()
	print '\n\n'

	#image = numpy.expand_dims(data, axis=0)
	#data = prepVGGimages(image, heat=True, scale=1.0, subFrac=0.0)[0]
	#print data.min(), data.mean(), data.max()


	slice = crop[int(p/2),:,:]
	slice = grey2color(slice)

	#imshow('image', slice*1)
	#waitKey(0)
	#crash



	#mayaPlot(crop, crop.mean())

	'''
	mean attenuation of 21 cbenign was 37 +- 7 HU
	mean attenuation of 42 malignant was 40 +- 8.7 HU


	air: -1000
	lung: -500
	fat: -100 to -50
	water: 0
	CSF	+15
	Kidney	+30
	Blood	+30 to +45
	Muscle	+10 to +40
	Grey matter	+37 to +45
	White matter	+20 to +30
	Liver	+40 to +60
	Soft Tissue, Contrast	+100 to +300
	Bone	+700 (cancellous bone) to +3000 (cortical bone)

	'''


	src = mlab.pipeline.scalar_field(crop)
	mlab.pipeline.iso_surface(src, name='bone', contours=[500, 1200 ], opacity=1.0, color=(1,1,1))
	mlab.pipeline.iso_surface(src, name='lung', contours=[-900, -600], opacity=0.3)
	mlab.pipeline.iso_surface(src, name='fat', contours=[-100, -50], opacity=0.3, color=(1,0.5,0))
	mlab.pipeline.iso_surface(src, name='other', contours=[50, 300], opacity=0.7, color=(0,1.0,0))
	mlab.show(stop=False)
	mlab.show_pipeline(engine=None, rich_view=True)
	#response = raw_input("hit enter to continue or ctrl-c to quit")
	#time.sleep(3)
	# mlab.pipeline.iso_surface(src, contours=[s.max()-0.1*s.ptp(), ],)

