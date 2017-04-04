import os

import numpy as np # linear algebra
import pandas
from tqdm import tqdm

# change later
#INPUT_FOLDER = '/data/datasets/lung/sample_images/'
INPUT_FOLDER = '/data/datasets/lung/data/stage1/'
OUTPUT_FOLDER = '/data/datasets/lung/resampled_order1/'

patients = os.listdir(INPUT_FOLDER)
patients.sort()



from preproc_dsb import load_scan, get_pixels_hu, labelsDF


log = open('spacings.tsv','w')


for imgNum, patientID in enumerate(tqdm(patients, total=len(patients))):

	rows = labelsDF[labelsDF.uuid == patientID]
	if len(rows)==1:
		row = rows.iloc[0]
	elif len(rows)==0:
		print 'missing'
		row = pandas.Series({
			'uuid': patientID,
			'cancer': -1
		})
	else: raise ValueError('Fuck')

	# only sample test set
	if row.cancer >= 0:
		#print 'skipping'
		continue

	#continue

	patientSlices = load_scan(INPUT_FOLDER + patientID)

	patientPixels = get_pixels_hu(patientSlices)

	originalShape = patientPixels.shape

	numSlices = originalShape[0]
	print numSlices

	originalSpacing = np.array([patientSlices[0].SliceThickness] + patientSlices[0].PixelSpacing, dtype=np.float32)
	x, y, z = originalSpacing
	s = '%s\t%s\t%s\n' % (x,y,z)


	print 'corner pixel: ', patientPixels[0,0,0], 'intercept/slope: ', patientSlices[0].RescaleIntercept, patientSlices[0].RescaleSlope


	log.write(s)
	log.flush()
