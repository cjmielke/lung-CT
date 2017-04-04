import os
import tables
from random import shuffle

import dicom
import numpy as np # linear algebra
import pandas
from tqdm import tqdm

from utils import resample

# change later
#INPUT_FOLDER = '/data/datasets/lung/sample_images/'
INPUT_FOLDER = '/data/datasets/lung/data/stage1/'
OUTPUT_FOLDER = '/data/datasets/lung/resampled_order1/'

#OUTPUT_FOLDER = '/ssd/lung/resampled_order1/'


patients = os.listdir(INPUT_FOLDER)
#patients.sort()
shuffle(patients)

RESAMPLED_IMG_SHAPE = (750, 750, 750)


# Load the scans in given folder path
def load_scan(path):
	slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
	slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))				# nice!

	# this assumes slice thickness is same between all slices! What if they are different?
	try:
		slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
	except:
		slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

	for s in slices:
		s.SliceThickness = slice_thickness

	return slices


def get_pixels_hu(slices):
	image = np.stack([s.pixel_array for s in slices])
	# Convert to int16 (from sometimes int16),
	# should be possible as values should always be low enough (<32k)
	image = image.astype(np.int16)

	# Set outside-of-scan pixels to 0
	# The intercept is usually -1024, so air is approximately 0
	#image[image <= -2000] = -3000

	# Convert to Hounsfield units (HU)
	for slice_number in range(len(slices)):

		intercept = slices[slice_number].RescaleIntercept
		slope = slices[slice_number].RescaleSlope

		if slope != 1:
			image[slice_number] = slope * image[slice_number].astype(np.float64)
			image[slice_number] = image[slice_number].astype(np.int16)

		image[slice_number] += np.int16(intercept)

	return np.array(image, dtype=np.int16)



labelsDF = pandas.read_csv('/data/datasets/lung/stage1_labels.csv', sep=',')
labelsDF.columns = ['uuid', 'cancer']
print labelsDF.columns



if __name__ == '__main__':


	tables.set_blosc_max_threads(4)
	#filters = tables.Filters(complevel=1, complib='blosc:lz4')      # 7.7sec / 1.2 GB   (14 sec 1015MB if precision is reduced)           140s 3.7GB
	filters = tables.Filters(complevel=5, complib='blosc:snappy')
	DB = tables.open_file(OUTPUT_FOLDER+'resampled.h5', mode='w', filters=None)
	#images = DB.create_earray(DB.root, 'resampled', atom=tables.Int16Atom(shape=RESAMPLED_IMG_SHAPE), shape=(0,), expectedrows=len(file_list), filters=filters)
	images = DB.create_carray(DB.root, 'resampled', atom=tables.Int16Atom(shape=RESAMPLED_IMG_SHAPE), shape=(len(patients),), filters=filters)




	imageDF = pandas.DataFrame()



	#first_patient = load_scan(INPUT_FOLDER + patients[0])


	#for imgNum, row in enumerate(tqdm(labelsDF.iterrows(), total=len(patients))):
	for imgNum, patientID in enumerate(tqdm(patients, total=len(patients))):

		rows = labelsDF[labelsDF.uuid == patientID]
		if len(rows)==1:
			row = rows.iloc[0]
		elif len(rows)==0:
			#print 'missing'
			row = pandas.Series({
				'uuid': patientID,
				'cancer': -1
			})
		else: raise ValueError('Fuck')



		#continue

		patientSlices = load_scan(INPUT_FOLDER + patientID)

		patientPixels = get_pixels_hu(patientSlices)

		originalShape = patientPixels.shape

		numSlices = originalShape[0]
		#print numSlices

		originalSpacing = np.array([patientSlices[0].SliceThickness] + patientSlices[0].PixelSpacing, dtype=np.float32)
		#print 'pixel spacing is ', originalSpacing

		resampled, newSpacing = resample(patientPixels, originalSpacing, new_spacing=[0.6, 0.6, 0.6])
		resampled[resampled < -1500] = -3000

		'''
		b = Counter(patientPixels.flatten().tolist())
		print 'raw image most common: ', b.most_common(10)

		b = Counter(resampled.flatten().tolist())
		print 'resampled image most common: ', b.most_common(10)
		'''


		#print("Shape before resampling\t", patientPixels.shape)
		#print("Shape after resampling\t", resampled.shape)


		s = pandas.Series({
			#'file': img_file,
			'spacingZ': newSpacing[0],
			'spacingY': newSpacing[1],
			'spacingX': newSpacing[2],
			'shapeZ': int(resampled.shape[0]),
			'shapeY': int(resampled.shape[1]),
			'shapeX': int(resampled.shape[2]),
			'oShapeZ': originalShape[0],
			'oShapeY': originalShape[1],
			'oShapeX': originalShape[2],
			'cancer': row.cancer,
			'uuid': row.uuid
		})
		s.name = imgNum
		imageDF = imageDF.append(s)
		#imageDF.imgNum = imageDF.imgNum.astype('int')
		imageDF.shapeZ = imageDF.shapeZ.astype('int')
		imageDF.shapeY = imageDF.shapeY.astype('int')
		imageDF.shapeX = imageDF.shapeX.astype('int')
		imageDF.cancer = imageDF.cancer.astype('int')
		imageDF.oShapeZ = imageDF.oShapeZ.astype('int')
		imageDF.oShapeY = imageDF.oShapeY.astype('int')
		imageDF.oShapeX = imageDF.oShapeX.astype('int')

		imageDF.to_csv(OUTPUT_FOLDER+'resampledImages.tsv', sep='\t', index_label='imgNum')


		resized = np.zeros(RESAMPLED_IMG_SHAPE, dtype='int16')
		xCopy = min(RESAMPLED_IMG_SHAPE[0], resampled.shape[0])
		yCopy = min(RESAMPLED_IMG_SHAPE[1], resampled.shape[1])
		zCopy = min(RESAMPLED_IMG_SHAPE[2], resampled.shape[2])
		resized[:xCopy, :yCopy, :zCopy] = resampled[:xCopy, :yCopy, :zCopy]

		images[imgNum] = resized


