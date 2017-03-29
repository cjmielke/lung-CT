import os

import dicom
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas
from skimage import measure
from tqdm import tqdm

from viz import showHist, mayaPlot

from utils import resample
import tables



# change later
INPUT_FOLDER = '/data/datasets/lung/sample_images/'
OUTPUT_FOLDER = '/data/datasets/lung/resampled_order1/'

patients = os.listdir(INPUT_FOLDER)
patients.sort()

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
	image[image == -2000] = 0

	# Convert to Hounsfield units (HU)
	for slice_number in range(len(slices)):

		intercept = slices[slice_number].RescaleIntercept
		slope = slices[slice_number].RescaleSlope

		if slope != 1:
			image[slice_number] = slope * image[slice_number].astype(np.float64)
			image[slice_number] = image[slice_number].astype(np.int16)

		image[slice_number] += np.int16(intercept)

	return np.array(image, dtype=np.int16)




tables.set_blosc_max_threads(4)
filters = tables.Filters(complevel=1, complib='blosc:lz4')      # 7.7sec / 1.2 GB   (14 sec 1015MB if precision is reduced)           140s 3.7GB
DB = tables.open_file('/data/datasets/luna/resampled.h5', mode='w', filters=None)
#images = DB.create_earray(DB.root, 'resampled', atom=tables.Int16Atom(shape=RESAMPLED_IMG_SHAPE), shape=(0,), expectedrows=len(file_list), filters=filters)
images = DB.create_carray(DB.root, 'resampled', atom=tables.Int16Atom(shape=RESAMPLED_IMG_SHAPE), shape=(len(patients),), filters=filters)




imageDF = pandas.DataFrame()



#first_patient = load_scan(INPUT_FOLDER + patients[0])


for imgNum, patientID in enumerate(tqdm(patients, total=len(patients))):

	patientSlices = load_scan(INPUT_FOLDER + patientID)

	patientPixels = get_pixels_hu(patientSlices)

	originalSpacing = np.array([patientSlices[0].SliceThickness] + patientSlices[0].PixelSpacing, dtype=np.float32)
	print 'pixel spacing is ', originalSpacing

	resampled, newSpacing = resample(patientPixels, originalSpacing, [1, 1, 1])

	print("Shape before resampling\t", patientPixels.shape)
	print("Shape after resampling\t", resampled.shape)


	s = pandas.Series({
		#'file': img_file,
		'spacingZ': newSpacing[0],
		'spacingY': newSpacing[1],
		'spacingX': newSpacing[2],
		'shapeZ': int(resampled.shape[0]),
		'shapeY': int(resampled.shape[1]),
		'shapeX': int(resampled.shape[2])
	})
	s.name = imgNum
	imageDF = imageDF.append(s)
	imageDF.imgNum = imageDF.imgNum.astype('int')
	imageDF.shapeZ = imageDF.shapeZ.astype('int')
	imageDF.spacingY = imageDF.spacingY.astype('int')
	imageDF.spacingX = imageDF.spacingX.astype('int')
	imageDF.to_csv(OUTPUT_FOLDER+'resampledImages.tsv', sep='\t', index_label='imgNum')


	resized = np.zeros(RESAMPLED_IMG_SHAPE, dtype='int16')
	xCopy = min(RESAMPLED_IMG_SHAPE[0], resampled.shape[0])
	yCopy = min(RESAMPLED_IMG_SHAPE[1], resampled.shape[1])
	zCopy = min(RESAMPLED_IMG_SHAPE[2], resampled.shape[2])
	resized[:xCopy, :yCopy, :zCopy] = resampled[:xCopy, :yCopy, :zCopy]

	images[imgNum] = resized












