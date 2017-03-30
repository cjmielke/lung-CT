import os

import dicom
import matplotlib.pyplot as plt
import numpy as np # linear algebra
from skimage import measure

from viz import showHist, mayaPlot

from utils import resample


INPUT_FOLDER = '/data/datasets/lung/sample_images/'


patients = os.listdir(INPUT_FOLDER)
patients.sort()




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
	#image[image == -2000] = 0

	# Convert to Hounsfield units (HU)
	for slice_number in range(len(slices)):

		intercept = slices[slice_number].RescaleIntercept
		slope = slices[slice_number].RescaleSlope

		if slope != 1:
			image[slice_number] = slope * image[slice_number].astype(np.float64)
			image[slice_number] = image[slice_number].astype(np.int16)

		image[slice_number] += np.int16(intercept)

	return np.array(image, dtype=np.int16)



#first_patient = load_scan(INPUT_FOLDER + patients[0])
patientSlices = load_scan(INPUT_FOLDER + '00cba091fa4ad62cc3200a657aeb957e')
first_patient_pixels = get_pixels_hu(patientSlices)

showHist(first_patient_pixels)



# Show some slice in the middle
plt.imshow(first_patient_pixels[80], cmap=plt.cm.gray)
plt.show()




#pix_resampled, spacing = resample(first_patient_pixels, first_patient, [1,1,1])
#pix_resampled, spacing = resample(first_patient_pixels, first_patient, [1,1,1])

spacing = np.array([patientSlices[0].SliceThickness] + patientSlices[0].PixelSpacing, dtype=np.float32)
print 'pixel spacing is ', spacing
pix_resampled, spacing = resample(first_patient_pixels, spacing, [1,1,1])





print("Shape before resampling\t", first_patient_pixels.shape)
print("Shape after resampling\t", pix_resampled.shape)

#plot_3d(pix_resampled, 400)
mayaPlot(pix_resampled, 400)			# ribcage




def largest_label_volume(im, bg=-1):
	vals, counts = np.unique(im, return_counts=True)

	counts = counts[vals != bg]
	vals = vals[vals != bg]

	if len(counts) > 0:
		return vals[np.argmax(counts)]
	else:
		return None





def segment_lung_mask(image, fill_lung_structures=True):

	# not actually binary, but 1 and 2.
	# 0 is treated as background, which we do not want
	binary_image = np.array(image > -320, dtype=np.int8)+1
	labels = measure.label(binary_image)

	# Pick the pixel in the very corner to determine which label is air.
	#   Improvement: Pick multiple background labels from around the patient
	#   More resistant to "trays" on which the patient lays cutting the air
	#   around the person in half
	background_label = labels[0,0,0]

	#Fill the air around the person
	binary_image[background_label == labels] = 2


	# Method of filling the lung structures (that is superior to something like
	# morphological closing)
	if fill_lung_structures:
		# For every slice we determine the largest solid structure
		for i, axial_slice in enumerate(binary_image):
			axial_slice = axial_slice - 1
			labeling = measure.label(axial_slice)
			l_max = largest_label_volume(labeling, bg=0)

			if l_max is not None: #This slice contains some lung
				binary_image[i][labeling != l_max] = 1


	binary_image -= 1 #Make the image actual binary
	binary_image = 1-binary_image # Invert it, lungs are now 1

	# Remove other air pockets insided body
	labels = measure.label(binary_image, background=0)
	l_max = largest_label_volume(labels, bg=0)
	if l_max is not None: # There are air pockets
		binary_image[labels != l_max] = 0

	return binary_image









segmented_lungs = segment_lung_mask(pix_resampled, fill_lung_structures=False)
segmented_lungs_fill = segment_lung_mask(pix_resampled, fill_lung_structures=True)
#plot_3d(segmented_lungs, 0)
#plot_3d(segmented_lungs_fill, 0)
#plot_3d(segmented_lungs_fill - segmented_lungs, 0)

mayaPlot(segmented_lungs, 0.01)
mayaPlot(segmented_lungs_fill, 0.01)
mayaPlot(segmented_lungs_fill - segmented_lungs, 0.01)














