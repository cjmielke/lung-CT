import os

import dicom
import matplotlib.pyplot as plt
import numpy as np # linear algebra
from skimage import measure

from viz import showHist, mayaPlot

from utils import resample

import tables, pandas

INPUT_FOLDER = '/data/datasets/lung/sample_images/'


DATADIR = '/data/datasets/luna/resampled_order1/'
tsvFile = DATADIR+'resampledImages.tsv'
arrayFile = DATADIR+'resampled.h5'


DB = tables.open_file(arrayFile, mode='r')
array = DB.root.resampled

DF = pandas.read_csv(tsvFile, sep='\t')
DF = DF.sample(frac=1)
#DF = DF[DF.cancer != -1]  # remove images from the submission set


row = DF.iloc[0]

from dsbTests import getImage
image, imgNum = getImage(array, row)

showHist(image)



# Show some slice in the middle
plt.imshow(image[80], cmap=plt.cm.gray)
plt.show()


from scipy import ndimage


image = ndimage.interpolation.zoom(image, 0.1)
print image.shape
#c

#plot_3d(pix_resampled, 400)
mayaPlot(image, 400)			# ribcage

c


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














