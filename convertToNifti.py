import nibabel
import numpy
import pandas
import tables
from scipy.ndimage import zoom
from utils import getImage

INPUT_FOLDER = '/data/datasets/lung/sample_images/'


DATADIR = '/data/datasets/lung/resampled_order1/'
tsvFile = DATADIR+'resampledImages.tsv'
arrayFile = DATADIR+'resampled.h5'


DB = tables.open_file(arrayFile, mode='r')
array = DB.root.resampled

DF = pandas.read_csv(tsvFile, sep='\t')
#DF = DF.sample(frac=1)
#DF = DF[DF.cancer != -1]  # remove images from the submission set

DF = DF[DF.cancer != -1]  # remove images from the submission set
DF = DF[DF.cancer == 1]  # remove images from the submission set
DF = DF.head(1)

row = DF.iloc[0]


image, imgNum = getImage(array, row)

def vol2Nifti(vol, filename, affine=None):
	if affine is None: affine = numpy.eye(4)
	#oImg = nibabel.Nifti1Image(image, affine, header=img.header)
	oImg = nibabel.Nifti1Image(vol, affine)
	oImg.update_header()
	oImg.to_filename(filename)


image = zoom(image, 0.3)
print image.shape

print image.dtype
vol2Nifti(image, 'firstcancer.nii.gz')




