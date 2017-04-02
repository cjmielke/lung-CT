import os

import dicom
import matplotlib.pyplot as plt
import nibabel
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


row = DF.iloc[3]

from dsbTests import getImage
image, imgNum = getImage(array, row)

def vol2Nifti(vol, filename):
	affine = tables.numpy.eye(4)
	#oImg = nibabel.Nifti1Image(image, affine, header=img.header)
	oImg = nibabel.Nifti1Image(vol, affine)
	oImg.to_filename(filename)



vol2Nifti(image, 'foob.nii')




