import pandas
import tables
from keras.callbacks import Callback
from pkg_resources import parse_version
from sklearn.metrics import roc_auc_score, roc_curve
from scipy import integrate
from keras import backend as K
import numpy
import os
import json
import pandas, tables
import itertools

from generators import getImage, scaleCube

DATADIR = '/data/datasets/lung/resampled_order1/'


#imagesDF = pandas.read_csv(DATADIR + 'resampledImages.tsv', sep='\t')

#DB = tables.open_file(DATADIR + 'resampled.h5', mode='r')
#imageArray = DB.root.resampled










def batch(iterable, n=1):
	l = len(iterable)
	for i in range(0, l, n):
		yield iterable[i: min(i+n, l)]

def batchGen(generator, n=1):
	batch = []
	for i in range(0, n):
		v = generator.next()
		batch.append(v)
	yield batch



# generator for image cubes, which might make batching easier to implement
def generateImageCubeBatches(image, positions, cubeSize, batchSize=8):

	l = len(positions)
	for i in range(0, l, batchSize):
		batchPositions = positions[i: min(i+batchSize, l)]

		for pos in batchPositions:
			pos = numpy.asarray(pos)
			pos *= cubeSize
			z, y, x = pos

			cube = image[z:z + cubeSize, y:y + cubeSize, x:x + cubeSize]
			assert cube.shape == (cubeSize, cubeSize, cubeSize)

			if cube.mean() < -3000: continue

			# apply same normalization as in training
			cube = scaleCube(cube)

			yield cube



def makeTheCall(image, model, cubeSize):
	'''
	
	:param lungImg: the image of the lung from the DSB data set 
	:param model: the current nodule detection model
	:return: probability
	'''

	

	# loop over the image, extracting cubes and applying model
	dim = numpy.asarray(image.shape)
	print 'dimension of image: ', dim

	nChunks = dim / cubeSize
	# print 'Number of chunks in each direction: ', nChunks

	positions = [p for p in itertools.product(*map(xrange, nChunks))]

	noduleScores = []

	for batch in generateImageCubeBatches(image, positions, cubeSize):

		# batch is now a list of cubes .... I think this is all we need for prediciton since its the only input
		# might need to expand dimensions ...

		pred = model.predict(batch)
		isNodule, diam, decodedImg = pred[0]

		# these should be lists


	pass







class testDSBdata(Callback):

	'''
	Loop over N images from the data science bowl stage1 dataset
	Apply nodule detector neural network to patches of image, and get output values
	Store those into an array
	Build a classifier on this, with something like K-Fold cross validation
	Report logloss of predictions
	'''

	def __init__(self, prefix='', batch=None, period=500, numImages = 1):
		self.batch = batch
		self.prefix = prefix
		self.period = period
		self.numImages = numImages

		self.DF = pandas.read_csv(DATADIR+'resampledImages.tsv', sep='\t')
		self.DB = tables.open_file(DATADIR+'resampled.h5', mode='r')
		self.array = self.DB.root.resampled
		self.imgGen = self._imgGen()


	# an internal generator looping through images
	def _imgGen(self):
		while True:
			for index, row in self.DF.iterrows():
				image = getImage(self.array)
				yield (row['canceer'], image)


	def on_epoch_end(self, epoch, logs={}):
		if epoch % self.period != 0: return

		#if not hasattr(self, 'model'): return

		cubeSize = 32		# FIXME, infer from model input size

		testY = []
		y_score = []
		for x in range(0, self.numImages):
			#X, Y = self.generator.next()
			#X, Y = self.batch(N=64)						# its not really a generator
			#if isinstance(Y, dict): Y = Y['predictions']

			cancer, image = self.imgGen.next()

			prob = makeTheCall(image, self.model, cubeSize)

			testY.append(cancer)
			y_score.append(prob)

		testY = numpy.concatenate(testY)
		y_score = numpy.concatenate(y_score)

		print testY.shape, y_score.shape


		# extract and integrate ROC curve manually
		fpr, tpr, thresholds = roc_curve(testY, y_score, pos_label=1, sample_weight=None, drop_intermediate=True)

		auc = integrate.trapz(tpr, fpr)

		if auc is None:
			print self.prefix, 'AUC is None'
			return
		if np.isnan(auc):
			print self.prefix, 'AUC is NaN'
			return

		# now shift the y points down
		tprShift = tpr - 0.8
		tprShift[tprShift < 0.0] = 0.0
		pauc = integrate.trapz(tprShift, fpr)

		predictedCancer = y_score > 0.5


		print '%s_AUC: %s  pAUC: %s   cancer: %s/%s   highest score: %s' % \
			  (self.prefix, auc, pauc, predictedCancer.sum(), len(predictedCancer), y_score.max())




		return {self.prefix+'AUC': auc, self.prefix+'pAUC': pauc}




