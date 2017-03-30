import pandas
import tables
from keras.callbacks import Callback
from pkg_resources import parse_version
from sklearn.metrics import roc_auc_score, roc_curve
from scipy import integrate
from keras import backend as K
import numpy as np
import os
import json





DATADIR = '/data/datasets/lung/resampled_order1/'






class testDSBdata(Callback):

	'''
	Loop over N images from the data science bowl stage1 dataset
	Apply nodule detector neural network to patches of image, and get output values
	Store those into an array
	Build a classifier on this, with something like K-Fold cross validation
	Report logloss of predictions
	'''

	def __init__(self, prefix='', batch=None, period=500, numImages = 40, ):
		self.batch = batch
		self.prefix = prefix
		self.period = period
		self.numImages = numImages

		self.DF = pandas.read_csv(DATADIR+'resampledImages.tsv', sep='\t')
		self.DB = tables.open_file(DATADIR+'resampled.h5', mode='r')
		self.array = DB.root.resampled


	def on_epoch_end(self, epoch, logs={}):
		if epoch % self.period != 0: return



		#if not hasattr(self, 'model'): return

		testY = []
		y_score = []
		for x in range(0, self.numImages):
			#X, Y = self.generator.next()
			X, Y = self.batch(N=64)						# its not really a generator
			if isinstance(Y, dict): Y = Y['predictions']

			s = self.model.predict(X)
			if isinstance(s, list): s = s[0]

			testY.append(Y)
			y_score.append(s)

		testY = np.concatenate(testY)
		y_score = np.concatenate(y_score)

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




