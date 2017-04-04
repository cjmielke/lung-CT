import numpy
import pandas
import tables
from keras.callbacks import Callback
from scipy import integrate
from sklearn.metrics import roc_curve
from gradCam import buildGradientFunction

from utils import getImage, getImageCubes, ImageArray

DATADIR = '/data/datasets/lung/resampled_order1/'
tsvFile = DATADIR+'resampledImages.tsv'
arrayFile = DATADIR+'resampled.h5'

arrayFile = DATADIR+'segmented.h5'





#imagesDF = pandas.read_csv(DATADIR + 'resampledImages.tsv', sep='\t')

#DB = tables.open_file(DATADIR + 'resampled.h5', mode='r')
#imageArray = DB.root.resampled










def makeBatches(iterable, n=1):
	l = len(iterable)
	for i in range(0, l, n):
		batch = iterable[i: min(i+n, l)]
		assert len(batch)
		batch = numpy.asarray(batch)
		yield batch






def makeTheCall(image, model, cubeSize):
	'''
	Classify an image
	:param lungImg: the image of the lung from the DSB data set 
	:param model: the current nodule detection model
	:return: probability
	'''

	batchSize = 128

	cubes, _ = getImageCubes(image, cubeSize, filterBackground=True)
	print 'got %s cubes' % len(cubes)
	gen = makeBatches(cubes, batchSize)

	nBatches = len(cubes)/batchSize
	#gen = generateImageCubeBatches(image, positions, cubeSize, batchSize=batchSize)
	predictions = model.predict_generator(gen, nBatches, pickle_safe=True, verbose=1,
										  max_q_size=1, workers=1 )

	isNodule, diam, decodedImg = predictions
	print isNodule.min(), isNodule.mean(), isNodule.max()
	print diam.min(), diam.mean(), diam.max()
	noduleScores = isNodule

	#noduleScores = numpy.asarray(noduleScores)
	print 'Nodule scores : ', noduleScores.shape, noduleScores.min(), noduleScores.mean(), noduleScores.max()
	print 'Image stats : ', image.min(), image.mean(), image.max()

	return noduleScores








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

		DF = pandas.read_csv(tsvFile, sep='\t')
		self.DF = DF[DF.cancer != -1]		# remove images from the submission set
		print 'Images in DSB dataset: ', len(DF), len(self.DF)


		self.DB = tables.open_file(arrayFile, mode='r')
		self.array = self.DB.root.resampled
		self.imgGen = self._imgGen()


	# an internal generator looping through images
	def _imgGen(self):
		while True:
			for index, row in self.DF.iterrows():
				image, imgNum = getImage(self.array, row)
				yield row['cancer'], image


	def on_epoch_end(self, epoch, logs={}):
		if epoch % self.period != 0: return

		#if not hasattr(self, 'model'): return

		#cubeSize = 32		# FIXME, infer from model input size
		_, x,y,z, channels = self.model.input_shape
		cubeSize = x

		testY = []
		y_score = []
		for x in range(0, self.numImages):
			#X, Y = self.generator.next()
			#X, Y = self.batch(N=64)						# its not really a generator
			#if isinstance(Y, dict): Y = Y['predictions']

			cancer, image = self.imgGen.next()

			noduleScores = makeTheCall(image, self.model, cubeSize)
			prob = noduleScores.max()
			print 'cancer/probability :', cancer, prob

			testY.append(cancer)
			y_score.append(prob)

		print testY, y_score
		testY = numpy.asarray(testY)
		y_score = numpy.asarray(y_score)

		print testY.shape, y_score.shape


		# extract and integrate ROC curve manually
		fpr, tpr, thresholds = roc_curve(testY, y_score, pos_label=1, sample_weight=None, drop_intermediate=True)

		auc = integrate.trapz(tpr, fpr)

		if auc is None:
			print self.prefix, 'AUC is None'
			return
		if numpy.isnan(auc):
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




class SparseImageSource():
	def __init__(self, arrayFile, tsvFile=None):
		sparseImages = ImageArray(arrayFile, tsvFile=tsvFile)
		self.DF = sparseImages.DF
		self.array = sparseImages.array
		self.cubeSize = self.array[0].shape[0]

	def getImageFromSparse(self, row, convertType=True):
		shape = row[['shapeZ', 'shapeY', 'shapeX']].as_matrix()
		imageNum = row['imageNum']
		image = numpy.zeros(shape)

		imageCubes = self.DF[self.DF.imgNum==imageNum]
		cs = self.cubeSize
		for cubeRow in imageCubes.iterrows():
			z, y, z = row[['realZ', 'realY', 'realX']].as_matrix()
			cubeNum = row['cubeNum']
			image[z:z+cs, y:y+cs, x:x+cs] = self.array[cubeNum] 

		if convertType:
			if image.dtype != K.floatx(): image = image.astype(K.floatx())

		return image


if __name__ == '__main__':
	import sys
	from keras.models import load_model

	modelFile = sys.argv[1]
	model = load_model(modelFile)
	_, x,y,z, channels = model.input_shape
	cubeSize = x

	gradient_function = buildGradientFunction(model)

	dsbImages = ImageArray(arrayFile, tsvFile=tsvFile)
	array, DF = dsbImages.array, dsbImages.DF

	DF = DF[DF.cancer != -1]  # remove images from the submission set
	DF = DF.head(10)

	testY, y_score = [], []

	for index, row in DF.iterrows():
		cancer = row['cancer']
		image, imgNum = getImage(array, row)

		print image.shape

		#cam = grad_cam(image, gradient_function)
		noduleScores = makeTheCall(image, model, cubeSize)
		prob = noduleScores.max()
		print 'cancer/probability :', cancer, prob

		testY.append(cancer)
		y_score.append(prob)

	print testY, y_score
	testY = numpy.asarray(testY)
	y_score = numpy.asarray(y_score)

	#print testY.shape, y_score.shape

	# extract and integrate ROC curve manually
	fpr, tpr, thresholds = roc_curve(testY, y_score, pos_label=1, sample_weight=None, drop_intermediate=True)

	auc = integrate.trapz(tpr, fpr)


	print 'AUC: ', auc




