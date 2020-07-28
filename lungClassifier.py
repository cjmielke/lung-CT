#!/usr/bin/env python

#np.random.seed(31337)
import random

import numpy
from keras import backend as K
from keras.callbacks import TensorBoard

K.set_floatx('float32')
K.set_image_dim_ordering('tf')
from callbacks import fgLogger





EXPERIMENT_ID = 'lungExp4'



from keras.layers import Dense, BatchNormalization, Flatten, Dropout, GlobalAveragePooling3D, GlobalMaxPooling3D, \
	AveragePooling3D
from keras.layers import Input, Convolution3D, MaxPooling3D, UpSampling3D
from keras.models import Model
from utils import ImageArray, prepOutDir, normalizeRange






class LungGenerator():
	def dfGen(self, DF):
		while True:
			for _, row in DF.iterrows():
				yield row

	def __init__(self, DATASET, args, valSplit=3, ignoreSource=True, bias=None):

		self.bias = bias
		self.autoencoder = args.autoencoder

		tsvFile = DATASET.replace('-resized', '').replace('.h5', '.tsv')

		print tsvFile, '-------------------------------------'
		camImages = ImageArray(DATASET, tsvFile=tsvFile, leafName='cams')
		self.array = camImages.array

		imageDF = camImages.DF.sample(frac=1)
		print len(imageDF)
		imageDF = imageDF[imageDF.cancer != -1]  # filter out missing labels!
		print len(imageDF)

		self.imgShape = self.array[0].shape
		if len(self.imgShape) == 3:
			x, y, z = self.imgShape
			self.imgShape = (x, y, z, 1)
			ignoreSource = False

		# imageDF = pandas.read_csv(DATADIR+'resampledImages.tsv', sep='\t')
		self.trainImagesDF = imageDF[imageDF.imgNum % valSplit != 0]
		self.valImagesDF = imageDF[imageDF.imgNum % valSplit == 0]

		self.trainRowGen = self.dfGen(self.trainImagesDF)
		if self.bias:
			self.trainPosGen = self.dfGen(self.trainImagesDF[self.trainImagesDF.cancer == 1])
			self.trainNegGen = self.dfGen(self.trainImagesDF[self.trainImagesDF.cancer == 0])

		# assure no train/validation overlap
		assert len(set(self.trainImagesDF.imgNum).intersection(set(self.valImagesDF.imgNum))) == 0
		print 'Training set size %s   validation %s' % (len(self.trainImagesDF), len(self.valImagesDF))

		self.ignoreSource = False
		if ignoreSource:
			x, y, z, nChannels = self.imgShape
			if nChannels > 1:
				self.ignoreSource = True
				self.imgShape = (x, y, z, nChannels - 1)


			# self.imgShape = (x, y, z, 1)

	def normalize(self, image):

		nChannels = image.shape[-1]
		if not self.ignoreSource: nChannels = nChannels - 1

		for channel in xrange(nChannels):
			cimg = image[:, :, :, channel]

			#MIN_BOUND = cimg.min()
			#MAX_BOUND = cimg.max()
			MIN_BOUND = -20
			MAX_BOUND = 20


			cimg = (cimg - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
			#cimg[cimg > 1] = 1.
			#cimg[cimg < 0] = 0.

			image[:, :, :, channel] = cimg

		if not self.ignoreSource:
			i = image[:, :, :, -1]
			i[i<-1000]=-1000
			image[:, :, :, -1] = normalizeRange(i, MIN_BOUND=-1000, MAX_BOUND=500)


		return image

	def getImg(self, row):
		imageNum = int(row['imgNum'])
		# print imageNum
		cam = self.array[imageNum]
		cam = cam.astype('float32')
		if len(cam.shape) == 3: cam = numpy.expand_dims(cam, axis=3)
		return cam

	def unpack(self, row, cam):

		if self.ignoreSource: cam = cam[:, :, :, :-1]

		# FIXME - TESTING
		# if self.ignoreSource: cam = (cam[:,:,:, 1]+cam[:,:,:, 0])
		# cam = numpy.expand_dims(cam, axis=3)		# emulate a batch with several instances

		#c = cam[:,:,:,0]
		#print c.min(), c.mean(), c.max()

		cam = self.normalize(cam)
		cam = numpy.expand_dims(cam, axis=0)  # emulate a batch with several instances

		cancer = row['cancer']
		# print cancer
		cancer = numpy.asarray([cancer])
		targets = {
			'cancer': cancer,
			'cancer1': cancer,
			'cancer2': cancer,
		}
		if self.autoencoder: targets['imgOut'] = cam

		#print cam.min(axis=(1,2,3)), cam.mean(axis=(1,2,3)), cam.max(axis=(1,2,3))
		return cam, targets

	def trainGen(self):
		while True:
			if self.bias:
				if random.random() < self.bias:
					row = self.trainPosGen.next()
				else:
					row = self.trainNegGen.next()
			else:
				row = self.trainRowGen.next()

			cam = self.getImg(row)
			# print '--------   cancer: ', row['cancer']
			#if cam[:, :, :, -1].mean() == 0: continue
			yield self.unpack(row, cam)

	def valGen(self):
		while True:
			for _, row in self.valImagesDF.iterrows():
				cam = self.getImg(row)
				#if cam[:, :, :, -1].mean() == 0: continue
				yield self.unpack(row, cam)






def buildModel(inputShape, args):

	print inputShape
	nChannels = inputShape[-1]

	input_img = Input(shape=inputShape)
	x = input_img

	if args.batchNorm: x = BatchNormalization()(x)



	fo = args.filters
	fi = fo*2

	b = args.bconv
	b=5

	firstPool = (4,4,4)
	secondPool = (2,2,2)


	#x = MaxPooling3D(pool_size=(2,2,2))(x)
	x = AveragePooling3D(pool_size=(2, 2, 2))(x)

	a=5
	x = Convolution3D(4, (a, a, a), activation='relu', padding='same')(x)
	x = MaxPooling3D(name='pool1', pool_size=(3,3,3))(x)

	s1 = Dense(4, activation='sigmoid')(Flatten()(x))
	nodule1 = Dense(1, activation='sigmoid', name='cancer1')(s1)


	b=3
	x = Convolution3D(4, (b, b, b), activation='relu', padding='same')(x)
	x = MaxPooling3D(pool_size=(3,3,3))(x)

	s2 = Dense(4, activation='sigmoid')(Flatten()(x))
	nodule2 = Dense(1, activation='sigmoid', name='cancer2')(s2)


	#x = Convolution3D(fi, (3, 3, 3), activation='relu', padding='same')(pool1)
	#if args.doubleLayers: x = Convolution3D(fi, (3, 3, 3), activation='relu', padding='same')(x)
	#pool2 = MaxPooling3D(name='pool2', pool_size=secondPool)(x)



	#x = Convolution3D(fi, (3, 3, 3), activation='relu', padding='same')(pool2)
	#if args.doubleLayers: x = Convolution3D(fi, (3, 3, 3), activation='relu', padding='same')(x)
	#pool3 = MaxPooling3D(pool_size=(2,2,2))(x)


	#l = Convolution3D(16, (1, 1, 1), activation='relu', padding='same')(pool3)


	#x = Convolution3D(fi, (3, 3, 3), activation='relu', padding='same')(pool2)
	#if args.doubleLayers: x = Convolution3D(fi, (3, 3, 3), activation='relu', padding='same')(x)
	#pool3 = MaxPooling3D(name='pool3', pool_size=(2,2,2))(x)

	#x = Convolution3D(fi, (5, 5, 5), activation='relu', padding='same')(pool3)
	#pool4 = MaxPooling3D(pool_size=(2,2,2))(x)


	encoded = x
	encoder = Model(inputs=[input_img], outputs=[encoded])

	'''
	if args.flat=='flat': flat = Flatten()(encoded)
	else:
		#flat = GlobalAveragePooling3D()(encoded)
		flat = GlobalMaxPooling3D()(encoded)
	'''

	#flat = Flatten()(encoded)
	#flat = GlobalAveragePooling3D()(encoded)
	flat = GlobalMaxPooling3D()(encoded)


	if args.dropout: flat = Dropout(args.dropout)(flat)

	shared = Dense(args.sharedNeurons, activation='sigmoid')(flat)
	#shared = Dense(args.sharedNeurons, bias_initializer='zero')(flat)
	#if args.dropout: shared = Dropout(args.dropout)(shared)
	nodule = Dense(1, activation='sigmoid', name='cancer')(shared)


	loss = {
		'cancer': 'binary_crossentropy',
		'cancer1': 'binary_crossentropy',
		'cancer2': 'binary_crossentropy'
	}
	metrics = {'cancer': 'accuracy'}

	if args.autoencoder:

		loss['imgOut'] = 'mse'
		metrics['imgOut'] = 'mae'


		x = UpSampling3D(size=secondPool)(pool2)
		#x = UpSampling3D(size=(5,5,5))(pool2)
		x = Convolution3D(fi, (3, 3, 3), activation='relu', padding='same')(x)
		if args.doubleLayers: x = Convolution3D(fi, (3, 3, 3), activation='relu', padding='same')(x)

		#x = UpSampling3D()(x)
		#x = Convolution3D(fi, (3, 3, 3), activation='relu', padding='same')(x)
		#if args.doubleLayers: x = Convolution3D(fi, (3, 3, 3), activation='relu', padding='same')(x)

		x = UpSampling3D(size=firstPool)(x)
		#x = UpSampling3D(size=(3,3,3))(x)
		if args.doubleLayers: x = Convolution3D(fo, (3, 3, 3), activation='relu', padding='same')(x)
		decoded = Convolution3D(nChannels, (3, 3, 3), activation='relu', padding='same', name='imgOut')(x)

		model = Model(inputs=[input_img], outputs=[nodule, decoded], name='multiOut')

	else:

		model = Model(inputs=[input_img], outputs=[nodule, nodule1, nodule2], name='multiOut')



	print 'hidden layer shape: ', encoder.output_shape


	# optomizers: adadelta, sgd, rmsprop, adam, nadam

	model.compile(optimizer=args.optimizer, loss=loss, metrics=metrics)


	print model.summary()

	return model








if __name__ == '__main__':

	import argparse, os
	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('-_id', dest='_id', default=None, type=str)
	parser.add_argument('-sharedNeurons', dest='sharedNeurons', default=2, type=int)
	parser.add_argument('-optimizer', dest='optimizer', default='nadam', type=str)
	parser.add_argument('-bconv', dest='bconv', default=3, type=int)
	parser.add_argument('-autoencoder', dest='autoencoder', default=0, type=int)
	parser.add_argument('-batchNorm', dest='batchNorm', default=0, type=int)
	parser.add_argument('-batchSize', dest='batchSize', default=64, type=int)
	parser.add_argument('-filters', dest='filters', default=8, type=int)
	parser.add_argument('-dropout', dest='dropout', default=0.0, type=float)
	parser.add_argument('-doubleLayers', dest='doubleLayers', default=1, type=int)
	parser.add_argument('-blocks', dest='blocks', default=3, type=int)

	#parser.add_argument('-flat', dest='flat', default='globalavgpool', type=str)
	parser.add_argument('-flat', dest='flat', default='flat', type=str)


	#parser.add_argument('-cam', default='cam1/cams', type=str)

	#parser.add_argument('-cam', default='cam2/cams', type=str)


	#parser.add_argument('-cam', default='cam3/cams', type=str)
	#parser.add_argument('-cam', default='cam3/cams-resized', type=str)

	#parser.add_argument('-cam', default='cam5/cams', type=str)
	parser.add_argument('-cam', default='cam5/cams-resized', type=str)

	parser.add_argument('-bias', default='0.5', type=float)

	#parser.add_argument('-optimizer', dest='optimizer', default='adadelta', type=str)

	#parser.add_argument('-scale', dest='scale', default="256.0", type=float)
	#parser.add_argument('-subFrac', dest='subFrac', default="0.0", type=float)

	args = parser.parse_args()
	print args



	if args._id: OUTDIR = '/home/cosmo/lung/jobs/%s/%s/' % ( EXPERIMENT_ID, args._id)
	else:  OUTDIR = '/modelState/'
	prepOutDir(OUTDIR, __file__)




	# DATASET = '/ssd/cams/cam1/cams.h5'
	# DATASET = '/ssd/cams/cam2/camsB.h5'

	#DATASET = '/ssd/cams/%s/cams.h5' % args.cam
	DATASET = '/ssd/cams/%s.h5' % args.cam
	print DATASET


	lungGen = LungGenerator(DATASET, args, ignoreSource=False, valSplit=5, bias=args.bias)

	imgShape = lungGen.imgShape
	model = buildModel(imgShape, args)

	'''
	stratifiedQ = Batcher(
		tGen, vGen,
		trainStratified=noduleGen.trainGen, valStratified=noduleGen.valGen, batchSize=args.batchSize)
	'''


	#tester = testDSBdata(period=10, numImages=2)

	#cAUC = ComputeAUC(batch=dq.valBatch, prefix='v_', period=50)
	#cAUCt = ComputeAUC(batch=dq.trainBatch, prefix='t_', period=50)
	tb = TensorBoard(log_dir=OUTDIR + 'tensorboard/', write_graph=False, histogram_freq=1)
	tb2 = TensorBoard(log_dir='/modelState/cams/', write_graph=False, histogram_freq=1)
	#tb = MyTensorBoard2(log_dir=OUTDIR+'tbfinetune/', write_graph=False, otherCallbacks=[cAUC, cAUCt])

	lh = fgLogger(OUTDIR)

	model.fit_generator(
		lungGen.trainGen(),
		verbose=2,
		epochs=30000,
		samples_per_epoch=args.batchSize,
		steps_per_epoch=8,
		callbacks=[tb, tb2, lh],
		#class_weight=classWeight,
		validation_data=lungGen.valGen(),
		validation_steps=16,
		#workers=1,
		#max_q_size=20
		#nb_val_samples=64
	)

	model.save(OUTDIR + 'model.h5')

	#lh.writeLog(lastN=100)

	lh.writeToSQL(OUTDIR+'exp.tsv', args, table=EXPERIMENT_ID)





