#!/usr/bin/env python

#np.random.seed(31337)
import tables

from keras import backend as K
from keras.callbacks import TensorBoard

from generators import imageCubeGen, Batcher, CubeGen
import numpy


K.set_floatx('float32')
K.set_image_dim_ordering('tf')
from keras.optimizers import SGD
from callbacks import fgLogger



DATADIR = '/ssd/'
DATASET = DATADIR + 'cams.h5'

from keras.losses import binary_crossentropy



import pandas

from keras.layers import Dense, Dropout, BatchNormalization, Flatten
from keras.layers import Input, Convolution3D, MaxPooling3D, UpSampling3D
from keras.models import Model
from utils import LeafLock, ImageArray


def getVGG19Encoder(inputShape, delLayers=0):
	from keras.applications.vgg19 import VGG19
	height, width = inputShape
	vggShape = (height, width, 3)
	encoder = VGG19(include_top=False, input_shape=vggShape)
	#encoder.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

	if delLayers:
		for k in xrange(0,delLayers):
			encoder.layers.pop()
		encoder.outputs = [encoder.layers[-1].output]
		encoder.layers[-1].outbound_nodes = []

	return encoder



def getDeeperEncoder(inputShape):
	height, width = inputShape
	shape = (height, width, 3)


	from nets import ResNet
	encoder = ResNet(include_top=False, input_shape=shape)

	return encoder



def buildClassifier(inputShape, optimizer='adadelta', filters=64, filterScale=2, batchNorm=True, sharedNeurons=16, bconv = 3):

	print inputShape

	x,y,z = inputShape
	input_img = Input(shape=(x, y, z, 1))
	x = input_img

	if batchNorm:
		x = BatchNormalization()(x)

	fo = filters
	fi = filters*filterScale

	b = bconv

	x = Convolution3D(fo, (b, b, b), activation='relu', padding='same')(x)
	x = Convolution3D(fo, (3, 3, 3), activation='relu', padding='same')(x)
	#x = MaxPooling3D((2, 2, 2), border_mode=border)(x)
	x = MaxPooling3D(name='pool1', pool_size=(3,3,3))(x)

	#x = Convolution3D(8, cs, cs, cs, activation='relu', border_mode=border)(x)
	#x = MaxPooling3D((2, 2, 2 ), border_mode=border)(x)

	x = Convolution3D(fi, (3, 3, 3), activation='relu', padding='same')(x)
	x = Convolution3D(fi, (3, 3, 3), activation='relu', padding='same')(x)
	x = MaxPooling3D(name='pool2')(x)

	#x = Convolution3D(8, cs, cs, cs, activation='relu', border_mode=border)(x)
	#x = Convolution3D(8, cs, cs, cs, activation='relu', border_mode=border)(x)
	#x = MaxPooling3D((2, 2, 2), border_mode=border)(x)

	encoded = x

	# at this point the representation is (nFilters, 13, 16, 13) i.e. 128-dimensional    (find with encoder.output_shape)

	#x = Convolution3D(nFilters, cs, cs, cs, activation='relu', border_mode=border)(x)
	#x = Convolution3D(nFilters, cs, cs, cs, activation='relu', border_mode=border)(x)
	#x = UpSampling3D((2, 2, 2))(x)
	#x = Convolution3D(8, cs, cs, cs, activation='relu', border_mode=border)(x)
	#x = Convolution3D(8, cs, cs, cs, activation='relu', border_mode=border)(x)

	if batchNorm:
		encoded = BatchNormalization()(x)


	flat = Flatten()(encoded)
	shared = Dense(sharedNeurons)(flat)

	nodule = Dense(1, activation='sigmoid', name='cancer')(shared)


	classifier = Model(inputs=[input_img], outputs=[nodule], name='multiOut')



	# optomizers: adadelta, sgd, rmsprop, adam, nadam

	# faster than nadam!
	classifier.compile(
		#optimizer='sgd',
		optimizer=optimizer,
		#optimizer='nadam',
		#loss='mse',
		#loss='binary_crossentropy',
		loss={
			'cancer': 'binary_crossentropy',
		},
		metrics={
			'cancer': 'accuracy',
		}
	)


	print classifier.summary()

	return classifier









def buildAutoencoder(inputShape, optimizer='adadelta', filters=64, filterScale=2, batchNorm=True, sharedNeurons=16, bconv = 3):

	print inputShape

	x,y,z = inputShape
	input_img = Input(shape=(x, y, z, 1))
	x = input_img

	if batchNorm:
		x = BatchNormalization()(x)



	fo = filters
	fi = filters*filterScale

	b = bconv


	x = Convolution3D(fo, (3, 3, 3), activation='relu', padding='same')(x)
	x = Convolution3D(fo, (3, 3, 3), activation='relu', padding='same')(x)
	#x = MaxPooling3D((2, 2, 2), border_mode=border)(x)
	x = MaxPooling3D(name='pool1', pool_size=(3,3,3))(x)

	#x = Convolution3D(8, cs, cs, cs, activation='relu', border_mode=border)(x)
	#x = MaxPooling3D((2, 2, 2 ), border_mode=border)(x)

	x = Convolution3D(fi, (3, 3, 3), activation='relu', padding='same')(x)
	x = Convolution3D(fi, (3, 3, 3), activation='relu', padding='same')(x)
	x = MaxPooling3D(name='pool2')(x)

	#x = Convolution3D(8, cs, cs, cs, activation='relu', border_mode=border)(x)
	#x = Convolution3D(8, cs, cs, cs, activation='relu', border_mode=border)(x)
	#x = MaxPooling3D((2, 2, 2), border_mode=border)(x)

	encoded = x
	encoder = Model(inputs=[input_img], outputs=[encoded])

	# at this point the representation is (nFilters, 13, 16, 13) i.e. 128-dimensional    (find with encoder.output_shape)

	#x = Convolution3D(nFilters, cs, cs, cs, activation='relu', border_mode=border)(x)
	#x = Convolution3D(nFilters, cs, cs, cs, activation='relu', border_mode=border)(x)
	#x = UpSampling3D((2, 2, 2))(x)
	#x = Convolution3D(8, cs, cs, cs, activation='relu', border_mode=border)(x)
	#x = Convolution3D(8, cs, cs, cs, activation='relu', border_mode=border)(x)

	if batchNorm:
		x = BatchNormalization()(x)

	x = UpSampling3D()(x)
	x = Convolution3D(fi, (3, 3, 3), activation='relu', padding='same')(x)
	x = Convolution3D(fi, (3, 3, 3), activation='relu', padding='same')(x)

	x = UpSampling3D(size=(3,3,3))(x)
	x = Convolution3D(fo, (3, 3, 3), activation='relu', padding='same')(x)
	decoded = Convolution3D(1, (3, 3, 3), activation='relu', padding='same', name='imgOut')(x)


	flat = Flatten()(encoded)
	shared = Dense(sharedNeurons)(flat)

	nodule = Dense(1, activation='sigmoid', name='cancer')(shared)



	autoencoder = Model(inputs=[input_img], outputs=[nodule, decoded], name='multiOut')

	print 'hidden layer shape: ', encoder.output_shape


	# optomizers: adadelta, sgd, rmsprop, adam, nadam

	# faster than nadam!
	autoencoder.compile(
		#optimizer='sgd',
		optimizer=optimizer,
		#optimizer='nadam',
		#loss='mse',
		#loss='binary_crossentropy',
		loss={
			'imgOut': 'mse',
			'cancer': 'binary_crossentropy',
		},
		metrics={
			'imgOut': 'mae',
			'cancer': 'accuracy',
		}
	)


	print autoencoder.summary()

	return encoder, autoencoder








if __name__ == '__main__':

	import argparse, os
	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('-_id', dest='_id', default=None, type=str)
	parser.add_argument('-optimizer', dest='optimizer', default='adadelta', type=str)
	#parser.add_argument('-scale', dest='scale', default="256.0", type=float)
	#parser.add_argument('-subFrac', dest='subFrac', default="0.0", type=float)
	parser.add_argument('-sharedNeurons', dest='sharedNeurons', default=16, type=int)
	parser.add_argument('-batchSize', dest='batchSize', default=32, type=int)
	parser.add_argument('-filters', dest='filters', default=8, type=int)
	parser.add_argument('-bconv', dest='bconv', default=8, type=int)
	parser.add_argument('-autoencoder', dest='autoencoder', default=1, type=int)
	parser.add_argument('-batchNorm', dest='batchNorm', default=1, type=int)


	args = parser.parse_args()
	print args

	autoencoder = args.autoencoder

	if args._id: OUTDIR = '/home/cosmo/lung/jobsLung/%s/' % args._id
	else:  OUTDIR = '/modelState/'
	if not os.path.exists(OUTDIR): os.makedirs(OUTDIR)


	camImages = ImageArray(DATASET, leafName='cams')
	imageDF = camImages.DF
	print len(imageDF)
	imageDF = imageDF[imageDF.cancer != -1]		# filter out missing labels!
	print len(imageDF)

	#imageDF = pandas.read_csv(DATADIR+'resampledImages.tsv', sep='\t')
	trainImagesDF = imageDF[imageDF.imgNum % 3 != 0]
	valImagesDF = imageDF[imageDF.imgNum % 3 == 0]
	assert len( set(trainImagesDF.imgNum).intersection(set(valImagesDF.imgNum))  ) == 0
	print len(imageDF), len(trainImagesDF), len(valImagesDF)


	def unpack(row):
		imageNum = row['imgNum']
		cam = camImages.array[imageNum]
		cam = numpy.expand_dims(cam, axis=3)
		cam = numpy.expand_dims(cam, axis=0)		# emulate a batch with several instances
		cancer = row['cancer']
		print cancer
		cancer = numpy.asarray([cancer])
		targets = {'cancer': cancer}
		if autoencoder: targets['imgOut'] = cam 
		return cam, targets

	def trainGen():
		while True:
			for _, row in trainImagesDF.iterrows():
				yield unpack(row) 

	def valGen():
		while True:
			for _, row in valImagesDF.iterrows():
				yield unpack(row)

	#candidateGen = CubeGen(DATADIR + 'candidateCubes.h5', trainImagesDF, valImagesDF, autoencoder=True, cubeSize=cubeSize)
	#noduleGen = CubeGen(DATADIR + 'noduleCubes.h5', trainImagesDF, valImagesDF, autoencoder=True, cubeSize=cubeSize)


	from gradCam import CAM_SHAPE

	if autoencoder:
		encoder, autoencoder = buildAutoencoder(CAM_SHAPE, optimizer=args.optimizer, filters=args.filters,
										batchNorm=args.batchNorm, sharedNeurons=args.sharedNeurons, bconv=args.bconv)
		model = autoencoder

	else:
		classifier = buildClassifier(CAM_SHAPE, optimizer=args.optimizer, filters=args.filters,
										batchNorm=args.batchNorm, sharedNeurons=args.sharedNeurons, bconv=args.bconv)
		model = classifier



	'''
	stratifiedQ = Batcher(
		tGen, vGen,
		trainStratified=noduleGen.trainGen, valStratified=noduleGen.valGen, batchSize=args.batchSize)
	'''


	#tester = testDSBdata(period=10, numImages=2)

	#cAUC = ComputeAUC(batch=dq.valBatch, prefix='v_', period=50)
	#cAUCt = ComputeAUC(batch=dq.trainBatch, prefix='t_', period=50)
	tb = TensorBoard(log_dir=OUTDIR + 'cams/', write_graph=False, histogram_freq=1)
	tb2 = TensorBoard(log_dir='/modelState/cams/', write_graph=False, histogram_freq=1)
	#tb = MyTensorBoard2(log_dir=OUTDIR+'tbfinetune/', write_graph=False, otherCallbacks=[cAUC, cAUCt])

	lh = fgLogger(OUTDIR)

	model.fit_generator(
		trainGen(),
		verbose=2,
		epochs=400,
		samples_per_epoch=args.batchSize,
		steps_per_epoch=8,
		callbacks=[tb, tb2, lh],
		#class_weight=classWeight,
		validation_data=valGen(),
		validation_steps=1,
		#workers=1,
		#max_q_size=20
		#nb_val_samples=64
	)

	model.save(OUTDIR + 'model.h5')

	#lh.writeLog(lastN=100)

	lh.writeToSQL('lungExp2', OUTDIR+'exp.tsv')









