#!/usr/bin/env python

#np.random.seed(31337)
import tables

from keras import backend as K
from keras.callbacks import TensorBoard

from callbacks import ComputeAUC, MyTensorBoard2#, ComputeBothAUC
from generators import imageCubeGen, Batcher, CubeGen

K.set_floatx('float32')
K.set_image_dim_ordering('tf')
from keras.optimizers import SGD
from callbacks import fgLogger
from generators import candidateGen

DATADIR = '/data/datasets/luna/resampled_order1/'
DATASET = DATADIR+'resampled.h5'

SSD_DATADIR = '/ssd/luna/resampled_order1/'
SSD_DATASET = SSD_DATADIR + 'resampled.h5'

import pandas

from keras.layers import Dense, Dropout, BatchNormalization, Flatten
from keras.layers import Input, Convolution3D, MaxPooling3D, UpSampling3D
from keras.models import Model
from utils import LeafLock

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



def buildClassifier(inputShape, delLayers=8, sharedNeurons=128, cancerNeurons=32, ageWeight=1.0):

	encoder = getEncoder(inputShape, delLayers=delLayers)
	#encoder = getDeeperEncoder(inputShape)

	input_img = encoder.input
	x = encoder.outputs


	x = BatchNormalization()(x)

	flat = GlobalAveragePooling2D()(x)
	#flat = Flatten(name='flatten')(x)


	x = BatchNormalization()(flat)
	x = Dense(sharedNeurons, activation='relu', name='shared')(x)
	shared = Dropout(0.5)(x)

	#shared = flat

	x = Dense(cancerNeurons, activation='relu', name='fc2c')(shared)
	x = Dropout(0.5)(x)
	cancerClass = Dense(1, activation='sigmoid', name='predictions')(x)

	x = Dense(cancerNeurons, activation='relu', name='fc2i')(shared)
	x = Dropout(0.5)(x)
	invasive = Dense(1, activation='sigmoid', name='invasive')(x)


	'''
	x = Dense(cancerNeurons, activation='relu', name='fc2fd')(shared)
	x = Dropout(0.5)(x)
	fdBC = Dense(1, activation='sigmoid', name='firstDegreeBc')(x)

	x = Dense(16, activation='relu', name='fc2fdy')(shared)
	x = Dropout(0.5)(x)
	fdBCy = Dense(1, activation='sigmoid', name='firstDegreeBcYoung')(x)
	'''

	x = Dense(cancerNeurons, activation='relu', name='fc2a')(shared)
	x = Dropout(0.5)(x) 
	age = Dense(1, activation='sigmoid', name='age')(x)

	x = Dense(cancerNeurons, activation='relu', name='fc2b')(shared)
	x = Dropout(0.5)(x) 
	bmi = Dense(1, activation='sigmoid', name='bmi')(x)

	#model = Model(input=[input_img], output=[cancerClass], name='multiOut')
	#model = Model(input=[input_img], output=[cancerClass, age], name='multiOut')
	#model = Model(input=[input_img], output=[cancerClass, age, bmi, invasive, fdBC, fdBCy], name='multiOut')
	#model = Model(input=[input_img], output=[cancerClass, age, bmi, invasive], name='multiOut')
	model = Model(input=[input_img], output=[cancerClass, invasive, age, bmi], name='multiOut')
	#model = Model(input=[input_img], output=[cancerClass, age, invasive], name='multiOut')
	#model = Model(input=[input_img], output=[cancerClass], name='multiOut')

	sgd = SGD(lr=0.1, decay=1e-06, momentum=0.9, nesterov=True)
	model.compile(
		#optimizer=sgd,
		#optimizer='adadelta',
		#optimizer='rmsprop',
		#optimizer='adamax',
		optimizer='nadam',
		#loss='binary_crossentropy',
		loss = {
			#'predictions': 'binary_crossentropy',
			'predictions': 'binary_crossentropy',
			#'predictions': binary_crossentropy_with_ranking,
			#'predictions': 'binary_crossentropy',
			'invasive': 'binary_crossentropy',
			#'firstDegreeBc' : 'binary_crossentropy',
			#'firstDegreeBcYoung': 'binary_crossentropy',
			#'age': 'kullback_leibler_divergence',
			#'age': 'mse',
			#'bmi': 'mse'
			'age': 'binary_crossentropy',
			'bmi': 'binary_crossentropy',
			#'bmi': binary_crossentropy_with_ranking
		},
		loss_weights={
			'predictions': 1.0,
			'invasive': 1.0,
			#'firstDegreeBc' : 0.5,
			#'firstDegreeBcYoung': 1.0,
			'age': ageWeight,
			'bmi': ageWeight
			#	'bmi': 1.0
		},
		#loss='msle',
		#loss='kld',
		metrics = {
			'age': 'accuracy',
			'bmi': 'accuracy',
			'predictions': 'fmeasure',
			'invasive': 'fmeasure'
			#'firstDegreeBc': 'accuracy',
			#'firstDegreeBcYoung': 'accuracy'
		}
		#metrics = {'imageOut': 'mse', 'predictions': ['accuracy', confusionFIXME]},
	)

	print model.summary()



	return model



def buildAutoencoder(inputShape, filters=64, filterScale=2, batchNorm=True, sharedNeurons=16):


	x,y,z = inputShape
	input_img = Input(shape=(x, y, z, 1))
	x = input_img

	if batchNorm:
		x = BatchNormalization()(x)



	fo = filters
	fi = filters*filterScale


	x = Convolution3D(fo, (3, 3, 3), activation='relu', padding='same')(x)
	x = Convolution3D(fo, (3, 3, 3), activation='relu', padding='same')(x)
	#x = MaxPooling3D((2, 2, 2), border_mode=border)(x)
	x = MaxPooling3D()(x)

	#x = Convolution3D(8, cs, cs, cs, activation='relu', border_mode=border)(x)
	#x = MaxPooling3D((2, 2, 2 ), border_mode=border)(x)

	x = Convolution3D(fi, (3, 3, 3), activation='relu', padding='same')(x)
	x = Convolution3D(fi, (3, 3, 3), activation='relu', padding='same')(x)
	x = MaxPooling3D()(x)

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

	x = UpSampling3D()(x)
	x = Convolution3D(fo, (3, 3, 3), activation='relu', padding='same')(x)
	decoded = Convolution3D(1, (3, 3, 3), activation='relu', padding='same', name='imgOut')(x)


	flat = Flatten()(encoded)
	shared = Dense(sharedNeurons)(flat)

	nodule = Dense(1, activation='sigmoid', name='nodule')(shared)

	db = Dense(16)(flat)
	diam = Dense(1, activation='relu', name='diam')(db)



	autoencoder = Model(inputs=[input_img], outputs=[nodule, diam, decoded], name='multiOut')

	print 'hidden layer shape: ', encoder.output_shape


	# optomizers: adadelta, sgd, rmsprop, adam, nadam

	# faster than nadam!
	autoencoder.compile(
		#optimizer='sgd',
		optimizer='adadelta',
		#optimizer='nadam',
		#loss='mse',
		#loss='binary_crossentropy',
		loss={
			'imgOut': 'mse',
			'nodule': 'binary_crossentropy',
			'diam': 'hinge'
		},
		metrics={
			'imgOut': 'mae',
			'nodule': 'accuracy',
			'diam': 'mae'
		}
	)


	print autoencoder.summary()

	return encoder, autoencoder








if __name__ == '__main__':

	import argparse, os
	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('-_id', dest='_id', default=None, type=str)
	parser.add_argument('-optimizer', dest='optimizer', default='nadam', type=str)

	parser.add_argument('-images', dest='images', default='imagesM', type=str)

	parser.add_argument('-delLayers', dest='delLayers', default="0", type=int)
	parser.add_argument('-ageWeight', dest='ageWeight', default="0.5", type=float)

	parser.add_argument('-scale', dest='scale', default="256.0", type=float)
	parser.add_argument('-subFrac', dest='subFrac', default="0.0", type=float)

	parser.add_argument('-sharedNeurons', dest='sharedNeurons', default=64, type=int)
	parser.add_argument('-cancerNeurons', dest='cancerNeurons', default=16, type=int)

	parser.add_argument('-batchSize', dest='batchSize', default=32, type=int)
	parser.add_argument('-cubeSize', dest='cubeSize', default=32, type=int)

	parser.add_argument('-filters', dest='filters', default=32, type=int)


	args = parser.parse_args()
	print args
	cubeSize = args.cubeSize


	if args._id: OUTDIR = '/home/cosmo/lung/jobs/%s/' % args._id
	else:  OUTDIR = '/modelState/'
	if not os.path.exists(OUTDIR): os.makedirs(OUTDIR)



	DB = tables.open_file(DATASET, mode='r')
	images = DB.root.resampled
	images = LeafLock(images)

	DBSSD = tables.open_file(DATASET, mode='r')
	imagesSSD = DBSSD.root.resampled
	#imagesSSD = LeafLock(imagesSSD)

	imageDF = pandas.read_csv(DATADIR+'resampledImages.tsv', sep='\t')
	trainImagesDF = imageDF[imageDF.index % 4 != 0]
	valImagesDF = imageDF[imageDF.imgNum % 4 == 0]
	print len(imageDF), len(trainImagesDF), len(valImagesDF)


	noduleDF = pandas.read_csv(DATADIR+'resampledNodules.tsv', sep='\t')#.merge(imageDF, on='imgNum')
	candidatesDF = pandas.read_csv(DATADIR+'resampledCandidates.tsv', sep='\t')#.merge(imageDF, on='imgNum')

	# generators of arbitrary cubes from images
	tGen = imageCubeGen(imagesSSD, trainImagesDF, noduleDF, candidatesDF, cubeSize=cubeSize, autoencoder=True)
	vGen = imageCubeGen(imagesSSD, valImagesDF, noduleDF, candidatesDF, cubeSize=cubeSize, autoencoder=True)


	candidateGen = CubeGen(DATADIR + 'cubes.h5', DATADIR + 'cubes.tsv', trainImagesDF, valImagesDF, autoencoder=True, cubeSize=cubeSize)
	noduleGen = CubeGen(DATADIR + 'nodules.h5', DATADIR + 'nodules.tsv', trainImagesDF, valImagesDF, autoencoder=True, cubeSize=cubeSize)


	#inputShape = images[0].shape

	#classifier = buildClassifier(inputShape, delLayers=args.delLayers, sharedNeurons=args.sharedNeurons, cancerNeurons=args.cancerNeurons, ageWeight=args.ageWeight)

	inputShape = (cubeSize, cubeSize, cubeSize)
	encoder, autoencoder = buildAutoencoder(inputShape, filters=args.filters)


	stratifiedQ = Batcher(tGen, vGen, trainStratified=noduleGen.trainGen, valStratified=noduleGen.valGen, batchSize=args.batchSize)
	#stratifiedQ = Batcher(candidateGen.trainGen, candidateGen.valGen, trainStratified=noduleGen.trainGen, valStratified=noduleGen.valGen, batchSize=args.batchSize)


	#cAUC = ComputeAUC(batch=dq.valBatch, prefix='v_', period=50)
	#cAUCt = ComputeAUC(batch=dq.trainBatch, prefix='t_', period=50)
	tb = TensorBoard(log_dir=OUTDIR + 'nodules/', write_graph=False, histogram_freq=1)
	#tb = MyTensorBoard2(log_dir=OUTDIR+'tbfinetune/', write_graph=False, otherCallbacks=[cAUC, cAUCt])

	lh = fgLogger(OUTDIR)

	autoencoder.fit_generator(
		stratifiedQ.trainGen,
		verbose=2,
		epochs=100,
		#samples_per_epoch=args.batchSize,
		steps_per_epoch=1,
		callbacks=[tb, lh],
		#class_weight=classWeight,
		validation_data=stratifiedQ.valGen,
		validation_steps=1,
		workers=1,
		#max_q_size=20
		#nb_val_samples=64
	)

	#classifier.save(OUTDIR + 'finetune.h5')

	lh.writeLog(lastN=100)





	DB.close()







