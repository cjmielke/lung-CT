#!/usr/bin/env python

#np.random.seed(31337)
import tables

from keras import backend as K
from keras.callbacks import TensorBoard

from generators import imageCubeGen, Batcher, CubeGen

K.set_floatx('float32')
K.set_image_dim_ordering('tf')

from callbacks import fgLogger


import pandas

from keras.layers import Dense, Dropout, BatchNormalization, Flatten
from keras.layers import Input, Convolution3D, MaxPooling3D, UpSampling3D
from keras.models import Model
from utils import LeafLock, prepOutDir, ImageArray


def buildModel(inputShape, args):


	x,y,z = inputShape
	input_img = Input(shape=(x, y, z, 1))
	x = input_img

	if args.batchNorm: x = BatchNormalization()(x)

	fo = args.filters
	fi = fo*2

	b = args.bconv
	x = Convolution3D(fo, (b, b, b), activation='relu', padding='same')(x)
	x = Convolution3D(fo, (3, 3, 3), activation='relu', padding='same')(x)
	#x = MaxPooling3D((2, 2, 2), border_mode=border)(x)
	x = MaxPooling3D(name='pool1')(x)

	#x = Convolution3D(8, cs, cs, cs, activation='relu', border_mode=border)(x)
	#x = MaxPooling3D((2, 2, 2 ), border_mode=border)(x)

	x = Convolution3D(fi, (3, 3, 3), activation='relu', padding='same')(x)
	if args.doubleLayers: x = Convolution3D(fi, (3, 3, 3), activation='relu', padding='same')(x)
	x = MaxPooling3D(name='pool2')(x)

	#x = Convolution3D(8, cs, cs, cs, activation='relu', border_mode=border)(x)
	#x = Convolution3D(8, cs, cs, cs, activation='relu', border_mode=border)(x)
	#x = MaxPooling3D((2, 2, 2), border_mode=border)(x)

	encoded = x
	encoder = Model(inputs=[input_img], outputs=[encoded])


	flat = Flatten()(encoded)
	if args.dropout: flat = Dropout(args.dropout)(flat)
	shared = Dense(args.sharedNeurons)(flat)
	nodule = Dense(1, activation='sigmoid', name='nodule')(shared)

	db = Dense(16)(flat)
	diam = Dense(1, activation='relu', name='diam')(db)

	loss = {
		'nodule': 'binary_crossentropy',
		'diam': 'hinge'
	}
	metrics = {
		'nodule': 'accuracy',
		'diam': 'mae'
	}

	if args.autoencoder:

		loss['imgOut'] = 'mse'
		metrics['imgOut'] = 'mae'

		x = UpSampling3D()(x)
		x = Convolution3D(fi, (3, 3, 3), activation='relu', padding='same')(x)
		if args.doubleLayers: x = Convolution3D(fi, (3, 3, 3), activation='relu', padding='same')(x)

		x = UpSampling3D()(x)
		if args.doubleLayers: x = Convolution3D(fo, (3, 3, 3), activation='relu', padding='same')(x)
		decoded = Convolution3D(1, (3, 3, 3), activation='relu', padding='same', name='imgOut')(x)

		model = Model(inputs=[input_img], outputs=[nodule, diam, decoded], name='multiOut')

	else:

		model = Model(inputs=[input_img], outputs=[nodule, diam], name='multiOut')



	print 'hidden layer shape: ', encoder.output_shape

	model.compile( optimizer='adadelta', loss=loss, metrics=metrics )

	print model.summary()
	return model








if __name__ == '__main__':

	import argparse, os
	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('-_id', dest='_id', default=None, type=str)
	parser.add_argument('-sharedNeurons', dest='sharedNeurons', default=64, type=int)
	parser.add_argument('-optimizer', dest='optimizer', default='nadam', type=str)
	parser.add_argument('-bconv', dest='bconv', default=3, type=int)
	parser.add_argument('-autoencoder', dest='autoencoder', default=1, type=int)
	parser.add_argument('-batchNorm', dest='batchNorm', default=0, type=int)
	parser.add_argument('-batchSize', dest='batchSize', default=32, type=int)
	parser.add_argument('-filters', dest='filters', default=16, type=int)
	parser.add_argument('-dropout', dest='dropout', default=0, type=float)
	parser.add_argument('-doubleLayers', dest='doubleLayers', default=1, type=int)

	#parser.add_argument('-optimizer', dest='optimizer', default='adadelta', type=str)

	#parser.add_argument('-scale', dest='scale', default="256.0", type=float)
	#parser.add_argument('-subFrac', dest='subFrac', default="0.0", type=float)



	parser.add_argument('-cubeSize', dest='cubeSize', default=32, type=int)



	args = parser.parse_args()
	print args
	cubeSize = args.cubeSize




	EXPERIMENT_ID = 'noduleExp2'

	if args._id: OUTDIR = '/home/cosmo/lung/jobs/%s/%s/' % ( EXPERIMENT_ID, args._id)
	else:  OUTDIR = '/modelState/'
	prepOutDir(OUTDIR, __file__)



	DATADIR = '/data/datasets/luna/resampled_order1/'
	DATASET = DATADIR + 'resampled.h5'

	SSD_DATADIR = '/ssd/luna/resampled_order1/'
	SSD_DATASET = SSD_DATADIR + 'resampled.h5'


	'''
	DB = tables.open_file(DATASET, mode='r')
	images = DB.root.resampled
	images = LeafLock(images)

	DBSSD = tables.open_file(DATASET, mode='r')
	imagesSSD = DBSSD.root.resampled
	#imagesSSD = LeafLock(imagesSSD)

	imageDF = pandas.read_csv(DATADIR+'resampledImages.tsv', sep='\t')
	'''


	resampledImages = ImageArray(SSD_DATADIR + 'resampled.h5', tsvFile=DATADIR+'resampledImages.tsv')
	imageDF, imagesSSD = resampledImages.DF, resampledImages.array
	#imagesSSD = LeafLock(imagesSSD)

	trainImagesDF = imageDF[imageDF.imgNum % 4 != 0]
	valImagesDF = imageDF[imageDF.imgNum % 4 == 0]
	assert len( set(trainImagesDF.imgNum).intersection(set(valImagesDF.imgNum))  ) == 0
	print len(imageDF), len(trainImagesDF), len(valImagesDF)


	noduleDF = pandas.read_csv(DATADIR+'resampledNodules.tsv', sep='\t')#.merge(imageDF, on='imgNum')
	candidatesDF = pandas.read_csv(DATADIR+'resampledCandidates.tsv', sep='\t')#.merge(imageDF, on='imgNum')

	# generators of arbitrary cubes from images
	tGen = imageCubeGen(imagesSSD, trainImagesDF, noduleDF, candidatesDF, cubeSize=cubeSize, autoencoder=True)
	vGen = imageCubeGen(imagesSSD, valImagesDF, noduleDF, candidatesDF, cubeSize=cubeSize, autoencoder=True)

	# generators from pre-extracted cubes
	candidateGen = CubeGen(DATADIR + 'candidateCubes.h5', trainImagesDF, valImagesDF, autoencoder=True, cubeSize=cubeSize)
	noduleGen = CubeGen(DATADIR + 'noduleCubes.h5', trainImagesDF, valImagesDF, autoencoder=True, cubeSize=cubeSize)

	#for imgNum, cube, targets in noduleGen.valGen:
	#	print imgNum, cube.shape, cube.mean(), targets['nodule']



	#inputShape = images[0].shape

	#classifier = buildClassifier(inputShape, delLayers=args.delLayers, sharedNeurons=args.sharedNeurons, cancerNeurons=args.cancerNeurons, ageWeight=args.ageWeight)

	inputShape = (cubeSize, cubeSize, cubeSize)
	model = buildModel(inputShape, args)


	#stratifiedQ = Batcher( tGen, vGen,
	#	trainStratified=noduleGen.trainGen,
	#	valStratified=noduleGen.valGen, batchSize=args.batchSize)


	stratifiedQ = Batcher(
		candidateGen.trainGen, candidateGen.valGen,
		trainStratified=noduleGen.trainGen, valStratified=noduleGen.valGen,
		batchSize=args.batchSize)

	#tester = testDSBdata(period=10, numImages=2)

	#cAUC = ComputeAUC(batch=dq.valBatch, prefix='v_', period=50)
	#cAUCt = ComputeAUC(batch=dq.trainBatch, prefix='t_', period=50)
	tb = TensorBoard(log_dir=OUTDIR + 'nodules/', write_graph=False, histogram_freq=1)
	#tb = MyTensorBoard2(log_dir=OUTDIR+'tbfinetune/', write_graph=False, otherCallbacks=[cAUC, cAUCt])

	lh = fgLogger(OUTDIR)

	model.fit_generator(
		stratifiedQ.trainGen,
		verbose=2,
		epochs=200,
		samples_per_epoch=args.batchSize,
		#steps_per_epoch=1,
		callbacks=[tb, lh],
		#class_weight=classWeight,
		validation_data=stratifiedQ.valGen,
		validation_steps=1,
		#workers=1,
		#max_q_size=20
		#nb_val_samples=64
	)

	model.save(OUTDIR + 'lungAutoencoder.h5')

	lh.writeLog(lastN=100)

	lh.writeToSQL(OUTDIR+'exp.tsv', args, table='noduleExp2')











