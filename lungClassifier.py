#!/usr/bin/env python

#np.random.seed(31337)

import numpy
from keras import backend as K
from keras.callbacks import TensorBoard

K.set_floatx('float32')
K.set_image_dim_ordering('tf')
from callbacks import fgLogger



DATADIR = '/ssd/'
DATASET = DATADIR + 'cams.h5'

from keras.layers import Dense, BatchNormalization, Flatten, Dropout
from keras.layers import Input, Convolution3D, MaxPooling3D, UpSampling3D
from keras.models import Model
from utils import ImageArray, prepOutDir



def buildModel(inputShape, args):

	print inputShape

	input_img = Input(shape=inputShape)
	x = input_img

	if args.batchNorm: x = BatchNormalization()(x)

	fo = args.filters
	fi = fo*2

	b = args.bconv

	x = Convolution3D(fo, (b, b, b), activation='relu', padding='same')(x)
	if args.doubleLayers: x = Convolution3D(fo, (3, 3, 3), activation='relu', padding='same')(x)
	x = MaxPooling3D(name='pool1')(x)
	#x = MaxPooling3D(name='pool1', pool_size=(3,3,3))(x)

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
	nodule = Dense(1, activation='sigmoid', name='cancer')(shared)


	loss = {'cancer': 'binary_crossentropy'}
	metrics = {'cancer': 'accuracy'}

	if args.autoencoder:

		loss['imgOut'] = 'mse'
		metrics['imgOut'] = 'mae'

		x = UpSampling3D()(x)
		x = Convolution3D(fi, (3, 3, 3), activation='relu', padding='same')(x)
		if args.doubleLayers: x = Convolution3D(fi, (3, 3, 3), activation='relu', padding='same')(x)

		x = UpSampling3D()(x)
		#x = UpSampling3D(size=(3,3,3))(x)
		if args.doubleLayers: x = Convolution3D(fo, (3, 3, 3), activation='relu', padding='same')(x)
		decoded = Convolution3D(1, (3, 3, 3), activation='relu', padding='same', name='imgOut')(x)

		model = Model(inputs=[input_img], outputs=[nodule, decoded], name='multiOut')

	else:

		model = Model(inputs=[input_img], outputs=[nodule], name='multiOut')



	print 'hidden layer shape: ', encoder.output_shape

	# optomizers: adadelta, sgd, rmsprop, adam, nadam

	model.compile(optimizer=args.optimizer, loss=loss, metrics=metrics)
	print model.summary()
	return model








if __name__ == '__main__':

	import argparse

	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('-_id', dest='_id', default=None, type=str)
	parser.add_argument('-sharedNeurons', dest='sharedNeurons', default=16, type=int)
	parser.add_argument('-optimizer', dest='optimizer', default='sgd', type=str)
	parser.add_argument('-bconv', dest='bconv', default=3, type=int)
	parser.add_argument('-autoencoder', dest='autoencoder', default=1, type=int)
	parser.add_argument('-batchNorm', dest='batchNorm', default=0, type=int)
	parser.add_argument('-batchSize', dest='batchSize', default=16, type=int)
	parser.add_argument('-filters', dest='filters', default=4, type=int)
	parser.add_argument('-dropout', dest='dropout', default=0, type=float)
	parser.add_argument('-doubleLayers', dest='doubleLayers', default=1, type=int)


	#parser.add_argument('-optimizer', dest='optimizer', default='adadelta', type=str)

	#parser.add_argument('-scale', dest='scale', default="256.0", type=float)
	#parser.add_argument('-subFrac', dest='subFrac', default="0.0", type=float)




	args = parser.parse_args()
	print args

	autoencoder = args.autoencoder

	EXPERIMENT_ID = 'lungExp2'

	if args._id: OUTDIR = '/home/cosmo/lung/jobs/%s/%s/' % ( EXPERIMENT_ID, args._id)
	else:  OUTDIR = '/modelState/'
	prepOutDir(OUTDIR, __file__)



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
		#print cancer
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

	imgShape = camImages.array[0].shape
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
		trainGen(),
		verbose=2,
		epochs=200,
		samples_per_epoch=args.batchSize,
		steps_per_epoch=8,
		callbacks=[tb, tb2, lh],
		#class_weight=classWeight,
		validation_data=valGen(),
		validation_steps=10,
		#workers=1,
		#max_q_size=20
		#nb_val_samples=64
	)

	model.save(OUTDIR + 'model.h5')

	#lh.writeLog(lastN=100)

	lh.writeToSQL(OUTDIR+'exp.tsv', args, table='lungExp2')








