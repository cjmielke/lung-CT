import collections
import threading
from Queue import Queue
from random import shuffle

import numpy
from keras import backend as K


'''
Current experiment : playing with subtracted means ..... its possibly working better now

'''


BATCH_SIZE = 32
VALIDATION_SIZE = 100

VALIDATION_SPLIT = 0.2

from matplotlib import cm


def myChannelMean(x, dim_ordering='default'):
	if dim_ordering == 'default':
		dim_ordering = K.image_dim_ordering()
	assert dim_ordering in {'tf', 'th'}
	# for jet
	c1 = 202.49
	c2 = 90.58
	c3 = 16.76

	if dim_ordering == 'th':
		# 'RGB'->'BGR'
		#x = x[:, ::-1, :, :]
		# Zero-center by mean pixel
		x[:, 0, :, :] -= c1
		x[:, 1, :, :] -= c2
		x[:, 2, :, :] -= c3
	else:
		# 'RGB'->'BGR'
		#x = x[:, :, :, ::-1]
		# Zero-center by mean pixel
		x[:, :, :, 0] -= c1
		x[:, :, :, 1] -= c2
		x[:, :, :, 2] -= c3
	return x




def prepVGGimages(imageBatch, divisor=4096.0, threshold=0, heat=False, scale=256.0, subFrac=0.0):
	#print imageBatch.shape
	#imageBatch = np.expand_dims(imageBatch, axis=1)
	#print imageBatch.shape
	inputImgs = []
	for i in imageBatch:
		if threshold:
			mask = i[i < threshold]

		i = i.astype(K.floatx())
		i = 1.0*i / (1.0*divisor)


		if heat:
			#i = np.power(i,4)
			#i = np.log(i+1.0)
			c = cm.jet(i)
			#print c[:,:,1].min(), c[:,:,1].max()
			#c = c*256.0
			#c = c - 125.0
			#b = c[:,:,2] - 103.939
			#g = c[:,:,1] - 116.779
			#r = c[:,:,0] - 123.68

			b = c[:,:,2]
			g = c[:,:,1]
			r = c[:,:,0]
			if threshold:
				b[mask] = 0
				g[mask] = 0
				r[mask] = 0
			#inputImgs.append([b, g, r])
			i = numpy.dstack([b, g, r])
			#print 'max pixel value : ', i.max()
		else:
			#i = i - 2048.0
			#i = i / divisor
			# print i.min(), i.max()		#about -100 to 100
			#inputImgs.append([i, i, i])
			i = numpy.dstack([i, i, i])

		i *= scale
		i -= subFrac*i.max()


		i = numpy.expand_dims(i, axis=0)
		#i = preprocess_input(i)
		#i = myChannelMean(i)
		i = i[0]
		#print i.min(), i.mean(), i.max()


		inputImgs.append(i)


	inputImgs = numpy.asarray(inputImgs)
	#print inputImgs.shape

	# print inputImgs.shape
	#inputImgs = np.reshape(inputImgs, (len(inputImgs), sx, sy, 3))
	#inputImgs = np.expand_dims(inputImgs, axis=0)

	#print inputImgs.shape

	return inputImgs




import time
def testGen():
	i=0
	while True:
		i+=1
		time.sleep(0.02)
		yield i



def threaded(fn):
	def wrapper(*args, **kwargs):
		thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
		thread.setDaemon(True)
		thread.start()
		return thread
	return wrapper



def dataGenAutoencoder(images, metadata, VGGprep=False, normalize=False, targets=False, binary=True):
	maxIndex = images.nrows
	#maxIndex = 1000
	while True:
		n = 0
		while n < maxIndex-10:
			image = images[n]
			r = metadata[n]

			#print image.shape

			if image.dtype!=K.floatx(): image = image.astype(K.floatx())
			if normalize:
				image = image*1.0/4096


			if VGGprep:
				image = numpy.expand_dims(image, axis=0)
				image = prepVGGimages(image)[0]

			image = numpy.expand_dims(image, axis=2)


			if targets:
				targets = {
					'predictions': numpy.float(r['thisBreastCancer']),
					'age': numpy.uint8(r['age'] + 60) * 1.0 - 60.0,
					'bmi': numpy.float(r['bmi']),
					'invasive': numpy.float(r['thisBreastInvasive']),
					'firstDegreeBc': numpy.float(r['firstDegreeBc']),
					'firstDegreeBcYoung': numpy.float(r['firstDegreeBcYoung']),
					'imgOut': image
				}

				if binary:
					age = targets['age']
					targets['age'] = (age > 0) * 1.0

					bmi = targets['bmi']
					targets['bmi'] = (bmi > 0) * 1.0

				yield [n, image, targets]

			else:
				yield [n, image, image]

			n+=1



def dataGenMultiTargetInterleaved(images, metadata, VGGprep=False, heat=True, binary=False, scale=256.0, subFrac=0.0, autoencoder=False):
	maxIndex = images.nrows
	#assert maxIndex > 1000, 'not enough data!'
	#maxIndex = 1000
	while True:
		n = 0
		while n < maxIndex-10:
			image = images[n]
			r = metadata[n]

			if image.dtype!=K.floatx(): image = image.astype(K.floatx())

			if VGGprep:
				image = numpy.expand_dims(image, axis=0)
				image = prepVGGimages(image, heat=heat, scale=scale, subFrac=subFrac)[0]

			if autoencoder: image = numpy.expand_dims(image, axis=2)

			#print image.shape

			targets = {
				'predictions': numpy.float(r['thisBreastCancer']),
				'age': numpy.uint8(r['age'] + 60) * 1.0 - 60.0,
				'bmi': numpy.float(r['bmi']),
				'invasive': numpy.float(r['thisBreastInvasive']),
				'firstDegreeBc': numpy.float(r['firstDegreeBc']),
				'firstDegreeBcYoung': numpy.float(r['firstDegreeBcYoung'])
			}

			if binary:
				age = targets['age']
				targets['age'] = (age>0) * 1.0

				bmi = targets['bmi']
				targets['bmi'] = (bmi > 0) * 1.0

			if autoencoder: targets['imgOut'] = image


			yield [n, image, targets]

			n+=1







class DataQueue:
	def __init__(self, gen, maxLength = 200, batchSize=64, stratified=False):

		self.sourceGen = gen
		self.trainQ = Queue(maxsize=maxLength)
		self.valQ = Queue(maxsize=maxLength)
		self.running = False
		self.batchSize = batchSize
		self.stratified = stratified			# stratified sampling

		if stratified:
			self.trainQpos = Queue(maxsize=maxLength)
			self.trainQneg = Queue(maxsize=maxLength)


		self.h = self.sourceLooper()
		self.running = True

		time.sleep(1)

		self.trainGen = self.trainingGen()
		self.valGen = self.validationGen()


	@threaded
	def sourceLooper(self):
		while True:
			if not self.running:
				time.sleep(1)
				continue

			tup = self.sourceGen.next()

			imageNum, image, targets = tup

			if imageNum%2==0:
				try: self.trainQ.put_nowait(tup)
				except: pass

				if self.stratified:
					try:
						if targets['predictions'] == 1:
							self.trainQpos.put_nowait(tup)
						else:
							self.trainQneg.put_nowait(tup)
					except: pass

			else:
				try: self.valQ.put_nowait(tup)
				except: continue



	def unpack(self, batch):
		_, X, Y = zip(*batch)
		X = numpy.asarray(X)
		#print X[:,:,:,0].min(), X[:,:,:,0].max(), X[:,:,:,0].mean(), X.shape

		if isinstance(Y[0], dict):
			keys = Y[0].keys()

			Yd = {}
			for k in keys:
				Yd[k] = numpy.asarray([r[k] for r in Y])

			Y=Yd
		else: Y = numpy.asarray(Y)

		return X, Y


	def trainBatch(self, N=None, stratified=False):
		if not N: N=self.batchSize

		if not stratified:
			batch = [self.trainQ.get() for _ in range(N)]
			for n, _, _ in batch: assert n%2==0
		else:
			N=N/2
			batchA = [self.trainQpos.get() for _ in range(N)]
			batchB = [self.trainQneg.get() for _ in range(N)]
			batch = batchA+batchB
			shuffle(batch)
			for n, _, _ in batch: assert n%2==0

		numPos = len([a for a in batch if a[2]['predictions']])
		#print numPos, len(batch)					# prove that batches are the right balance for training/testing

		return self.unpack(batch)

	def trainingGen(self):
		while True:
			yield self.trainBatch(stratified=self.stratified)



	def valBatch(self, N=None):
		if not N: N=self.batchSize
		batch = [self.valQ.get() for _ in range(N)]
		for n, _, _ in batch: assert n%2==1
		return self.unpack(batch)

	def validationGen(self):
		while True:
			yield self.valBatch()







class DataRing:
	def __init__(self, gen, maxLength = 100, delay=0.001, batchSize=64):

		#self.sourceGen = dataGenMultiTargetInterleaved(images, metadata, VGGprep=VGGprep)
		self.sourceGen = gen
		self.deque = collections.deque(maxlen=maxLength)
		self.running = False
		self.delay = delay
		self.batchSize = batchSize
		#self.lock = threading.Lock()

		self.h = self.sourceLooper()
		self.running = True

		time.sleep(1)

	@threaded
	def sourceLooper(self):
		while True:
			if not self.running:
				time.sleep(1)
				continue
			tup = self.sourceGen.next()

			self.deque.appendleft(tup)
			#if self.delay: time.sleep(self.delay)


	def unpack(self, batch):
		_, X, Y = zip(*batch)
		X = numpy.asarray(X)
		#print X[:,:,:,0].min(), X[:,:,:,0].max(), X[:,:,:,0].mean(), X.shape
		#print X[:,:,:,1].min(), X[:,:,:,1].max(), X[:,:,:,1].mean(), X.shape
		#print X[:,:,:,2].min(), X[:,:,:,2].max(), X[:,:,:,2].mean(), X.shape
		#print

		if isinstance(Y[0], dict):
			keys = Y[0].keys()

			Yd = {}
			for k in keys:
				Yd[k] = numpy.asarray([r[k] for r in Y])

			Y=Yd
		else: Y = numpy.asarray(Y)

		return X, Y

	def trainingGen(self):
		while True:
			batch = [(n, x, y) for n, x, y in list(self.deque) if (n/1)%2==0][0:self.batchSize]
			yield self.unpack(batch)

	def validationGen(self):
		while True:
			batch = [(n, x, y) for n, x, y in list(self.deque) if (n/1)%2==1][0:self.batchSize]
			#indexen = [n for n,_,_ in batch]
			#print 'range :', min(indexen), max(indexen)
			yield self.unpack(batch)




	def trainingBatch(self):
		while True:
			batch = [(n, x, y) for n, x, y in list(self.deque) if (n / 1) % 2 == 0]
			return self.unpack(batch)


	def validationBatch(self):
		while True:
			batch = [(n, x, y) for n, x, y in list(self.deque) if (n / 1) % 2 == 1]
			return self.unpack(batch)





def dataGen(images, metadata, validation=False, batchSize=BATCH_SIZE, VGGprep=False):
	while True:

		if validation:								# validation set is ending segment of dataset
			n = images.nrows - VALIDATION_SIZE
			maxIndex = images.nrows
		else:
			n=0
			maxIndex = images.nrows - VALIDATION_SIZE - batchSize

		# intentionally break it - this sanity test indeed produces magical classifiers
		#n = images.nrows - VALIDATION_SIZE
		#maxIndex = images.nrows

		while n < maxIndex:
			to = n + batchSize

			imageBatch = images[n:to]
			if imageBatch.dtype!=K.floatx(): imageBatch = imageBatch.astype(K.floatx())
			#print imageBatch.dtype



			if VGGprep: imageBatch = prepVGGimages(imageBatch)


			#metadata = 

			predictions = [row['thisBreastCancer'] for row in metadata[n:to]]
			predictions = numpy.asarray(predictions)
			predictions = numpy.reshape(predictions, len(predictions), 1)
			#print predictions.shape, imageBatch.shape

			imageIDs = [r['imageNum'] for r in metadata[n:to]]

			'''
			if validation:
				validationIDs.update(imageIDs)
				assert len(set(imageIDs).intersection(trainingIDs)) == 0 
				assert len(validationIDs.intersection(trainingIDs)) == 0 
			else: trainingIDs.update(imageIDs)
			'''



			yield [imageBatch, predictions]			# autoencoder

			n = to






def dataGenMultiTarget(images, metadata, validation=False, batchSize=BATCH_SIZE, VGGprep=False):
	while True:

		VALIDATION_SIZE = 250
		#print 'Validation size : ', VALIDATION_SIZE

		if validation:		# validation is beginning of dataset
			n=0
			maxIndex = VALIDATION_SIZE
		else:
			n = VALIDATION_SIZE
			maxIndex = metadata.nrows


		while n < maxIndex-1:
			to = n + batchSize
			to = min(to, maxIndex-1)

			#print validation, n, to

			imageBatch = images[n:to]

			if imageBatch.dtype!=K.floatx(): imageBatch = imageBatch.astype(K.floatx())
			#print imageBatch.dtype

			if VGGprep: imageBatch = prepVGGimages(imageBatch)

			predictions = numpy.asarray([r['thisBreastCancer'] for r in metadata[n:to]])#.astype(K.floatx())
			invasive = numpy.asarray([r['thisBreastInvasive'] for r in metadata[n:to]])#.astype(K.floatx())
			bmi = numpy.asarray([r['bmi'] for r in metadata[n:to]])
			age = numpy.asarray([numpy.uint8(r['age'] + 60) * 1.0 - 60.0 for r in metadata[n:to]]).astype(K.floatx())

			#imageIDs = [r['imageNum'] for r in metadata[n:to]]

			'''
			if validation:
				validationIDs.update(imageIDs)
				assert len(set(imageIDs).intersection(trainingIDs)) == 0 
				assert len(validationIDs.intersection(trainingIDs)) == 0 
			else: trainingIDs.update(imageIDs)
			'''

			targets = {
				'predictions': predictions,
				'age': age,
				'bmi': bmi,
				'invasive': invasive
			}


			yield [imageBatch, targets]			# autoencoder

			n = to








