import tables
from random import shuffle

import numpy
import pandas
from numpy.random import random

from utils import getImage, prepCube, getImageCubes
from utils import normalizeStd, findNodules

BATCH_SIZE = 32
VALIDATION_SIZE = 100

VALIDATION_SPLIT = 0.2


def getNoduleDiameter(row):
	if row is None: return 0.0
	else: diam = row.get('diameter_mm',0.0)
	return diam


def candidateGen(imageArray, candidatesDF, cubeSize=20, autoencoder=False):
	'''
	this generator loops over candidates only
	SLOW AS FUCK! but only if training and validation are simultaneously used ....
	'''
	oldImgNum = None
	while True:
		for index, row in candidatesDF.iterrows():

			if row['imgNum'] != oldImgNum:
				print ' new image loded =============='
				image, imageNum = getImage(imageArray, row)
				oldImgNum = imageNum

			normImg = normalizeStd(image.clip(min=-1000, max=700))

			voxelC = row[['voxelZ', 'voxelY', 'voxelX']].as_matrix()

			p = cubeSize/2
			for i in [0,1,2]: voxelC[i] = numpy.clip(voxelC[i], p, image.shape[i]-p )
			zMin, zMax = int(voxelC[0]-p), int(voxelC[0]+p)
			yMin, yMax = int(voxelC[1]-p), int(voxelC[1]+p)
			xMin, xMax = int(voxelC[2]-p), int(voxelC[2]+p)

			cube = image[ zMin: zMax, yMin: yMax, xMin: xMax ]
			assert cube.shape == (cubeSize, cubeSize, cubeSize)

			#print cube.min(), cube.mean(), cube.max()

			if cube.mean() < -3000:
				# print cube.mean()
				# print 'empty cube'
				continue

			cube = normImg[zMin: zMax, yMin: yMax, xMin: xMax]
			# print cube.min(), cube.mean(), cube.max()

			isCancerous = row['class']

			targets = {
				'nodule': isCancerous,
				#'candidate': 1
			}

			imgOut = numpy.expand_dims(cube, axis=3)
			if autoencoder: targets['imgOut'] = imgOut

			cube = numpy.expand_dims(cube, axis=3)

			# print pos, cube.mean(), numNodules, numCandidates
			#gc.collect()
			yield [imageNum, cube, targets]








class CubeGen:

	def __init__(self, arrayFile, trainImagesDF, valImagesDF, autoencoder=False, cubeSize=None, tsvFile=None):

		if not tsvFile: tsvFile = arrayFile.replace('.h5', '.tsv')

		noduleDF = pandas.read_csv(tsvFile, sep='\t')#.merge(imageDF, on='imgNum')
		self.DB = tables.open_file(arrayFile, mode='r')
		self.array = self.DB.root.cubes

		print 'pytables array length: ', len(self.array)

		if len(noduleDF) < 1500:
			print 'storing in ram!'
			self.array = []
			for _, row in noduleDF.iterrows():
				noduleNum = int(row['noduleNum'])
				cube = self.DB.root.cubes[noduleNum]
				#print cube.mean()
				self.array.append(cube)

			#self.array[4] *= 0		# tag a single array to see where it pops out

		storedCubeSize = self.array[0].shape[0]
		if cubeSize: self.cubeSize = cubeSize
		else: self.cubeSize = storedCubeSize		# else, infer from whats stored
		assert self.cubeSize <= storedCubeSize

		self.tNodulesDF = noduleDF.merge(trainImagesDF, on='imgNum').copy(deep=True)
		self.vNodulesDF = noduleDF.merge(valImagesDF, on='imgNum').copy(deep=True)

		# initialize the generators!
		self.trainGen = self._trainGen()
		self.valGen = self._valGen()

		print 'Length of train/val cubes sets : ', len(self.tNodulesDF), len(self.vNodulesDF)


		if 'class' in noduleDF.columns:
			print 'candidates file given, filtering out nodules'
			noduleDF = noduleDF[noduleDF['class']==0.0]
			self.allNodules=False
			print 'number of nodules in training: ', len(self.tNodulesDF[self.tNodulesDF['class']==1.0])
			print 'number of non-nodules in training: ', len(self.tNodulesDF[self.tNodulesDF['class']!=1.0])
			print 'number of nodules in val: ', len(self.vNodulesDF[self.vNodulesDF['class']==1.0])
			print 'number of non-nodules in val: ', len(self.vNodulesDF[self.vNodulesDF['class']!=1.0])
		else: self.allNodules=True

		trainingNodules = set(self.tNodulesDF.noduleNum)
		valNodules = set(self.vNodulesDF.noduleNum)
		assert len(trainingNodules.intersection(valNodules))==0

		self.autoencoder = autoencoder

	def ret(self, row):
		noduleNum = int(row['noduleNum'])
		assert noduleNum < len(self.array)
		cube = self.array[noduleNum]
		cube = prepCube(cube)
		#normImg = normalizeStd(image.clip(min=-1000, max=700))

		if self.allNodules: nodule=1.0
		else: nodule = row['class']

		targets = {
			'nodule': nodule,
			'diam': getNoduleDiameter(row),
			#'candidate': 1.0
		}

		if self.autoencoder: targets['imgOut'] = cube
		return [row['imgNum'], cube, targets]

	def _trainGen(self):
		while True:
			for _, row in self.tNodulesDF.iterrows():
				res = self.ret(row)
				#print res[1].mean()
				yield res

	def _valGen(self):
		while True:
			for _, row in self.vNodulesDF.iterrows():
				res = self.ret(row)
				#i, c, t = res			# sanity checks
				#c = numpy.flip(c, axis=0)
				#c = numpy.flip(c, axis=1)
				yield res



# outer loop is images - inner loop is over cubes ... returning labels based on dataframe lookups

def imageCubeGen(imageArray, imageDF, noduleDF, candidatesDF, cubeSize=32, autoencoder=False, atMost=200):
	while True:
		for _, row in imageDF.iterrows():

			image, imageNum = getImage(imageArray, row)

			imgNodules = noduleDF[noduleDF.imgNum==imageNum]
			imgCandidates = candidatesDF[candidatesDF.imgNum==imageNum]

			#if autoencoder: image = numpy.expand_dims(image, axis=2)


			cubes, positions = getImageCubes(image, cubeSize, prep=True)
			for cube, pos in zip(cubes, positions):
				realPos = pos*cubeSize
				z, y, x = realPos

				#if random() > 0.4 and cube.mean() < 300: continue			# doesn't  work! its normalized now

				# is there a nodule or candidates in this cube?
				nodulesInCube = findNodules(imgNodules, x, y, z, cubeSize)
				numNodules = len(nodulesInCube)
				candidatesInCube = findNodules(imgCandidates, x, y, z, cubeSize)
				numCandidates = len(candidatesInCube)

				if (numNodules or numCandidates): print 'Candidates: %s Nodules: %s' % (numCandidates, numNodules)
				#else: print '-'*10
				#if numCandidates: plot_nodule(cube)

				if len(nodulesInCube):
					diam = getNoduleDiameter(nodulesInCube.iloc[0])
				else: diam = getNoduleDiameter(None)

				targets = {
					'nodule': (numNodules>0)*1.0,
					#'candidate': (numCandidates>0)*1.0,
					'diam' : diam
				}

				if autoencoder: targets['imgOut'] = cube


				#print pos, cube.mean(), numNodules, numCandidates

				yield [imageNum, cube, targets]




class Batcher:
	'''
	Handles train/validation splits, as well as stratification of labels, from a myriad of generators
	'''


	def __init__(self, trainG, valG, trainStratified=None, valStratified=None, batchSize=8):

		self.trainG = trainG
		self.valG = valG
		self.batchSize = batchSize
		self.trainStratified = trainStratified
		self.valStratified = valStratified

		self.trainGen = self.trainingGen()
		self.valGen = self.validationGen()


	def unpack(self, batch):
		_, X, Y = zip(*batch)
		#print len(X), X[0].shape
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
		if not N: N = self.batchSize

		if not stratified:
			batch = [self.trainG.next() for _ in range(N)]
		else:
			N=N/2
			batchA = [self.trainG.next() for _ in range(N)]
			batchB = [self.trainStratified.next() for _ in range(N)]
			batch = batchA+batchB
			del batchA
			del batchB
			shuffle(batch)
			numPos = len([a for a in batch if a[2]['nodule']])
			#assert numPos * 2 == len(batch)

		#for n, _, _ in batch: assert n % 2 == 0
		#for n, c, t in batch: assert c.mean() != 0.0
		#numPos = len([a for a in batch if a[2]['nodule']])
		print 'POS/BATCHSIZE %s/%s' % (numPos, len(batch))					# prove that batches are the right balance for training/testing
		#print len(batch)
		return self.unpack(batch)

	def trainingGen(self):
		while True: yield self.trainBatch(stratified=self.trainStratified)



	def valBatch(self, N=None, stratified=False):
		if not N: N=self.batchSize

		if not stratified:
			batch = [self.valG.next() for _ in range(N)]
		else:
			N=N/2
			batchA = [self.valG.next() for _ in range(N)]
			batchB = [self.valStratified.next() for _ in range(N)]
			batch = batchA+batchB
			del batchA
			del batchB
			shuffle(batch)
			numPos = len([a for a in batch if a[2]['nodule']])
			#assert numPos*2==len(batch)

		#numPos = len([a for a in batch if a[2]['nodule']])
		#for n, c, t in batch: assert c.mean() != 0.0
		print 'val POS/BATCHSIZE %s/%s' % (numPos, len(batch))					# prove that batches are the right balance for training/testing

		#for n, _, _ in batch: assert n%2==1
		return self.unpack(batch)

	def validationGen(self):
		while True: yield self.valBatch(stratified=self.valStratified)




'''


def camGen(imageArray, imageDF, noduleDF, candidatesDF, cubeSize=32, autoencoder=False, atMost=200):
	while True:
		for _, row in imageDF.iterrows():

			image, imageNum = getImage(imageArray, row)

			imgNodules = noduleDF[noduleDF.imgNum==imageNum]
			imgCandidates = candidatesDF[candidatesDF.imgNum==imageNum]

			#if autoencoder: image = numpy.expand_dims(image, axis=2)


			cubes, positions = getImageCubes(image, cubeSize, prep=True)
			for cube, pos in zip(cubes, positions):
				realPos = pos*cubeSize
				z, y, x = realPos

				#if random() > 0.4 and cube.mean() < 300: continue			# doesn't  work! its normalized now

				# is there a nodule or candidates in this cube?
				nodulesInCube = findNodules(imgNodules, x, y, z, cubeSize)
				numNodules = len(nodulesInCube)
				candidatesInCube = findNodules(imgCandidates, x, y, z, cubeSize)
				numCandidates = len(candidatesInCube)

				if (numNodules or numCandidates): print 'Candidates: %s Nodules: %s' % (numCandidates, numNodules)
				#else: print '-'*10
				#if numCandidates: plot_nodule(cube)

				if len(nodulesInCube):
					diam = getNoduleDiameter(nodulesInCube.iloc[0])
				else: diam = getNoduleDiameter(None)

				targets = {
					'nodule': (numNodules>0)*1.0,
					#'candidate': (numCandidates>0)*1.0,
					'diam' : diam
				}

				if autoencoder: targets['imgOut'] = cube


				#print pos, cube.mean(), numNodules, numCandidates

				yield [imageNum, cube, targets]







'''


