import json
import os
from pkg_resources import parse_version

import numpy as np
from keras import backend as K
from keras.callbacks import Callback
from scipy import integrate
from sklearn.metrics import roc_auc_score, roc_curve


def confusionFIXME(y_true, y_pred):
	y_pred_pos = K.round(K.clip(y_pred, 0, 1))
	y_pred_neg = 1 - y_pred_pos
	y_pos = K.round(K.clip(y_true, 0, 1))
	y_neg = 1 - y_pos
	tp = K.sum(y_pos * y_pred_pos) / (K.sum(y_pos) + K.epsilon())
	tn = K.sum(y_neg * y_pred_neg) / (K.sum(y_neg) + K.epsilon())
	return {'TPOS': tp, 'TNEG': tn}


# FIXME - doesn't work ... need to express things in tensorflow!
def AUC(y_true, y_score):
	y_true = y_true.eval()
	y_score = y_score.eval()


	auc = roc_auc_score(y_true, y_score, average='macro', sample_weight=None)
	print 'AUC : ', auc, 'y_score mean: ', y_score.mean()

	# extract and integrate ROC curve manually
	fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=None, sample_weight=None, drop_intermediate=True)

	auc = integrate.trapz(tpr, fpr)

	# now shift the y points down
	tprShift = tpr - 0.8
	tprShift[tprShift < 0.0] = 0.0

	pauc = integrate.trapz(tprShift, fpr)

	return {'AUC': auc, 'pAUC': pauc}




class fgLogger(Callback):
	def __init__(self, logDir):
		self.logDir = logDir
		self.log = {}

	def on_train_begin(self, logs={}):
		print 'keys in logs: ', logs.keys()

	def on_epoch_end(self, epoch, logs={}):
		for k in logs.keys():
			if k not in self.log: self.log[k] = []
			self.log[k].append(logs[k])

		#print self.log.keys()
		#print self.log

		self.writeLog(lastN=1)


	def writeLog(self, lastN=100):
		if 'loss' not in self.log: return
		if 'val_loss' not in self.log: return
		f = os.path.join(self.logDir, 'scores.json')
		with open(f, 'w') as lf:
			o = {
				'_scores': {
					'loss_t': round(np.asarray(self.log['loss'])[-lastN:].mean(), 2),
					'loss_v':  round(np.asarray(self.log['val_loss'])[-lastN:].mean(), 2)
				}
			}
			if 't_AUC' in self.log:
				o['_scores']['AUC_t'] = round(np.asarray(self.log['t_AUC']).mean(), 2)
			if 'v_AUC' in self.log:
				o['_scores']['AUC_v'] = round(np.asarray(self.log['v_AUC']).mean(), 2)
			if 'predictions_fmeasure' in self.log:
				o['_scores']['F_t'] = round(np.asarray(self.log['predictions_fmeasure'])[-lastN:].mean(), 2)
			if 'val_predictions_fmeasure' in self.log:
				o['_scores']['F_v'] = round(np.asarray(self.log['val_predictions_fmeasure'])[-lastN:].mean(), 2)

			#print o
			j = json.dumps(o)
			lf.write(j)





		f = os.path.join(self.logDir, 'chart.json')
		with open(f, 'w') as lf:
			o = {
				'_charts': [
					{
						'axis': {
							'x': {'label': 'Iteration'},
							'y': {'label': 'Loss'}
						},
						'columnNames': ['train', 'val'],
						'data': {
							'columns': [
								self.log['loss'],
								self.log['val_loss']
							]
						}
					},

				]
			}


			if 'predictions_fmeasure' in self.log:
				chart = {
						'axis': {
							'x': {'label': 'Iteration'},
							'y': {'label': 'fmeasure'}
						},
						'columnNames': ['train', 'val'],
						'data': {
							'columns': [
								self.log['predictions_fmeasure'],
								self.log['val_predictions_fmeasure']
							]
						}
					}
				o['_charts'].append(chart)

			if 't_AUC' in self.log and 'v_AUC' in self.log:
				chart = {
						'axis': {
							'x': {'label': 'Iteration'},
							'y': {'label': 'AUC'}
						},
						'columnNames': ['train', 'val'],
						'data': {
							'columns': [
								self.log['t_AUC'],
								self.log['v_AUC']
							]
						}
					}
				o['_charts'].append(chart)


			j = json.dumps(o)
			lf.write(j)

















class ComputeAUC(Callback):
	def __init__(self, prefix='', batch=None, period=500, numBatches = 40):
		self.batch = batch
		self.prefix = prefix
		self.period = period
		self.numBatches = numBatches

	def on_epoch_end(self, epoch, logs={}):
		if epoch % self.period != 0: return 
		#if not hasattr(self, 'model'): return

		testY = []
		y_score = []
		for x in range(0,self.numBatches):
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











class MyTensorBoard2(Callback):
	''' Tensorboard basic visualizations.

	This callback writes a log for TensorBoard, which allows
	you to visualize dynamic graphs of your training and test
	metrics, as well as activation histograms for the different
	layers in your model.

	TensorBoard is a visualization tool provided with TensorFlow.

	If you have installed TensorFlow with pip, you should be able
	to launch TensorBoard from the command line:
	```
	tensorboard --logdir=/full_path_to_your_logs
	```
	You can find more information about TensorBoard
	[here](https://www.tensorflow.org/versions/master/how_tos/summaries_and_tensorboard/index.html).

	# Arguments
		log_dir: the path of the directory where to save the log
			files to be parsed by Tensorboard
		histogram_freq: frequency (in epochs) at which to compute activation
			histograms for the layers of the model. If set to 0,
			histograms won't be computed.
		write_graph: whether to visualize the graph in Tensorboard.
			The log file can become quite large when
			write_graph is set to True.
	'''

	def __init__(self, log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False, otherCallbacks=[], skipFirst=1):
		super(MyTensorBoard2, self).__init__()
		if K._BACKEND != 'tensorflow':
			raise RuntimeError('TensorBoard callback only works '
							   'with the TensorFlow backend.')
		self.log_dir = log_dir
		self.histogram_freq = histogram_freq
		self.merged = None
		self.write_graph = write_graph
		self.write_images = write_images
		self.skipFirst = skipFirst

		self.otherCallbacks = otherCallbacks

		import tensorflow as tf

		if parse_version(tf.__version__) >= parse_version('0.12.0'):
			self.merged = tf.summary.merge_all()
		else:
			self.merged = tf.merge_all_summaries()
		if self.write_graph:
			if parse_version(tf.__version__) >= parse_version('0.12.0'):
				self.writer = tf.summary.FileWriter(self.log_dir,
													self.sess.graph)
			elif parse_version(tf.__version__) >= parse_version('0.8.0'):
				self.writer = tf.train.SummaryWriter(self.log_dir,
													 self.sess.graph)
			else:
				self.writer = tf.train.SummaryWriter(self.log_dir,
													 self.sess.graph_def)
		else:
			if parse_version(tf.__version__) >= parse_version('0.12.0'):
				self.writer = tf.summary.FileWriter(self.log_dir)
			else:
				self.writer = tf.train.SummaryWriter(self.log_dir)
		print 'model set'

	def _set_model(self, model):
		import tensorflow as tf
		import keras.backend.tensorflow_backend as KTF

		self.model = model
		self.sess = KTF.get_session()
		if self.histogram_freq and self.merged is None:
			for layer in self.model.layers:

				for weight in layer.weights:
					tf.histogram_summary(weight.name, weight)

					if self.write_images:
						w_img = tf.squeeze(weight)

						shape = w_img.get_shape()
						if len(shape) > 1 and shape[0] > shape[1]:
							w_img = tf.transpose(w_img)

						if len(shape) == 1:
							w_img = tf.expand_dims(w_img, 0)

						w_img = tf.expand_dims(tf.expand_dims(w_img, 0), -1)

						tf.image_summary(weight.name, w_img)

				if hasattr(layer, 'output'):
					tf.histogram_summary('{}_out'.format(layer.name),
										 layer.output)


	def on_epoch_end(self, epoch, logs={}):

		if epoch < self.skipFirst:
			print 'Skipping first %s epochs' % self.skipFirst
			return 

		import tensorflow as tf


		for cb in self.otherCallbacks:
			cb.model = self.model
			ret = cb.on_epoch_end(epoch)
			if not ret:
				#print 'AUC ----  No return!'		# happens because of period!
				continue
			for k, v in ret.iteritems():
				logs[k] = v



		if self.model.validation_data and self.histogram_freq:
			if epoch % self.histogram_freq == 0:
				# TODO: implement batched calls to sess.run
				# (current call will likely go OOM on GPU)
				if self.model.uses_learning_phase:
					cut_v_data = len(self.model.inputs)
					val_data = self.model.validation_data[:cut_v_data] + [0]
					tensors = self.model.inputs + [K.learning_phase()]
				else:
					val_data = self.model.validation_data
					tensors = self.model.inputs
				feed_dict = dict(zip(tensors, val_data))
				result = self.sess.run([self.merged], feed_dict=feed_dict)
				summary_str = result[0]
				self.writer.add_summary(summary_str, epoch)

		for name, value in logs.items():
			if name in ['batch', 'size']:
				continue
			summary = tf.Summary()
			summary_value = summary.value.add()
			summary_value.simple_value = value.item()
			summary_value.tag = name
			self.writer.add_summary(summary, epoch)
		self.writer.flush()

	def on_train_end(self, _):
		self.writer.close()











