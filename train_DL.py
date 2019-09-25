from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
import pickle
from keras import layers, optimizers, models
from keras.utils import to_categorical
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, scale, MinMaxScaler
from sklearn.model_selection import train_test_split
import sys
from keras.utils import plot_model
from keras.regularizers import l2
from utils import DataGenerator
import tensorflow as tf

sys.path.insert(0, '/home/go96bix/projects/Masterarbeit/ML')
import DataParsing
import matplotlib

matplotlib.rcParams['backend'] = 'Agg'
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.utils import class_weight as clw
import sklearn.metrics as metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from scipy.optimize import minimize
from keras import backend as K
import math
from sklearn.metrics import roc_auc_score
from keras.engine import Layer
import tensorflow_hub as hub


class ElmoEmbeddingLayer(Layer):
	def __init__(self, **kwargs):
		self.dimensions = 1024
		self.trainable = True
		super(ElmoEmbeddingLayer, self).__init__(**kwargs)

	def build(self, input_shape):
		self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
							   name="{}_module".format(self.name))

		self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
		super(ElmoEmbeddingLayer, self).build(input_shape)

	def call(self, x, mask=None):
		result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1),
						   as_dict=True,
						   signature='default',
						   )['default']
		return result

	def compute_mask(self, inputs, mask=None):
		return K.not_equal(inputs, '--PAD--')

	def compute_output_shape(self, input_shape):
		return (input_shape[0], self.dimensions)


def set_trainability(model, trainable=False):
	model.trainable = trainable
	for layer in model.layers:
		layer.trainable = trainable


def auc_10_perc_fpr(y_true, y_pred):
	def my_roc_auc_score(y_true_np, y_pred_np, max_fpr=0.1):
		return roc_auc_score(y_true_np, y_pred_np, max_fpr=max_fpr)

	return tf.py_func(my_roc_auc_score, (y_true, y_pred), tf.double)


def calc_n_plot_ROC_curve(y_true, y_pred, name="best", plot=True):
	fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
	auc = metrics.roc_auc_score(y_true, y_pred)
	# fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=2)
	optimal_idx = np.argmax(np.abs(tpr - fpr))

	# alternatives
	# optimal = tpr / (tpr + fpr)
	fnr = 1 - tpr
	tnr = 1 - fpr
	# precision = tpr / (tpr + fpr)
	# recall = tpr / 1
	# optimal = 2*((precision*recall)/(precision+recall))
	# acc = (tpr[i] + (1 - fpr[i])) / 2
	# optimal = tpr/(tpr+fpr+fnr)
	# wenn gewuenscht das relatativ hohe precision
	# acc mit hoeher gewichteter fpr
	# optimal = (tpr+tnr)/((fpr*5)+fnr+tpr+tnr)
	# optimal = (tpr+tnr)/((fpr)+fnr+tpr+tnr)
	# optimal[np.isnan(optimal)] = 0
	# optimal_idx = np.argmax(optimal)
	# optimal_threshold = thresholds[i][optimal_idx]
	optimal_threshold = thresholds[optimal_idx]
	print(optimal_threshold)
	if plot:
		plt.plot(fpr, tpr, label=f'ROC curve {name.replace("_", " ")} (area = {auc:0.2f})')
		plt.scatter(fpr[optimal_idx], tpr[optimal_idx], c="green")
		plt.plot([0, 1], [0, 1], 'k--', lw=2)
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Receiver operating characteristic for multiple classes')
		plt.legend(loc="lower right")
		plt.savefig(directory + f"/roc_curve_{name}.pdf")
		plt.close()
	return optimal_threshold


class X_Data():
	def __init__(self, sequences, table):
		self.sequences = np.array(sequences)
		self.table = table
		# self.table = np.swapaxes(table,1,2)
		# self.table = np.reshape(table,(-1,table.shape[1]*table.shape[2]))
		self.seq_length = self.sequences.shape[1]


def parse_amino(x, generator):
	# amino = "GALMFWKQESPVICYHRNDTU"
	amino = "GALMFWKQESPVICYHRNDTUOBZX"
	encoder = LabelEncoder()
	encoder.fit(list(amino))
	print(encoder.classes_)
	print(encoder.transform(encoder.classes_))
	out = []
	for i in x:
		if generator:
			dnaSeq = i[1].upper()
		else:
			dnaSeq = i[0].upper()
		encoded_X = encoder.transform(list(dnaSeq))
		out.append(encoded_X)
	return np.array(out)


# hydrophilicity by parker
hydrophilicity_scores = {'A': 2.1, 'C': 1.4, 'D': 10.0, 'E': 7.8, 'F': -9.2, 'G': 5.7, 'H': 2.1, 'I': -8.0, 'K': 5.7,
						 'L': -9.2, 'M': -4.2, 'N': 7.0, 'P': 2.1, 'Q': 6.0, 'R': 4.2, 'S': 6.5, 'T': 5.2, 'V': -3.7,
						 'W': -10.0, 'Y': -1.9}
# Chou Fasman beta turn prediction (avg = 1)
betaturn_scores = {'A': 0.66, 'C': 1.19, 'D': 1.46, 'E': 0.74, 'F': 0.6, 'G': 1.56, 'H': 0.95, 'I': 0.47, 'K': 1.01,
				   'L': 0.59, 'M': 0.6, 'N': 1.56, 'P': 1.52, 'Q': 0.98, 'R': 0.95, 'S': 1.43, 'T': 0.96, 'V': 0.5,
				   'W': 0.96, 'Y': 1.14}
# Emini surface accessibility scale (avg = 0.62)
surface_accessibility_scores = {'A': 0.49, 'C': 0.26, 'D': 0.81, 'E': 0.84, 'F': 0.42, 'G': 0.48, 'H': 0.66, 'I': 0.34,
								'K': 0.97, 'L': 0.4, 'M': 0.48, 'N': 0.78, 'P': 0.75, 'Q': 0.84, 'R': 0.95, 'S': 0.65,
								'T': 0.7, 'V': 0.36, 'W': 0.51, 'Y': 0.76}
# Kolaskar and Tongaokar antigenicity scale (avg = 1.0)
antigenicity_scores = {'A': 1.064, 'C': 1.412, 'D': 0.866, 'E': 0.851, 'F': 1.091, 'G': 0.874, 'H': 1.105, 'I': 1.152,
					   'K': 0.93, 'L': 1.25, 'M': 0.826, 'N': 0.776, 'P': 1.064, 'Q': 1.015, 'R': 0.873, 'S': 1.012,
					   'T': 0.909, 'V': 1.383, 'W': 0.893, 'Y': 1.161}


def normalize_dict(in_dict):
	"""
	normalizes values in dict to range [0, 1]
	:param in_dict:
	:return:
	"""
	keys = []
	values = []
	for key, value in dict(in_dict).items():
		keys.append(key)
		values.append(value)

	values_nestedlist = np.array([[i] for i in values])
	min_max_scaler = MinMaxScaler()
	# feed in a numpy array
	values = min_max_scaler.fit_transform(values_nestedlist).flatten()

	out_dict = {}
	for i in range(len(values)):
		out_dict.update({keys[i]: values[i]})

	return out_dict


# hydrophilicity by parker
hydrophilicity_scores = normalize_dict(hydrophilicity_scores)
# Chou Fasman beta turn prediction (avg = 1)
betaturn_scores = normalize_dict(betaturn_scores)
# Emini surface accessibility scale (avg = 0.62)
surface_accessibility_scores = normalize_dict(surface_accessibility_scores)
# Kolaskar and Tongaokar antigenicity scale (avg = 1.0)
antigenicity_scores = normalize_dict(antigenicity_scores)


def load_data(complex, directory, val_size=0.3, generator=False, sequence_length=50, full_seq_embedding=False,
			  final_set=True, include_raptorx_iupred=False, include_dict_scores=False, non_binary=False):
	def load_raptorx_iupred(samples):
		out = []
		shift = 20
		for sample in samples:
			start = int(sample[0])
			stop = int(sample[1])
			file = sample[2]
			# print(os.path.join("/home/le86qiz/Documents/Konrad/tool_comparison/raptorx/flo_files", f"{file}.csv"))
			try:
				table_numpy = pd.read_csv(
					os.path.join("/home/le86qiz/Documents/Konrad/tool_comparison/raptorx/flo_files", f"{file}.csv"),
					sep="\t", index_col=None).values
				seq_len = table_numpy.shape[0]
				table_numpy_big = np.zeros((seq_len + (shift * 2), 7))
				table_numpy_big[shift:shift + seq_len] = table_numpy
				table_numpy_sliced = table_numpy_big[start + shift:stop + shift]

			except:
				print(f"not able to load {file}")
				print(start)
				table_numpy_sliced = np.zeros((49, 7))
			# table_numpy_sliced[0:7] = 0.33
			# table_numpy_sliced[7] = 0.5

			out.append(table_numpy_sliced)
		return np.array(out)

	def get_dict_scores(seqs):
		out_arr = []
		for index_seq, seq in enumerate(seqs):
			seq_arr = np.zeros((49, 4))
			for index, char in enumerate(seq):
				char = char.upper()
				# check value for char in dicts if char not in dict give value 0,5
				hydro = hydrophilicity_scores.get(char, 0.5)
				beta = betaturn_scores.get(char, 0.5)
				surface = surface_accessibility_scores.get(char, 0.5)
				antigen = antigenicity_scores.get(char, 0.5)
				features = np.array([hydro, beta, surface, antigen])
				seq_arr[index] = features
			out_arr.append(seq_arr)
		return np.array(out_arr)

	if full_seq_embedding:
		Y_train_old = pd.read_csv(directory + '/Y_train.csv', delimiter='\t', dtype='str', header=None).values
		Y_test_old = pd.read_csv(directory + '/Y_test.csv', delimiter='\t', dtype='str', header=None).values
		# X_train_old = np.array(pickle.load(open(directory + '/X_train.pkl', "rb")))
		X_train_old = []
		X_test_old = np.array(pickle.load(open(directory + '/X_test.pkl', "rb")))

		try:
			Y_val_old = pd.read_csv(directory + '/Y_val.csv', delimiter='\t', dtype='str', header=None).values
			X_val_old = np.array(pickle.load(open(directory + '/X_val.pkl', "rb")))
			print("loaded validation set from: " + directory + '/Y_val.pkl')
		except:
			assert generator == False, "if generator is in use, don't great validation set from train, this would lead to overfitting of the validation set"
			print("create validation set from train")
			X_train_old, X_val_old, Y_train_old, Y_val_old = train_test_split(X_train_old, Y_train_old,
																			  test_size=val_size,
																			  shuffle=True)
		if final_set:
			if include_raptorx_iupred:
				samples_test = np.array([i[1:] for i in X_test_old])
				table_test = load_raptorx_iupred(samples_test)
				sequences_test = np.array([i[0] for i in X_test_old])
				X_test = X_Data(sequences=sequences_test,
								table=table_test)

				samples_val = np.array([i[1:] for i in X_val_old])
				table_val = load_raptorx_iupred(samples_val)
				sequences_val = np.array([i[0] for i in X_val_old])
				X_val = X_Data(sequences=sequences_val,
							   table=table_val)
				X_train = X_train_old

			elif include_dict_scores:
				X_test_old_seq = pd.read_csv(directory + '/X_test.csv', delimiter='\t', dtype='str', header=None).values
				X_val_old_seq = pd.read_csv(directory + '/X_val.csv', delimiter='\t', dtype='str', header=None).values
				X_test_old_seq = X_test_old_seq[:, 0]
				X_val_old_seq = X_val_old_seq[:, 0]
				table_test = get_dict_scores(X_test_old_seq)
				table_val = get_dict_scores(X_val_old_seq)

				X_train = X_train_old
				X_test = np.stack(X_test_old[:, 0])
				X_val = np.stack(X_val_old[:, 0])

				X_test = X_Data(sequences=X_test,
								table=table_test)
				X_val = X_Data(sequences=X_val,
							   table=table_val)

			elif non_binary:
				X_train = X_train_old
				X_test = X_test_old
				X_val = X_val_old
			else:
				X_train = X_train_old
				X_test = np.stack(X_test_old[:, 0])
				X_val = np.stack(X_val_old[:, 0])
		else:
			X_train = X_train_old
			X_test = X_test_old
			X_val = X_val_old
		if non_binary:
			Y_train =Y_train_old[:, 0]
			Y_test = np.array(Y_test_old[:, 0],np.float)
			Y_test = np.array([1-Y_test,Y_test]).swapaxes(0,1)
			Y_val = np.array(Y_val_old[:, 0],np.float)
			Y_val = np.array([1-Y_val,Y_val]).swapaxes(0,1)
		else:
			Y_train, y_encoder = DataParsing.encode_string(y=Y_train_old[:, 0])
			Y_test = DataParsing.encode_string(y=Y_test_old[:, 0], y_encoder=y_encoder)
			Y_val = DataParsing.encode_string(y=Y_val_old[:, 0], y_encoder=y_encoder)
		return X_train, X_val, X_test, Y_train, Y_val, Y_test, None

	if not complex:
		Y_train_old = pd.read_csv(directory + '/Y_train.csv', delimiter='\t', dtype='str', header=None).values
		Y_test_old = pd.read_csv(directory + '/Y_test.csv', delimiter='\t', dtype='str', header=None).values
		X_train_old = pd.read_csv(directory + '/X_train.csv', delimiter='\t', dtype='str', header=None).values
		X_test_old = pd.read_csv(directory + '/X_test.csv', delimiter='\t', dtype='str', header=None).values

		try:
			Y_val_old = pd.read_csv(directory + '/Y_val.csv', delimiter='\t', dtype='str', header=None).values
			X_val_old = pd.read_csv(directory + '/X_val.csv', delimiter='\t', dtype='str', header=None).values
			print("loaded validation set from: " + directory + '/Y_val.csv')
		except:
			assert generator == False, "if generator is in use, don't great validation set from train, this would lead to overfitting of the validation set"
			print("create validation set from train")
			X_train_old, X_val_old, Y_train_old, Y_val_old = train_test_split(X_train_old, Y_train_old,
																			  test_size=val_size, random_state=42)

		if final_set:
			if include_raptorx_iupred:
				samples_test = np.array([i[1:] for i in X_test_old])
				table_test = load_raptorx_iupred(samples_test)
				sequences_test = np.array([i[0] for i in X_test_old])

				samples_val = np.array([i[1:] for i in X_val_old])
				table_val = load_raptorx_iupred(samples_val)
				sequences_val = np.array([i[0] for i in X_val_old])

				elmo_embedder = DataGenerator.Elmo_embedder()

				sequences_test = np.array([list(i) for i in sequences_test])
				sequences_val = np.array([list(i) for i in sequences_val])

				original_length = 49
				start_float = (original_length - sequence_length) / 2
				start = math.floor(start_float)
				stop = original_length - math.ceil(start_float)

				print("embedding test")
				sequences_test = elmo_embedder.elmo_embedding(sequences_test, start, stop)
				print("embedding val")
				sequences_val = elmo_embedder.elmo_embedding(sequences_val, start, stop)

				X_test = X_Data(sequences=sequences_test, table=table_test)
				X_val = X_Data(sequences=sequences_val, table=table_val)
				X_train = X_train_old

				Y_train, y_encoder = DataParsing.encode_string(y=Y_train_old)
				Y_test = DataParsing.encode_string(y=Y_test_old, y_encoder=y_encoder)
				Y_val = DataParsing.encode_string(y=Y_val_old, y_encoder=y_encoder)

				return X_train, X_val, X_test, Y_train, Y_val, Y_test, None

			elif include_dict_scores:
				pass

			else:
				X_train_old = X_train_old[:, 0]
				X_test_old = X_test_old[:, 0]
				X_val_old = X_val_old[:, 0]

		pos_y_test = Y_test_old[:, 1:]
		if sequence_length != 50:
			print("WARNING not using full length of seq")

		# cut out middle if necessary cut more away from end
		# e.g. sequence = ABCD, new_seq_length=1-> B
		original_length = 49
		# original_length = 50
		start_float = (original_length - sequence_length) / 2
		start = math.floor(start_float)
		stop = original_length - math.ceil(start_float)
		if not generator:
			# X_train = parse_amino(x=[[i[0][start:stop]] for i in X_train_old], generator=generator)
			# X_test = parse_amino(x=[[i[0][start:stop]] for i in X_test_old], generator=generator)
			# X_val = parse_amino(x=[[i[0][start:stop]] for i in X_val_old], generator=generator)
			elmo_embedder = DataGenerator.Elmo_embedder()
			X_train = elmo_embedder.elmo_embedding(X_train_old[:, 1], start, stop)
			print(X_train.shape)
			X_test = elmo_embedder.elmo_embedding(X_test_old[:, 1], start, stop)
			X_val = elmo_embedder.elmo_embedding(X_val_old[:, 1], start, stop)
			print(X_val.shape)

		else:
			elmo_embedder = DataGenerator.Elmo_embedder()
			# X_train = elmo_embedder.elmo_embedding(X_train_old[:,1],start,stop)
			print("embedding test")
			# X_test = elmo_embedder.elmo_embedding(X_test_old[:, 1], start, stop)
			if final_set:
				X_test_old = np.array([list(i) for i in X_test_old])
				# X_test_old = np.array([list(i.upper()) for i in X_test_old])
				X_val_old = np.array([list(i) for i in X_val_old])
			# X_val_old = np.array([list(i.upper()) for i in X_val_old])
			else:
				X_test_old = np.array([list(i) for j in X_test_old for i in j])
				X_val_old = np.array([list(i) for j in X_val_old for i in j])
			# X_test_old, mapping_X = DataGenerator.split_embedding_seq_n_times(X_test_old, sequence_length, 1)
			X_test = elmo_embedder.elmo_embedding(X_test_old, start, stop)
			print("embedding val")
			X_val = elmo_embedder.elmo_embedding(X_val_old, start, stop)

			# X_val = elmo_embedder.elmo_embedding(X_val_old[:, 1], start, stop)
			print("embedding train")
			X_train = []
	# X_train = parse_amino(x=[[i[1][start:stop]] for i in X_train_old], generator=False)
	# X_test = parse_amino(x=[[i[1][start:stop]] for i in X_test_old], generator=False)
	# X_val = parse_amino(x=[[i[1][start:stop]] for i in X_val_old], generator=False)

	else:
		Y_train_old = pd.read_csv(directory + '/Y_train.csv', delimiter='\t', dtype='str', header=None).values
		Y_test_old = pd.read_csv(directory + '/Y_test.csv', delimiter='\t', dtype='str', header=None).values
		X_train_old = pickle.load(open(directory + '/X_train.pkl', "rb"))
		X_test_old = pickle.load(open(directory + '/X_test.pkl', "rb"))

		try:
			Y_val_old = pd.read_csv(directory + '/Y_val.csv', delimiter='\t', dtype='str', header=None).values
			X_val_old = pickle.load(open(directory + '/X_val.pkl', "rb"))
			print("loaded validation set from: " + directory + '/Y_val.pkl')
		except:
			assert generator == False, "if generator is in use, don't great validation set from train, this would lead to overfitting of the validation set"
			print("create validation set from train")
			X_train_old, X_val_old, Y_train_old, Y_val_old = train_test_split(X_train_old, Y_train_old,
																			  test_size=val_size,
																			  shuffle=True)
		original_length = 50
		start_float = (original_length - sequence_length) / 2
		start = math.floor(start_float)
		stop = original_length - math.ceil(start_float)
		elmo_embedder = DataGenerator.Elmo_embedder()
		amino = "GALMFWKQESPVICYHRNDT"
		encoder = LabelEncoder()
		encoder.fit(list(amino))

		X_train_old_seq = np.array(list(map(encoder.inverse_transform, np.array(X_train_old[:, 0:50], dtype=np.int))))
		X_test_old_seq = np.array(list(map(encoder.inverse_transform, np.array(X_test_old[:, 0:50], dtype=np.int))))
		X_val_old_seq = np.array(list(map(encoder.inverse_transform, np.array(X_val_old[:, 0:50], dtype=np.int))))

		# elmo_embedder = elmo_embedder.elmo_embedding(X_train_old, start, stop)
		print("burn in")
		# elmo_embedder.elmo_embedding(X_train_old_seq, start, stop)
		elmo_embedder.elmo_embedding(X_test_old_seq, start, stop)
		# elmo_embedder.elmo_embedding(X_train_old_seq, start, stop)
		# elmo_embedder.elmo_embedding(X_val_old_seq, start, stop)
		print("embedding")
		X_train = X_Data(sequences=elmo_embedder.elmo_embedding(X_train_old_seq, start, stop),
						 table=X_train_old[:, sequence_length:])
		X_test = X_Data(sequences=elmo_embedder.elmo_embedding(X_test_old_seq, start, stop),
						table=X_test_old[:, sequence_length:])
		X_val = X_Data(sequences=elmo_embedder.elmo_embedding(X_val_old_seq, start, stop),
					   table=X_val_old[:, sequence_length:])
		# X_train = X_Data(sequences=X_train_old[:, 0:sequence_length], table=X_train_old[:, sequence_length:])
		# X_test = X_Data(sequences=X_test_old[:, 0:sequence_length], table=X_test_old[:, sequence_length:])
		# X_val = X_Data(sequences=X_val_old[:, 0:sequence_length], table=X_val_old[:, sequence_length:])

		# sum_float = np.sum(X_train_old[0, 0:sequence_length])
		# assert sum_float == int(sum_float), "Warning: different sequence length in dataset than defined by user"
		# X_train = X_Data(sequences=X_train_old[:, 0, :], table=X_train_old[:, 1:, :])
		# X_test = X_Data(sequences=X_test_old[:, 0, :], table=X_test_old[:, 1:, :])
		# X_val = X_Data(sequences=X_val_old[:, 0, :], table=X_val_old[:, 1:, :])

		pos_y_test = Y_test_old[:, 1:]

	if generator:
		# Y_train, y_encoder = DataParsing.encode_string(y=Y_train_old[:, 1])
		# Y_test = DataParsing.encode_string(y=Y_test_old[:, 1], y_encoder=y_encoder)
		# Y_val = DataParsing.encode_string(y=Y_val_old[:, 1], y_encoder=y_encoder)
		if non_binary:
			Y_train =Y_train_old[:, 0]
			Y_test = np.array(Y_test_old[:, 0],np.float)
			Y_test = np.array([1-Y_test,Y_test]).swapaxes(0,1)
			Y_val = np.array(Y_val_old[:, 0],np.float)
			Y_val = np.array([1-Y_val,Y_val]).swapaxes(0,1)
		else:
			Y_train, y_encoder = DataParsing.encode_string(y=Y_train_old)
			Y_test = DataParsing.encode_string(y=Y_test_old, y_encoder=y_encoder)
			Y_val = DataParsing.encode_string(y=Y_val_old, y_encoder=y_encoder)
	# Y_train = np.argmax(Y_train, axis=1)
	# Y_test = np.argmax(Y_test, axis=1)
	# Y_val = np.argmax(Y_val, axis=1)

	elif complex:
		Y_train = to_categorical(np.array(Y_train_old[:, 0], dtype=np.float))
		Y_test = to_categorical(np.array(Y_test_old[:, 0], dtype=np.float))
		Y_val = to_categorical(np.array(Y_val_old[:, 0], dtype=np.float))
	else:
		Y_train, y_encoder = DataParsing.encode_string(y=Y_train_old[:, 1])
		Y_test = DataParsing.encode_string(y=Y_test_old[:, 1], y_encoder=y_encoder)
		Y_val = DataParsing.encode_string(y=Y_val_old[:, 1], y_encoder=y_encoder)

	#
	# X_train = DataParsing.encode_string(maxLen=19, x=X_train_old, repeat=False, use_spacer=False,
	#                                     online_Xtrain_set=False)
	# X_test = DataParsing.encode_string(maxLen=19, x=X_test_old, repeat=False, use_spacer=False)
	# X_val = DataParsing.encode_string(maxLen=19, x=X_val_old, repeat=False, use_spacer=False)

	return X_train, X_val, X_test, Y_train, Y_val, Y_test, pos_y_test


# def build_model(nodes, dropout, seq_length, weight_decay_lstm=0, weight_decay_dense=0):
def build_model(nodes, dropout, seq_length, weight_decay_lstm=1e-6, weight_decay_dense=1e-3,  non_binary=False):
	#     model = models.Sequential()
	inputs = layers.Input(shape=(seq_length, 1024))
	# embedding_matrix = pickle.load(open('/home/go96bix/projects/deep_eve/seqvec/uniref50_v2/embeddings.pkl', "rb"))
	# model.add(layers.Embedding(embedding_matrix.shape[0],embedding_matrix.shape[1], weights=[embedding_matrix], input_length=seq_length,trainable=False))
	# model.add(layers.Embedding(25, 1024, input_length=seq_length))
	# nodes, return_sequences=True, dropout=dropout, recurrent_dropout=0.2,kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))
	hidden = layers.Bidirectional(
		layers.LSTM(nodes, input_shape=(seq_length, 1024), return_sequences=True, dropout=dropout,
					recurrent_dropout=0.2, kernel_regularizer=l2(weight_decay_lstm),
					recurrent_regularizer=l2(weight_decay_lstm), bias_regularizer=l2(weight_decay_lstm)))(inputs)
	# model.add(layers.Bidirectional(layers.LSTM(nodes, return_sequences=True, dropout=dropout, recurrent_dropout=0.2, kernel_regularizer=l2(weight_decay_lstm), recurrent_regularizer=l2(weight_decay_lstm), bias_regularizer=l2(weight_decay_lstm))))
	hidden = layers.Bidirectional(
		layers.LSTM(nodes, dropout=dropout, recurrent_dropout=0.2, kernel_regularizer=l2(weight_decay_lstm),
					recurrent_regularizer=l2(weight_decay_lstm), bias_regularizer=l2(weight_decay_lstm)))(hidden)
	hidden = layers.Dense(nodes, kernel_regularizer=l2(weight_decay_dense), bias_regularizer=l2(weight_decay_dense))(
		hidden)
	hidden = layers.LeakyReLU(alpha=0.01)(hidden)
	# model.add(layers.Dense(1))
	# model.compile(optimizer='rmsprop',
	#               loss='mean_absolute_error',
	#               # loss='mean_squared_error',
	#               metrics=['mean_squared_error'])

	out = layers.Dense(2, activation='softmax', kernel_regularizer=l2(weight_decay_dense),
					   bias_regularizer=l2(weight_decay_dense))(hidden)
	# adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False)
	# model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['acc'])
	model = models.Model(inputs=inputs, outputs=out)

	if non_binary:
		model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[])
	else:
		model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', auc_10_perc_fpr])
	# model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['acc'])
	# metrics=['accuracy'])
	model.summary()
	return model


def build_complex_model(nodes, dropout, seq_length=19, vector_dim=1, table_columns=0):
	# seq_input = layers.Input(shape=(seq_length, vector_dim,), name="seq_input")

	seq_input = layers.Input(shape=(seq_length,), name="seq_input")
	left = layers.Embedding(21, 10, input_length=seq_length)(seq_input)
	# This returns a tensor
	# inputs = layers.Input(shape=(timesteps, X_test.shape[-1]))

	left = layers.Bidirectional(layers.LSTM(nodes, return_sequences=True, dropout=dropout, recurrent_dropout=0.2))(left)
	left = layers.Bidirectional(layers.LSTM(nodes, dropout=dropout, recurrent_dropout=0.2))(left)
	# left = layers.Dense(nodes)(left)
	left = layers.Dense(nodes)(left)
	out_left = layers.LeakyReLU(alpha=0.01)(left)
	Seq_model = models.Model(seq_input, out_left)
	# out_left = layers.Dense(1)(left)
	# out_left = layers.Dense(2, activation='softmax')(left)

	# variant 1
	# auxiliary_input = layers.Input(shape=(table_columns,seq_length,), dtype='float', name='aux_input')
	# middle_input = layers.concatenate([out_left, auxiliary_input])
	# variant 2
	# input array with feature vectors, each dim in Vector is one position in sequence
	# input.shape(features, seq_length)
	# auxiliary_input = layers.Input(shape=(table_columns, seq_length,), dtype='float', name='aux_input')
	auxiliary_input = layers.Input(shape=(table_columns,), dtype='float', name='aux_input')

	right = layers.Dense(nodes)(auxiliary_input)
	# right = layers.Flatten()(right)
	right = layers.LeakyReLU(alpha=0.01)(right)

	Table_model = models.Model(auxiliary_input, right)

	# right = layers.Conv1D()
	middle_input = layers.concatenate([Seq_model(seq_input), Table_model(auxiliary_input)])
	# middle_input = layers.concatenate([out_left, right])
	middle = layers.Dense(nodes)(middle_input)
	middle = layers.LeakyReLU(alpha=0.01)(middle)
	# out = layers.Dense(1)(middle)

	# output = layers.Dense(1, name='output')(middle)
	output = layers.Dense(2, activation='softmax', name='output')(middle)
	model = models.Model(inputs=[seq_input, auxiliary_input], outputs=output)
	# model = models.Model(inputs=[seq_input, auxiliary_input], outputs=output)

	# model.compile(optimizer='rmsprop',
	#               loss='mean_absolute_error',
	#               # loss='mean_squared_error',
	#               metrics=['mean_squared_error'])
	set_trainability(Table_model, False)
	# set_trainability(Seq_model,False)
	# set_trainability(model,False)
	Seq_model.compile(optimizer="adam", loss='binary_crossentropy')
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
	Table_model.compile(optimizer="adam", loss='binary_crossentropy')
	model.summary()
	return model, Seq_model, Table_model


# right = layers.Conv1D(nodes, 9, activation='relu')(auxiliary_input)
# right = layers.MaxPooling1D(3)(right)
# right = layers.Conv1D(nodes, 9, activation='relu')(right)
# right = layers.MaxPooling1D(3)(right)
# right3 = layers.Conv1D(nodes, 9, activation='relu')(right)
# right_flat = layers.Flatten()(right3)
#
# joined = layers.Concatenate()([left2, right_flat])
# predictions = layers.Dense(Y_train.shape[-1], activation='softmax')(joined)
#
# model = models.Model(inputs=inputs, outputs=predictions)


# sequences = []
# table = []
def acc_with_cutoffs(Y_true, Y_pred, cutoffs):
	assert Y_true.shape[1] == len(cutoffs), "number of cutoffs and classes must be equal"
	# binarize the pred with cutoffs
	Y_pred_bin = np.zeros(Y_true.shape)
	for i in range(0, Y_true.shape[1]):
		Y_pred_bin[:, i] = Y_pred[:, i] > cutoffs[i]

	for index, sample in enumerate(Y_pred_bin):
		# if no entry over threshold use highest value as prediction
		if sum(sample) == 0:
			Y_pred_bin[index, np.argmax(sample)] = 1

	# find multiple predictions
	positions = np.argwhere(Y_pred_bin == 1)
	Y_true_extended = np.array([])
	Y_pred_extended = np.array([])
	for i in positions:
		true = np.argmax(Y_true[i[0]])
		pred = i[1]
		Y_true_extended = np.append(Y_true_extended, true)
		Y_pred_extended = np.append(Y_pred_extended, pred)

	print(len(Y_true_extended))
	acc = sum(Y_true_extended == Y_pred_extended) / len(Y_true_extended)
	print(f"acc: {acc}")


def compare_quality(model, path, X_test, Y_test, X_val, Y_val, pos_y_test, complex_model=False,
					include_raptorx_iupred=False):
	def calc_quality(model, X_test, Y_test, X_val, Y_val, pos_y_test, complex_model=False, path=False, middle_name="",
					 include_raptorx_iupred=False):
		if path:
			print("load model:")
			print(model_path)
			model.load_weights(model_path)

		if complex_model:
			pred = model.predict({'seq_input': X_test.sequences, 'aux_input': X_test.table})
			pred_val = model.predict({'seq_input': X_val.sequences, 'aux_input': X_val.table})
			complex_model = False
		elif include_raptorx_iupred:
			pred = model.predict({'seq_input': X_test.sequences, 'aux_input': X_test.table})
			pred_val = model.predict({'seq_input': X_val.sequences, 'aux_input': X_val.table})
		else:
			pred = model.predict(X_test)
			pred_val = model.predict(X_val)
		error = metrics.mean_absolute_error(Y_test, pred)
		print(f"error {error}")

		# for i in range(20):
		# 	print(f"prediction: {pred[i]} true:{Y_test[i]}")

		# if complex_model:
		#     pred = np.array(pred > 0.5,dtype=int)
		#     pred_flat = pred.flatten()
		#     table = pd.crosstab(
		#         pd.Series(Y_test),
		#         pd.Series(pred_flat),
		#         rownames=['True'],
		#         colnames=['Predicted'],
		#         margins=True)
		#     print(table)
		#     acc = sum(pred_flat == Y_test)/len(Y_test)
		#
		# else:
		accuracy, error = calculate_weighted_accuracy([1], [pred], [pred_val], 2,
													  Y=Y_test, Y_val=Y_val,
													  ROC=True, name=f"{middle_name}")

	# table = pd.crosstab(
	# 	pd.Series(np.argmax(Y_test, axis=1)),
	# 	pd.Series(np.argmax(pred, axis=1)),
	# 	rownames=['True'],
	# 	colnames=['Predicted'],
	# 	margins=True)
	# print(table)
	# acc = sum(np.argmax(pred, axis=1) == np.argmax(Y_test, axis=1)) / len(np.argmax(Y_test, axis=1))
	#
	# print(f"Accuracy: {acc}")
	# # if path:
	# # 	pred_n_pos = np.append(pred, pos_y_test, axis=1)
	# # 	np.savetxt(directory + '/pred.csv', pred_n_pos, delimiter='\t', fmt='%s')

	def calc_acc_with_cutoff(name="best"):
		if complex_model:
			Y_pred_test = model.predict({'seq_input': X_test.sequences, 'aux_input': X_test.table})
			Y_pred_val = model.predict({'seq_input': X_val.sequences, 'aux_input': X_val.table})

		else:
			Y_pred_test = model.predict(X_test)
			Y_pred_val = model.predict(X_val)

		cutoff = calc_n_plot_ROC_curve(y_true=Y_val[:, 1], y_pred=Y_pred_val[:, 1], name=name)
		# acc_with_cutoffs(Y_test, Y_pred_test, [cutoff]*2)

		"""cutoff adapted"""
		print(name)
		print("cutoff adapted")
		Y_pred_test[:, 1] = Y_pred_test[:, 1] > cutoff
		Y_pred_test[:, 0] = Y_pred_test[:, 1] == 0
		table = pd.crosstab(
			pd.Series(np.argmax(Y_test, axis=1)),
			pd.Series(np.argmax(Y_pred_test, axis=1)),
			rownames=['True'],
			colnames=['Predicted'],
			margins=True)
		print(table)
		acc = sum(np.argmax(Y_test, axis=1) == np.argmax(Y_pred_test, axis=1)) / len(np.argmax(Y_pred_test, axis=1))
		print(f"Accuracy: {acc}")

	# last weights
	calc_quality(model, X_test, Y_test, X_val, Y_val, pos_y_test, complex_model, path=False,
				 middle_name=f"last_model_{suffix}",
				 include_raptorx_iupred=include_raptorx_iupred)
	# calc_acc_with_cutoff(name="last")
	# best weights
	if path:
		for middle_name in ("loss", "acc", "auc10"):
			model_path = f"{path}/weights.best.{middle_name}.{suffix}.hdf5"

			calc_quality(model, X_test, Y_test, X_val, Y_val, pos_y_test, complex_model, path=path,
						 middle_name=f"{middle_name}_{suffix}", include_raptorx_iupred=include_raptorx_iupred)


# calc_acc_with_cutoff()


def build_multi_length_model(nodes, dropout):
	models_multi_length = []
	inputs_multi_length = []

	for sequence_length in range(9, 51, 10):
		input_name = f"seq_input_{sequence_length}"

		# build model
		seq_input = layers.Input(shape=(sequence_length,), name=input_name)
		left = layers.Embedding(21, 10, input_length=sequence_length)(seq_input)

		left = layers.Bidirectional(layers.LSTM(nodes, return_sequences=True, dropout=dropout, recurrent_dropout=0.2))(
			left)
		left = layers.Bidirectional(layers.LSTM(nodes, dropout=dropout, recurrent_dropout=0.2))(left)
		left = layers.Dense(nodes)(left)
		left = layers.LeakyReLU(alpha=0.01)(left)
		out_left = layers.Dense(2, activation='softmax')(left)
		Seq_model = models.Model(seq_input, out_left)
		Seq_model.compile(optimizer="adam", loss='binary_crossentropy')

		models_multi_length.append(Seq_model(seq_input))
		inputs_multi_length.append(seq_input)

	middle_input = layers.concatenate(models_multi_length)
	middle = layers.Dense(nodes)(middle_input)
	middle = layers.LeakyReLU(alpha=0.01)(middle)

	output = layers.Dense(2, activation='softmax', name='output')(middle)
	model = models.Model(inputs=inputs_multi_length, outputs=output)
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
	model.summary()
	plot_model(model, to_file='multi_length.png')
	return model  # , models_multi_length, inputs_multi_length


def build_elmo_embedding_model():
	import tensorflow as tf
	import tensorflow_hub as hub
	elmo = hub.Module("/home/go96bix/projects/deep_eve/elmo", trainable=False)

	# with tf.Graph().as_default():
	#     sentences = tf.placeholder(tf.string)
	#     embed = hub.Module(module)
	#     embeddings = embed(sentences)
	#     session = tf.train.MonitoredSession()
	# return lambda x: session.run(embeddings, {sentences: x})
	def ELMoEmbedding(x):
		return elmo(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]

	input_text = layers.Input(shape=(1,), dtype="string")
	embedding = layers.Lambda(ELMoEmbedding, output_shape=(1024,))(input_text)


def build_model_with_raptorx_iupred(nodes, dropout, seq_length=50, weight_decay_lstm=0, weight_decay_dense=0):
	seq_input = layers.Input(shape=(seq_length, 1024), name="seq_input")
	left = layers.Bidirectional(
		layers.LSTM(nodes, input_shape=(seq_length, 1024), return_sequences=True, dropout=dropout,
					recurrent_dropout=0.2, kernel_regularizer=l2(weight_decay_lstm),
					recurrent_regularizer=l2(weight_decay_lstm), bias_regularizer=l2(weight_decay_lstm)))(seq_input)
	left = layers.Bidirectional(
		layers.LSTM(nodes, dropout=dropout, recurrent_dropout=0.2, kernel_regularizer=l2(weight_decay_lstm),
					recurrent_regularizer=l2(weight_decay_lstm), bias_regularizer=l2(weight_decay_lstm)))(left)
	left = layers.Dense(nodes, kernel_regularizer=l2(weight_decay_dense), bias_regularizer=l2(weight_decay_dense))(left)
	out_left = layers.LeakyReLU(alpha=0.01)(left)
	Seq_model = models.Model(seq_input, out_left)

	auxiliary_input = layers.Input(shape=(seq_length, 7), dtype='float', name='aux_input')
	right = layers.Bidirectional(
		layers.LSTM(nodes, input_shape=(seq_length, 1024), return_sequences=True, dropout=dropout,
					recurrent_dropout=0.2, kernel_regularizer=l2(weight_decay_lstm),
					recurrent_regularizer=l2(weight_decay_lstm), bias_regularizer=l2(weight_decay_lstm)))(
		auxiliary_input)
	right = layers.Bidirectional(
		layers.LSTM(nodes, dropout=dropout, recurrent_dropout=0.2, kernel_regularizer=l2(weight_decay_lstm),
					recurrent_regularizer=l2(weight_decay_lstm), bias_regularizer=l2(weight_decay_lstm)))(right)
	right = layers.Dense(nodes, kernel_regularizer=l2(weight_decay_dense), bias_regularizer=l2(weight_decay_dense))(
		right)
	out_right = layers.LeakyReLU(alpha=0.01)(right)
	Table_model = models.Model(auxiliary_input, out_right)

	middle_input = layers.concatenate([Seq_model(seq_input), Table_model(auxiliary_input)])
	middle = layers.Dense(nodes)(middle_input)
	middle = layers.LeakyReLU(alpha=0.01)(middle)
	output = layers.Dense(2, activation='softmax', name='output')(middle)
	model = models.Model(inputs=[seq_input, auxiliary_input], outputs=output)

	set_trainability(Table_model, False)
	# set_trainability(Seq_model,False)
	# set_trainability(model,False)
	Seq_model.compile(optimizer="adam", loss='binary_crossentropy')
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', auc_10_perc_fpr])
	Table_model.compile(optimizer="adam", loss='binary_crossentropy')
	model.summary()
	return model, Seq_model, Table_model


def build_model_with_table(nodes, dropout, seq_length=50, weight_decay_lstm=0, weight_decay_dense=0):
	seq_input = layers.Input(shape=(seq_length, 1024), name="seq_input")
	left = layers.Bidirectional(
		layers.LSTM(nodes, input_shape=(seq_length, 1024), return_sequences=True, dropout=dropout,
					recurrent_dropout=0.2, kernel_regularizer=l2(weight_decay_lstm),
					recurrent_regularizer=l2(weight_decay_lstm), bias_regularizer=l2(weight_decay_lstm)))(seq_input)
	left = layers.Bidirectional(
		layers.LSTM(nodes, dropout=dropout, recurrent_dropout=0.2, kernel_regularizer=l2(weight_decay_lstm),
					recurrent_regularizer=l2(weight_decay_lstm), bias_regularizer=l2(weight_decay_lstm)))(left)
	left = layers.Dense(nodes, kernel_regularizer=l2(weight_decay_dense), bias_regularizer=l2(weight_decay_dense))(left)
	out_left = layers.LeakyReLU(alpha=0.01)(left)
	Seq_model = models.Model(seq_input, out_left)

	auxiliary_input = layers.Input(shape=(seq_length, 4), dtype='float', name='aux_input')
	right = layers.Bidirectional(
		layers.LSTM(nodes // 2, input_shape=(seq_length, 1024), return_sequences=True, dropout=dropout,
					recurrent_dropout=0.2, kernel_regularizer=l2(weight_decay_lstm),
					recurrent_regularizer=l2(weight_decay_lstm), bias_regularizer=l2(weight_decay_lstm)))(
		auxiliary_input)
	right = layers.Bidirectional(
		layers.LSTM(nodes // 2, dropout=dropout, recurrent_dropout=0.2, kernel_regularizer=l2(weight_decay_lstm),
					recurrent_regularizer=l2(weight_decay_lstm), bias_regularizer=l2(weight_decay_lstm)))(right)
	right = layers.Dense(nodes // 2, kernel_regularizer=l2(weight_decay_dense),
						 bias_regularizer=l2(weight_decay_dense))(
		right)
	out_right = layers.LeakyReLU(alpha=0.01)(right)
	Table_model = models.Model(auxiliary_input, out_right)

	middle_input = layers.concatenate([Seq_model(seq_input), Table_model(auxiliary_input)])
	middle = layers.Dense(nodes)(middle_input)
	middle = layers.LeakyReLU(alpha=0.01)(middle)
	output = layers.Dense(2, activation='softmax', name='output')(middle)
	model = models.Model(inputs=[seq_input, auxiliary_input], outputs=output)

	set_trainability(Table_model, False)
	# set_trainability(Seq_model,False)
	# set_trainability(model,False)
	Seq_model.compile(optimizer="adam", loss='binary_crossentropy')
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', auc_10_perc_fpr])
	Table_model.compile(optimizer="adam", loss='binary_crossentropy')
	model.summary()
	return model, Seq_model, Table_model


def build_broad_complex_model(nodes, dropout, sequence_length=50, weight_decay=1e-6):
	adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	# adam=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
	# adam=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.001, amsgrad=False)

	tensors_models = []
	list_models_table_cols = []
	inputs_table_cols = []

	"""sequence model"""
	# seq_input = layers.Input(shape=(sequence_length,), name="seq_input")
	seq_input = layers.Input(shape=(sequence_length, 1024), name="seq_input")
	# embedding_matrix = pickle.load(open('/home/go96bix/projects/deep_eve/seqvec/uniref50_v2/embeddings.pkl', "rb"))
	# left = layers.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix],
	#                            input_length=sequence_length, trainable=False)(seq_input)

	# left = layers.Embedding(21, 10, input_length=sequence_length)(seq_input)
	left = layers.Bidirectional(layers.LSTM(nodes, return_sequences=True, dropout=dropout, recurrent_dropout=0.2,
											kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay),
											bias_regularizer=l2(weight_decay)))(seq_input)
	left = layers.Bidirectional(
		layers.LSTM(nodes, dropout=dropout, recurrent_dropout=0.2, kernel_regularizer=l2(weight_decay),
					recurrent_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay)))(left)
	left = layers.Dense(nodes, kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(left)
	out_left = layers.LeakyReLU(alpha=0.01)(left)
	Seq_model = models.Model(seq_input, out_left)
	Seq_model.summary()

	tensors_models.append(Seq_model(seq_input))
	inputs_table_cols.append(seq_input)

	"""loop creating multiple table input models"""
	length_arr = [sequence_length] * 9
	length_arr.append(13)
	for index, length in enumerate(length_arr):
		input_name = f"table_input_{index}"

		#     build model
		if length == sequence_length:
			table_input = layers.Input(shape=(length, 1,), name=input_name)
			middle = layers.Bidirectional(
				layers.LSTM(nodes, return_sequences=True, dropout=dropout, recurrent_dropout=0.2,
							kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay),
							bias_regularizer=l2(weight_decay)))(table_input)
			middle = layers.Bidirectional(
				layers.LSTM(nodes, dropout=dropout, recurrent_dropout=0.2, kernel_regularizer=l2(weight_decay),
							recurrent_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay)))(middle)
			middle = layers.Dense(nodes, kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(middle)
			middle_out = layers.LeakyReLU(alpha=0.01)(middle)
			Table_model = models.Model(table_input, middle_out)

		else:
			table_input = layers.Input(shape=(length,), dtype='float', name=input_name)
			# table_input = layers.Flatten()(table_input)
			right = layers.Dense(nodes, kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(
				table_input)
			right_out = layers.LeakyReLU(alpha=0.01)(right)
			Table_model = models.Model(table_input, right_out)

		# Table_model.compile(optimizer="adam", loss='binary_crossentropy')
		Table_model.compile(optimizer=adam, loss='binary_crossentropy')
		list_models_table_cols.append(Table_model)
		tensors_models.append(Table_model(table_input))
		inputs_table_cols.append(table_input)

	middle_input = layers.concatenate(tensors_models)
	middle = layers.Dense(nodes, kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(middle_input)
	middle = layers.LeakyReLU(alpha=0.01)(middle)

	output = layers.Dense(2, activation='softmax', name='output', kernel_regularizer=l2(weight_decay),
						  bias_regularizer=l2(weight_decay))(middle)
	model = models.Model(inputs=inputs_table_cols, outputs=output)
	# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', auc_10_perc_fpr])
	model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['acc', auc_10_perc_fpr])
	model.summary()
	plot_model(model, to_file='broad_complex.png')
	return model, Seq_model, list_models_table_cols


def test_broad_complex_model(path, suffix, complex_model=True, shuffleTraining=True,
							 nodes=32, use_generator=True, epochs=100, dropout=0.0, faster=False, batch_size=32,
							 sequence_length=50, tensorboard=False, gpus=False, cross_val=True,
							 modular_training=True, weight_decay=1e-6, **kwargs):
	inputs_train = {}
	inputs_val = {}
	inputs_test = {}

	# input_X = {}

	X_train, X_val, X_test, Y_train, Y_val, Y_test, pos_y_test = load_data(complex_model, path, val_size=0.3,
																		   generator=use_generator,
																		   sequence_length=sequence_length)
	# define inputs
	for col_num in range(10):
		input_name = f"table_input_{col_num}"
		if col_num == 0:
			inputs_train.update({"seq_input": X_train.sequences})
			inputs_val.update({"seq_input": X_val.sequences})
			inputs_test.update({"seq_input": X_test.sequences})
		# input_X.update({"seq_input": X_i})

		col_train = X_train.table[:, 0:sequence_length]
		col_val = X_val.table[:, 0:sequence_length]
		col_test = X_test.table[:, 0:sequence_length]

		if col_num != 9:
			col_train = np.array(col_train).reshape(X_train.table.shape[0], -1, 1)
			col_val = np.array(col_val).reshape(X_val.table.shape[0], -1, 1)
			col_test = np.array(col_test).reshape(X_test.table.shape[0], -1, 1)

		X_train.table = X_train.table[:, sequence_length::]
		X_val.table = X_val.table[:, sequence_length::]
		X_test.table = X_test.table[:, sequence_length::]

		inputs_train.update({input_name: col_train})
		inputs_val.update({input_name: col_val})
		inputs_test.update({input_name: col_test})

	# if cross_val:
	#     # X_i = np.append(X_train_i, X_val_i, axis=0)
	#     # Y_i = np.append(Y_train_i, Y_val_i, axis=0)
	#     # if you want weighted esemble dont join train and val set
	#     X_i = X_train_i
	#     Y_i = Y_train_i
	#     input_X.update({input_name: X_i})

	model, Seq_model, list_models_table_cols = build_broad_complex_model(nodes, dropout, sequence_length, weight_decay)

	filepath = path + "/weights.best.acc." + suffix + ".hdf5"
	filepath2 = path + "/weights.best.loss." + suffix + ".hdf5"
	filepath3 = path + "/weights.best.auc10." + suffix + ".hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True,
								 mode='max')
	checkpoint2 = ModelCheckpoint(filepath2, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True,
								  mode='min')
	checkpoint3 = ModelCheckpoint(filepath3, monitor='val_auc_10_perc_fpr', verbose=1, save_best_only=True,
								  save_weights_only=True, mode='max')
	tensorboard = TensorBoard(f'./tensorboard_log_dir')
	callbacks_list = [checkpoint, checkpoint2, checkpoint3, tensorboard]

	if cross_val:
		# define 10-fold cross validation test harness
		kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1337)
		cvscores = []
		for train, val in kfold.split(col_train, np.argmax(Y_train, axis=1)):
			checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
										 save_weights_only=True, mode='max')
			checkpoint2 = ModelCheckpoint(filepath2, monitor='val_loss', verbose=1, save_best_only=True,
										  save_weights_only=True, mode='min')
			checkpoint3 = ModelCheckpoint(filepath3, monitor='val_auc_10_perc_fpr', verbose=1, save_best_only=True,
										  save_weights_only=True, mode='max')
			# tensorboard = TensorBoard(f'{path}/tensorboard_log_dir')
			if tf.gfile.Exists(f"./tensorboard_log_dir/run{len(cvscores)}"):
				tf.gfile.DeleteRecursively(f"./tensorboard_log_dir/run{len(cvscores)}")
			tensorboard = TensorBoard(f'./tensorboard_log_dir/run{len(cvscores)}')
			callbacks_list = [checkpoint, checkpoint2, checkpoint3, tensorboard]

			K.clear_session()
			del model

			model, Seq_model, list_models_table_cols = build_broad_complex_model(nodes, dropout, sequence_length)

			inputs_train_K_fold = {}
			inputs_val_K_fold = {}
			for item in inputs_train.items():
				inputs_train_K_fold.update({item[0]: item[1][train]})
				inputs_val_K_fold.update({item[0]: item[1][val]})

			if modular_training:
				seq_train_length = (epochs // 14) * 3
				col_train_length = (epochs - seq_train_length) // 11

				epo = 0
				while epo < epochs:
					# for epo in range(epochs):
					if epo == 0:
						for model_table_col in list_models_table_cols:
							set_trainability(model_table_col, False)
							model_table_col.compile(optimizer='adam', loss='binary_crossentropy')
						set_trainability(Seq_model, True)
						Seq_model.layers[0].trainable = False
						set_trainability(model, True)

						Seq_model.compile(optimizer="adam", loss='binary_crossentropy')
						model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', auc_10_perc_fpr])


					elif epo >= seq_train_length and epo < col_train_length * 12 and epo % col_train_length == 0:
						# for model_table_col in list_models_table_cols:
						model_table_col = list_models_table_cols[(epo // col_train_length) - 2]
						set_trainability(model_table_col, True)
						model_table_col.compile(optimizer='adam', loss='binary_crossentropy')
						model_table_col = list_models_table_cols[(epo // col_train_length) - 3]
						set_trainability(model_table_col, True)
						model_table_col.compile(optimizer='adam', loss='binary_crossentropy')
						set_trainability(Seq_model, False)
						set_trainability(model, True)

						Seq_model.compile(optimizer="adam", loss='binary_crossentropy')
						model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', auc_10_perc_fpr])

					elif epo == epochs - (col_train_length * 2):
						adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0,
											   amsgrad=False)
						for model_table_col in list_models_table_cols:
							set_trainability(model_table_col, True)
							model_table_col.compile(optimizer=adam, loss='binary_crossentropy')
						set_trainability(Seq_model, True)
						set_trainability(model, True)

						Seq_model.compile(optimizer=adam, loss='binary_crossentropy')
						model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['acc', auc_10_perc_fpr])

					model.fit(inputs_train_K_fold, {'output': Y_train[train]}, callbacks=callbacks_list,
							  validation_data=(inputs_val_K_fold, {'output': Y_train[val]}),
							  epochs=epo + col_train_length, batch_size=batch_size, shuffle=shuffleTraining, verbose=2,
							  initial_epoch=epo)

					epo += col_train_length
			else:
				model.fit(inputs_train_K_fold, {'output': Y_train[train]}, callbacks=callbacks_list,
						  validation_data=(inputs_val_K_fold, {'output': Y_train[val]}),
						  epochs=epochs, batch_size=batch_size, shuffle=shuffleTraining, verbose=1)

				# load "best" model and train finer
				model.load_weights(filepath2)
				adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0,
									   amsgrad=False)
				for model_table_col in list_models_table_cols:
					set_trainability(model_table_col, True)
					model_table_col.compile(optimizer=adam, loss='binary_crossentropy')
				set_trainability(Seq_model, True)
				set_trainability(model, True)

				Seq_model.compile(optimizer=adam, loss='binary_crossentropy')
				model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['acc', auc_10_perc_fpr])

				# model.load_weights(filepath2)
				model.fit(inputs_train_K_fold, {'output': Y_train[train]}, callbacks=callbacks_list,
						  validation_data=(inputs_val_K_fold, {'output': Y_train[val]}),
						  epochs=5, batch_size=batch_size, shuffle=shuffleTraining, verbose=1)

			# scores = model.evaluate(inputs_val_K_fold, Y_train[val], verbose=0,batch_size=len(val))
			scores = model.evaluate(inputs_val, Y_val, verbose=0, batch_size=len(Y_val))
			print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
			cvscores.append(scores[1] * 100)
			model.save_weights(f"{path}/weights_model_k-fold_run_{len(cvscores)}_{suffix}.hdf5")
			# if real val_set exists, use models with lowest val_loss as models in ensemble
			os.rename(filepath, f"{path}/weights_model_highest_val_acc_k-fold_run_{len(cvscores)}_{suffix}.hdf5")
			os.rename(filepath2, f"{path}/weights_model_lowest_val_loss_k-fold_run_{len(cvscores)}_{suffix}.hdf5")
			os.rename(filepath3, f"{path}/weights_model_highest_val_auc10_k-fold_run_{len(cvscores)}_{suffix}.hdf5")

		print("summary")
		print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

		model, Seq_model, list_models_table_cols = build_broad_complex_model(nodes, dropout, sequence_length)
		set_trainability(Seq_model, True)
		Seq_model.compile(optimizer='adam', loss='binary_crossentropy')
		model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', auc_10_perc_fpr])
		Seq_model.summary()
		model.summary()

		validate_cross_val_models(model, path, inputs_test, inputs_val, Y_test, Y_val)
	else:
		model.fit(inputs_train,
				  {'output': Y_train}, callbacks=callbacks_list,
				  validation_data=(inputs_val, {'output': Y_val}),
				  epochs=epochs, batch_size=batch_size, shuffle=shuffleTraining, verbose=2)

		compare_quality(model, path, inputs_test, Y_test, inputs_val, Y_val, pos_y_test, complex_model=False)


def test_multiple_length(path, suffix, complex_model=False, online_training=False, shuffleTraining=True,
						 one_hot_encoding=True, val_size=0.3, design=1, sampleSize=1, nodes=32, use_generator=True,
						 snapShotEnsemble=False, epochs=100, dropout=0.0, faster=False, batch_size=32,
						 sequence_length=50,
						 voting=False, tensorboard=False, gpus=False, titel='', x_axes='', y_axes='', accuracy=False,
						 loss=False, runtime=False, label1='', label2='', label3='', label4='', cross_val=True,
						 modular_training=True, **kwargs):
	# create input for different models
	# X_train, X_val, X_test, Y_train, Y_val, Y_test, pos_y_test = [], [], [], [], [], [], []
	# models_multi_length = []
	# inputs_multi_length = []
	inputs_train = {}
	inputs_val = {}
	inputs_test = {}

	input_X = {}

	for sequence_length in range(9, 51, 10):
		X_train_i, X_val_i, X_test_i, Y_train_i, Y_val_i, Y_test_i, pos_y_test_i = load_data(complex_model, path,
																							 val_size=0.3,
																							 generator=use_generator,
																							 sequence_length=sequence_length)

		# define inputs
		input_name = f"seq_input_{sequence_length}"
		inputs_train.update({input_name: X_train_i})
		inputs_val.update({input_name: X_val_i})
		inputs_test.update({input_name: X_test_i})

		if cross_val:
			# X_i = np.append(X_train_i, X_val_i, axis=0)
			# Y_i = np.append(Y_train_i, Y_val_i, axis=0)
			# if you want weighted esemble dont join train and val set
			X_i = X_train_i
			Y_i = Y_train_i
			input_X.update({input_name: X_i})

	model = build_multi_length_model(nodes, dropout)

	filepath2 = path + "/weights.best.loss." + suffix + ".hdf5"
	checkpoint2 = ModelCheckpoint(filepath2, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
	callbacks_list = [checkpoint2]

	if cross_val:
		# define 10-fold cross validation test harness
		kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1337)
		cvscores = []
		for train, val in kfold.split(X_i, np.argmax(Y_i, axis=1)):
			checkpoint2 = ModelCheckpoint(filepath2, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
			callbacks_list = [checkpoint2]
			K.clear_session()
			del model
			model = build_multi_length_model(nodes, dropout)
			inputs_train_K_fold = {}
			inputs_val_K_fold = {}
			for item in input_X.items():
				inputs_train_K_fold.update({item[0]: item[1][train]})
				inputs_val_K_fold.update({item[0]: item[1][val]})

			model.fit(inputs_train_K_fold, {'output': Y_i[train]}, callbacks=callbacks_list,
					  validation_data=(inputs_val_K_fold, {'output': Y_i[val]}),
					  epochs=epochs, batch_size=batch_size, shuffle=shuffleTraining, verbose=2)

			scores = model.evaluate(inputs_val_K_fold, Y_i[val], verbose=0)
			print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
			cvscores.append(scores[1] * 100)
			model.save_weights(f"{path}/weights_model_k-fold_run_{len(cvscores)}.hdf5")
			# if real val_set exists, use models with lowest val_loss as models in ensemble
			os.rename(filepath2, f"{path}/weights_model_lowest_val_loss_k-fold_run_{len(cvscores)}.hdf5")

		print("summary")
		print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

		for middle_name in ("", "lowest_val_loss_"):
			models_filenames = []
			for file in sorted(os.listdir(path)):
				if file.endswith(".hdf5") and file.startswith(f"weights_model_{middle_name}k-fold_run_"):
					print(file)
					models_filenames.append(os.path.join(path, file))

			preds = []
			preds_val = []
			for fn in models_filenames:
				print("load model and predict")
				model.load_weights(fn)
				pred = model.predict(inputs_test)
				preds.append(pred)
				pred_val = model.predict(inputs_val)
				preds_val.append(pred_val)

			prediction_weights = [1. / len(models_filenames)] * len(models_filenames)
			accuracy, error = calculate_weighted_accuracy(prediction_weights, preds, preds_val, 2,
														  Y=Y_test_i, Y_val=Y_val_i,
														  ROC=True, name=f"{middle_name}ensemble")

			best_weights = weighted_ensemble(preds_val, 2, nb_models=len(models_filenames), Y=Y_val_i)
			accuracy, error = calculate_weighted_accuracy(best_weights, preds, preds_val, 2,
														  Y=Y_test_i, Y_val=Y_val_i,
														  ROC=True, name=f"{middle_name}ensemble_weighted")

	else:
		model.fit(inputs_train,
				  {'output': Y_train_i}, callbacks=callbacks_list,
				  validation_data=(inputs_val, {'output': Y_val_i}),
				  epochs=epochs, batch_size=batch_size, shuffle=shuffleTraining, verbose=2)

		compare_quality(model, path, inputs_test, Y_test_i, inputs_val, Y_val_i, pos_y_test_i, complex_model=False)


def test_and_plot(path, suffix, complex_model=False, online_training=False, shuffleTraining=True,
				  full_seq_embedding=False,
				  one_hot_encoding=True, val_size=0.3, design=1, sampleSize=1, nodes=32, use_generator=True,
				  snapShotEnsemble=False, epochs=100, dropout=0.0, faster=False, batch_size=32, sequence_length=50,
				  voting=False, tensorboard=False, gpus=False, titel='', x_axes='', y_axes='', accuracy=False,
				  loss=False, runtime=False, label1='', label2='', label3='', label4='', cross_val=True,
				  modular_training=True, include_raptorx_iupred=False, include_dict_scores=False, non_binary=False,
				  **kwargs):
	# GET SETTINGS AND PREPARE DATA
	# global X_train, X_test, X_val, Y_train, Y_test, Y_val, batch_size, SEED, directory

	# SAVE SETTINGS
	with open(path + '/' + suffix + "_config.txt", "w") as file:
		for i in list(locals().items()):
			file.write(str(i) + '\n')

	X_train, X_val, X_test, Y_train, Y_val, Y_test, pos_y_test = load_data(complex_model, path, val_size=0.2,
																		   generator=use_generator,non_binary=non_binary,
																		   sequence_length=sequence_length,
																		   full_seq_embedding=full_seq_embedding,
																		   include_raptorx_iupred=include_raptorx_iupred,
																		   include_dict_scores=include_dict_scores)
	filepath = path + "/weights.best.acc." + suffix + ".hdf5"
	filepath2 = path + "/weights.best.loss." + suffix + ".hdf5"
	filepath2_model = path + "/weights.best.loss." + suffix + "_complete_model.hdf5"
	filepath3 = path + "/weights.best.auc10." + suffix + ".hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True,
								 mode='max')
	checkpoint2 = ModelCheckpoint(filepath2, monitor='val_loss', verbose=1, save_best_only=True,
								  save_weights_only=True, mode='min')
	checkpoint2_model = ModelCheckpoint(filepath2_model, monitor='val_loss', verbose=1, save_best_only=True,
										save_weights_only=False, mode='min')
	checkpoint3 = ModelCheckpoint(filepath3, monitor='val_auc_10_perc_fpr', verbose=1, save_best_only=True,
								  save_weights_only=True, mode='max')
	tensorboard = TensorBoard(f'./tensorboard_log_dir')
	callbacks_list = [checkpoint, checkpoint2, checkpoint2_model, checkpoint3, tensorboard]
	# callbacks_list = []

	# filepath2 = path + "/weights.best.loss." + suffix + ".hdf5"
	# checkpoint2 = ModelCheckpoint(filepath2, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
	# callbacks_list = [checkpoint2]
	# filepath = path + "/weights.best.acc." + suffix + ".hdf5"
	# checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	# callbacks_list = [checkpoint]

	if not complex_model:
		if include_raptorx_iupred:
			model, Seq_model, Table_model = build_model_with_raptorx_iupred(nodes, dropout, seq_length=sequence_length)
			inputs_val = {}
			inputs_test = {}
			for col_num in range(2):
				if col_num == 0:
					inputs_val.update({"seq_input": X_val.sequences})
					inputs_test.update({"seq_input": X_test.sequences})

				if col_num == 1:
					inputs_val.update({"aux_input": X_val.table})
					inputs_val.update({"aux_input": X_val.table})
		elif include_dict_scores:
			model, Seq_model, Table_model = build_model_with_table(nodes, dropout, seq_length=sequence_length)
			inputs_val = {}
			inputs_test = {}
			for col_num in range(2):
				if col_num == 0:
					inputs_val.update({"seq_input": X_val.sequences})
					inputs_test.update({"seq_input": X_test.sequences})

				if col_num == 1:
					inputs_val.update({"aux_input": X_val.table})
					inputs_val.update({"aux_input": X_val.table})
		else:
			model = build_model(nodes, dropout, seq_length=sequence_length, non_binary=non_binary)  # X_train.shape[1])
		if use_generator:
			if include_raptorx_iupred or include_dict_scores:
				params = {"number_subsequences": 1, "dim": X_test.sequences.shape[1],
						  "n_channels": X_test.sequences.shape[-1],
						  "n_classes": Y_test.shape[-1], "shuffle": shuffleTraining, "online_training": online_training,
						  "seed": 1, "faster": batch_size}
			else:
				params = {"number_subsequences": 1, "dim": X_test.shape[1], "n_channels": X_test.shape[-1],
						  "n_classes": Y_test.shape[-1], "shuffle": shuffleTraining, "online_training": online_training,
						  "seed": 1, "faster": batch_size}

			training_generator = DataGenerator.DataGenerator(directory=directory + "/train",
															 sequence_length=sequence_length, non_binary = non_binary,
															 full_seq_embedding=full_seq_embedding,
															 include_raptorx_iupred=include_raptorx_iupred,
															 include_dict_scores=include_dict_scores,
															 **params, **kwargs)
			model.fit_generator(generator=training_generator, epochs=epochs, callbacks=callbacks_list,
			                    validation_data=(X_val, Y_val),
			                    shuffle=shuffleTraining, verbose=1)
			if include_raptorx_iupred or include_dict_scores:
				model.fit_generator(generator=training_generator, epochs=epochs, callbacks=callbacks_list,
									validation_data=((inputs_val, {'output': Y_val})), shuffle=shuffleTraining,
									verbose=1)

				model.load_weights(filepath)
				adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0,
									   amsgrad=False)
				set_trainability(Table_model, True)
				Table_model.compile(optimizer=adam, loss='binary_crossentropy')
				set_trainability(Seq_model, False)
				set_trainability(model, True)

				Seq_model.compile(optimizer=adam, loss='binary_crossentropy')
				model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['acc', auc_10_perc_fpr])

				# model.load_weights(filepath2)
				model.fit_generator(generator=training_generator, epochs=5, callbacks=callbacks_list,
									validation_data=((inputs_val, {'output': Y_val})), shuffle=shuffleTraining,
									verbose=1)

				model.load_weights(filepath)
				adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0,
									   amsgrad=False)
				set_trainability(Table_model, True)
				Table_model.compile(optimizer=adam, loss='binary_crossentropy')
				set_trainability(Seq_model, True)
				set_trainability(model, True)

				Seq_model.compile(optimizer=adam, loss='binary_crossentropy')
				model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['acc', auc_10_perc_fpr])

				# model.load_weights(filepath2)
				model.fit_generator(generator=training_generator, epochs=5, callbacks=callbacks_list,
									validation_data=((inputs_val, {'output': Y_val})), shuffle=shuffleTraining,
									verbose=1)

			model.save_weights(path + "/weights.last_model." + suffix + ".hdf5")
			# else:
				# model.fit_generator(generator=training_generator, epochs=epochs, callbacks=callbacks_list,
				# 					validation_data=(X_val, Y_val), shuffle=shuffleTraining, verbose=1)
		# set_trainability(model,True)
		# adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
		# model.compile(adam, 'binary_crossentropy', ['acc'])
		#
		# model.fit_generator(generator=training_generator, epochs=epochs//2, callbacks=callbacks_list,
		#                     validation_data=(X_val, Y_val), shuffle=shuffleTraining, verbose=2)
		# model.fit_generator(generator=training_generator, epochs=epochs//2, callbacks=callbacks_list,
		#                     validation_data=(X_val, Y_val), shuffle=shuffleTraining, verbose=2)
		# embeddings = model.layers[0].get_weights()[0]

		else:

			# define 10-fold cross validation test harness
			kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1337)
			cvscores = []
			# X = np.append(X_train, X_val, axis=0)
			# Y = np.append(Y_train, Y_val, axis=0)
			X = X_train
			Y = Y_train

			if cross_val:
				for train, test in kfold.split(X, np.argmax(Y, axis=1)):
					# checkpoint2 = ModelCheckpoint(filepath2, monitor='val_loss', verbose=1, save_best_only=True,
					#                               mode='min')
					# callbacks_list = [checkpoint2]
					checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
												 save_weights_only=True, mode='max')
					checkpoint2 = ModelCheckpoint(filepath2, monitor='val_loss', verbose=1, save_best_only=True,
												  save_weights_only=True, mode='min')
					checkpoint3 = ModelCheckpoint(filepath3, monitor='val_auc_10_perc_fpr', verbose=1,
												  save_best_only=True, save_weights_only=True, mode='max')
					# tensorboard = TensorBoard(f'{path}/tensorboard_log_dir')
					if tf.gfile.Exists(f"./tensorboard_log_dir/run{len(cvscores)}"):
						tf.gfile.DeleteRecursively(f"./tensorboard_log_dir/run{len(cvscores)}")
					tensorboard = TensorBoard(f'./tensorboard_log_dir/run{len(cvscores)}')
					callbacks_list = [checkpoint, checkpoint2, checkpoint3, tensorboard]

					K.clear_session()
					del model

					model = build_model(nodes, dropout, seq_length=sequence_length)
					model.fit(X[train], Y[train], epochs=epochs, batch_size=batch_size, verbose=2,
							  validation_data=(X_val, Y_val),
							  callbacks=callbacks_list, shuffle=shuffleTraining)
					scores = model.evaluate(X[test], Y[test], verbose=0)
					print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
					cvscores.append(scores[1] * 100)

					model.save_weights(f"{path}/weights_model_k-fold_run_{len(cvscores)}_{suffix}.hdf5")
					# if real val_set exists, use models with lowest val_loss as models in ensemble
					os.rename(filepath,
							  f"{path}/weights_model_highest_val_acc_k-fold_run_{len(cvscores)}_{suffix}.hdf5")
					os.rename(filepath2,
							  f"{path}/weights_model_lowest_val_loss_k-fold_run_{len(cvscores)}_{suffix}.hdf5")
					os.rename(filepath3,
							  f"{path}/weights_model_highest_val_auc10_k-fold_run_{len(cvscores)}_{suffix}.hdf5")

				print("summary")
				print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

				# global std_div, acc, length
				# std_div.append(np.std(cvscores))
				# acc.append(np.mean(cvscores))
				# length.append(sequence_length)
				# cvscores = []
				validate_cross_val_models(model, path, X_test, X_val, Y_test, Y_val)

			else:
				model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, Y_val),
						  verbose=2, callbacks=callbacks_list, shuffle=shuffleTraining)
	else:
		class_weight = clw.compute_class_weight('balanced', np.unique(np.argmax(Y_train, axis=1)),
												np.argmax(Y_train, axis=1))
		print(class_weight)
		model = build_complex_model(nodes, dropout, seq_length=X_train.seq_length,
									table_columns=X_train.table.shape[1])

		# define 10-fold cross validation test harness
		kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1337)
		cvscores = []
		# X = X_Data(np.append(X_train.sequences, X_val.sequences, axis=0), np.append(X_train.table, X_val.table, axis=0))
		# Y = np.append(Y_train, Y_val, axis=0)
		X = X_train
		Y = Y_train

		for train, test in kfold.split(X.sequences, np.argmax(Y, axis=1)):
			if cross_val:
				class_weight = clw.compute_class_weight('balanced', np.unique(np.argmax(Y[train], axis=1)),
														np.argmax(Y[train], axis=1))
				print(class_weight)
			# class_weight = clw.compute_class_weight('balanced', np.unique(np.argmax(Y[test], axis=1)),
			#                                         np.argmax(Y[test], axis=1))
			# print(class_weight)

			# checkpoint2 = ModelCheckpoint(filepath2, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
			checkpoint2 = ModelCheckpoint(filepath2, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
			callbacks_list = [checkpoint2]
			K.clear_session()
			del model
			model, Seq_model, Table_model = build_complex_model(nodes, dropout, seq_length=sequence_length,
																table_columns=X_train.table.shape[1])
			# plot graph
			plot_model(model, to_file='multiple_inputs.png')
			modular_training = True

			if modular_training:
				for epo in range(epochs):
					if epo == 50:
						set_trainability(Seq_model, False)
						set_trainability(Table_model, True)
						# set_trainability(model, False)

						Seq_model.compile(optimizer="adam", loss='binary_crossentropy')
						# adam = optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.1,
						#                        amsgrad=False)
						Table_model.compile(optimizer='adam', loss='binary_crossentropy')
						model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

					if cross_val:
						# model.fit({'seq_input': X.sequences[train], 'aux_input': X.table[train]},
						#           {'output': Y[train]}, callbacks=callbacks_list,
						#           epochs=epo + 1, batch_size=batch_size, shuffle=shuffleTraining, verbose=2,
						#           class_weight=class_weight, initial_epoch=epo)
						model.fit({'seq_input': X.sequences[train], 'aux_input': X.table[train]},
								  {'output': Y[train]}, callbacks=callbacks_list, validation_data=(
								{'seq_input': X_val.sequences, 'aux_input': X_val.table}, {'output': Y_val}),
								  epochs=epo + 1, batch_size=batch_size, shuffle=shuffleTraining, verbose=2,
								  class_weight=class_weight, initial_epoch=epo)
					else:
						model.fit({'seq_input': X_train.sequences, 'aux_input': X_train.table},
								  {'output': Y_train}, callbacks=callbacks_list,
								  validation_data=(
									  {'seq_input': X_val.sequences, 'aux_input': X_val.table}, {'output': Y_val}),
								  epochs=epo + 1, batch_size=batch_size, shuffle=shuffleTraining, verbose=2,
								  class_weight=class_weight, initial_epoch=epo)
			else:
				if cross_val:
					model.fit({'seq_input': X.sequences[train], 'aux_input': X.table[train]},
							  {'output': Y[train]}, callbacks=callbacks_list,
							  epochs=epochs, batch_size=batch_size, shuffle=shuffleTraining, verbose=2,
							  class_weight=class_weight)
				else:
					model.fit({'seq_input': X_train.sequences, 'aux_input': X_train.table},
							  {'output': Y_train}, callbacks=callbacks_list,
							  validation_data=(
								  {'seq_input': X_val.sequences, 'aux_input': X_val.table}, {'output': Y_val}),
							  epochs=epochs, batch_size=batch_size, shuffle=shuffleTraining, verbose=2,
							  class_weight=class_weight)

			if cross_val:
				scores = model.evaluate({'seq_input': X.sequences[test], 'aux_input': X.table[test]},
										{'output': Y[test]}, verbose=0)
				print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
				cvscores.append(scores[1] * 100)
				model.save_weights(f"{path}/weights_model_k-fold_run_{len(cvscores)}.hdf5")
				# if real val_set exists, use models with lowest val_loss as models in ensemble
				os.rename(filepath2, f"{path}/weights_model_lowest_val_loss_k-fold_run_{len(cvscores)}.hdf5")

		if cross_val:
			print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

			# single last model test acc
			pred = model.predict({'seq_input': X_test.sequences, 'aux_input': X_test.table})
			table = pd.crosstab(
				pd.Series(np.argmax(Y_test, axis=1)),
				pd.Series(np.argmax(pred, axis=1)),
				rownames=['True'],
				colnames=['Predicted'],
				margins=True)
			print(table)
			acc = sum(np.argmax(pred, axis=1) == np.argmax(Y_test, axis=1)) / len(np.argmax(Y_test, axis=1))
			print(f"Test accuracy last model: {acc}")

			for middle_name in ("", "lowest_val_loss_"):
				models_filenames = []
				for file in sorted(os.listdir(path)):
					if file.endswith(".hdf5") and file.startswith(f"weights_model_{middle_name}k-fold_run_"):
						print(file)
						models_filenames.append(os.path.join(path, file))

				preds = []
				preds_val = []
				for fn in models_filenames:
					print("load model and predict")
					model.load_weights(fn)
					pred = model.predict({'seq_input': X_test.sequences, 'aux_input': X_test.table})
					preds.append(pred)
					pred_val = model.predict({'seq_input': X_val.sequences, 'aux_input': X_val.table})
					preds_val.append(pred_val)

				prediction_weights = [1. / len(models_filenames)] * len(models_filenames)
				accuracy, error = calculate_weighted_accuracy(prediction_weights, preds, preds_val, 2,
															  Y=Y_test, Y_val=Y_val,
															  ROC=True, name=f"{middle_name}ensemble")

				best_weights = weighted_ensemble(preds_val, 2, nb_models=len(models_filenames), Y=Y_val)
				accuracy, error = calculate_weighted_accuracy(best_weights, preds, preds_val, 2,
															  Y=Y_test, Y_val=Y_val,
															  ROC=True, name=f"{middle_name}ensemble_weighted")

	if not cross_val:
		compare_quality(model, path, X_test, Y_test, X_val, Y_val, pos_y_test, complex_model=complex_model,
						include_raptorx_iupred=include_raptorx_iupred or include_dict_scores)


def validate_cross_val_models(model, path, inputs_test, inputs_val, Y_test, Y_val):
	for middle_name in ("", "lowest_val_loss_", "highest_val_acc_", "highest_val_auc10_"):
		models_filenames = []
		for file in sorted(os.listdir(path)):
			if file.endswith(f"_{suffix}.hdf5") and file.startswith(f"weights_model_{middle_name}k-fold_run_"):
				print(file)
				models_filenames.append(os.path.join(path, file))

		preds = []
		preds_val = []
		for fn in models_filenames:
			print("load model and predict")
			model.load_weights(fn, by_name=True)
			pred = model.predict(inputs_test)
			preds.append(pred)
			pred_val = model.predict(inputs_val)
			preds_val.append(pred_val)

		prediction_weights = [1. / len(models_filenames)] * len(models_filenames)
		accuracy, error = calculate_weighted_accuracy(prediction_weights, preds, preds_val, 2,
													  Y=Y_test, Y_val=Y_val,
													  ROC=True, name=f"{middle_name}ensemble")

		best_weights = weighted_ensemble(preds_val, 2, nb_models=len(models_filenames), Y=Y_val)
		accuracy, error = calculate_weighted_accuracy(best_weights, preds, preds_val, 2,
													  Y=Y_test, Y_val=Y_val,
													  ROC=True, name=f"{middle_name}ensemble_weighted")


def calculate_weighted_accuracy(prediction_weights, preds, preds_val, nb_classes, Y, Y_val, ROC=True,
								name="ensemble"):
	"""
	equally weighted model prediction accuracy
	:param prediction_weights: array with weights of single models e.g. [0,0.6,0.4]
	:param preds: array with the predicted classes/labels of the models
	:param nb_classes: how many different classes/labels exist
	:param X: raw-data which should be predicted
	:param Y: true labels for X
	:return: y_true_small == True labels for complete sequences, yTrue == True labels for complete subsequences, y_pred_mean == with mean predicted labels for complete sequences, y_pred_voted == voted labels for complete sequences, y_pred == predicted labels for complete subsequences
	"""
	weighted = any([i != prediction_weights[0] for i in prediction_weights])

	weighted_predictions = np.zeros((Y.shape[0], nb_classes), dtype='float32')

	for weight, prediction in zip(prediction_weights, preds):
		weighted_predictions += weight * np.array(prediction)

	yPred = np.argmax(weighted_predictions, axis=1)
	yTrue = np.argmax(Y, axis=-1)
	accuracy = metrics.accuracy_score(yTrue, yPred) * 100
	error = 100 - accuracy

	if ROC:
		# plot histogram of predictions
		yPred_0 = weighted_predictions[:, 1][yTrue == 0]
		yPred_1 = weighted_predictions[:, 1][yTrue == 1]
		yPred_total = np.array([yPred_0, yPred_1])
		import matplotlib.pyplot as plt

		plt.hist(yPred_total.swapaxes(0, 1), bins=20, range=(0, 1), stacked=False, label=['no Epitope', 'true Epitope'])
		plt.legend()
		plt.savefig(directory + f"/{name}.png")
		plt.close()

		weighted_predictions_val = np.zeros((Y_val.shape[0], nb_classes), dtype='float32')

		for weight, prediction in zip(prediction_weights, preds_val):
			weighted_predictions_val += weight * np.array(prediction)

		cutoff = calc_n_plot_ROC_curve(y_true=Y_val[:, 1], y_pred=weighted_predictions_val[:, 1], name=name, plot=False)
		calc_n_plot_ROC_curve(y_true=Y[:, 1], y_pred=weighted_predictions[:, 1], name=name)
		table = pd.crosstab(
			pd.Series(yTrue),
			pd.Series(yPred),
			rownames=['True'],
			colnames=['Predicted'],
			margins=True)
		print(table)
		print(f"Accuracy ensemble " + (not weighted) * f"not " + f"weighted: {accuracy}")
		print(f"Error ensemble: {error}")

		"""cutoff adapted"""
		print("cutoff adapted")
		weighted_predictions2 = weighted_predictions.copy()
		weighted_predictions[:, 1] = weighted_predictions[:, 1] > cutoff
		weighted_predictions[:, 0] = weighted_predictions[:, 1] == 0
		yPred = np.argmax(weighted_predictions, axis=1)
		table = pd.crosstab(
			pd.Series(yTrue),
			pd.Series(yPred),
			rownames=['True'],
			colnames=['Predicted'],
			margins=True)
		print(table)
		accuracy = metrics.accuracy_score(yTrue, yPred) * 100
		error = 100 - accuracy
		print(f"Accuracy ensemble " + (not weighted) * f"not " + f"weighted: {accuracy}")
		print(f"Error ensemble: {error}")

		"""cutoff adapted"""
		print("cutoff adapted 50/50")
		print(f"new cutoff {cutoff}")
		cutoff_median = np.median(weighted_predictions2[:, 1])
		weighted_predictions2[:, 1] = weighted_predictions2[:, 1] > cutoff_median
		weighted_predictions2[:, 0] = weighted_predictions2[:, 1] == 0
		yPred = np.argmax(weighted_predictions2, axis=1)
		table = pd.crosstab(
			pd.Series(yTrue),
			pd.Series(yPred),
			rownames=['True'],
			colnames=['Predicted'],
			margins=True)
		print(table)
		accuracy = metrics.accuracy_score(yTrue, yPred) * 100
		error = 100 - accuracy
		print(f"Accuracy ensemble " + (not weighted) * f"not " + f"weighted: {accuracy}")
		print(f"Error ensemble: {error}")

	return accuracy, error


def weighted_ensemble(preds, nb_classes, nb_models, Y, NUM_TESTS=250):
	"""
	calculates the best weights
	:param preds: array with predicted labels for X
	:param nb_classes: how many different classes/labels exist
	:param nb_models: how many different models exist
	:param X: raw-data which should be predicted
	:param Y: true labels for X
	:param NUM_TESTS: how many test should be done for the derteming the best weight
	:return: array with best weights
	"""

	# Create the loss metric
	def log_loss_func(weights, Y, preds, nb_classes):
		''' scipy minimize will pass the weights as a numpy array
		https://github.com/titu1994/Snapshot-Ensembles/blob/master/optimize_cifar100.ipynb
		'''
		final_prediction = np.zeros((Y.shape[0], nb_classes), dtype='float32')

		for weight, prediction in zip(weights, preds):
			final_prediction += weight * np.array(prediction)

		return log_loss(np.argmax(Y, axis=-1), final_prediction)

	best_acc = 0.0
	best_weights = None

	# Parameters for optimization
	constraints = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
	bounds = [(0, 1)] * len(preds)
	foo = []
	# Check for NUM_TESTS times
	for iteration in range(NUM_TESTS):
		# Random initialization of weights
		prediction_weights = np.random.random(nb_models)

		# Minimise the loss
		result = minimize(log_loss_func, prediction_weights, args=(Y, preds, nb_classes), method='SLSQP',
						  bounds=bounds, constraints=constraints)
		# print('Best Ensemble Weights: {weights}'.format(weights=result['x']))
		weights = result['x']
		foo.append(weights)
		accuracy, error = calculate_weighted_accuracy(prediction_weights, preds, None, 2, Y=Y, Y_val=None,
													  ROC=False)

		if accuracy > best_acc:
			best_acc = accuracy
			best_weights = weights

	print("Best accuracy: " + str(best_acc))
	print("Best weigths: " + str(best_weights))
	return best_weights


length = []
acc = []
std_div = []

if __name__ == "__main__":
	# directory = os.getcwd() + "/epitope_data_small"
	# directory = "/home/go96bix/projects/epitop_pred/data_complex_3_len30"
	# directory = "/home/go96bix/projects/epitop_pred/data_complex_2"
	# directory = "/home/go96bix/projects/epitop_pred/Full_data_len50"
	# directory = "/home/go96bix/projects/epitop_pred/Full_data_len50_rawSeq"
	# directory = "/home/go96bix/projects/epitop_pred/Full_data_len50_2"
	# directory = "/home/go96bix/projects/epitop_pred/Full_data"
	# directory = "/home/go96bix/projects/epitop_pred/data_no_polymorph"
	# directory = "/home/go96bix/projects/epitop_pred/data_only_polymorph"
	# directory = "/home/go96bix/projects/epitop_pred/validated_data"
	# directory = "/home/go96bix/projects/epitop_pred/data_seq_balanced"
	# directory = "/home/go96bix/projects/epitop_pred/epitope_data"
	# directory = "/home/go96bix/projects/epitop_pred/data_no_poly_no_raptorx"
	# directory = "/home/go96bix/projects/epitop_pred/data_generator/full_embedding_len50_more_noEpi"
	# directory = "/home/go96bix/projects/epitop_pred/data_generator_final/validated_non_epitopes/local_embedding"
	# directory = "/home/go96bix/projects/epitop_pred/data_generator_final/both_non_epitopes/global_embedding"
	# directory = "/home/go96bix/projects/epitop_pred/data_generator_bepipred/global_embedding"
	# directory = "/home/go96bix/projects/epitop_pred/data_generator_bepipred/cirkular_filling/local_embedding"
	# directory = "/home/go96bix/projects/epitop_pred/data_generator_bepipred_final/global_embedding"
	directory = "/home/go96bix/projects/epitop_pred/data_generator_bepipred_non_binary_0.9_seqID"
	suffix = "250_nodes_with_decay_global_1000epochs"
	complex_model = False
	nodes = 250
	dropout = 0.4
	sequence_length = 49
	full_seq_embedding = True
	include_raptorx_iupred = False
	include_dict_scores = False
	non_binary=True
	# test_multiple_length(path=directory, suffix=sufcutoff adaptedfix, complex_model=complex_model, nodes=nodes, dropout=dropout,
	#                      epochs=60,use_generator=False, batch_size=200, sequence_length=sequence_length,cross_val=True)
	# exit()
	# for sequence_length in range(10, 51, 2):
	#     test_and_plot(path=directory, suffix=suffix, complex_model=complex_model, nodes=nodes, dropout=dropout,
	#                   epochs=60,
	#                   use_generator=False, batch_size=200, sequence_length=sequence_length)
	test_and_plot(path=directory, suffix=suffix, complex_model=complex_model, nodes=nodes, dropout=dropout,
				  epochs=1000, use_generator=True, batch_size=64, sequence_length=sequence_length, cross_val=False,
				  full_seq_embedding=full_seq_embedding, include_raptorx_iupred=include_raptorx_iupred,
				  include_dict_scores=include_dict_scores, non_binary=non_binary)
	# test_broad_complex_model(path=directory, suffix=suffix, complex_model=complex_model, nodes=nodes, dropout=dropout,
	#                          epochs=240, use_generator=False, batch_size=500, sequence_length=sequence_length,
	#                          cross_val=True, modular_training=True, weight_decay = 1e-4)
	# build_elmo_embedding_model()

	"""test single model"""
	# X_train, X_val, X_test, Y_train, Y_val, Y_test, pos_y_test = load_data(complex_model, directory, val_size=0.3,
	#                                                                        generator=False,
	#                                                                        sequence_length=sequence_length)
	# model, Seq_model, list_models_table_cols =build_broad_complex_model(nodes,dropout,sequence_length)
	# model.load_weights("/home/go96bix/projects/epitop_pred/Full_data_len50_2/weights_model_lowest_val_loss_k-fold_run_8.hdf5")
	#
	# inputs_train = {}
	# inputs_val = {}
	# inputs_test = {}
	#
	# for col_num in range(10):
	#     input_name = f"table_input_{col_num}"
	#     if col_num == 0:
	#         inputs_train.update({"seq_input": X_train.sequences})
	#         inputs_val.update({"seq_input": X_val.sequences})
	#         inputs_test.update({"seq_input": X_test.sequences})
	#         # input_X.update({"seq_input": X_i})
	#
	#     col_train = X_train.table[:, 0:sequence_length]
	#     col_val = X_val.table[:, 0:sequence_length]
	#     col_test = X_test.table[:, 0:sequence_length]
	#
	#     if col_num != 9:
	#         col_train = np.array(col_train).reshape(X_train.table.shape[0], -1, 1)
	#         col_val = np.array(col_val).reshape(X_val.table.shape[0], -1, 1)
	#         col_test = np.array(col_test).reshape(X_test.table.shape[0], -1, 1)
	#
	#     X_train.table = X_train.table[:, sequence_length::]
	#     X_val.table = X_val.table[:, sequence_length::]
	#     X_test.table = X_test.table[:, sequence_length::]
	#
	#     inputs_train.update({input_name: col_train})
	#     inputs_val.update({input_name: col_val})
	#     inputs_test.update({input_name: col_test})
	#
	# preds_val = model.predict(inputs_val)
	# preds = model.predict(inputs_test)
	# accuracy, error = calculate_weighted_accuracy([1], [preds], [preds_val], 2,
	#                                               Y=Y_test, Y_val=Y_val,
	#                                               ROC=True, name=f"lowest_val_single_model_8")

	"""check if NN only checks distribution of AA. 
	# It does not."""
	# X_test_centerEpi = pd.read_csv("/home/le86qiz/Documents/Konrad/general_epitope_analyses/artificial_epitopes/center_epitopes.txt", delimiter='\t', dtype='str', header=None).values
	# X_test_completeEpi = pd.read_csv("/home/le86qiz/Documents/Konrad/general_epitope_analyses/artificial_epitopes/complete_epitopes.txt", delimiter='\t', dtype='str', header=None).values
	# X_test_random = pd.read_csv("/home/le86qiz/Documents/Konrad/general_epitope_analyses/artificial_epitopes/random_sequences.txt", delimiter='\t', dtype='str', header=None).values
	#
	# X_test_centerEpi = parse_amino(X_test_centerEpi,generator=False)
	# X_test_completeEpi = parse_amino(X_test_completeEpi,generator=False)
	# X_test_random = parse_amino(X_test_random,generator=False)
	#
	# model =build_model(nodes,dropout,sequence_length)
	# model.load_weights("/home/go96bix/projects/epitop_pred/epitope_data/weights.best.loss.generator_final.hdf5")
	# pred_centerEpi = model.predict(X_test_centerEpi)
	# pred_completeEpi = model.predict(X_test_completeEpi)
	# pred_random = model.predict(X_test_random)
	# true = np.array([0]*100000)
	# thresh = 0.6
	#
	# print("center Epi")
	# table = pd.crosstab(
	#     pd.Series(true),
	#     pd.Series(pred_centerEpi[:,1]>thresh),
	#     rownames=['True'],
	#     colnames=['Predicted'],
	#     margins=True)
	# print(table)
	# print(f"acc: {metrics.accuracy_score(true,pred_centerEpi[:,1]>thresh)}")
	# print()
	# print("complete Epi")
	# table = pd.crosstab(
	#     pd.Series(true),
	#     pd.Series(pred_completeEpi[:,1]>thresh),
	#     rownames=['True'],
	#     colnames=['Predicted'],
	#     margins=True)
	# print(table)
	# print(f"acc: {metrics.accuracy_score(true,pred_completeEpi[:,1]>thresh)}")
	# print()
	# print("random")
	# table = pd.crosstab(
	#     pd.Series(true),
	#     pd.Series(pred_random[:,1]>thresh),
	#     rownames=['True'],
	#     colnames=['Predicted'],
	#     margins=True)
	# print(table)
	# print(f"acc: {metrics.accuracy_score(true,pred_random[:,1]>thresh)}")

	exit()
	df = pd.DataFrame()
	df["length"] = length
	df["acc"] = acc
	df["std_dic"] = std_div
	df.to_csv(f"{directory}/length_test_results.csv", sep=",")
	plt.errorbar(length, acc, std_div)
	plt.xlabel('length of input sequence')
	plt.ylabel('accuracy')
	plt.title('importance of input length')
	plt.ylim((50, 100))
	plt.savefig(f"{directory}/length_test_results.pdf")
	exit()
	"""test curve of protein 220200"""
	# model = build_model(nodes, dropout, seq_length=50)
	X_train, X_val, X_test, Y_train, Y_val, Y_test, pos_y_test = load_data(complex_model, directory, val_size=0.3,
																		   generator=False)
	model = build_complex_model(nodes, dropout, seq_length=50)
	model_path = "/home/go96bix/projects/epitop_pred/epitope_data/weights.best.loss.test_generator.hdf5"
	model.load_weights(model_path)
	print(model.get_weights())
	exit()

	X_train, X_val, X_test, Y_train, Y_val, Y_test, pos_y_test = load_data(complex_model, directory, val_size=0.3,
																		   generator=True)

	# X_test_old = pd.read_csv("/home/go96bix/projects/epitop_pred/220200_test.csv", delimiter='\t', dtype='str',
	#                          header=None).values
	# X_220200_test = parse_amino(x=X_test_old, generator=True)

	Y_pred_test = model.predict(X_test)
	Y_pred_val = model.predict(X_val)
	# calc_n_plot_ROC_curve(y_true=Y_test.ravel(),y_pred=Y_pred.ravel())

	cutoff = calc_n_plot_ROC_curve(y_true=Y_val[:, 1], y_pred=Y_pred_val[:, 1])

	"""normal version with cutoff 0.5"""
	table = pd.crosstab(
		pd.Series(np.argmax(Y_test, axis=1)),
		pd.Series(np.argmax(Y_pred_test, axis=1)),
		rownames=['True'],
		colnames=['Predicted'],
		margins=True)
	print(table)

	acc = sum(np.argmax(Y_test, axis=1) == np.argmax(Y_pred_test, axis=1)) / len(np.argmax(Y_pred_test, axis=1))
	print(f"acc: {acc}")

	"""cutoff adapted"""
	Y_pred_test = Y_pred_test > cutoff
	table = pd.crosstab(
		pd.Series(np.argmax(Y_test, axis=1)),
		pd.Series(np.argmax(Y_pred_test, axis=1)),
		rownames=['True'],
		colnames=['Predicted'],
		margins=True)
	print(table)
	acc = sum(np.argmax(Y_test, axis=1) == np.argmax(Y_pred_test, axis=1)) / len(np.argmax(Y_pred_test, axis=1))
	print(f"acc: {acc}")

#
# prediction_protein = model.predict(X_220200_test)
# np.array(prediction_protein[:,1]).tofile("pred_220200_3.csv","\n")
