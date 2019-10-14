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
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from scipy.optimize import minimize
from keras import backend as K
import math
from keras.engine import Layer
import tensorflow_hub as hub
from scipy import interp


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
		y_true_np = np.append(y_true_np, np.array([1,1])-y_true_np[0])
		y_pred_np = np.append(y_pred_np, np.array([1,1])-y_true_np[0])
		return roc_auc_score(y_true_np, y_pred_np, max_fpr=max_fpr)

	return tf.py_func(my_roc_auc_score, (y_true, y_pred), tf.double)

def auc_10_perc_fpr_binary(y_true, y_pred):
	def my_roc_auc_score(y_true_np, y_pred_np, max_fpr=0.1):
		y_true_np = np.append(y_true_np, np.array([1,1])-y_true_np[0])
		y_pred_np = np.append(y_pred_np, np.array([1,1])-y_true_np[0])
		y_bin = np.array(y_true_np >= 0.5, np.int)
		return roc_auc_score(y_bin, y_pred_np, max_fpr=max_fpr)

	return tf.py_func(my_roc_auc_score, (y_true, y_pred), tf.double)

def accuracy_binary(y_true, y_pred):
	def my_acc_score(y_true_np, y_pred_np):
		y_true_bin = np.array(y_true_np >= 0.5, np.int)
		y_pred_bin = np.array(y_pred_np >= 0.5, np.int)
		acc = accuracy_score(y_true_bin, y_pred_bin)
		return acc

	return tf.py_func(my_acc_score, (y_true, y_pred), tf.double)


def calc_n_plot_ROC_curve(y_true, y_pred, name="best", plot=True):
	fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
	auc = metrics.roc_auc_score(y_true, y_pred)
	optimal_idx = np.argmax(np.abs(tpr - fpr))

	# alternatives
	# optimal = tpr / (tpr + fpr)
	fnr = 1 - tpr
	tnr = 1 - fpr
	optimal_threshold = thresholds[optimal_idx]
	print(optimal_threshold)
	print(len(fpr))
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
		self.seq_length = self.sequences.shape[1]


def parse_amino(x, generator):
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
              final_set=True, include_raptorx_iupred=False, include_dict_scores=False, non_binary=False, own_embedding=False):
	def load_raptorx_iupred(samples):
		out = []
		shift = 20
		for sample in samples:
			start = int(sample[0])
			stop = int(sample[1])
			file = sample[2]
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
		X_train_old = np.array(pickle.load(open(directory + '/X_train.pkl', "rb")))
		# X_train_old = []
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
				protein_mapping = {}
				train_val_proteins = np.append(X_train_old[:,3],X_val_old[:, 3])
				train_val_proteins = np.append(train_val_proteins, X_test_old[:, 3])
				for index, i in enumerate(np.unique(train_val_proteins)):
					protein_mapping[index] = np.where(train_val_proteins == i)

				X_train = np.stack(X_train_old[:, 0])
				# X_train = X_train_old
				X_test = np.stack(X_test_old[:, 0])
				X_val = np.stack(X_val_old[:, 0])
		else:
			X_train = X_train_old
			X_test = X_test_old
			X_val = X_val_old
		if non_binary:
			Y_train = Y_train_old[:, 0]
			Y_test = np.array(Y_test_old[:, 0], np.float)
			Y_test = np.array([1 - Y_test, Y_test]).swapaxes(0, 1)
			Y_val = np.array(Y_val_old[:, 0], np.float)
			Y_val = np.array([1 - Y_val, Y_val]).swapaxes(0, 1)
		else:
			Y_train, y_encoder = DataParsing.encode_string(y=Y_train_old[:, 0])
			Y_test = DataParsing.encode_string(y=Y_test_old[:, 0], y_encoder=y_encoder)
			Y_val = DataParsing.encode_string(y=Y_val_old[:, 0], y_encoder=y_encoder)

		# original_length = 49
		# start_float = (original_length - sequence_length) / 2
		# start = math.floor(start_float)
		# stop = original_length - math.ceil(start_float)
		# X_test = X_test[:,start:stop]
		# X_train = X_train[:,start:stop]
		# X_val = X_val[:,start:stop]
		return X_train, X_val, X_test, Y_train, Y_val, Y_test, None, protein_mapping

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

				return X_train, X_val, X_test, Y_train, Y_val, Y_test, None, None

			elif include_dict_scores:
				pass

			else:
				protein_mapping = {}
				train_val_proteins = np.append(X_train_old[:,3],X_val_old[:, 3])
				train_val_proteins = np.append(train_val_proteins, X_test_old[:, 3])
				for index, i in enumerate(np.unique(train_val_proteins)):
					protein_mapping[index] = np.where(train_val_proteins == i)

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
			if own_embedding:
				amino = "ABCDEFGHIJKLMNOPQRSTUVWXYZ-"
				encoder = LabelEncoder()
				encoder.fit(list(amino))
				# print(np.unique(np.array([list(i.upper()) for i in X_train_old]).flatten()))
				X_train = np.array(list(map(encoder.transform, np.array([list(i.upper()) for i in X_train_old]))))
				# print(X_train)
				X_test = np.array(list(map(encoder.transform,  np.array([list(i.upper()) for i in X_test_old]))))
				X_val = np.array(list(map(encoder.transform,  np.array([list(i.upper()) for i in X_val_old]))))
			else:
				elmo_embedder = DataGenerator.Elmo_embedder()
				X_train = elmo_embedder.elmo_embedding(X_train_old[:, 1], start, stop)
				print(X_train.shape)
				X_test = elmo_embedder.elmo_embedding(X_test_old[:, 1], start, stop)
				X_val = elmo_embedder.elmo_embedding(X_val_old[:, 1], start, stop)
				print(X_val.shape)

		else:
			elmo_embedder = DataGenerator.Elmo_embedder()
			print("embedding test")
			if final_set:
				X_test_old = np.array([list(i) for i in X_test_old])
				X_val_old = np.array([list(i) for i in X_val_old])
			else:
				X_test_old = np.array([list(i) for j in X_test_old for i in j])
				X_val_old = np.array([list(i) for j in X_val_old for i in j])
			X_test = elmo_embedder.elmo_embedding(X_test_old, start, stop)
			print("embedding val")
			X_val = elmo_embedder.elmo_embedding(X_val_old, start, stop)

			print("embedding train")
			X_train = []

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

		print("burn in")
		elmo_embedder.elmo_embedding(X_test_old_seq, start, stop)
		print("embedding")
		X_train = X_Data(sequences=elmo_embedder.elmo_embedding(X_train_old_seq, start, stop),
		                 table=X_train_old[:, sequence_length:])
		X_test = X_Data(sequences=elmo_embedder.elmo_embedding(X_test_old_seq, start, stop),
		                table=X_test_old[:, sequence_length:])
		X_val = X_Data(sequences=elmo_embedder.elmo_embedding(X_val_old_seq, start, stop),
		               table=X_val_old[:, sequence_length:])

		pos_y_test = Y_test_old[:, 1:]

	if generator:
		if non_binary:
			Y_train = Y_train_old[:, 0]
			Y_test = np.array(Y_test_old[:, 0], np.float)
			Y_test = np.array([1 - Y_test, Y_test]).swapaxes(0, 1)
			Y_val = np.array(Y_val_old[:, 0], np.float)
			Y_val = np.array([1 - Y_val, Y_val]).swapaxes(0, 1)
		else:
			Y_train, y_encoder = DataParsing.encode_string(y=Y_train_old)
			Y_test = DataParsing.encode_string(y=Y_test_old, y_encoder=y_encoder)
			Y_val = DataParsing.encode_string(y=Y_val_old, y_encoder=y_encoder)

	elif complex:
		Y_train = to_categorical(np.array(Y_train_old[:, 0], dtype=np.float))
		Y_test = to_categorical(np.array(Y_test_old[:, 0], dtype=np.float))
		Y_val = to_categorical(np.array(Y_val_old[:, 0], dtype=np.float))
	else:
		if own_embedding:
			Y_train, y_encoder = DataParsing.encode_string(y=Y_train_old)
			Y_test = DataParsing.encode_string(y=Y_test_old, y_encoder=y_encoder)
			Y_val = DataParsing.encode_string(y=Y_val_old, y_encoder=y_encoder)
		else:
			Y_train, y_encoder = DataParsing.encode_string(y=Y_train_old[:, 1])
			Y_test = DataParsing.encode_string(y=Y_test_old[:, 1], y_encoder=y_encoder)
			Y_val = DataParsing.encode_string(y=Y_val_old[:, 1], y_encoder=y_encoder)

	return X_train, X_val, X_test, Y_train, Y_val, Y_test, pos_y_test, protein_mapping


def build_model(nodes, dropout, seq_length, weight_decay_lstm=1e-6, weight_decay_dense=1e-3, non_binary=False, own_embedding=False, both_embeddings=False):
	if own_embedding:
		inputs = layers.Input(shape=(seq_length,))
		seq_input = layers.Embedding(27, 10, input_length=seq_length)(inputs)
		hidden = layers.Bidirectional(
			layers.LSTM(nodes, return_sequences=True, dropout=dropout,
			            recurrent_dropout=0.2, kernel_regularizer=l2(weight_decay_lstm),
			            recurrent_regularizer=l2(weight_decay_lstm), bias_regularizer=l2(weight_decay_lstm)))(seq_input)
		hidden = layers.Bidirectional(
			layers.LSTM(nodes, dropout=dropout, recurrent_dropout=0.2, kernel_regularizer=l2(weight_decay_lstm),
			            recurrent_regularizer=l2(weight_decay_lstm), bias_regularizer=l2(weight_decay_lstm)))(hidden)

	elif both_embeddings:
		embedding_input=layers.Input(shape=(seq_length, 1024))
		left = layers.Bidirectional(
			layers.LSTM(nodes, input_shape=(seq_length, 1024), return_sequences=True, dropout=dropout,
			            recurrent_dropout=0.2, kernel_regularizer=l2(weight_decay_lstm),
			            recurrent_regularizer=l2(weight_decay_lstm), bias_regularizer=l2(weight_decay_lstm)))(embedding_input)
		left = layers.Dense(nodes)(left)
		left = layers.LeakyReLU(alpha=0.01)(left)
		out_left = layers.Flatten()(left)
		big_model = models.Model(embedding_input, out_left)

		seq_input = layers.Input(shape=(seq_length,))
		right = layers.Embedding(27, 10, input_length=seq_length)(seq_input)
		right = layers.Bidirectional(
			layers.LSTM(nodes, return_sequences=True, dropout=dropout,
			            recurrent_dropout=0.2, kernel_regularizer=l2(weight_decay_lstm),
			            recurrent_regularizer=l2(weight_decay_lstm), bias_regularizer=l2(weight_decay_lstm)))(right)
		right = layers.Dense(nodes)(right)
		right = layers.LeakyReLU(alpha=0.01)(right)
		out_right = layers.Flatten()(right)
		small_model = models.Model(seq_input, out_right)

		hidden = layers.concatenate([big_model(embedding_input),small_model(seq_input)])

	else:
		inputs = layers.Input(shape=(seq_length, 1024))
		hidden = layers.Bidirectional(
			layers.LSTM(nodes, input_shape=(seq_length, 1024), return_sequences=True, dropout=dropout,
			            recurrent_dropout=0.2, kernel_regularizer=l2(weight_decay_lstm),
			            recurrent_regularizer=l2(weight_decay_lstm), bias_regularizer=l2(weight_decay_lstm)))(inputs)
		hidden = layers.Bidirectional(
			layers.LSTM(nodes, dropout=dropout, recurrent_dropout=0.2, kernel_regularizer=l2(weight_decay_lstm),
			            recurrent_regularizer=l2(weight_decay_lstm), bias_regularizer=l2(weight_decay_lstm)))(hidden)

	# hidden = layers.Dense(nodes, kernel_regularizer=l2(weight_decay_dense), bias_regularizer=l2(weight_decay_dense))(
	# 	inputs)
	# hidden = layers.LeakyReLU(alpha=0.01)(hidden)
	# hidden = layers.Flatten()(hidden)
	hidden = layers.Dense(nodes, kernel_regularizer=l2(weight_decay_dense), bias_regularizer=l2(weight_decay_dense))(
		hidden)

	hidden = layers.LeakyReLU(alpha=0.01)(hidden)

	out = layers.Dense(2, activation='softmax', kernel_regularizer=l2(weight_decay_dense),
	                   bias_regularizer=l2(weight_decay_dense))(hidden)
	if both_embeddings:
		model = models.Model(inputs=[embedding_input,seq_input], outputs=out)
	else:
		model = models.Model(inputs=inputs, outputs=out)

	adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	if non_binary:
		model.compile(optimizer="adam", loss='binary_crossentropy', metrics=[accuracy_binary, auc_10_perc_fpr_binary])
	else:
		if both_embeddings:
			set_trainability(big_model, False)
			small_model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['acc', auc_10_perc_fpr])
			big_model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['acc', auc_10_perc_fpr])
			model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['acc', auc_10_perc_fpr])
			model.summary()
			return model, small_model, big_model
		model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['acc', auc_10_perc_fpr])
	model.summary()
	return model, None, None


def build_complex_model(nodes, dropout, seq_length=19, vector_dim=1, table_columns=0):
	seq_input = layers.Input(shape=(seq_length,), name="seq_input")
	left = layers.Embedding(21, 10, input_length=seq_length)(seq_input)

	left = layers.Bidirectional(layers.LSTM(nodes, return_sequences=True, dropout=dropout, recurrent_dropout=0.2))(left)
	left = layers.Bidirectional(layers.LSTM(nodes, dropout=dropout, recurrent_dropout=0.2))(left)
	left = layers.Dense(nodes)(left)
	out_left = layers.LeakyReLU(alpha=0.01)(left)
	Seq_model = models.Model(seq_input, out_left)
	auxiliary_input = layers.Input(shape=(table_columns,), dtype='float', name='aux_input')

	right = layers.Dense(nodes)(auxiliary_input)
	right = layers.LeakyReLU(alpha=0.01)(right)

	Table_model = models.Model(auxiliary_input, right)

	middle_input = layers.concatenate([Seq_model(seq_input), Table_model(auxiliary_input)])
	middle = layers.Dense(nodes)(middle_input)
	middle = layers.LeakyReLU(alpha=0.01)(middle)

	output = layers.Dense(2, activation='softmax', name='output')(middle)
	model = models.Model(inputs=[seq_input, auxiliary_input], outputs=output)

	set_trainability(Table_model, False)
	Seq_model.compile(optimizer="adam", loss='binary_crossentropy')
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
	Table_model.compile(optimizer="adam", loss='binary_crossentropy')
	model.summary()
	return model, Seq_model, Table_model


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

		accuracy, error = calculate_weighted_accuracy([1], [pred], [pred_val], 2,
		                                              Y=Y_test, Y_val=Y_val,
		                                              ROC=True, name=f"{middle_name}")

	def calc_acc_with_cutoff(name="best"):
		if complex_model:
			Y_pred_test = model.predict({'seq_input': X_test.sequences, 'aux_input': X_test.table})
			Y_pred_val = model.predict({'seq_input': X_val.sequences, 'aux_input': X_val.table})

		else:
			Y_pred_test = model.predict(X_test)
			Y_pred_val = model.predict(X_val)

		cutoff = calc_n_plot_ROC_curve(y_true=Y_val[:, 1], y_pred=Y_pred_val[:, 1], name=name)

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
	if path:
		for middle_name in ("loss", "acc", "auc10"):
			model_path = f"{path}/weights.best.{middle_name}.{suffix}.hdf5"

			calc_quality(model, X_test, Y_test, X_val, Y_val, pos_y_test, complex_model, path=path,
			             middle_name=f"{middle_name}_{suffix}", include_raptorx_iupred=include_raptorx_iupred)


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
	return model


def build_elmo_embedding_model():
	import tensorflow as tf
	import tensorflow_hub as hub
	elmo = hub.Module("/home/go96bix/projects/deep_eve/elmo", trainable=False)

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
	Seq_model.compile(optimizer="adam", loss='binary_crossentropy')
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', auc_10_perc_fpr])
	Table_model.compile(optimizer="adam", loss='binary_crossentropy')
	model.summary()
	return model, Seq_model, Table_model


def build_broad_complex_model(nodes, dropout, sequence_length=50, weight_decay=1e-6):
	adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

	tensors_models = []
	list_models_table_cols = []
	inputs_table_cols = []

	"""sequence model"""
	seq_input = layers.Input(shape=(sequence_length, 1024), name="seq_input")
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
			right = layers.Dense(nodes, kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(
				table_input)
			right_out = layers.LeakyReLU(alpha=0.01)(right)
			Table_model = models.Model(table_input, right_out)

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

				model.fit(inputs_train_K_fold, {'output': Y_train[train]}, callbacks=callbacks_list,
				          validation_data=(inputs_val_K_fold, {'output': Y_train[val]}),
				          epochs=5, batch_size=batch_size, shuffle=shuffleTraining, verbose=1)

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
                  own_embedding=False, both_embeddings=False, **kwargs):
	# SAVE SETTINGS
	with open(path + '/' + suffix + "_config.txt", "w") as file:
		for i in list(locals().items()):
			file.write(str(i) + '\n')

	X_train, X_val, X_test, Y_train, Y_val, Y_test, pos_y_test, protein_mapping = load_data(complex_model, path, val_size=0.2,
	                                                                       generator=use_generator,
	                                                                       non_binary=non_binary,
	                                                                       sequence_length=sequence_length,
	                                                                       full_seq_embedding=full_seq_embedding,
	                                                                       include_raptorx_iupred=include_raptorx_iupred,
	                                                                       include_dict_scores=include_dict_scores,
	                                                                       own_embedding=own_embedding)
	# X_test_2 = X_val.copy()
	# Y_test_2 = Y_val.copy()
	# X_val = X_test.copy()
	# Y_val = Y_test.copy()
	# X_test = X_test_2
	# Y_test = Y_test_2

	filepath = path + "/weights.best.acc." + suffix + ".hdf5"
	filepath2 = path + "/weights.best.loss." + suffix + ".hdf5"
	filepath2_model = path + "/weights.best.loss." + suffix + "_complete_model.hdf5"
	filepath3 = path + "/weights.best.auc10." + suffix + ".hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True,
	                             mode='max')
	# checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy_binary', verbose=1, save_best_only=True, save_weights_only=True,
	#                              mode='max')
	# checkpoint = EarlyStopping('val_loss', min_delta=0, patience=epochs//10, restore_best_weights=True)

	checkpoint2 = ModelCheckpoint(filepath2, monitor='val_loss', verbose=1, save_best_only=True,
	                              save_weights_only=True, mode='min')
	# checkpoint2_model = ModelCheckpoint(filepath2_model, monitor='val_loss', verbose=1, save_best_only=True,
	#                                     save_weights_only=False, mode='min')
	checkpoint3 = ModelCheckpoint(filepath3, monitor='val_auc_10_perc_fpr', verbose=1, save_best_only=True,
								  save_weights_only=True, mode='max')
	# checkpoint3 = ModelCheckpoint(filepath3, monitor='val_auc_10_perc_fpr_binary', verbose=1, save_best_only=True,
	# 							  save_weights_only=True, mode='max')

	# tensorboard = TensorBoard(f'./tensorboard_log_dir')
	# checkpoint2 = ModelCheckpoint(filepath2, monitor='loss', verbose=1, save_best_only=True,
	#                               save_weights_only=True, mode='min')
	# callbacks_list = [checkpoint2]
	callbacks_list = [checkpoint, checkpoint2, checkpoint3]

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
					inputs_test.update({"aux_input": X_test.table})
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
					inputs_test.update({"aux_input": X_test.table})
		else:
			model = build_model(nodes, dropout, seq_length=sequence_length, non_binary=non_binary, own_embedding=own_embedding, both_embeddings=both_embeddings)  # X_train.shape[1])
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
			                                                 sequence_length=sequence_length, non_binary=non_binary,
			                                                 full_seq_embedding=full_seq_embedding,
			                                                 include_raptorx_iupred=include_raptorx_iupred,
			                                                 include_dict_scores=include_dict_scores,
			                                                 **params, **kwargs)

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
			else:
				model.fit_generator(generator=training_generator, epochs=epochs, callbacks=callbacks_list,
				                    validation_data=(X_val, Y_val),
				                    shuffle=shuffleTraining, verbose=1)

			model.save_weights(path + "/weights.last_model." + suffix + ".hdf5")

		else:
			join_train_test_val= True
			protein_list = list(protein_mapping.keys())
			# define 10-fold cross validation test harness
			kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1337)
			cvscores = []
			if cross_val:
				if join_train_test_val:

					if both_embeddings:
						X_train_seq, X_val_seq, X_test_seq, Y_train, Y_val, Y_test, pos_y_test, protein_mapping = load_data(
							complex_model, path, val_size=0.2,
							generator=use_generator,
							non_binary=non_binary,
							sequence_length=sequence_length,
							full_seq_embedding=False,
							include_raptorx_iupred=include_raptorx_iupred,
							include_dict_scores=include_dict_scores,
							own_embedding=True)
						X_seq = np.append(X_train_seq, X_val_seq, axis=0)
						X_seq = np.append(X_seq, X_test_seq, axis=0)
						del X_train_seq, X_val_seq, X_test_seq
					# inputs_train ={}
					# inputs_train.update({"seq_input": X_seq, "embedding_input": X})

					X = np.append(X_train, X_val, axis=0)
					del X_train, X_val
					X = np.append(X, X_test, axis=0)
					del X_test
					Y = np.append(Y_train, Y_val, axis=0)
					Y = np.append(Y, Y_test, axis=0)

					n_proteins = np.zeros(len(protein_list))

					tprs = []
					aucs = []
					mean_fpr = np.linspace(0, 1, 100)

					for train, test in kfold.split(protein_list, n_proteins):
						checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
						                             save_weights_only=True, mode='max')
						checkpoint2 = ModelCheckpoint(filepath2, monitor='val_loss', verbose=1, save_best_only=True,
						                              save_weights_only=True, mode='min')
						checkpoint3 = ModelCheckpoint(filepath3, monitor='val_auc_10_perc_fpr', verbose=1,
						                              save_best_only=True, save_weights_only=True, mode='max')
						if tf.gfile.Exists(f"./tensorboard_log_dir/run{len(cvscores)}"):
							tf.gfile.DeleteRecursively(f"./tensorboard_log_dir/run{len(cvscores)}")
						tensorboard = TensorBoard(f'./tensorboard_log_dir/run{len(cvscores)}')
						callbacks_list = [checkpoint, checkpoint2, checkpoint3, tensorboard]

						K.clear_session()
						del model

						model, small_model, big_model = build_model(nodes, dropout, seq_length=sequence_length, own_embedding=own_embedding, both_embeddings=both_embeddings)

						train = [k for i in train for j in protein_mapping[i] for k in j]
						test = [k for i in test for j in protein_mapping[i] for k in j]
						assert len(test) == len(np.unique(test)), "duplicates found in k_fold test split"
						if both_embeddings:
							model.fit([X[train],X_seq[train]], Y[train], epochs=epochs, batch_size=batch_size, verbose=2,
							          validation_data=([X[test], X_seq[test]], Y[test]),
							          callbacks=callbacks_list, shuffle=shuffleTraining)

							# model.load_weights(filepath3)
							adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0,
												   amsgrad=False)
							set_trainability(big_model, True)
							big_model.compile(optimizer="adam", loss='binary_crossentropy')
							set_trainability(small_model, False)
							set_trainability(model, True)

							small_model.compile(optimizer="adam", loss='binary_crossentropy')
							model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['acc', auc_10_perc_fpr])

							model.fit([X[train], X_seq[train]], Y[train], epochs=10, batch_size=batch_size, verbose=2,
									  validation_data=([X[test], X_seq[test]], Y[test]),
									  callbacks=callbacks_list, shuffle=shuffleTraining)

						else:
							model.fit(X[train], Y[train], epochs=epochs, batch_size=batch_size, verbose=2,
							          validation_data=(X[test], Y[test]),
							          callbacks=callbacks_list, shuffle=shuffleTraining)
						if both_embeddings:
							scores = model.evaluate([X[test], X_seq[test]], Y[test], verbose=0, batch_size=batch_size)
						else:
							scores = model.evaluate(X[test], Y[test], verbose=0, batch_size=batch_size)
						print("%s: %.4f" % (model.metrics_names[2], scores[2]))
						cvscores.append(scores[2])

						model.save_weights(f"{path}/weights_model_k-fold_run_{len(cvscores)}_{suffix}.hdf5")
						os.rename(filepath,
						          f"{path}/weights_model_highest_val_acc_k-fold_run_{len(cvscores)}_{suffix}.hdf5")
						os.rename(filepath2,
						          f"{path}/weights_model_lowest_val_loss_k-fold_run_{len(cvscores)}_{suffix}.hdf5")
						os.rename(filepath3,
						          f"{path}/weights_model_highest_val_auc10_k-fold_run_{len(cvscores)}_{suffix}.hdf5")

						# Compute ROC curve and area the curve
						if both_embeddings:
							pred = model.predict([X[test], X_seq[test]])
						else:
							pred = model.predict(X[test])
						fpr, tpr, thresholds = metrics.roc_curve(Y[:, 1][test], pred[:, 1])
						tprs.append(interp(mean_fpr, fpr, tpr))
						tprs[-1][0] = 0.0
						roc_auc = metrics.auc(fpr, tpr)
						aucs.append(roc_auc)
						plt.plot(fpr, tpr, lw=1, alpha=0.3,
						         label='ROC fold %d (AUC = %0.2f)' % (len(cvscores), roc_auc))

					mean_tpr = np.mean(tprs, axis=0)
					mean_tpr[-1] = 1.0
					mean_auc = metrics.auc(mean_fpr, mean_tpr)
					std_auc = np.std(aucs)

					std_tpr = np.std(tprs, axis=0)
					tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
					tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
					plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
					                 label=r'$\pm$ 1 std. dev.')

					plt.plot(mean_fpr, mean_tpr, color='b',
					         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
					         lw=2, alpha=.8)
					plt.plot([0, 1], [0, 1], 'k--', lw=2)
					plt.xlim([0.0, 1.0])
					plt.ylim([0.0, 1.05])
					plt.xlabel('False Positive Rate')
					plt.ylabel('True Positive Rate')
					plt.title('Receiver operating characteristic for multiple classes')
					plt.legend(loc="lower right")
					plt.savefig(directory + f"/roc_curve_{suffix}_ensemble.pdf")
					plt.close()

					print("%.4f (+/- %.4f)" % (np.mean(cvscores), np.std(cvscores)))
				else:
					X = X_train
					Y = Y_train

					for train, test in kfold.split(X, np.argmax(Y, axis=1)):
						checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
						                             save_weights_only=True, mode='max')
						checkpoint2 = ModelCheckpoint(filepath2, monitor='val_loss', verbose=1, save_best_only=True,
						                              save_weights_only=True, mode='min')
						checkpoint3 = ModelCheckpoint(filepath3, monitor='val_auc_10_perc_fpr', verbose=1,
						                              save_best_only=True, save_weights_only=True, mode='max')
						if tf.gfile.Exists(f"./tensorboard_log_dir/run{len(cvscores)}"):
							tf.gfile.DeleteRecursively(f"./tensorboard_log_dir/run{len(cvscores)}")
						tensorboard = TensorBoard(f'./tensorboard_log_dir/run{len(cvscores)}')
						callbacks_list = [checkpoint, checkpoint2, checkpoint3, tensorboard]

						K.clear_session()
						del model

						model = build_model(nodes, dropout, seq_length=sequence_length, own_embedding=own_embedding)

						model.fit(X[train], Y[train], epochs=epochs, batch_size=batch_size, verbose=2,
						          validation_data=(X_val, Y_val),
						          callbacks=callbacks_list, shuffle=shuffleTraining)
						scores = model.evaluate(X[test], Y[test], verbose=0)
						scores = model.evaluate(X[test], Y[test], verbose=0, batch_size=batch_size)

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

					validate_cross_val_models(model, path, X_test, X_val, Y_test, Y_val)

			else:
				model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, Y_val),
				          verbose=1, callbacks=callbacks_list, shuffle=shuffleTraining)
	else:
		class_weight = clw.compute_class_weight('balanced', np.unique(np.argmax(Y_train, axis=1)),
		                                        np.argmax(Y_train, axis=1))
		print(class_weight)
		model = build_complex_model(nodes, dropout, seq_length=X_train.seq_length,
		                            table_columns=X_train.table.shape[1])

		# define 10-fold cross validation test harness
		kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1337)
		cvscores = []
		X = X_train
		Y = Y_train

		for train, test in kfold.split(X.sequences, np.argmax(Y, axis=1)):
			if cross_val:
				class_weight = clw.compute_class_weight('balanced', np.unique(np.argmax(Y[train], axis=1)),
				                                        np.argmax(Y[train], axis=1))
				print(class_weight)
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

						Seq_model.compile(optimizer="adam", loss='binary_crossentropy')
						Table_model.compile(optimizer='adam', loss='binary_crossentropy')
						model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

					if cross_val:
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
	yTrue_val = np.argmax(Y_val, axis=-1)
	accuracy = metrics.accuracy_score(yTrue, yPred) * 100
	error = 100 - accuracy

	if ROC:
		# plot histogram of predictions
		yPred_0 = weighted_predictions[:, 1][yTrue == 0]
		yPred_1 = weighted_predictions[:, 1][yTrue == 1]
		yPred_total = [yPred_0, yPred_1]
		import matplotlib.pyplot as plt

		plt.hist(yPred_total, bins=20, range=(0, 1), stacked=False, label=['no Epitope', 'true Epitope'])
		plt.legend()
		plt.savefig(directory + f"/{name}.png")
		plt.close()

		weighted_predictions_val = np.zeros((Y_val.shape[0], nb_classes), dtype='float32')

		for weight, prediction in zip(prediction_weights, preds_val):
			weighted_predictions_val += weight * np.array(prediction)

		cutoff = calc_n_plot_ROC_curve(y_true=yTrue_val, y_pred=weighted_predictions_val[:, 1], name=name, plot=False)
		calc_n_plot_ROC_curve(y_true=yTrue, y_pred=weighted_predictions[:, 1], name=name)
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
	directory = "/home/go96bix/projects/epitop_pred/data_generator_bepipred_binary_0.5_seqID"
	# directory = "/home/go96bix/projects/epitop_pred/data_generator_bepipred_binary_0.8_seqID_checked_output"
	# directory = "/home/go96bix/projects/epitop_pred/data_generator_bepipred_binary_allProteins"
	suffix = "both_embeddings_small_big_approach_keep_learning_long"
	complex_model = False
	nodes = 10
	dropout = 0.4
	sequence_length = 49
	full_seq_embedding = True
	include_raptorx_iupred = False
	include_dict_scores = False
	non_binary = False
	own_embedding = False
	both_embeddings = True
	if both_embeddings:
		own_embedding=False
	test_and_plot(path=directory, suffix=suffix, complex_model=complex_model, nodes=nodes, dropout=dropout,
	              epochs=50, use_generator=False, batch_size=640, sequence_length=sequence_length, cross_val=True,
	              full_seq_embedding=full_seq_embedding, include_raptorx_iupred=include_raptorx_iupred,
	              include_dict_scores=include_dict_scores, non_binary=non_binary, own_embedding=own_embedding, both_embeddings=both_embeddings)
