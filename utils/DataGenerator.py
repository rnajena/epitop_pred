from pathlib import Path
import pickle
import numpy as np
import keras
import os
import multiprocessing.pool
from functools import partial
import keras_preprocessing.image.utils as utils
from random import sample as randsomsample
import pandas as pd
import math
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from allennlp.commands.elmo import ElmoEmbedder
import torch

from train_DL import X_Data


class Elmo_embedder():
	def __init__(self, model_dir='/home/go96bix/projects/deep_eve/seqvec/uniref50_v2', weights="/weights.hdf5",
	             options="/options.json"):
		torch.set_num_threads(multiprocessing.cpu_count() // 2)
		self.model_dir = model_dir
		self.weights = self.model_dir + weights
		self.options = self.model_dir + options
		self.seqvec = ElmoEmbedder(self.options, self.weights, cuda_device=-1)

	def elmo_embedding(self, X, start=None, stop=None):
		assert start != None and stop != None, "deprecated to use start stop, please trim seqs beforehand"

		# X_trimmed = X[:, start:stop]
		# X_parsed = self.seqvec.embed_sentences(X_trimmed)
		# X_parsed = (np.array(list(X_parsed)).mean(axis=1))
		# return X_parsed
		if type(X[0]) == str:
			np.array([list(i.upper()) for i in X])
		X_parsed = []
		# X.sort(key=len)
		embedding = self.seqvec.embed_sentences(X)
		for i in embedding:
			print(i.shape)
			X_parsed.append(np.array(i).sum(axis=0))
			print(X_parsed[-1].shape)

		return X_parsed


class DataGenerator(keras.utils.Sequence):
	'Generates data for Keras'

	def __init__(self, directory, classes=None, number_subsequences=32, dim=(32, 32, 32), n_channels=6,
	             n_classes=10, shuffle=True, n_samples=None, seed=None, faster=True, online_training=False, repeat=True,
	             use_spacer=False, randomrepeat=False, sequence_length=50, full_seq_embedding=False, final_set=True,
	             include_raptorx_iupred=False, include_dict_scores=False, non_binary=False, **kwargs):
		'Initialization'
		self.directory = directory
		self.classes = classes
		self.dim = dim
		self.labels = None
		self.list_IDs = None
		self.n_channels = n_channels
		self.shuffle = shuffle
		self.seed = seed
		self.online_training = online_training
		self.repeat = repeat
		self.use_spacer = use_spacer
		self.randomrepeat = randomrepeat
		self.maxLen = kwargs.get("maxLen", None)
		self.sequence_length = sequence_length
		self.full_seq_embedding = full_seq_embedding
		self.final_set = final_set
		self.include_raptorx_iupred = include_raptorx_iupred
		self.include_dict_scores = include_dict_scores
		self.non_binary = non_binary

		if full_seq_embedding:
			file_format = 'pkl'
		else:
			file_format = 'csv'

		if number_subsequences == 1:
			self.shrink_timesteps = False
		else:
			self.shrink_timesteps = True

		self.number_subsequences = number_subsequences

		if faster == True:
			self.faster = 16
		elif type(faster) == int and faster > 0:
			self.faster = faster
		else:
			self.faster = 1

		self.number_samples_per_batch = self.faster

		self.number_samples_per_class_to_pick = n_samples

		if not classes:
			classes = []
			for subdir in sorted(os.listdir(directory)):
				if os.path.isdir(os.path.join(directory, subdir)):
					classes.append(subdir)
			self.classes = classes

		self.n_classes = len(classes)
		self.class_indices = dict(zip(classes, range(len(classes))))
		print(self.class_indices)
		# want a dict which contains dirs and number usable files
		pool = multiprocessing.pool.ThreadPool()
		function_partial = partial(_count_valid_files_in_directory,
		                           white_list_formats={file_format},
		                           follow_links=None,
		                           split=None)
		self.samples = pool.map(function_partial, (os.path.join(directory, subdir) for subdir in classes))
		self.samples = dict(zip(classes, self.samples))

		results = []

		for dirpath in (os.path.join(directory, subdir) for subdir in classes):
			results.append(pool.apply_async(utils._list_valid_filenames_in_directory,
			                                (dirpath, {file_format}, None, self.class_indices, None)))

		self.filename_dict = {}
		for res in results:
			classes, filenames = res.get()
			for index, class_i in enumerate(classes):
				self.filename_dict.update({f"{class_i}_{index}": filenames[index]})

		pool.close()
		pool.join()

		if not n_samples:
			self.number_samples_per_class_to_pick = min(self.samples.values())

		self.elmo_embedder = Elmo_embedder()

		self.on_epoch_end()

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.list_IDs) / self.number_samples_per_batch))

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		indexes = self.indexes[index * self.number_samples_per_batch:(index + 1) * self.number_samples_per_batch]

		# Find list of IDs
		list_IDs_temp = [self.list_IDs[k] for k in indexes]

		# Generate data
		X, y, sample_weight = self.__data_generation(list_IDs_temp, indexes)

		return X, y, sample_weight

	def on_epoch_end(self):
		'make X-train sample list'
		"""
		1. go over each class
		2. select randomly #n_sample samples of each class
		3. add selection list to dict with class as key 
		"""

		self.class_selection_path = np.array([])
		self.labels = np.array([])
		for class_i in self.classes:
			samples_class_i = randsomsample(range(0, self.samples[class_i]), self.number_samples_per_class_to_pick)
			self.class_selection_path = np.append(self.class_selection_path,
			                                      [self.filename_dict[f"{self.class_indices[class_i]}_{i}"] for i in
			                                       samples_class_i])
			self.labels = np.append(self.labels, [self.class_indices[class_i] for i in samples_class_i])

		self.list_IDs = self.class_selection_path

		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.list_IDs))
		if self.shuffle == True:
			if self.seed:
				np.random.seed(self.seed)
			np.random.shuffle(self.indexes)

	def __data_generation(self, list_IDs_temp, indexes):
		'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
		# Initialization
		X = np.empty((self.number_samples_per_batch), dtype=object)
		Y = np.empty((self.number_samples_per_batch), dtype=int)
		X_seq = np.empty((self.number_samples_per_batch), dtype=object)
		sample_weight = np.array([])

		# Generate data
		for i, ID in enumerate(list_IDs_temp):
			# Store sample
			# load tsv, parse to numpy array, get str and set as value in X[i]
			sample_weight = np.append(sample_weight, 1)
			if self.full_seq_embedding:
				if self.final_set:
					if self.include_raptorx_iupred:
						X[i] = np.array(pickle.load(open(os.path.join(self.directory, ID), "rb")))

					elif self.include_dict_scores:
						X[i] = np.array(pickle.load(open(os.path.join(self.directory, ID), "rb")))
						X_seq[i] = \
						pd.read_csv(os.path.join(self.directory, ID[:-4] + ".csv"), delimiter='\t', dtype='str',
						            header=None).values[0][0]

					else:
						if self.non_binary:
							X[i] = pickle.load(open(os.path.join(self.directory, ID), "rb"))
						else:
							X[i] = pickle.load(open(os.path.join(self.directory, ID), "rb"))[0]
				else:
					X[i] = pickle.load(open(os.path.join(self.directory, ID), "rb"))

			else:
				if self.final_set:
					if self.include_raptorx_iupred:
						X[i] = \
						pd.read_csv(os.path.join(self.directory, ID), delimiter='\t', dtype='str', header=None).values[
							0]
					else:
						if self.non_binary:
							print(os.path.join(self.directory, ID))
							X[i] = pd.read_csv(os.path.join(self.directory, ID), delimiter='\t', dtype='str',
							                   header=None).values
						else:
							X[i] = pd.read_csv(os.path.join(self.directory, ID), delimiter='\t', dtype='str',
							                   header=None).values[0][0]
				else:
					X[i] = \
					pd.read_csv(os.path.join(self.directory, ID), delimiter='\t', dtype='str', header=None)[1].values[0]
			# Store class
			Y[i] = self.labels[indexes[i]]

		sample_weight = np.array([[i] * self.number_subsequences for i in sample_weight]).flatten()
		if self.include_raptorx_iupred:
			samples_test = [i[1:] for i in X]
			table_X, filtered = load_raptorx_iupred(samples_test)
			X = np.array([i[0] for i in X])
		elif self.include_dict_scores:
			X = np.array([i[0] for i in X])
			table_X = get_dict_scores(X_seq)

		if self.non_binary:
			slicesize = self.sequence_length
			X_2 = []
			Y_2 = []
			for x_i in X:
				if self.full_seq_embedding:
					y_i = x_i[1].split("\t")
				else:
					y_i = x_i[1]
				possible_postions = np.where(np.array(y_i) != "-")[0]
				selection = np.random.permutation(possible_postions)
				for selection_index, i in enumerate(selection):
					if selection_index >= 5:
						break
					start = i - slicesize // 2
					stop = start + slicesize
					X_2.append(x_i[0][start:stop])
					Y_2.append([1 - float(y_i[i]), float(y_i[i])])
			X = np.array(X_2)
			Y_2 = np.array(Y_2)
			sample_weight = np.ones(len(Y_2))

		if not self.full_seq_embedding:
			if self.final_set:
				original_length = 49
			else:
				original_length = 50
			start_float = (original_length - self.sequence_length) / 2
			start = math.floor(start_float)
			stop = original_length - math.ceil(start_float)

			X = np.array([list(j) for j in X])
			X, mapping_X, slice_position = split_seq_n_times(X, self.sequence_length, self.number_subsequences)
			X = self.elmo_embedder.elmo_embedding(X, start, stop)
		else:
			X, mapping_X, slice_position = split_embedding_seq_n_times(X, self.sequence_length,
			                                                           self.number_subsequences)

		if self.non_binary:
			return X, Y_2, sample_weight
		if self.include_raptorx_iupred:
			table_sliced = np.empty((len(table_X), 49, 7))
			for index, i in enumerate(slice_position):
				if len(table_X[index]) == 49:
					table_sliced[index] = table_X[index]
				else:
					table_sliced[index] = table_X[index][i:i + 49]

			X_dict = {}
			X_dict.update({"seq_input": X})
			X_dict.update({"aux_input": table_sliced})
			return X_dict, keras.utils.to_categorical(Y, num_classes=self.n_classes), sample_weight
		elif self.include_dict_scores:
			table_sliced = np.empty((len(table_X), 49, 4))
			for index, i in enumerate(slice_position):
				if len(table_X[index]) == 49:
					table_sliced[index] = table_X[index]
				else:
					table_sliced[index] = table_X[index][i:i + 49]

			X_dict = {}
			X_dict.update({"seq_input": X})
			X_dict.update({"aux_input": table_sliced})
			return X_dict, keras.utils.to_categorical(Y, num_classes=self.n_classes), sample_weight
		else:
			return X, keras.utils.to_categorical(Y, num_classes=self.n_classes), sample_weight


def _count_valid_files_in_directory(directory, white_list_formats, split,
                                    follow_links):
	"""
	Copy from keras 2.1.5
	Count files with extension in `white_list_formats` contained in directory.

	Arguments:
		directory: absolute path to the directory
			containing files to be counted
		white_list_formats: set of strings containing allowed extensions for
			the files to be counted.
		split: tuple of floats (e.g. `(0.2, 0.6)`) to only take into
			account a certain fraction of files in each directory.
			E.g.: `segment=(0.6, 1.0)` would only account for last 40 percent
			of images in each directory.
		follow_links: boolean.

	Returns:
		the count of files with extension in `white_list_formats` contained in
		the directory.
	"""
	num_files = len(
		list(utils._iter_valid_files(directory, white_list_formats, follow_links)))
	if split:
		start, stop = int(split[0] * num_files), int(split[1] * num_files)
	else:
		start, stop = 0, num_files
	return stop - start


def parse_amino(x, encoder):
	out = []
	for i in x:
		dnaSeq = i[0].upper()
		encoded_X = encoder.transform(list(dnaSeq))
		out.append(encoded_X)
	return np.array(out)


def split_embedding_seq_n_times(embeddings, slicesize, amount_samples=10):
	splited_em_seqs = []
	mapping_slices_to_protein = []
	slice_position = []

	for index, protein in enumerate(embeddings):
		if len(protein) < slicesize:
			protein_pad = np.zeros((slicesize, 1024))
			for i in range(0, len(protein)):
				protein_pad[i] = protein[i]
			protein = protein_pad
		for i in np.random.choice(len(protein) - slicesize + 1, amount_samples):
			splited_em_seqs.append(protein[i:i + slicesize])
			mapping_slices_to_protein.append(index)
			slice_position.append(i)
	return np.array(splited_em_seqs), np.array(mapping_slices_to_protein), slice_position


def split_seq_n_times(seqs, slicesize, amount_samples=10):
	splited_em_seqs = []
	mapping_slices_to_protein = []
	slice_position = []

	for index, protein in enumerate(seqs):
		if len(protein) < slicesize:
			protein_pad = ['-'] * slicesize
			for i in range(0, len(protein)):
				protein_pad[i] = protein[i]
			protein = protein_pad
		for i in np.random.choice(len(protein) - slicesize + 1, amount_samples):
			splited_em_seqs.append(protein[i:i + slicesize])
			mapping_slices_to_protein.append(index)
			slice_position.append(i)
	return np.array(splited_em_seqs), np.array(mapping_slices_to_protein), slice_position


def load_raptorx_iupred(samples):
	out = []
	filtered = []
	shift = 20
	for index, sample in enumerate(samples):
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
			filtered.append(index)
			print(f"not able to load {file}")
			table_numpy_sliced = np.zeros((49, 7))

		out.append(table_numpy_sliced)
	return np.array(out), np.array(filtered)


def get_dict_scores(seqs):
	out_arr = []
	for index_seq, seq in enumerate(seqs):
		seq_arr = np.zeros((49, 4))
		for index, char in enumerate(seq):
			char = char.upper()
			hydro = hydrophilicity_scores.get(char, 0.5)
			beta = betaturn_scores.get(char, 0.5)
			surface = surface_accessibility_scores.get(char, 0.5)
			antigen = antigenicity_scores.get(char, 0.5)
			features = np.array([hydro, beta, surface, antigen])
			seq_arr[index] = features
		out_arr.append(seq_arr)
	return np.array(out_arr)


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

# hydrophilicity by parker
hydrophilicity_scores = normalize_dict(hydrophilicity_scores)
# Chou Fasman beta turn prediction (avg = 1)
betaturn_scores = normalize_dict(betaturn_scores)
# Emini surface accessibility scale (avg = 0.62)
surface_accessibility_scores = normalize_dict(surface_accessibility_scores)
# Kolaskar and Tongaokar antigenicity scale (avg = 1.0)
antigenicity_scores = normalize_dict(antigenicity_scores)
