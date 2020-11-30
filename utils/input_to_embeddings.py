"""Load pre-trained model:"""
from allennlp.commands.elmo import ElmoEmbedder
from allennlp.modules.elmo import Elmo, batch_to_ids
from pathlib import Path
import os
import numpy as np
import pickle
import time
import torch.nn as nn
import torch
import subprocess
import collections, gc, resource, torch

class Elmo_embedder():
	def __init__(self, model_dir='/home/go96bix/projects/deep_eve/seqvec/uniref50_v2', weights="/weights.hdf5",
				 options="/options.json"):
		torch.set_num_threads(multiprocessing.cpu_count() // 2)
		self.model_dir = model_dir
		self.weights = self.model_dir + weights
		self.options = self.model_dir + options
		self.seqvec = ElmoEmbedder(self.options, self.weights, cuda_device=-1)

	def elmo_embedding(self, X, start=None, stop=None):
		print(X.shape)
		if start != None and stop != None:
			X_trimmed = X[:, start:stop]
			X_parsed = self.seqvec.embed_sentences(X_trimmed)
			X_parsed = (np.array(list(X_parsed)).mean(axis=1))

		else:
			X_parsed = []
			# X.sort(key=len)
			embedding = self.seqvec.embed_sentences(X,batch_size=2)
			for i in embedding:
				X_parsed.append(np.array(i).sum(axis=0))
		return X_parsed

def debug_memory():
	print('maxrss = {}'.format(
		resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
	tensors = collections.Counter((str(o.device), o.dtype, tuple(o.shape))
	                              for o in gc.get_objects()
	                              if torch.is_tensor(o))

	for line in tensors.items():
		print('{}\t{}'.format(*line))


def dump_tensors(gpu_only=True):
	torch.cuda.empty_cache()
	for obj in gc.get_objects():
		try:
			if torch.is_tensor(obj):
				if not gpu_only or obj.is_cuda:
					del obj
					gc.collect()
			elif hasattr(obj, "data") and torch.is_tensor(obj.data):
				if not gpu_only or obj.is_cuda:
					del obj
					gc.collect()
		except Exception as e:
			pass


def get_gpu_memory_map():
	result = subprocess.check_output(
		[
			'nvidia-smi', '--query-gpu=memory.used',
			'--format=csv,nounits,noheader'
		])

	return result


class Elmo_embedder():
	def __init__(self, model_dir='/home/go96bix/projects/deep_eve/seqvec/uniref50_v2', weights="/weights.hdf5",
	             options="/options.json"):
		# torch.set_num_threads(multiprocessing.cpu_count()//2)
		self.model_dir = model_dir
		self.weights = self.model_dir + weights
		self.options = self.model_dir + options
		self.seqvec = ElmoEmbedder(self.options, self.weights, cuda_device=-1)

	def elmo_embedding(self, X):
		X_parsed = self.seqvec.embed_sentences(X, 100)
		return list(X_parsed)


class DL_embedding():
	def __init__(self, header, seq_embedding):
		self.sequences = np.array(seq_embedding)
		self.header = header
		self.seq_length = self.sequences.shape[0]

def file_len(fname):
	with open(fname) as f:
		for i, l in enumerate(f):
			pass
	return i + 1


def sort_file(fname, start_index=0, line_batch_size=1000000):
	"""
	CAUTION longest 1% will not be returned
	1. open file
	2. collect header and seqs from line start_index to start_index + line_batch_size
	3. sort by length
	4. return sorted lists of header and seqs
	:param fname: input file path
	:param start_index: number line were to start
	:param line_batch_size: how many line to sort
	:return: sorted list of header, seqs
	"""

	with open(fname) as f:
		start = time.time()
		header = []
		seqs = []
		end = False
		for index, line in enumerate(f):
			if index < start_index:
				pass
			elif index < start_index + line_batch_size:
				if line.startswith(">"):
					header.append(line.strip())
				else:
					seqs.append(line.strip())
				end = True
			else:
				end = False
				break

		lengths = [len(i) for i in seqs]
		# sorting = np.argsort(lengths)[int(0.99 * len(lengths)):]
		sorting = np.argsort(lengths)[int(0.955 * len(lengths)):int(0.965 * len(lengths))]
		# sorting = np.argsort(lengths)[0:int(0.99 * len(lengths))]
		header = np.array([header[i] for i in sorting])
		seqs = np.array([seqs[i] for i in sorting])
		stop = time.time()
		print(stop - start)
	return header, seqs, index, end


def parse_file(path='../../input/Archeae_rep_seq.fa', max_array_size=65000):
	"""
	1. open fasta file
	2. sort batch of lines by length of sequences
	3. build batches smaller max_array_size (65.000 ~ 32GB GPU RAM)
	4. embed batch and save each sequence as single file

	:param path: path to input fasta file
	:param max_array_size: size of data proccessed on same time on GPU should be set as high as possible for faster runtime
	:return: save each seq in path as "number".pkl
	"""

	def embedd2(num_tokens, startindex, seq, header):
		print(num_tokens)
		character_ids = batch_to_ids(seq)
		# if device != "cpu":
		#     torch.cuda.empty_cache()
		character_ids.to(device)
		embedding = elmo(character_ids)
		tensors = embedding['elmo_representations']
		del character_ids, embedding
		# print(f"GPU MEMORY: {get_gpu_memory_map()}")
		embedding = [tensor.detach().cpu().numpy() for tensor in tensors]
		embedding = (np.array(embedding).mean(axis=0))

		for index, i in enumerate(embedding):
			# with open(f"{directory}/{startindex+index}.pkl", "wb") as outfile:
				# embedding_object = DL_embedding(header[index], i)
				# pickle.dump(embedding_object, outfile)
			with open(f"{directory}/{header[index][1:]}.pkl", "wb") as outfile:
				embedding_i = (header[index],i[0:len(seq[index])])
				pickle.dump(embedding_i, outfile)

	def embedd3(num_tokens, startindex, seq, header):
		# print("HELLOOOOOOO")
		print(num_tokens)
		seq_nested_list = np.array([list(i.upper()) for i in seq])
		# print(seq_nested_list)
		embedding = elmo_embedder.seqvec.embed_sentences(seq_nested_list)
		for index, i in enumerate(embedding):
			with open(f"{directory}/{header[index][1:]}.pkl", "wb") as outfile:
				embedding_i = (header[index],(np.array(i).sum(axis=0)))
				pickle.dump(embedding_i, outfile)

	elmo_embedder = Elmo_embedder()
	number_lines = file_len(path)
	end = False
	startindex_file = 0
	index_output = 0
	k = 0
	while end == False:
		headers, seqs, index, end = sort_file(path, start_index=startindex_file)
		startindex_file = index
		header = []
		seq = []
		print(f"max length: {len(seqs[-1])}")
		start = time.time()
		start_origin = start
		# num_tokens = 0
		max_length = 0
		array_size = 0
		for index, seq_i in enumerate(seqs):
			directory = os.path.dirname(path)
			# num_tokens += len(seq_i)
			if len(seq_i) > max_length:
				max_length = len(seq_i)
			k += 1
			array_size_old = array_size
			array_size = max_length * (len(seq) + 1)
			if array_size > max_array_size:
				# if num_tokens > 60000:
				# header_last = header[-1]
				print(array_size_old)
				embedd3(array_size_old, index_output, seq, header)
				# embedd2(num_tokens - len(seq_i), startindex, seq, header)
				stop = time.time()
				print(stop - start)
				print(f"{k} ca. {k*100/(number_lines/2):>.3f}%")
				start = stop
				seq = []
				header = []
				# header.append(header_last)
				# num_tokens = len(seq_i)
				index_output = k
			header.append(headers[index])
			seq.append(seq_i)
		embedd3(array_size, index_output, seq, header)
		index_output = k
		print("finish")
		totalTime = time.time() - start_origin
		print(f"total time {totalTime} s")

if __name__ == "__main__":
	from input_to_embeddings import DL_embedding
	cwd = os.getcwd()
	# model_dir = '/home/go96bix/virus_detection/seqvec/uniref50_v2/'
	model_dir = '/home/go96bix/projects/deep_eve/seqvec/uniref50_v2/'
	weights = 'weights.hdf5'
	options = 'options.json'
	#
	elmo = Elmo(model_dir + options, model_dir + weights, 3)
	device = "cpu"
	# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	if torch.cuda.device_count() > 1:
		print("Let's use", torch.cuda.device_count(), "GPUs!")
		elmo = nn.DataParallel(elmo)

	elmo.to(device)
	# elmo_embedder = Elmo_embedder(model_dir=model_dir,weights=weights,options=options)

	# parse_file("/home/go96bix/virus_detection/Viruses/Viruses.fa")
	# parse_file("/home/go96bix/virus_detection/input/Archeae_rep_seq.fa")
	# parse_file("/mnt/local/uniprot_taxonomy/Eukaryota/Eukaryota_rep_seq.fa")
	# parse_file("/home/go96bix/virus_detection/input_files/Eukaryota/Eukaryota_rep_seq.fa")
	parse_file("/home/go96bix/projects/raw_data/embeddings_bepipred_samples/iedb_linear_epitopes.fasta", max_array_size=50000)
	# parse_file("/home/go96bix/virus_detection/input_files/Bacteria/Bacteria_rep_seq.fa")
