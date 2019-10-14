import os
import pandas as pd
import re
import numpy as np
import sys

sys.path.insert(0, '/home/go96bix/projects/epitop_pred/')
from utils import DataGenerator
import pickle
import glob
from sklearn.metrics.pairwise import cosine_similarity

"""
embedd all proteins
gehe uber blast treffer
mache zwei dicts (epi_dict; non_epi_dict)
 jeweils als value array [tuple (start stop) von window]
gehe ueber epi_dict 
	per value slice window embedding into non_epi_area_protein and epi
gehe ueber non_epi_dict
	slice out non_epi
"""

# SETTINGS
slicesize = 49
shift = 22
global_embedding_bool = True
# non_epi_in_protein_bool = False
# circular filling == if windows with to short entries (frame goes over start or end of protein) fill with AAs of start or end of protein
use_circular_filling = False
big_set = True

cwd = "/home/go96bix/projects/epitop_pred"
# directory = os.path.join(cwd, "data_generator_bepipred_binary_allProteins")
directory = os.path.join(cwd, "data_generator_bepipred_binary_0.8_seqID_checked_output")
# input_dir = "/home/go96bix/projects/raw_data/bepipred_proteins_with_marking"
input_dir = "/home/go96bix/projects/raw_data/bepipred_proteins_with_marking_0.8_seqID"

def readFasta_extended(file):
	## read fasta file
	header = ""
	seq = ""
	values = []
	with open(file, "r") as infa:
		for index, line in enumerate(infa):
			line = line.strip()
			if index == 0:
				header = line[1:].split("\t")
			elif index == 1:
				seq += line
			elif index == 2:
				pass
			else:
				values = line.split("\t")
	return header, seq, values


def readFasta(file):
	## read fasta file
	seq = ""
	with open(file, "r") as infa:
		for line in infa:
			line = line.strip()
			if re.match(">", line):
				pass
			else:
				seq += line
	return seq


def prepare_sequences(seq_local, header):
	if use_circular_filling:
		protein_pad_local = list(seq_local[-shift:] + seq_local + seq_local[0:shift])
	else:
		protein_pad_local = ["-"] * (seq_len + (shift * 2))

	if global_embedding_bool:
		if big_set:
			file_name = header[0].split("_")
			assert len(file_name) == 4, f"filename of unexpected form, expected epi_1234_100_123 but got {header[0]}"
			file_name = file_name[0] + "_" + file_name[1]
			seq_global_tuple = pickle.load(
				open(os.path.join("/home/go96bix/projects/raw_data/embeddings_bepipred_samples",
				                  file_name + ".pkl"), "rb"))
			seq_global = seq_global_tuple[1]
			# sample_embedding = elmo_embedder.seqvec.embed_sentence(seq_local)
			# sample_embedding = sample_embedding.sum(axis=0)
			# seq_global2 = sample_embedding
		else:
			print(seq_local)
			sample_embedding = elmo_embedder.seqvec.embed_sentence(seq_local)
			sample_embedding = sample_embedding.sum(axis=0)
			seq_global = sample_embedding

		protein_pad_global = np.zeros((seq_len + (shift * 2), 1024), dtype=np.float32)
		if use_circular_filling:
			protein_pad_global[0:shift] = seq_global[-shift:]
			protein_pad_global[-shift:] = seq_global[0:shift]

	for i in range(0, seq_len, 1):
		protein_pad_local[i + (shift)] = seq_local[i]

		if global_embedding_bool:
			protein_pad_global[i + (shift)] = seq_global[i]
			# print(cosine_similarity([seq_global[i], seq_global2[i]]))

	protein_pad_local = "".join(protein_pad_local)
	# epitope_arr_local.append([epitope, values, header, file])

	# if global_embedding_bool:
	# 	epitope_arr_global.append([protein_pad_global, values, header, file])

	if global_embedding_bool:
		return protein_pad_local, protein_pad_global
	else:
		return protein_pad_local




elmo_embedder = DataGenerator.Elmo_embedder()

epitope_arr_local = []
epitope_arr_global = []
non_epitope_arr_local = []
non_epitope_arr_global = []
non_epi_part_in_protein_arr_local = []
non_epi_part_in_protein_arr_global = []

# blast_df = pd.DataFrame.from_csv(
# 	"/home/go96bix/projects/epitop_pred/utils/bepipred_samples_like_filtered_blast_table.tsv",
# 	sep="\t", index_col=None)

epi_protein_hits_dict = {}
none_epi_protein_hits_dict = {}

protein_file_list = np.array(glob.glob(f"{input_dir}/*.fasta"))
shuffle = np.random.permutation(range(len(protein_file_list)))
protein_file_list = protein_file_list[shuffle]
protein_seq_dict = {}

X_train_global = []
X_train_local = []
X_val_global = []
X_val_local = []
X_test_global = []
X_test_local = []
Y_train = []
Y_val = []
Y_test = []
test_roc = []
do_val = True
do_test = False

min_samples = (len(protein_file_list) // 10) * 2
for index, file in enumerate(protein_file_list):
	Y = []
	X_local = []
	X_global = []
	if file.endswith("protein_130.fasta"):
		print()
	headers, seq_local, values = readFasta_extended(file)
	# protein_seq_dict.update({file:seq_local})

	seq_local = seq_local.upper()
	seq_len = len(seq_local)
	if seq_len < 25:
		continue
	if global_embedding_bool:
		protein_pad_local, protein_pad_global = prepare_sequences(seq_local, headers)
	else:
		protein_pad_local = prepare_sequences(seq_local, headers)

	for head in headers:
		head_arr = head.split("_")
		name = head_arr[0] + "_" + head_arr[1]
		start = int(head_arr[2]) + shift
		stop = int(head_arr[3]) + shift
		median_pos = (start+stop-1)//2
		slice_start = median_pos - slicesize // 2
		slice_stop = slice_start + slicesize

		if head.startswith("Pos"):
			hits = epi_protein_hits_dict.get(file, [])
			if (start, stop) not in hits:
				hits.append((start, stop))
				Y.append("true_epitope")
				if global_embedding_bool:
					X_global.append([protein_pad_global[slice_start:slice_stop],start,stop,file,head])
				X_local.append([protein_pad_local[slice_start:slice_stop],start,stop,file,head])

			else:
				print(f"doublicated: {head}")

			# save all epitopes
			epi_protein_hits_dict.update({file: hits})
		else:
			hits = none_epi_protein_hits_dict.get(file, [])
			if (start, stop) not in hits:
				hits.append((start, stop))
				Y.append("non_epitope")
				if global_embedding_bool:
					X_global.append([protein_pad_global[slice_start:slice_stop],start,stop,file,head])
				X_local.append([protein_pad_local[slice_start:slice_stop],start,stop,file,head])
			else:
				print(f"doublicated: {head}")

			none_epi_protein_hits_dict.update({file: hits})

	if do_val:
		if index > min_samples:
			do_val = False
			do_test = True
		else:
			if global_embedding_bool:
				X_val_global.extend(X_global)
			X_val_local.extend(X_local)
			Y_val.extend(Y)
	if do_test:
		if index > 2*min_samples:
			do_val = False
			do_test = False
		else:
			test_roc.append(file)
			if global_embedding_bool:
				X_test_global.extend(X_global)
			X_test_local.extend(X_local)
			Y_test.extend(Y)
	if not do_test and not do_val:
		if global_embedding_bool:
			X_train_global.extend(X_global)
		X_train_local.extend(X_local)
		Y_train.extend(Y)

for file in X_train_global:
	arr = file[0]
	start = file[1]
	stop = file[2]
	path = file[3]
	header = file[4]
	headers, seq_local, values = readFasta_extended(path)
	assert header in headers, f"header {header} not in {headers}, for file {path}"
	assert f"{start-shift}_{stop-shift}" in header, f"{start}_{stop} not in {header} for file {path}"

for i in np.unique(Y_train):
	directory2 = directory + f"/train/{i}"
	if not os.path.exists(directory2):
		os.makedirs(directory2)

X_test_local = np.array(X_test_local)
X_test_global = np.array(X_test_global)
X_val_local = np.array(X_val_local)
X_val_global = np.array(X_val_global)
X_train_local = np.array(X_train_local)
X_train_global = np.array(X_train_global)

Y_test = np.array(Y_test)
Y_val = np.array(Y_val)
Y_train = np.array(Y_train)

with open(directory + '/samples_for_ROC.csv', "w")as outfile:
	for sample in test_roc:
		outfile.write(f"{sample}\n")

for index, sample in enumerate(Y_train):
	directory2 = directory + f"/train/{sample}/{index}.csv"
	f = open(directory2, 'w')
	f.write(
		f"{X_train_local[index][0]}\t{X_train_local[index][1]}\t{X_train_local[index][2]}\t{X_train_local[index][3]}\t{X_train_local[index][4]}")

	if global_embedding_bool:
		with open(directory + f"/train/{sample}/{index}.pkl", "wb") as outfile:
			pickle.dump(X_train_global[index], outfile)

for index, i in enumerate((X_test_local, X_val_local, X_train_local)):
	len_i = i.shape[0]
	shuffle = np.random.permutation(range(len_i))
	if index == 0:
		if global_embedding_bool:
			pickle.dump(X_test_global[shuffle], open(directory + '/X_test.pkl', 'wb'))
		pd.DataFrame(X_test_local[shuffle]).to_csv(directory + '/X_test.csv', sep='\t', encoding='utf-8', header=None,
		                                           index=None)
		pd.DataFrame(Y_test[shuffle]).to_csv(directory + '/Y_test.csv', sep='\t', encoding='utf-8', header=None,
		                                     index=None)

	elif index == 1:
		if global_embedding_bool:
			pickle.dump(X_val_global[shuffle], open(directory + '/X_val.pkl', 'wb'))
		pd.DataFrame(X_val_local[shuffle]).to_csv(directory + '/X_val.csv', sep='\t', encoding='utf-8', header=None,
		                                          index=None)
		pd.DataFrame(Y_val[shuffle]).to_csv(directory + '/Y_val.csv', sep='\t', encoding='utf-8', header=None,
		                                    index=None)

	elif index == 2:
		if global_embedding_bool:
			pickle.dump(X_train_global[shuffle], open(directory + '/X_train.pkl', 'wb'))
		pd.DataFrame(X_train_local[shuffle]).to_csv(directory + '/X_train.csv', sep='\t', encoding='utf-8', header=None,
		                                            index=None)
		pd.DataFrame(Y_train[shuffle]).to_csv(directory + '/Y_train.csv', sep='\t', encoding='utf-8', header=None,
		                                      index=None)
