import os
import pandas as pd
import re
import numpy as np
import sys

sys.path.insert(0, '/home/go96bix/projects/epitop_pred/')
from utils import DataGenerator
import pickle

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


# SETTINGS
slicesize = 49
shift = 22
global_embedding_bool = True
non_epi_in_protein_bool = False
# circular filling == if windows with to short entries (frame goes over start or end of protein) fill with AAs of start or end of protein
use_circular_filling = False
big_set = True

cwd = "/home/go96bix/projects/epitop_pred"
directory = os.path.join(cwd, "data_generator_bepipred_final")

elmo_embedder = DataGenerator.Elmo_embedder()

epitope_arr_local = []
epitope_arr_global = []
non_epitope_arr_local = []
non_epitope_arr_global = []
non_epi_part_in_protein_arr_local = []
non_epi_part_in_protein_arr_global = []

blast_df = pd.DataFrame.from_csv(
	"/home/go96bix/projects/epitop_pred/utils/bepipred_samples_like_filtered_blast_table.tsv",
	sep="\t", index_col=None)

epi_protein_hits_dict = {}
none_epi_protein_hits_dict = {}
foo = 0

for index, row in blast_df.iterrows():
	type_hit = row['#qseqid']
	file = type_hit + "_" + str(row['sseqid']).split("|")[1]
	start = row['sstart'] - row['qstart']
	stop = start + slicesize
	if type_hit.startswith("epi"):
		hits = epi_protein_hits_dict.get(file, [])
		if (start, stop) not in hits:
			hits.append((start, stop))
		else:
			print(f"doublicated: {index}")

		# save all epitopes
		epi_protein_hits_dict.update({file: hits})
	else:
		hits = none_epi_protein_hits_dict.get(file, [])
		if (start, stop) not in hits:
			hits.append((start, stop))
		else:
			print(f"doublicated: {index}")

		# save all epitopes
		none_epi_protein_hits_dict.update({file: hits})

for index, (key, values) in enumerate(epi_protein_hits_dict.items()):

	if big_set:
		seq_local = readFasta(os.path.join(
			"/home/go96bix/projects/raw_data/bepipred_sequences",
			key + ".fasta"))
	else:
		seq_local = readFasta(os.path.join(
			"/home/le86qiz/Documents/Konrad/deepipred_training_data/complete_protein_sequences/",
			key + ".fasta"))
	seq_local = seq_local.lower()
	seq_len = len(seq_local)
	if seq_len < 25:
		continue

	if use_circular_filling:
		protein_pad_local = list(seq_local[-shift:] + seq_local + seq_local[0:shift])
	else:
		protein_pad_local = ["-"] * (seq_len + (shift * 2))

	if global_embedding_bool:
		if big_set:
			file_name = key.split("_")
			assert len(file_name) == 4, f"filename of unexpected form, expected epi_1234_ID_123.fasta, but got {key}"
			file_name = file_name[2] + "_" + file_name[3]
			seq_global_tuple = pickle.load(
				open(os.path.join("/home/go96bix/projects/raw_data/embeddings_bepipred_samples",
				                  file_name + ".pkl"), "rb"))
			seq_global = seq_global_tuple[1]

		else:
			print(seq_local)
			sample_embedding = elmo_embedder.seqvec.embed_sentence(seq_local)
			sample_embedding = sample_embedding.mean(axis=0)
			seq_global = sample_embedding

		protein_pad_global = np.zeros((seq_len + (shift * 2), 1024), dtype=np.float32)
		if use_circular_filling:
			protein_pad_global[0:shift] = seq_global[-shift:]
			protein_pad_global[-shift:] = seq_global[0:shift]

	for i in range(0, seq_len, 1):
		protein_pad_local[i + (shift)] = seq_local[i]

		if global_embedding_bool:
			protein_pad_global[i + (shift)] = seq_global[i]

	seq_len = len(protein_pad_local)
	not_epi_mask = np.ones(seq_len, dtype=np.int)
	for val in values:
		not_epi_mask[val[0] + shift:val[1] + shift] = 0

		epitope = protein_pad_local[val[0] + shift:val[1] + shift]
		epitope = "".join(epitope)
		assert len(epitope) == slicesize, f"error {epitope} in {key} is not {slicesize} long but {len(epitope)}"
		epitope_arr_local.append([epitope, val[0], val[1], key])

		if global_embedding_bool:
			epitope = protein_pad_global[val[0] + shift:val[1] + shift]
			epitope_arr_global.append([epitope, val[0], val[1], key])

	if non_epi_in_protein_bool:
		start_bool = False
		start = 0
		stop = False
		for index, i in enumerate(not_epi_mask):
			if i == 1 and start_bool == False:
				start = index
				start_bool = True
			elif i == 0 and start_bool == True:
				stop = index
				if stop - start > slicesize:

					non_epitope = protein_pad_local[start:stop]
					non_epitope = "".join(non_epitope)
					non_epi_part_in_protein_arr_local.append([non_epitope, start - shift, stop - shift, key])

					if global_embedding_bool:
						non_epitope = protein_pad_global[start:stop]
						non_epi_part_in_protein_arr_global.append([non_epitope, start - shift, stop - shift, key])
					start_bool = False
					stop = False
			else:
				pass
		if start_bool == True:
			stop = index + 1
			if stop - start > slicesize:
				non_epitope = protein_pad_local[start:stop]
				non_epitope = "".join(non_epitope)
				non_epi_part_in_protein_arr_local.append([non_epitope, start - shift, stop - shift, key])

				if global_embedding_bool:
					non_epitope = protein_pad_global[start:stop]
					non_epi_part_in_protein_arr_global.append([non_epitope, start - shift, stop - shift, key])

# include non-epitopes presented in other papers
for key, values in none_epi_protein_hits_dict.items():
	if big_set:
		seq_local = readFasta(os.path.join(
			"/home/go96bix/projects/raw_data/bepipred_sequences",
			key + ".fasta"))
	else:
		seq_local = readFasta(os.path.join(
			"/home/le86qiz/Documents/Konrad/deepipred_training_data/complete_protein_sequences/",
			key + ".fasta"))
	seq_local = seq_local.lower()
	seq_len = len(seq_local)
	if seq_len < 25:
		continue

	if use_circular_filling:
		protein_pad_local = list(seq_local[-shift:] + seq_local + seq_local[0:shift])
	else:
		protein_pad_local = ["-"] * (seq_len + (shift * 2))

	if global_embedding_bool:
		if big_set:
			file_name = key.split("_")
			assert len(file_name) == 4, f"filename of unexpected form, expected epi_1234_ID_123.fasta, but got {key}"
			file_name = file_name[2] + "_" + file_name[3]
			seq_global_tuple = pickle.load(
				open(os.path.join("/home/go96bix/projects/raw_data/embeddings_bepipred_samples",
				                  file_name + ".pkl"), "rb"))
			seq_global = seq_global_tuple[1]


		else:
			print(seq_local)
			sample_embedding = elmo_embedder.seqvec.embed_sentence(seq_local)
			sample_embedding = sample_embedding.mean(axis=0)
			seq_global = sample_embedding

		protein_pad_global = np.zeros((seq_len + (shift * 2), 1024), dtype=np.float32)
		if use_circular_filling:
			protein_pad_global[0:shift] = seq_global[-shift:]
			protein_pad_global[-shift:] = seq_global[0:shift]

	for i in range(0, seq_len, 1):
		protein_pad_local[i + (shift)] = seq_local[i]

		if global_embedding_bool:
			protein_pad_global[i + (shift)] = seq_global[i]

	for val in values:
		non_epitope = protein_pad_local[val[0] + shift:val[1] + shift]
		non_epitope = "".join(non_epitope)
		assert len(epitope) == slicesize, f"error {epitope} in {key} is not {slicesize} long but {len(epitope)}"
		non_epitope_arr_local.append([non_epitope, val[0], val[1], key])

		if global_embedding_bool:
			non_epitope = protein_pad_global[val[0] + shift:val[1] + shift]
			non_epitope_arr_global.append([non_epitope, val[0], val[1], key])

num_samples = []
all_samples = []

if non_epi_in_protein_bool:
	non_epi_all_arr = [i for j in (non_epi_part_in_protein_arr_local, non_epitope_arr_local) for i in j]
else:
	non_epi_all_arr = non_epitope_arr_local

for arr in [non_epi_all_arr, epitope_arr_local]:
	count_non_overlapping_windows_samples = [len(i[0]) // slicesize for i in arr]
	num_samples.append(sum(count_non_overlapping_windows_samples))
min_samples = min(num_samples)

val_df = pd.DataFrame()
test_df = pd.DataFrame()
train_df = pd.DataFrame()

X_train_global = []
X_train_local = []
X_val_global = []
X_val_local = []
X_test_global = []
X_test_local = []
Y_train = []
Y_val = []
Y_test = []

if non_epi_in_protein_bool:
	local_arrays = [non_epi_part_in_protein_arr_local, non_epitope_arr_local, epitope_arr_local]
else:
	local_arrays = [non_epitope_arr_local, epitope_arr_local]
if global_embedding_bool:
	if non_epi_in_protein_bool:
		global_arrays = [non_epi_part_in_protein_arr_global, non_epitope_arr_global, epitope_arr_global]
	else:
		global_arrays = [non_epitope_arr_global, epitope_arr_global]

# make sure number samples per class in val and test set have straight number
min_samples = (min_samples // 10) * 2
for index, arr in enumerate(local_arrays):
	do_val = True
	do_test = False
	samples = 0
	selection = np.random.permutation(range(len(arr)))

	if non_epi_in_protein_bool:
		y = ["non_epitope", "non_epitope", "true_epitope"][index]
	else:
		y = ["non_epitope", "true_epitope"][index]

	for i in selection:
		len_sample = len(arr[i][0])
		max_shift = len_sample % slicesize
		start_pos = np.random.random_integers(0, max_shift)
		if do_val:
			fraction = 1
			if y == "non_epitope" and non_epi_in_protein_bool:
				fraction = 0.5

			for j in range(start_pos, len_sample - slicesize + 1, slicesize):
				X_val_local.append([local_arrays[index][i][0][j:j + slicesize], local_arrays[index][i][1] + j,
				                    local_arrays[index][i][1] + j + slicesize, local_arrays[index][i][3]])
				if global_embedding_bool:
					X_val_global.append([global_arrays[index][i][0][j:j + slicesize], global_arrays[index][i][1] + j,
					                     global_arrays[index][i][1] + j + slicesize, global_arrays[index][i][3]])
				Y_val.append(y)
				samples += 1
				if samples >= int(fraction * min_samples):
					do_test = True
					do_val = False
					samples = 0
					break
		elif do_test:

			fraction = 1
			if y == "non_epitope" and non_epi_in_protein_bool:
				fraction = 0.5
			for j in range(start_pos, len_sample - slicesize + 1, slicesize):
				X_test_local.append([local_arrays[index][i][0][j:j + slicesize], local_arrays[index][i][1] + j,
				                     local_arrays[index][i][1] + j + slicesize, local_arrays[index][i][3]])
				if global_embedding_bool:
					X_test_global.append([global_arrays[index][i][0][j:j + slicesize], global_arrays[index][i][1] + j,
					                      global_arrays[index][i][1] + j + slicesize, global_arrays[index][i][3]])

				Y_test.append(y)
				samples += 1
				if samples >= int(fraction * min_samples):
					do_test = False
					do_val = False
					samples = 0
					break
		else:
			X_train_local.append(local_arrays[index][i])
			if global_embedding_bool:
				X_train_global.append(global_arrays[index][i])

			Y_train.append(y)

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

for index, sample in enumerate(Y_train):
	directory2 = directory + f"/train/{sample}/{index}.csv"
	f = open(directory2, 'w')
	f.write(
		f"{X_train_local[index][0]}\t{X_train_local[index][1]}\t{X_train_local[index][2]}\t{X_train_local[index][3]}")

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
