import os
import pandas as pd
import re
import numpy as np
import sys
sys.path.insert(0, '/home/go96bix/projects/epitop_pred/')
from utils import DataGenerator
import pickle

import glob


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

# SETTINGS
slicesize = 49
shift = 24
global_embedding_bool = True
# circular filling == if windows with to short entries (frame goes over start or end of protein) fill with AAs of start or end of protein
use_circular_filling = False
big_set = True

cwd = "/home/go96bix/projects/epitop_pred"
directory = os.path.join(cwd, "data_generator_bepipred_non_binary_0.9_seqID")
if not os.path.isdir(directory):
	os.makedirs(directory)

elmo_embedder = DataGenerator.Elmo_embedder()

epitope_arr_local = []
epitope_arr_global = []

protein_arr = []

for file in glob.glob("/home/go96bix/projects/raw_data/bepipred_proteins_with_marking_0.9_seqID/*.fasta"):
	header, seq_local, values = readFasta_extended(file)
	# protein_arr.append((header,seq,values))

	seq_local = seq_local.lower()
	seq_len = len(seq_local)
	if seq_len < 25:
		continue

	values = ["-"]*shift+values+["-"]*shift
	if use_circular_filling:
		protein_pad_local = list(seq_local[-shift:]+seq_local+seq_local[0:shift])
	else:
		protein_pad_local = ["-"] * (seq_len+(shift*2))

	if global_embedding_bool:
		if big_set:
			# file_name = key.split("_")
			file_name = header[0].split("_")
			assert len(file_name)==4, f"filename of unexpected form, expected epi_1234_100_123 but got {header[0]}"
			file_name = file_name[0] + "_" + file_name[1]
			seq_global_tuple = pickle.load(open(os.path.join("/home/go96bix/projects/raw_data/embeddings_bepipred_samples",
			                                            file_name+".pkl"),"rb"))
			seq_global = seq_global_tuple[1]

		else:
			print(seq_local)
			sample_embedding = elmo_embedder.seqvec.embed_sentence(seq_local)
			sample_embedding = sample_embedding.mean(axis=0)
			seq_global = sample_embedding

		protein_pad_global = np.zeros((seq_len + (shift * 2), 1024),dtype=np.float32)
		if use_circular_filling:
			protein_pad_global[0:shift] = seq_global[-shift:]
			protein_pad_global[-shift:] = seq_global[0:shift]

	for i in range(0, seq_len, 1):
		protein_pad_local[i + (shift)] = seq_local[i]

		if global_embedding_bool:
			protein_pad_global[i + (shift)] = seq_global[i]

	epitope = "".join(protein_pad_local)
	epitope_arr_local.append([epitope,values,header,file])

	if global_embedding_bool:
		epitope_arr_global.append([protein_pad_global,values,header,file])


num_samples = []

for arr in epitope_arr_local:
	# weiss noch nciht in wie weit ich das brauche
	possible_samples = sum(np.array(arr[1]) != "-")
	num_samples.append(possible_samples)
max_samples = sum(num_samples)

print(f"all predictable positions are {max_samples}")
# min_samples = len(epitope_arr_local)

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

shuffle = np.random.permutation(range(len(epitope_arr_local)))
epitope_arr_local = np.array(epitope_arr_local)[shuffle]
if global_embedding_bool:
	epitope_arr_global = np.array(epitope_arr_global)[shuffle]

# make sure number samples per class in val and test set have straight number
min_samples = (max_samples//10)*2
samples = 0
test_roc = []
do_val = True
do_test = False
for index, arr in enumerate(epitope_arr_local):

	possible_postions = np.where(np.array(arr[1]) != "-")[0]
	selection = np.random.permutation(possible_postions)
	if do_test or do_val:
		for index_selection, i in enumerate(selection):
			if index_selection > len(possible_postions)//10:
				break
			start = i-slicesize//2
			stop = start+slicesize
			assert start >= 0, f"error calculating start position: start {start}, {header[0]}"

			# len_sample = len(arr[0])
			# max_shift = len_sample % slicesize
			if do_val:
				X_val_local.append(epitope_arr_local[index][0][start:stop])
				if global_embedding_bool:
					X_val_global.append(epitope_arr_global[index][0][start:stop])
				Y_val.append(epitope_arr_local[index][1][i])
				samples += 10

				if samples >= int(min_samples):
					do_test = True
					do_val = False
					samples = 0
					break
			elif do_test:
				if index_selection==0:
					test_roc.append(arr[3])
				X_test_local.append(epitope_arr_local[index][0][start:stop])
				# X_test.append(arr[i][j:j + slicesize])
				if global_embedding_bool:
					X_test_global.append(epitope_arr_global[index][0][start:stop])

				Y_test.append(epitope_arr_local[index][1][i])
				samples += 10
				if samples >= int(min_samples):
					do_test = False
					do_val = False
					samples = 0
					break
	Y_str = "\t".join(epitope_arr_local[index][1])
	X_train_local.append([epitope_arr_local[index][0],Y_str])
	if global_embedding_bool:
		X_train_global.append([epitope_arr_global[index][0],Y_str])

	Y_train.append(epitope_arr_local[index][1])


directory2 = directory + f"/train/all/"
if not os.path.exists(directory2):
	os.makedirs(directory2)
# test_roc = np.array(test_roc)
X_test_local = np.array(X_test_local)
X_test_global = np.array(X_test_global)
X_val_local = np.array(X_val_local)
X_val_global = np.array(X_val_global)
X_train_local = np.array(X_train_local)
X_train_global = np.array(X_train_global)

Y_test = np.array(Y_test)
Y_val = np.array(Y_val)
Y_train = np.array(Y_train)

with open(directory + '/samples_for_ROC.csv',"w")as outfile:
	for sample in test_roc:
		outfile.write(f"{sample}\n")

for index, sample in enumerate(Y_train):
	# directory2 = directory + f"/train/{index}.csv"
	f = open(os.path.join(directory2,f"{index}.csv"), 'w')
	# np.savetxt(directory2,X_train[index],delimiter="\t",fmt='%d')
	# f.write(f"{X_train_local[index][0]}\t{X_train_local[index][1]}\t{X_train_local[index][2]}\t{X_train_local[index][3]}")
	seq = '\t'.join(X_train_local[index][0])
	values = X_train_local[index][1]
	f.write(f"{seq}\n{values}")
	# f.close()

	if global_embedding_bool:
		with open(os.path.join(directory2,f"{index}.pkl"), "wb") as outfile:
			pickle.dump(X_train_global[index], outfile)


for index, i in enumerate((X_test_local,X_val_local,X_train_local)):
	len_i = i.shape[0]
	shuffle = np.random.permutation(range(len_i))
	if index == 0:
		if global_embedding_bool:
			pickle.dump(X_test_global[shuffle], open(directory + '/X_test.pkl','wb'))
		pd.DataFrame(X_test_local[shuffle]).to_csv(directory + '/X_test.csv', sep='\t', encoding='utf-8', header=None,
			                                     index=None)
		pd.DataFrame(Y_test[shuffle]).to_csv(directory + '/Y_test.csv', sep='\t', encoding='utf-8', header=None, index=None)

	elif index == 1:
		if global_embedding_bool:
			pickle.dump(X_val_global[shuffle], open(directory + '/X_val.pkl', 'wb'))
		pd.DataFrame(X_val_local[shuffle]).to_csv(directory + '/X_val.csv', sep='\t', encoding='utf-8', header=None, index=None)
		pd.DataFrame(Y_val[shuffle]).to_csv(directory + '/Y_val.csv', sep='\t', encoding='utf-8', header=None, index=None)

	elif index == 2:
		if global_embedding_bool:
			pickle.dump(X_train_global[shuffle], open(directory + '/X_train.pkl', 'wb'))
		pd.DataFrame(X_train_local[shuffle]).to_csv(directory + '/X_train.csv', sep='\t', encoding='utf-8', header=None, index=None)
		pd.DataFrame(Y_train[shuffle]).to_csv(directory + '/Y_train.csv', sep='\t', encoding='utf-8', header=None, index=None)
