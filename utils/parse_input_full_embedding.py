import os
import pandas as pd
import re
import numpy as np

from utils import DataGenerator
import pickle

"""
embedd all proteins
gehe uber balst treffer
mache dict mit array [tuple (start stop) von window]
gehe ueber dict 
	pro value slice window embedding
	
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


cwd = "/home/go96bix/projects/epitop_pred"
directory = os.path.join(cwd, "data_generator")
elmo_embedder = DataGenerator.Elmo_embedder()

"""
what to put in test set
"""

in_test_set = []
test_df = pd.DataFrame.from_csv(
	"/home/le86qiz/Documents/Konrad/general_epitope_analyses/bepipred_evaluation/deepipred_results/test_samples.csv",
	sep=",", header=None, index_col=None)
test_df = test_df[test_df[2] == 'true_epitopes']
for index, row in test_df.iterrows():
	seq_id = int(row[0]) + 1
	name = f"seq_{seq_id}"
	in_test_set.append(name)

epitope_arr = []
epitope_arr_test = []
non_epitope_arr = []

slicesize = 49

blast_df = pd.DataFrame.from_csv(
	"/home/le86qiz/Documents/Konrad/prediction_pipeline/raptorx_pipeline/epitopes/epitope_results/filtered_blast_results.csv",
	sep="\t")

# generate an dict which holds for each protein the start and stop position of each epitope window
# generate the same dict only containing the test set protein epitopes
protein_hits_dict = {}
protein_hits_dict_test_set = {}
foo = 0
for index, row in blast_df.iterrows():
	file = str(row['sseqid']).split("|")[1]
	start = row['sstart'] - row['qstart']
	stop = start + slicesize
	hits = protein_hits_dict.get(file, [])
	if (start, stop) not in hits:
		hits.append((start, stop))
	else:
		print(f"doublicated: {index}")

	# save only epitopes in test set
	if index in in_test_set:
		hits_test = protein_hits_dict_test_set.get(file, [])
		hits_test.append((start, stop))
		protein_hits_dict_test_set.update({file: hits_test})
	# save all epitopes
	protein_hits_dict.update({file: hits})

# embedd each protein
# extend proteins with 20 zero columns at each site
# make a mask were the protein is NOT an epitope
# save epitopes and not epitopes in different arrays
for key, values in protein_hits_dict.items():
	seq = readFasta(os.path.join(
		"/home/le86qiz/Documents/Konrad/prediction_pipeline/raptorx_pipeline/epitopes/epitope_results/complete_epitope_protein_sequences",
		key + ".txt"))
	seq_len = len(seq)
	shift = 20
	protein_pad = np.zeros((seq_len + (shift * 2), 1024))

	sample_embedding = elmo_embedder.seqvec.embed_sentence(seq)
	sample_embedding = sample_embedding.mean(axis=0)
	seq = sample_embedding

	for i in range(0, seq_len, 1):
		protein_pad[i + (shift)] = seq[i]

	seq_len = len(protein_pad)
	not_epi_mask = np.ones(seq_len)
	# print(values)
	for val in values:
		not_epi_mask[val[0] + shift:val[1] + shift] = 0

		epitope = protein_pad[val[0] + shift:val[1] + shift]

		if val in protein_hits_dict_test_set.get(key, []):
			epitope_arr_test.append(epitope)
		else:
			epitope_arr.append(epitope)

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
				non_epitope = protein_pad[start:stop]
				non_epitope_arr.append(non_epitope)
				start_bool = False
				stop = False
		else:
			pass
	if start_bool == True:
		stop = index + 1
		if stop - start > slicesize:
			non_epitope = protein_pad[start:stop]
			# non_epitope = "".join(non_epitope)
			non_epitope_arr.append(non_epitope)

num_samples = []
all_samples = []
for arr in [non_epitope_arr, epitope_arr]:
	count_non_overlapping_windows_samples = [len(i) // slicesize for i in arr]
	num_samples.append(sum(count_non_overlapping_windows_samples))

# wenn gleiche test set wie frueher gewollt
min_samples = 5 * len(epitope_arr_test)
# klassisch
# min_samples = min(num_samples)
val_df = pd.DataFrame()
test_df = pd.DataFrame()
train_df = pd.DataFrame()

X_train = []
X_val = []
X_test = []
Y_train = []
Y_val = []
Y_test = []

# generate different sets
for index, arr in enumerate([non_epitope_arr, epitope_arr]):
	do_val = True
	do_test = False
	samples = 0
	selection = np.random.permutation(range(len(arr)))

	y = ["non_epitope", "true_epitope"][index]

	for i in selection:
		len_sample = len(arr[i])
		max_shift = len_sample % slicesize
		start_pos = np.random.random_integers(0, max_shift)
		if do_val:
			for j in range(start_pos, len_sample - slicesize + 1, slicesize):

				X_val.append(arr[i][j:j + slicesize])
				Y_val.append(y)
				samples += 1
				if samples >= int(0.2 * min_samples):
					do_test = True
					do_val = False
					samples = 0
					break
		elif do_test:
			if y == "true_epitope":
				for j in epitope_arr_test:
					X_test.append(j)
					Y_test.append(y)
				do_test = False
				do_val = False
				samples = 0
			else:
				for j in range(start_pos, len_sample - slicesize, slicesize):
					X_test.append(arr[i][j:j + slicesize])
					Y_test.append(y)
					samples += 1
					if samples >= len(epitope_arr_test):
						do_test = False
						do_val = False
						samples = 0
						break
		else:
			X_train.append(arr[i])
			Y_train.append(y)

for i in np.unique(Y_train):
	directory2 = directory + f"/train/{i}"
	if not os.path.exists(directory2):
		os.makedirs(directory2)

X_test = np.array(X_test)
X_val = np.array(X_val)
X_train = np.array(X_train)

Y_test = np.array(Y_test)
Y_val = np.array(Y_val)
Y_train = np.array(Y_train)

for index, sample in enumerate(Y_train):
	with open(directory + f"/train/{sample}/{index}.pkl", "wb") as outfile:
		pickle.dump(X_train[index], outfile)

for index, i in enumerate((X_test, X_val, X_train)):
	len_i = i.shape[0]
	shuffle = np.random.permutation(range(len_i))
	if index == 0:
		pickle.dump(X_test[shuffle], open(directory + '/X_test.pkl', 'wb'))
		pd.DataFrame(Y_test[shuffle]).to_csv(directory + '/Y_test.csv', sep='\t', encoding='utf-8', header=None,
		                                     index=None)
	elif index == 1:
		pickle.dump(X_val[shuffle], open(directory + '/X_val.pkl', 'wb'))
		pd.DataFrame(Y_val[shuffle]).to_csv(directory + '/Y_val.csv', sep='\t', encoding='utf-8', header=None,
		                                    index=None)
	elif index == 2:
		pickle.dump(X_train[shuffle], open(directory + '/X_train.pkl', 'wb'))
		pd.DataFrame(Y_train[shuffle]).to_csv(directory + '/Y_train.csv', sep='\t', encoding='utf-8', header=None,
		                                      index=None)
