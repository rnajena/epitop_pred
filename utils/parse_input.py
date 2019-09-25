import os
from builtins import enumerate

import pandas as pd
import numpy as np
from utils import DataGenerator
import pickle

cwd = "/home/go96bix/projects/epitop_pred"
directory = os.path.join(cwd, "data_generator")
classes = 0
num_samples = []
all_samples = []
classic = False
embedding = False
elmo_embedder = DataGenerator.Elmo_embedder()
slicesize = 49
use_old_test_set = True

if use_old_test_set:
	test_df_old = pd.DataFrame.from_csv(
		"/home/le86qiz/Documents/Konrad/general_epitope_analyses/bepipred_evaluation/deepipred_results/test_samples.csv",
		sep=",", header=None, index_col=None)
	test_df_old_y = test_df_old[2].values
	test_df_old = test_df_old[1].values
else:
	test_df_old = []

for root, dirs, files in os.walk(directory):
	for file in files:
		if file.endswith(".csv"):
			classes += 1
			df_input = pd.DataFrame.from_csv(os.path.join(directory, file), header=None, index_col=False)
			df_input["y"] = file[:-4]
			df_input = df_input.drop_duplicates(keep='first')
			mask_not_old_test_set = np.array([seq not in test_df_old for seq in df_input[0]])
			df_input = df_input[mask_not_old_test_set]
			count_non_overlapping_windows_samples = [len(i) // slicesize for i in df_input[0].values]
			num_samples.append(sum(count_non_overlapping_windows_samples))
			all_samples.append(df_input)

min_samples = min(num_samples)
val_df = pd.DataFrame()
test_df = pd.DataFrame()
train_df = pd.DataFrame()

if use_old_test_set:
	min_samples_test_set = len(test_df_old) / 2
else:
	min_samples_test_set = min_samples

if classic:
	for index, df_class in enumerate(all_samples):
		# validation set
		val_df_class = df_class.sample(n=int(0.2 * min_samples))
		val_df = val_df.append(val_df_class)
		df_help = df_class.drop(val_df_class.index)

		test_df_class = df_help.sample(n=int(0.2 * min_samples))
		test_df = test_df.append(test_df_class)
		df_help = df_help.drop(test_df_class.index)
		# train set
		train_df = train_df.append(df_help.sample(frac=1))

	for i in train_df['y'].unique():
		directory2 = directory + f"/train/{i}"
		if not os.path.exists(directory2):
			os.makedirs(directory2)

	for index, sample in train_df.iterrows():
		directory2 = directory + f"/train/{sample['y']}/{index}.csv"
		f = open(directory2, 'w')
		f.write(f"{sample.name}\t{sample[0]}")
		f.close()

	# shuffle
	test_df = test_df.sample(frac=1)
	X_test = test_df[0]
	Y_test = test_df["y"]
	X_test.to_csv(directory + '/X_test.csv', sep='\t', encoding='utf-8')
	Y_test.to_csv(directory + '/Y_test.csv', sep='\t', encoding='utf-8')

	train_df = train_df.sample(frac=1)
	X_train = train_df[0]
	Y_train = train_df["y"]
	X_train.to_csv(directory + '/X_train.csv', sep='\t', encoding='utf-8')
	Y_train.to_csv(directory + '/Y_train.csv', sep='\t', encoding='utf-8')

	val_df = val_df.sample(frac=1)
	X_val = val_df[0]
	Y_val = val_df["y"]
	X_val.to_csv(directory + '/X_val.csv', sep='\t', encoding='utf-8')
	Y_val.to_csv(directory + '/Y_val.csv', sep='\t', encoding='utf-8')

else:
	X_train = []
	X_val = []
	X_test = []
	Y_train = []
	Y_val = []
	Y_test = []

	for index, df_class in enumerate(all_samples):
		do_val = True
		do_test = False
		samples = 0
		selection = np.random.permutation(range(len(df_class.index)))
		np_class_0 = df_class[0].values
		np_class_y = df_class['y'].values
		for i in selection:
			len_sample = len(np_class_0[i])
			if embedding:
				sample_embedding = elmo_embedder.seqvec.embed_sentence(np_class_0[i])
				sample_embedding = sample_embedding.mean(axis=0)

			max_shift = len_sample % slicesize
			# start from random position and generate non overlapping windows
			start_pos = np.random.random_integers(0, max_shift)
			if do_val:
				for j in range(start_pos, len_sample - slicesize + 1, slicesize):
					if embedding:
						X_val.append(sample_embedding[j:j + slicesize])
					else:
						X_val.append(np_class_0[i][j:j + slicesize])
					Y_val.append(np_class_y[i])
					samples += 1

					if samples >= int(0.2 * min_samples):
						do_test = True
						do_val = False
						samples = 0
						break
			elif do_test:
				for j in range(start_pos, len_sample - slicesize, slicesize):
					if embedding:
						X_test.append(sample_embedding[j:j + slicesize])
					else:
						if use_old_test_set:
							if len(X_test) < min_samples_test_set:
								for index, j in enumerate(test_df_old):
									X_test.append(j[0:slicesize])
									class_i = test_df_old_y[index]
									if class_i == 'true_epitopes':
										class_i = 'true_epitope'
									else:
										class_i = 'non_epitope'
									Y_test.append(class_i)
							do_test = False
							do_val = False
							samples = 0
							break
						else:
							X_test.append(np_class_0[i][j:j + slicesize])
							Y_test.append(np_class_y[i])
							samples += 1

					if not use_old_test_set:
						if samples >= int(min_samples_test_set):
							do_test = False
							do_val = False
							samples = 0
							break
			else:
				if embedding:
					X_train.append(sample_embedding)
				else:
					X_train.append(np_class_0[i])
				Y_train.append(np_class_y[i])

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

	if embedding:
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
				pd.DataFrame(Y_train[shuffle]).to_csv(directory + '/Y_train.csv', sep='\t', encoding='utf-8',
				                                      header=None, index=None)

	else:
		for index, sample in enumerate(Y_train):
			directory2 = directory + f"/train/{sample}/{index}.csv"
			f = open(directory2, 'w')
			f.write(f"{index}\t{X_train[index]}")
			f.close()

		for index, i in enumerate((X_test, X_val, X_train)):
			len_i = i.shape[0]
			shuffle = np.random.permutation(range(len_i))
			if index == 0:
				pd.DataFrame(X_test[shuffle]).to_csv(directory + '/X_test.csv', sep='\t', encoding='utf-8', header=None,
				                                     index=None)
				pd.DataFrame(Y_test[shuffle]).to_csv(directory + '/Y_test.csv', sep='\t', encoding='utf-8', header=None,
				                                     index=None)

			if index == 1:
				pd.DataFrame(X_val[shuffle]).to_csv(directory + '/X_val.csv', sep='\t', encoding='utf-8', header=None,
				                                    index=None)
				pd.DataFrame(Y_val[shuffle]).to_csv(directory + '/Y_val.csv', sep='\t', encoding='utf-8', header=None,
				                                    index=None)

			if index == 2:
				pd.DataFrame(X_train[shuffle]).to_csv(directory + '/X_train.csv', sep='\t', encoding='utf-8',
				                                      header=None, index=None)
				pd.DataFrame(Y_train[shuffle]).to_csv(directory + '/Y_train.csv', sep='\t', encoding='utf-8',
				                                      header=None, index=None)
