import sys

sys.path.insert(0, '/home/go96bix/projects/Masterarbeit/ML')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle


def semicol_to_rows(data):
	data_total = []
	for j in data:
		data_row = np.array([])
		for i in j:
			data_cell = np.array(i.split(';'))
			data_row = np.append(data_row, data_cell)
		data_total.append(data_row)

	data_total = np.array(data_total, dtype=str)
	return data_total


def parse_amino(x):
	amino = "GALMFWKQESPVICYHRNDTUOBZX"
	encoder = LabelEncoder()
	encoder.fit(list(amino))
	out = []
	for i in x:
		dnaSeq = i.upper()
		encoded_X = encoder.transform(list(dnaSeq))
		out.append(encoded_X)
	return np.array(out)


def simple_seq(df, directory):
	"""make train test set"""
	"""sequence x-set"""
	data_x = df["seq_slice"]
	data_y = df[['episcore', '#gene_ID', 'episcore_pos']]
	data = df.values
	x = data_x.values

	y = data_y.values

	print(directory + '/X_train.csv')
	X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, shuffle=False)
	np.savetxt(directory + '/X_train.csv', X_train, delimiter='\t', fmt='%s')
	np.savetxt(directory + '/X_test.csv', X_test, delimiter='\t', fmt='%s')
	np.savetxt(directory + '/Y_train.csv', Y_train, delimiter='\t', fmt='%s')
	np.savetxt(directory + '/Y_test.csv', Y_test, delimiter='\t', fmt='%s')


def seq_balanced(df_true, df_false, directory):
	df_true = df_true.drop_duplicates("seq_slice", keep="first")
	df_false = df_false.drop_duplicates("seq_slice", keep="first")
	samples_true = df_true.shape[0]
	samples_false = df_false.shape[0]
	number_total_samples_per_class = min(samples_false, samples_true)

	data_x = pd.DataFrame()
	data_y = pd.DataFrame()

	for df in (df_false, df_true):
		df = df.sample(n=number_total_samples_per_class)
		data_x = data_x.append(df[["seq_slice"]])
		data_y = data_y.append(df[['episcore', '#gene_ID', 'episcore_pos']])

	x = data_x.values
	y = data_y.values
	print(directory + '/X_train.csv')
	X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
	np.savetxt(directory + '/X_train.csv', X_train, delimiter='\t', fmt='%s')
	np.savetxt(directory + '/X_test.csv', X_test, delimiter='\t', fmt='%s')
	np.savetxt(directory + '/Y_train.csv', Y_train, delimiter='\t', fmt='%s')
	np.savetxt(directory + '/Y_test.csv', Y_test, delimiter='\t', fmt='%s')


def complex(df_true, df_false, directory):
	# 3D Version, [Sample,Feature, Pos]
	def expand_columns_3D(df):
		samples = df.shape[0]
		out_arr = []
		for col in df.columns:
			arr = np.zeros((samples, seq_length))
			column = df[col]
			if col == "seq_slice":
				arr = column.values
				# arr = parse_amino(column.values)
				out_arr.append(arr)

			elif col in ["#homologs", "deeplocvalues"]:
				column = column.str.split(";", expand=False)
				number_values_in_Col = len(column.values[0])
				for value_index in range(number_values_in_Col):
					arr = np.zeros((samples, seq_length))
					for sample_index, values in enumerate(column.values):
						for seq_index in range(seq_length):
							arr[sample_index, seq_index] = values[value_index]
					out_arr.append(arr)

			else:
				column = column.str.split(";", expand=False)
				for sample_index, values in enumerate(column.values):
					for value_index, value in enumerate(values):
						arr[sample_index, value_index] = value
				out_arr.append(arr)
		return out_arr

	# 2D Version [Sample, Feature_pos] advantage homologs and deeplocvalues just one time inside
	def expand_columns_2D(df):

		samples = df.shape[0]
		out_arr = []
		for index, row in df.iterrows():
			arr = np.array([])
			for index_value, value in enumerate(row):
				if index_value == 0:
					arr = np.append(arr, list(value))
				else:
					value = str(value).split(";")
					arr = np.append(arr, value)
			out_arr.append(arr)

		return np.array(out_arr)

	samples = min(df_false.shape[0], df_true.shape[0])
	df_test = pd.DataFrame()
	df_val = pd.DataFrame()
	df_train = pd.DataFrame()
	for df in (df_true, df_false):
		seq_length = 50

		df_test_class = df.sample(n=int(0.2 * samples))
		df_test = df_test.append(df_test_class)
		df = df.drop(df_test_class.index)

		df_val_class = df.sample(n=int(0.2 * samples))
		df_val = df_val.append(df_val_class)
		df = df.drop(df_val_class.index)

		df_train = df_train.append(df)

	for index, df in enumerate([df_test, df_val, df_train]):
		data_y = df[['episcore', '#gene_ID', 'episcore_pos']]
		df = df.drop(['episcore', '#gene_ID', 'episcore_pos'], axis=1)
		make_3D_arr = False
		if make_3D_arr:
			out_arr = expand_columns_3D(df)
		else:
			out_arr = expand_columns_2D(df)

		if index == 0:
			Y_test = data_y.values
			np.savetxt(directory + '/Y_test.csv', Y_test, delimiter='\t', fmt='%s')

			x = np.array(out_arr)
			if make_3D_arr:
				x = np.swapaxes(x, 0, 1)
			X_test_output = open(directory + '/X_test.pkl', 'wb')
			pickle.dump(x, X_test_output)

		if index == 1:
			Y_val = data_y.values
			np.savetxt(directory + '/Y_val.csv', Y_val, delimiter='\t', fmt='%s')

			x = np.array(out_arr)
			if make_3D_arr:
				x = np.swapaxes(x, 0, 1)
			X_val_output = open(directory + '/X_val.pkl', 'wb')
			pickle.dump(x, X_val_output)

		if index == 2:
			Y_train = data_y.values
			np.savetxt(directory + '/Y_train.csv', Y_train, delimiter='\t', fmt='%s')

			x = np.array(out_arr)
			if make_3D_arr:
				x = np.swapaxes(x, 0, 1)
			X_train_output = open(directory + '/X_train.pkl', 'wb')
			pickle.dump(x, X_train_output)


path = "/home/go96bix/projects/epitop_pred/Full_data"
df_true = pd.read_csv(path + '/epitopes_edited.csv', delimiter=',', dtype='str')
df_false = pd.read_csv(path + '/non_epitopes_edited.csv', delimiter=',', dtype='str')

df_true = df_true.dropna()
df_false = df_false.dropna()
directory = "/home/go96bix/projects/epitop_pred/Full_data_len50_rawSeq"
complex(df_true, df_false, directory)
