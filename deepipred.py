from theano.scan_module.scan_utils import scan_args

print('\nLoading packages.')

# suppresses anaconda FutureWarnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pickle
# import all needed packages
import os
os.environ['KMP_WARNINGS'] = 'off'
from keras import models
from keras import layers
from keras.regularizers import l2
import argparse
import time
from sklearn.preprocessing import LabelEncoder
import sys
import numpy as np
# deeploc needs theano v1.0.4
# conda install -c conda-forge theano 
os.environ['THEANO_FLAGS']='device=cpu,floatX=float32,optimizer=fast_compile'
# from DeepLoc.models import *
# from DeepLoc.utils import *
from math import pi
from bokeh.models import ColumnDataSource, Plot, LinearAxis, Grid, Range1d, Label, BoxAnnotation
from bokeh.layouts import column
from bokeh.models.glyphs import Text
from bokeh.models import Legend
#from bokeh.io import show
from bokeh.plotting import figure, output_file, save
import tensorflow as tf
from multiprocessing import Pool
import glob
from utils import DataGenerator

#### Argument parser

parser = argparse.ArgumentParser(description='Runs the epitope predicion Pipeline', epilog='')
parser.add_argument('-e', '-epitopes', help='File containing a list of known Epitope sequences for plotting.]',metavar='<File>')
parser.add_argument('-n', '-nonepitopes', help='File containing a list of non Epitope sequences for plotting.]',metavar='<File>')
parser.add_argument('-i', '-infile', help='Multi- or Singe- Fasta file with protein sequences.',metavar='<File>')
parser.add_argument('-o','-outdir', help='Specifies output directory. Default = .',metavar='<Folder>')
parser.add_argument('-delim', help='Delimiter char for fasta header. Default = White space.',metavar='<String>')
parser.add_argument('-idpos', help='Position of gene ID in fasta header. Zero based. Default = 0.',metavar='<Integer>')
parser.add_argument('-t','-threshold', help='Threshold for epitope score. Default 0.75.', metavar='<Float>')
parser.add_argument('-p', '-processes', help='Number of processes used for predictions. Default 1.', metavar='<Int>')
args = parser.parse_args()
########################


######## Flo stuff ######
class Protein_seq():
	def __init__(self, sequence, score, over_threshold, positions=None):
		self.sequence = sequence
		self.score = score
		self.over_threshold = over_threshold
		if positions == None:
			self.positions = list(range(1,len(self.sequence)+1))
		else:
			self.positions = positions


def build_model_old(nodes, seq_length, dropout=0):
	model = models.Sequential()
	model.add(layers.Embedding(21, 10, input_length=seq_length))
	model.add(layers.Bidirectional(layers.LSTM(nodes, return_sequences=True, dropout=dropout, recurrent_dropout=0.2)))
	model.add(layers.Bidirectional(layers.LSTM(nodes, dropout=dropout, recurrent_dropout=0.2)))
	model.add(layers.Dense(nodes))
	model.add(layers.LeakyReLU(alpha=0.01))
	model.add(layers.Dense(2, activation='softmax'))

	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
	model.summary()
	return model

def build_model(nodes, dropout, seq_length, weight_decay_lstm= 0, weight_decay_dense=0):
	""" model with elmo embeddings for amino acids"""
	inputs = layers.Input(shape=(seq_length, 1024))
	hidden = layers.Bidirectional(layers.LSTM(nodes, input_shape=(seq_length,1024), return_sequences=True, dropout=dropout, recurrent_dropout=0.2, kernel_regularizer=l2(weight_decay_lstm), recurrent_regularizer=l2(weight_decay_lstm), bias_regularizer=l2(weight_decay_lstm)))(inputs)
	hidden = layers.Bidirectional(layers.LSTM(nodes, dropout=dropout, recurrent_dropout=0.2, kernel_regularizer=l2(weight_decay_lstm), recurrent_regularizer=l2(weight_decay_lstm), bias_regularizer=l2(weight_decay_lstm)))(hidden)
	hidden = layers.Dense(nodes, kernel_regularizer=l2(weight_decay_dense), bias_regularizer=l2(weight_decay_dense))(hidden)
	hidden = layers.LeakyReLU(alpha=0.01)(hidden)

	out = layers.Dense(2, activation='softmax', kernel_regularizer=l2(weight_decay_dense), bias_regularizer=l2(weight_decay_dense))(hidden)
	model= models.Model(inputs=inputs,outputs=out)

	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
	model.summary()
	return model

def parse_amino(x):
	"""
	Takes amino acid sequence and parses it to a numerical sequence.
	"""
	amino = "GALMFWKQESPVICYHRNDTU"
	encoder = LabelEncoder()
	encoder.fit(list(amino))
	out = []
	for i in x:
		dnaSeq = i[1].upper()
		encoded_X = encoder.transform(list(dnaSeq))
		out.append(encoded_X)
	return np.array(out)


def split_AA_seq(seq, slicesize, shift):
	"""
	Takes input sequence and slicesize: Returns slices of that sequence with a slice length of 'slicesize' with a sliding window of 1.
	"""
	splited_AA_seqs = []
	for i in range(0, len(seq) - slicesize):
		splited_AA_seqs.append([i + (slicesize // 2) - shift, seq[i:i + slicesize]])
	return np.array(splited_AA_seqs)

def split_embedding_seq(embeddings, slicesize, shift):
	assert len(embeddings) == 1, "splitting of embeddings not intended for multiple proteins (state of affairs 12.06.19)"
	splited_em_seqs = []
	for protein in embeddings:
		splited_em_seq = []
		for i in range(0, len(protein) - slicesize):
			splited_em_seq.append([i + (slicesize // 2) - shift, protein[i:i + slicesize]])
		splited_em_seqs.append(splited_em_seq)
	return np.array(splited_em_seqs[0])

# filters tensor flow output (the higher the number the more ist filtered)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # {0, 1, 2 (warnings), 3 (errors)}

starttime = time.time()

if not args.i:
	print('Error: No input file given.')
	exit()

multifasta = args.i
idpos = 0
outdir = ''

if args.idpos:
	idpos = int(args.idpos)

if args.o:
	outdir = args.o
	if not os.path.isabs(outdir):
		outdir = f'{os.getcwd()}/{args.o}'
	if not os.path.exists(outdir):
		os.makedirs(outdir)
else:
	outdir = os.getcwd()

if not os.path.exists(outdir + '/results'):
	os.makedirs(outdir + '/results')
outdir = outdir + '/results'
print(f'\nOut dir set to : {outdir}')

processes = 1
if args.p:
	processes = int(args.p)
	print(f'Processes set to: {processes}.')

epitope_threshold = 0.75

if args.t:
	epitope_threshold = float(args.t)
print(f'The epitope threshold is set to: {epitope_threshold}\n')
# slicesize is the amount of AA which are used as input to predict liklelyhood of epitope
slicesize = 49

##### reading input fasta file #####

fasta = {}
fastaheader = {}
print('Reading input fasta.')
with open(multifasta,'r') as infile:
	acNumber = ''
	for line in infile:
		if line.startswith('>'):
			if args.delim:
				acNumber = line.split(args.delim)[idpos].strip().strip('>')
				fastaheader[acNumber] = line.strip()
			else:
				acNumber = line.split()[idpos].strip().strip('>')
				fastaheader[acNumber] = line.strip()
		else:
			if acNumber in fasta:
				fasta[acNumber] += line.strip()
			else:
				fasta[acNumber] = line.strip()


'''
##############################################
###### Deeploc localisation prediction #######
##############################################

##### progress vars ####
filecounter = 1
total = str(len(fasta))
########################

# function to call and save deeploc results
def run_deeploc(geneid, sequence, outdir, filecounter):
	print(f'Started Deeploc for {geneid}    File: {filecounter} / {total}')
	out_ids, out_loc, out_mem = prediction([geneid], [sequence], 1)
	deeploc_out = np.insert(out_loc[0], 0, out_mem[0]).tolist()
	outfile = f'{outdir}/deeploc/{geneid}.csv'
	with open(outfile, 'w') as outfile:
		outfile.write('#Membrane	Nucleus	Cytoplasm	Extracellular	Mitochondrion	Cell_membrane	Endoplasmic_reticulum	Plastid	Golgi_apparatus	Lysosome/Vacuole	Peroxisome\n')
		outfile.write('\t'.join([str(x) for x in deeploc_out]))
	print(f'Finished Deeploc for {geneid}    File: {filecounter} / {total}')

if not os.path.exists(outdir + '/deeploc'):
	os.makedirs(outdir + '/deeploc')
# run deeploc in multiple processes
pool = Pool(processes=processes)
print('\nRunning Deeploc protein localisation prediction.')
deeploc_dict = {}
for geneid in fasta:
	pool.apply_async(run_deeploc, args=(geneid, fasta[geneid], outdir, filecounter))
	filecounter += 1
pool.close()
pool.join()
print()
# read deeploc result files
print('Reading Deeploc results.')
deeploc_dict = {}
deeplocfiles = glob.glob(f'{outdir}/deeploc/*')
for file in deeplocfiles:
	geneid = file.rsplit('/',1)[1][:-4]
	with open(file, 'r') as infile:
		for line in infile:
			if not line.startswith('#'):
				deeploc_dict[geneid] = [float(x) for x in line.strip().split('\t')]

##############################################
'''

'''
##### remove sequences that are too short #####
removed = []
for geneid in fasta:
	if len(fasta[geneid]) <= slicesize:
		removed.append(geneid)
if len(removed):
	print(f'{len(removed)} genes have been removed for beeing shorter than {slicesize} amino acids.')
	open(outdir + '/removed_genes.csv', 'w').close()
	with open(outdir + '/removed_genes.csv', 'w') as outfile:
		for geneid in removed:
			del fasta[geneid]
			outfile.write(f'{geneid}\n')
	print()
'''
##### reading provided epitope lists #######
epitopes = list()
if args.e:
	print('Reading provided epitope sequences.')
	with open(args.e, 'r') as infile:
		for line in infile:
			epitopes.append(line.strip())
	print('There were ' + str(len(epitopes)) + ' epitope sequences provided.')

nonepitopes = list()
if args.n:
	print('Reading provided non-epitope sequences.')
	with open(args.n, 'r') as infile:
		for line in infile:
			nonepitopes.append(line.strip())
	print('There were ' + str(len(nonepitopes)) + ' non-epitope sequences provided.')
print()

# TODO
def ensemble_prediction(model, path, inputs_test, suffix, middle_name = "", prediction_weights = False, nb_classes = 2):
	models_filenames = []
	for file in sorted(os.listdir(path)):
		if file.endswith(f"_{suffix}.hdf5") and file.startswith(f"weights_model_{middle_name}k-fold_run_"):
			# print(file)
			models_filenames.append(os.path.join(path, file))

	preds = []
	for fn in models_filenames:
		model.load_weights(fn, by_name=True)
		pred = model.predict(inputs_test)
		preds.append(pred)

	if not prediction_weights:
		prediction_weights = [1. / len(models_filenames)] * len(models_filenames)
	weighted_predictions = np.zeros((inputs_test.shape[0], nb_classes), dtype='float32')
	for weight, prediction in zip(prediction_weights, preds):
		weighted_predictions += weight * np.array(prediction)

	return weighted_predictions
"""
OLD
# location of weights from the previously trained model
#model_path = "/home/go96bix/projects/epitop_pred/epitope_data/weights.best.loss.test_generator.hdf5"
model_path = "/home/go96bix/projects/epitop_pred/epitope_data/weights.best.loss.only_seq_data_no_weight_decay.hdf5"
# load weights, after this step the model behaves as we trained it
model.load_weights(model_path)
"""
#model_path = "/home/go96bix/projects/epitop_pred/data_generator_final/both_non_epitopes/global_embedding/weights.best.auc10.2nd_try_final.hdf5"	# 25 nodes
# model_path = "/home/go96bix/projects/epitop_pred/data_generator_final/validated_non_epitopes/global_embedding/weights.best.loss.test_final_local_validated_non_epi.hdf5"	# 100 nodes
# model_path = "/home/go96bix/projects/epitop_pred/data_generator_final/both_non_epitopes/global_embedding/weights.best.loss.2nd_try_final.hdf5"	# 25 nodes
# model_path = "/home/go96bix/projects/epitop_pred/data_generator_bepipred_final/local_embedding/weights.best.auc10.250_nodes_with_decay_local.hdf5"	# 250 nodes
# model_path = "/home/go96bix/projects/epitop_pred/data_generator_bepipred/local_embedding/weights.best.loss.250nodes_weight_decay.hdf5"	# 250 nodes
model_path_local = "/home/go96bix/projects/epitop_pred/data_generator_bepipred_final/local_embedding/weights.best.auc10.250_nodes_with_decay_local.hdf5"	# 250 nodes
model_path_global = "/home/go96bix/projects/epitop_pred/data_generator_bepipred_final/global_embedding/weights.best.auc10.250_nodes_with_decay.hdf5"	# 250 nodes
shift = 22
local_embedding = False
use_circular_filling = False

print('Deep Neural Network model summary:')
nodes = 250
elmo_embedder = DataGenerator.Elmo_embedder()
model_local = build_model(nodes, dropout=0, seq_length=slicesize)
model_global = build_model(nodes, dropout=0, seq_length=slicesize)
model_local.load_weights(model_path_local)
model_global.load_weights(model_path_global)

holydict = {}


##############################################
######### DeEpiPred score prediction #########
##############################################

##### progress vars ####
filecounter = 1
printlen = 1
total = str(len(fasta))
########################

print('\nPredicting DeEpiPred scores.')
# go over all entries in dict
for geneid in fasta:
	score_both_models = []
	for embedding_version in ["local","global"]:
		if embedding_version == "local":
			local_embedding = True
		else:
			local_embedding = False
		############### progress ###############
		elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.time()-starttime))
		printstring = f'Predicting scores for: {geneid}    File: {filecounter} / {total}   Elapsed time: {elapsed_time}'
		if len(printstring) < printlen:
			print(' '*printlen, end='\r')
		print(printstring, end='\r')
		printlen = len(printstring)
		filecounter += 1
		#######################################
		""" OLD version
		# slice the long AA in short segments so it can be used as input for the neural network
		seq_slices = split_AA_seq(fasta[geneid], slicesize)
		# parse input to numerical values
	
		X_test = parse_amino(seq_slices)
		# finally predict the epitopes
		Y_pred_test = model.predict(X_test)	
		"""
		# embedding per slice version

		if local_embedding:
			seq = fasta[geneid].lower()

			if use_circular_filling:
				seq_extended = list(seq[-shift:] + seq + seq[0:shift])
			else:
				seq_extended = np.array(["-"] * (len(seq) + (shift * 2)))
				seq_extended[shift:-shift] = np.array(list(seq))

			seq_slices = split_AA_seq(seq_extended, slicesize,shift)
			positions = seq_slices[:,0]
			seq_slices_input = np.array([list(i) for i in seq_slices[:,1]])

			# start = time.time()
			X_test = elmo_embedder.elmo_embedding(seq_slices_input, 0, slicesize)
			# stop = time.time()
			# print(stop-start)

		#embedding whole protein version
		else:
			# seq_slices = split_AA_seq(fasta[geneid], slicesize)
			seq = np.array([list(fasta[geneid].lower())])
			name = geneid.split("_")[2:]
			name = "_".join(name)
			X_test = pickle.load(open(f"/home/go96bix/projects/raw_data/embeddings_bepipred_samples/{name}.pkl","rb"))[1]
			# X_test = elmo_embedder.elmo_embedding(seq, 0, len(seq[0]))
			seq_extended = np.zeros((1,seq.shape[1] + (shift * 2), 1024), dtype=np.float32)
			if use_circular_filling:
				seq_extended[0,0:shift] = X_test[-shift:]
				seq_extended[0,-shift:] = X_test[0:shift]
			seq_extended[0,shift:-shift] = X_test
			seq_slices = split_embedding_seq(seq_extended,slicesize,shift)
			positions = seq_slices[:, 0]
			X_test = np.stack(seq_slices[:,1])

		# # finally predict the epitopes
		# path_weights = "/home/go96bix/projects/epitop_pred/epitope_data"
		# suffix_weights = "big_embeddings"
		# Y_pred_test = ensemble_prediction(model,path=path_weights,inputs_test=X_test,suffix=suffix_weights)
		if local_embedding:
			Y_pred_test = model_local.predict(X_test)
		else:
			Y_pred_test = model_global.predict(X_test)
		# the column 0 in Y_pred_test is the likelihood that the slice is NOT a Epitope, for us mostly interesting
		# is col 1 which contain the likelihood of being a epitope
		epi_score = Y_pred_test[:, 1]

		# use leading and ending zeros so that the score array has the same length as the input sequence
		score = np.zeros(len(fasta[geneid]))
		# leading AAs which are not predictable get value of first predicted value (were this AA where involved)
		score[0:int(positions[0])] = epi_score[0]
		# last AAs which are not predictable get value of last predicted value (were this AA where involved)
		score[int(positions[-1]):]=epi_score[-1]
		score[np.array(positions,dtype=int)] = epi_score
		#print(score)
		score_both_models.append(score)

	score_both_models = np.array(score_both_models)
	score = score_both_models.mean(axis=0)
	score_bool = score > epitope_threshold

	protein = Protein_seq(sequence=fasta[geneid], score=score, over_threshold=score_bool)
	holydict.update({geneid:protein})

# function for frame average extends frame by frame_extend to left AND right, thus frame_extend 3 results in a total window of 7
def frame_avg(values, frame_extend = 2):
	averages = []
	protlen = len(values)
	for pos in range(protlen):
		framelist = []
		for shift in range(-frame_extend,frame_extend+1,1):
			if not (pos+shift) < 0 and not (pos+shift) > (protlen -1):
				framelist.append(float(values[pos+shift]))
		averages.append(sum(framelist)/len(framelist))
	return averages

# calculate parker hydrophilicity scores
def parker_avg(seq):
	parker_scores = {'A':2.1,'C':1.4,'D':10.0,'E':7.8,'F':-9.2,'G':5.7,'H':2.1,'I':-8.0,'K':5.7,'L':-9.2,'M':-4.2,'N':7.0,'P':2.1,'Q':6.0,'R':4.2,'S':6.5,'T':5.2,'V':-3.7,'W':-10.0,'Y':-1.9}
	score = 0
	for AA in seq:
		score += parker_scores.get(AA, 0)
	score = (score / len(seq))/20 + 0.5		# deviding by 20 and adding 0.5 to normalize between 0 and 1
	return score


'''
########### calculate amino acid k-mer scores ############
# all lines added therefore are marked with		# aa k-mer score
aa_scores = {}
with open('/home/le86qiz/Documents/Konrad/general_epitope_analyses/aminoacid_epitope_scores.csv') as infile:
	for line in infile:
		aa = line.split('\t')[0]
		score = float(line.strip().split('\t')[1])
		aa_scores[aa] = score
'''

# aa_scoredict = {}
# non_polar_min = {} # GVCLIMWF
# hydrophilic_max = {} # STDNQRH
# verzweigt_min = {} # VLIE
# strong_pos = {} # PDR
# strong_neg = {} # VLIF
# max_min_diff_score = {}
hyrophilicity_parker = {}
for geneid in holydict:
	seq = holydict[geneid].sequence
	# scores = []
	# for aa in seq:
	# 	scores.append(aa_scores.get(aa,0))
	# aa_scoredict[geneid] = scores
	# frame_extend = 10
	# normalizer = frame_extend * 2 + 1
	protlen = len(seq)
	# non_polar_min_scores = []
	# hydrophilic_max_scores = []
	# verzweigt_min_scores = []
	# strong_pos_scores = []
	# strong_neg_scores = []
	hydrophilicity_parker_scores = []
	for pos in range(protlen):
		# framelist = []
		# non_polar_min_score = hydrophilic_max_score = verzweigt_min_score = strong_pos_score = strong_neg_score = 0
		# for shift in range(-frame_extend,frame_extend+1,1):
		# 	if not (pos+shift) < 0 and not (pos+shift) > (protlen - 1):
		# 		framelist.append(seq[pos+shift])
		# for aa in framelist:
		# 	if aa in 'GVCLIMWF':
		# 		non_polar_min_score += 1
		# 	if aa in 'STDNQRH':
		# 		hydrophilic_max_score += 1
		# 	if aa in 'VLIE':
		# 		verzweigt_min_score += 1
		# 	if aa in 'PDR':
		# 		strong_pos_score += 1
		# 	if aa in 'VLIF':
		# 		strong_neg_score += 1
		# non_polar_min_score = non_polar_min_score / normalizer
		# hydrophilic_max_score = hydrophilic_max_score / normalizer
		# verzweigt_min_score = verzweigt_min_score / normalizer
		# strong_pos_score = strong_pos_score / normalizer
		# strong_neg_score = strong_neg_score / normalizer
		# non_polar_min_scores.append(non_polar_min_score)
		# hydrophilic_max_scores.append(hydrophilic_max_score)
		# verzweigt_min_scores.append(verzweigt_min_score)
		# strong_pos_scores.append(strong_pos_score)
		# strong_neg_scores.append(strong_neg_score)
		# hydrophilicity by parker
		framelist_parker = []
		for shift in range(-3,3+1,1):
			if not (pos+shift) < 0 and not (pos+shift) > (protlen - 1):
				framelist_parker.append(seq[pos+shift])
		hydrophilicity_parker_scores.append(parker_avg(framelist_parker))
	hyrophilicity_parker[geneid] = hydrophilicity_parker_scores
	# non_polar_min[geneid] = non_polar_min_scores
	# hydrophilic_max[geneid] = hydrophilic_max_scores
	# verzweigt_min[geneid] = verzweigt_min_scores
	# strong_pos[geneid] = strong_pos_scores
	# strong_neg[geneid] = strong_neg_scores
	# max_min_diff_score[geneid] = (np.array(hydrophilic_max_scores) - np.array(non_polar_min_scores)) + 0.5


########### calculate amino acid k-mer scores end ############


###### calculate position weighted frame average scores ######

def pwa(scores = [], frame_extend = 5):
	seqlen = len(scores)
	# calculate positon weight matrix
	weight = frame_extend + 1
	weights = []
	for i in range(weight):
		weights.append(i)
	weights = weights[frame_extend:0:-1] + weights
	weights = [(weight -x)/(weight) for x in weights]
	pwm = []
	pwm_adapted =[]
	for i in range(seqlen):
		out = []
		for j in range(len(weights)):
			if i + j - frame_extend >= 0 and j - frame_extend + i < seqlen:
				out.append(weights[j])
		if len(out) < seqlen:
			if i + frame_extend< len(out):
				out = out + [0] * (seqlen - len(out))
			elif frame_extend + i  + 1 >= seqlen:
				out = [0] * (seqlen - len(out)) + out
			else:
				out = [0] * (i - len(out) + frame_extend + 1) + out + [0] * (seqlen - i - frame_extend -1)
		pwm.append(out)
	pwm_adapted = pwm.copy()
	# multiply scores to pwm
	for i in range(len(scores)):
		pwm_adapted[i] = [scores[i] * pwm_score for pwm_score in pwm[i]]
	# sum up and normalize per position
	pwm_scores = np.array(pwm_adapted).sum(axis=0) / np.array(pwm).sum(axis=0)
	return(pwm_scores)
'''
pwm_scoredict = {}
for geneid in holydict:
	scores = holydict[geneid].score
		# sum up and normalize per position
	pwm_scoredict[geneid] = pwa(scores, frame_extend = 24)
'''



##############################################
############### Output results ###############
##############################################

if not os.path.exists(outdir + '/deepipred'):
	os.makedirs(outdir + '/deepipred')



######## epitope table #########
predicted_epitopes = {}
epitope_slicelen = 15
slice_shiftsize = 5

print(f'\n\nWriting predicted epitopes to:\n{outdir}/predicted_epitopes.csv\n{outdir}/predicted_epitopes_sliced.faa')
open(f'{outdir}/predicted_epitopes.csv', 'w').close()
open(f'{outdir}/predicted_epitopes_sliced.faa', 'w').close()
open(f'{outdir}/deepipred_scores.csv', 'w').close()
open(f'{outdir}/hydrophilicity_parker_scores.csv', 'w').close()
# open(f'{outdir}/max_min_diff_scores.csv', 'w').close()
with open(f'{outdir}/predicted_epitopes.csv','w') as outfile:
	with open(f'{outdir}/predicted_epitopes_sliced.faa','w') as outfile2:
		with open(f'{outdir}/deepipred_scores.csv', 'w') as outfile3:
			with open(f'{outdir}/hydrophilicity_parker_scores.csv', 'w') as outfile4:
				with open(f'{outdir}/max_min_diff_scores.csv', 'w') as outfile5:
					outfile.write('#Gene_ID\tstart\tend\tsequence\tscore')
					for geneid in holydict:
						#scores = pwa(holydict[geneid].score, 24)
						scores = holydict[geneid].score
						scores_hydrophilicity_parker = frame_avg(hyrophilicity_parker[geneid],frame_extend = 10)
						# scores_max_min_diff = frame_avg(max_min_diff_score[geneid], frame_extend = 10)
						seq = holydict[geneid].sequence
						predicted_epis = set()
						predicted_epitopes[geneid] = []
						newepi = True
						start = 0
						end = 0
						i = 0
						out = f'{outdir}/deepipred/{geneid}.csv'
						with open(out,'w') as outfile6:
							# write complete scores to file
							outfile6.write('#Aminoacid\tDeepipred\n')
							outfile3.write(f'>{geneid}\n')
							outfile4.write(f'>{geneid}\n')
							# outfile5.write(f'>{geneid}\n')
							for x in range(len(seq)):
								outfile3.write(f'{seq[x]}\t{scores[x]}\n')
								outfile4.write(f'{seq[x]}\t{scores_hydrophilicity_parker[x]}\n')
								# outfile5.write(f'{seq[x]}\t{scores_max_min_diff[x]}\n')
								outfile6.write(f'{seq[x]}\t{scores[x]}\n')
							for score in scores:
								if score >= epitope_threshold:
									if newepi:
										start = i
										newepi = False
									else:
										end = i
								else:
									newepi = True
									if end - start >= 8:
										#predicted_epis.add((start + 1, end + 1, seq[start:end+1], np.median(scores[start:end+1]) * deeploc_score))
										predicted_epis.add((start + 1, end + 1, seq[start:end+1], np.median(scores[start:end+1])))
								i += 1
							if end - start >= 8:
								#predicted_epis.add((start + 1, end + 1, seq[start:end+1], np.median(scores[start:end+1]) * deeploc_score))
								predicted_epis.add((start + 1, end + 1, seq[start:end+1], np.median(scores[start:end+1])))
							predicted_epis = sorted(predicted_epis)
							epiout = ''
							for epi in predicted_epis:
								epiout = f'{epiout}\n{geneid}\t{epi[0]}\t{epi[1]}\t{epi[2]}\t{epi[3]}'
								predicted_epitopes[geneid].append(epi[2])
								# print slices to blast table
				### sliced epitope regions
								if len(epi[2]) > epitope_slicelen:
									for i in range(0,len(epi[2]) - (epitope_slicelen -1),slice_shiftsize):
										outfile2.write(f'>{geneid}|pos_{i+epi[0]}:{i+epi[0]+epitope_slicelen}\n{epi[2][i:i+epitope_slicelen]}\n')
				### complete epitope regions
				#				outfile2.write(f'>{geneid}|pos_{epi[0]}:{epi[1]}|score_{epi[3]}\n{epi[2]}\n')
				### complete sequence
				#				outfile2.write(f'>{geneid}|pos_{epi[0]}:{epi[1]}|score_{epi[3]}\n{seq}\n')
							outfile.write(f'{epiout}')



######## Plots #########
print('\nPlotting.')

##### progress vars ####
filecounter = 1
printlen = 1
total = str(len(fasta))
########################

for geneid in holydict:

	############### progress ###############
	elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.time()-starttime))
	printstring = f'Plotting: {geneid}    File: {filecounter} / {total}   Elapsed time: {elapsed_time}'
	if len(printstring) < printlen:
		print(' '*printlen, end='\r')
	print(printstring, end='\r')
	printlen = len(printstring)
	filecounter += 1
	#######################################


	# make output dir and create output filename
	if not os.path.exists(outdir + '/plots'):
		os.makedirs(outdir + '/plots')
	out = f'{outdir}/plots/{geneid}.html'
	output_file(out)

	seq = holydict[geneid].sequence
	pos = holydict[geneid].positions
	score = holydict[geneid].score
	flag = holydict[geneid].over_threshold
	pwa_score = pwa(score, frame_extend = 24)
	protlen = len(seq)
	hyrophilicity_parker_score = frame_avg(hyrophilicity_parker[geneid], frame_extend = 10)


	# create a new plot with a title and axis labels
	p = figure(title=fastaheader[geneid][1:], y_range = (-0.03,1.03), y_axis_label='Scores',plot_width=1200,plot_height=460,tools='xpan,xwheel_zoom,reset', toolbar_location='above')
	p.min_border_left = 80

	# add a line renderer with legend and line thickness
	l1 = p.line(range(1,protlen+1), score, line_width=1,color='black', visible = True)
	l2 = p.line(range(1,protlen+1), ([epitope_threshold] * protlen), line_width=1,color='red', visible = True)
#	l10 = p.line(range(1,protlen+1), pwa_score, line_width=1,color='darkgreen', visible = False)
#	l12 = p.line(range(1,protlen+1), hyrophilicity_parker_score, line_width=1,color='black', visible = False)

	#legend = Legend(items=[('DeEpiPred',[l1]), ('epitope_threshold',[l2]) ] )
	legend = Legend(items=[('DeEpiPred',[l1]),
	('epitope_threshold',[l2]) ])
#	('pwa_score',[l10]),
#	('hydrophilicity by parker', [l12]) ] )	# aa k-mer score and pwm

	p.add_layout(legend,'right')
#	p.legend.orientation = 'vertical'
#	p.legend.location = 'right'
	p.xaxis.visible = False
	p.legend.click_policy="hide"

	p.x_range.bounds = (-50, protlen+51)

	### plot for sequence
	# symbol based plot stuff

	plot = Plot(title=None, x_range=p.x_range, y_range=Range1d(0,9), plot_width=1200, plot_height=50, min_border=0, toolbar_location=None)

	y = [1]*protlen
	source = ColumnDataSource(dict(x=list(pos), y=y, text=list(seq)))
	glyph = Text(x="x", y="y", text="text", text_color='black', text_font_size='8pt')
	plot.add_glyph(source, glyph)
	label = Label(x=-80,y=y[1],x_units='screen',y_units='data',text = 'Sequence', render_mode='css', background_fill_color='white',background_fill_alpha=1.0)
	plot.add_layout(label)

	xaxis = LinearAxis()
	plot.add_layout(xaxis, 'below')
	plot.add_layout(Grid(dimension=0, ticker=xaxis.ticker))


	# add predicted epitope boxes
	if predicted_epitopes[geneid]:
		for epi in predicted_epitopes[geneid]:
			if seq.find(epi) > -1:
				start = seq.find(epi) + 1
				end = start + len(epi) + 1
				non_epitope = [-0.02] * (start - 1) + [1.02] * len(epi) + [-0.02] * ((protlen - (start-1) - len(epi)))
				p.vbar(x = list(pos), bottom = -0.02, top = non_epitope, width = 1, alpha = 0.2, line_alpha = 0, color = 'darkgreen', legend = 'predicted_epitopes', visible = True)

	# add known epitope boxes
	if epitopes:
		for epi in epitopes:
			if seq.find(epi) > -1:
				start = seq.find(epi) + 1
				end = start + len(epi) + 1
				epitope = [-0.02] * (start - 1) + [1.02] * len(epi) + [-0.02] * ((protlen - (start-1) - len(epi)))
				p.vbar(x = list(pos), bottom = -0.02, top = epitope, width = 1, alpha = 0.2, line_alpha = 0, color = 'blue', legend = 'provided_epitope', visible = False)
#				output_file(f'{outdir}/plots/{geneid}_epi.html') # adds _epi suffix to outfile if a supplied epitope was provided

	# add non-epitope boxes
	if nonepitopes:
		for epi in nonepitopes:
			if seq.find(epi) > -1:
				start = seq.find(epi) + 1
				end = start + len(epi) + 1
				non_epitope = [-0.02] * (start - 1) + [1.02] * len(epi) + [-0.02] * ((protlen - (start-1) - len(epi)))
				p.vbar(x = list(pos), bottom = -0.02, top = non_epitope, width = 1, alpha = 0.2, line_alpha = 0, color = 'darkred', legend = 'provided_non_epitope', visible = False)

	save(column(p,plot))
'''
	# DeepLoc barplot
	deeploclocations = ['Membrane','Nucleus','Cytoplasm','Extracellular','Mitochondrion','Cell_membrane','Endoplasmic_reticulum','Plastid','Golgi_apparatus','Lysosome/Vacuole','Peroxisome']
	deepplot = figure(x_range=deeploclocations, plot_height=350, title="DeepLoc", toolbar_location=None, tools="")
	deepplot.vbar(x = deeploclocations, top=deeploc_dict[geneid], width = 0.8)
	deepplot.xgrid.grid_line_color = None
	deepplot.xaxis.major_label_orientation = pi/2
	deepplot.y_range.start = 0

	save(column(p,plot,deepplot))
'''

