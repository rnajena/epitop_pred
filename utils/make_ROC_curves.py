#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 15:35:52 2019

@author: le86qiz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
import os
from scipy import interp
import pickle

deepipred_results_dir = f'/home/go96bix/projects/raw_data/binary_25_nodes_100_epochs_08DO_0.5_seqID_new/results/'

# read test file table
# testfiletable = '/home/go96bix/projects/epitop_pred/with_errors/data_generator_bepipred_binary_0.5_seqID/samples_for_ROC.csv'
# testfiletable = '/home/go96bix/projects/raw_data/allprotein.csv'
testfiletable = '/home/go96bix/projects/raw_data/08_allprotein.csv'
# testfiletable = '/home/go96bix/projects/raw_data/05_allprotein.csv'
# kfoldtable_dir = "/home/go96bix/projects/epitop_pred/data_generator_bepipred_binary_allProteins"
kfoldtable_dir = "/home/go96bix/projects/epitop_pred/data_generator_bepipred_binary_double_cluster_0.8_0.5_seqID"
# kfoldtable_dir = "/home/go96bix/projects/epitop_pred/data_generator_bepipred_binary_0.5_seqID"
# testfiletable = '/home/le86qiz/Documents/Konrad/tool_comparison/comparison3/samples_for_ROC.csv'

thresh = 1

def lbtope_results():
	lbtope_results_dict = dict()
	with open("/home/go96bix/projects/paper/EpiDope/raw_data/all_seq_used_for_training/training.fasta", "r") as input:
		header = [i.strip() for i in input.readlines() if i.startswith(">")]
	# with open("/home/go96bix/projects/epitop_pred/lbtope_results.txt","r") as input:
	with open("/home/go96bix/projects/epitop_pred/lbtope_results_bepipred_dataset.txt","r") as input:
		header_index = 0
		scores = []
		for line in input:
			if line.startswith("Epitopes"):
				if header_index > 0:
					lbtope_results_dict.update({testid:scores})
				head = header[header_index]
				header_index += 1
				head = head[1:].split("\t")
				testid = head[0]
				scores = []

			else:
				scores.append(float(line.strip().split("\t")[-1]))

		lbtope_results_dict.update({testid: scores})

		return lbtope_results_dict

def _binary_roc_auc_score(fpr, tpr, max_fpr=1):
	if max_fpr is None or max_fpr == 1:
		return metrics.auc(fpr, tpr)
	if max_fpr <= 0 or max_fpr > 1:
		raise ValueError("Expected max_frp in range ]0, 1], got: %r"
		                 % max_fpr)

	# Add a single point at max_fpr by linear interpolation
	stop = np.searchsorted(fpr, max_fpr, 'right')
	x_interp = [fpr[stop - 1], fpr[stop]]
	y_interp = [tpr[stop - 1], tpr[stop]]
	tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
	fpr = np.append(fpr[:stop], max_fpr)
	partial_auc = metrics.auc(fpr, tpr)

	# McClish correction: standardize result to be 0.5 if non-discriminant
	# and 1 if maximal
	min_area = 0.5 * max_fpr ** 2
	max_area = max_fpr
	return 0.5 * (1 + (partial_auc - min_area) / (max_area - min_area))

def get_testproteinIDS(testfiletable):
	testproteinIDs = []
	with open(testfiletable) as infile:
		for line in infile:
			# file = line.strip().rsplit('/', 1)[1]
			# testproteinIDs.append(file[:-6])

			if line.startswith("/"):
				file = line.strip().rsplit('/', 1)[-1]
				testproteinIDs.append(file[:-6])
			elif line.startswith("Cluster"):
				files = cluster_dict[line.strip()]
				for file in files:
					file = file.strip().rsplit('/', 1)[-1]
					testproteinIDs.append(file[:-6])
	return testproteinIDs

def cluster_to_dict(file="/home/go96bix/projects/raw_data/clustered_protein_seqs/my_double_cluster0.8_05/0.5_seqID.fasta.clstr",
                    directory_fasta="/home/go96bix/projects/raw_data/bepipred_proteins_with_marking"):
	out_dict = {}
	with open(file, "r") as infile:
		allLines = infile.read()
		clusters = allLines.split(">Cluster")
		for cluster in clusters:
			if len(cluster) > 0:
				proteins = cluster.strip().split("\n")
				files = []
				for index, protein in enumerate(proteins):
					if index == 0:
						cluster_name = "Cluster_" + protein
					else:
						filename = protein.split(" ")[1][1:-3] + ".fasta"
						protein_file = os.path.join(directory_fasta, filename)
						files.append(protein_file)
				out_dict.update({cluster_name:files})
	return out_dict

cluster_dict = cluster_to_dict()
testproteinIDs = get_testproteinIDS(testfiletable)

# get start/stop postions of epitopes/nonepitopes
startstop_epi = {}
startstop_nonepi = {}
counter = 0
counter_pos = 0
counter_neg = 0
length = np.array([])
length_pos = np.array([])
length_neg = np.array([])

lbtope_train = set()
with open("/home/go96bix/projects/epitop_pred/LBtope_Variable_Negative_epitopes.txt", "r") as input:
	lbtope_train.update(set([i.strip() for i in input.readlines()]))
with open("/home/go96bix/projects/epitop_pred/LBtope_Variable_Positive_epitopes.txt", "r") as input:
	lbtope_train.update(set([i.strip() for i in input.readlines()]))

for testid in testproteinIDs:
	file = f'/home/le86qiz/Documents/Konrad/tool_comparison/comparison3/bepipred_proteins_with_marking/{testid}.fasta'
	with open(file) as infile:
		for index, line in enumerate(infile):
			if index == 0:
				head = line[1:].strip().split()

			elif index ==1:
				for epiID in head:
					epiID = epiID.split('_')
					flag = epiID[0]
					start = int(epiID[2])
					stop = int(epiID[3])
					# if stop - start <= 11:
					# 	continue
					if line[start:stop] in lbtope_train:
						continue
					counter += 1
					length = np.append(length, stop-start)
					if flag == 'PositiveID':
						counter_pos += 1
						length_pos = np.append(length_pos, stop - start)

						if testid in startstop_epi:
							startstop_epi[testid].append([start, stop])
						else:
							startstop_epi[testid] = [[start, stop]]
					else:
						counter_neg += 1
						length_neg = np.append(length_neg, stop - start)

						if testid in startstop_nonepi:
							startstop_nonepi[testid].append([start, stop])
						else:
							startstop_nonepi[testid] = [[start, stop]]

			else:
				break

print("all",counter)
print("positive",counter_pos)
print("negative",counter_neg)
print("mean length", np.mean(length))
print("median length", np.median(length))
print("mean length positive", np.mean(length_pos))
print("median length positive", np.median(length_pos))
print("mean length negative", np.mean(length_neg))
print("median length negative", np.median(length_neg))
# exit()
# read bepipred
bepipred_scores = []
bepipred_flag = []
for testid in testproteinIDs:
	bepipred_file = f'/home/le86qiz/Documents/Konrad/tool_comparison/comparison3/bepipred/results/{testid}.csv'
	bepipred_table = pd.read_csv(bepipred_file, sep="\t", index_col=None, skiprows=1).values
	bepipred_table = bepipred_table[:, 7]
	for startstop in startstop_epi.get(testid, []):
		start = startstop[0]
		stop = startstop[1]
		scores = bepipred_table[start:stop]
		score = sum(scores) / len(scores)
		bepipred_scores.append(score)
		# bepipred_scores.extend(scores)
		bepipred_flag.append(1)
		# bepipred_flag.extend([1]*len(scores))
	for startstop in startstop_nonepi.get(testid, []):
		start = startstop[0]
		stop = startstop[1]
		scores = bepipred_table[start:stop]
		score = sum(scores) / len(scores)
		bepipred_scores.append(score)
		# bepipred_scores.extend(scores)
		bepipred_flag.append(0)
		# bepipred_flag.extend([0] * len(scores))

bepipred_scores = np.array(bepipred_scores)
bepipred_flag = np.array(bepipred_flag)

# read antigenicity
antigenicity_scores = []
antigenicity_flag = []
for testid in testproteinIDs:
	antigenicity_file = f'/home/le86qiz/Documents/Konrad/tool_comparison/comparison3/aa_scores/antigenicity/{testid}.csv'
	antigenicity_table = pd.read_csv(antigenicity_file, sep="\t", index_col=None).values
	antigenicity_table = antigenicity_table[:, 1]
	antigenicity_table = antigenicity_table - np.mean(antigenicity_table)
	for startstop in startstop_epi.get(testid, []):
		start = startstop[0]
		stop = startstop[1]
		scores = antigenicity_table[start:stop]
		score = sum(scores) / len(scores)
		antigenicity_scores.append(score)
		antigenicity_flag.append(1)
	for startstop in startstop_nonepi.get(testid, []):
		start = startstop[0]
		stop = startstop[1]
		scores = antigenicity_table[start:stop]
		score = sum(scores) / len(scores)
		antigenicity_scores.append(score)
		antigenicity_flag.append(0)
antigenicity_scores = np.array(antigenicity_scores)
antigenicity_flag = np.array(antigenicity_flag)

# read betaturn
betaturn_scores = []
betaturn_flag = []
for testid in testproteinIDs:
	betaturn_file = f'/home/le86qiz/Documents/Konrad/tool_comparison/comparison3/aa_scores/betaturn/{testid}.csv'
	betaturn_table = pd.read_csv(betaturn_file, sep="\t", index_col=None).values
	betaturn_table = betaturn_table[:, 1]
	for startstop in startstop_epi.get(testid, []):
		start = startstop[0]
		stop = startstop[1]
		scores = betaturn_table[start:stop]
		score = sum(scores) / len(scores)
		betaturn_scores.append(score)
		betaturn_flag.append(1)
	for startstop in startstop_nonepi.get(testid, []):
		start = startstop[0]
		stop = startstop[1]
		scores = betaturn_table[start:stop]
		score = sum(scores) / len(scores)
		betaturn_scores.append(score)
		betaturn_flag.append(0)
betaturn_scores = np.array(betaturn_scores)
betaturn_flag = np.array(betaturn_flag)

# read hydrophilicity
hydrophilicity_scores = []
hydrophilicity_flag = []
for testid in testproteinIDs:
	hydrophilicity_file = f'/home/le86qiz/Documents/Konrad/tool_comparison/comparison3/aa_scores/hydrophilicity/{testid}.csv'
	hydrophilicity_table = pd.read_csv(hydrophilicity_file, sep="\t", index_col=None).values
	hydrophilicity_table = hydrophilicity_table[:, 1]
	for startstop in startstop_epi.get(testid, []):
		start = startstop[0]
		stop = startstop[1]
		scores = hydrophilicity_table[start:stop]
		score = sum(scores) / len(scores)
		hydrophilicity_scores.append(score)
		hydrophilicity_flag.append(1)
	for startstop in startstop_nonepi.get(testid, []):
		start = startstop[0]
		stop = startstop[1]
		scores = hydrophilicity_table[start:stop]
		score = sum(scores) / len(scores)
		hydrophilicity_scores.append(score)
		hydrophilicity_flag.append(0)
hydrophilicity_scores = np.array(hydrophilicity_scores)
hydrophilicity_flag = np.array(hydrophilicity_flag)

# read accessibility
accessibility_scores = []
accessibility_flag = []
for testid in testproteinIDs:
	accessibility_file = f'/home/le86qiz/Documents/Konrad/tool_comparison/comparison3/aa_scores/accessibility/{testid}.csv'
	accessibility_table = pd.read_csv(accessibility_file, sep="\t", index_col=None).values
	accessibility_table = accessibility_table[:, 1]
	for startstop in startstop_epi.get(testid, []):
		start = startstop[0]
		stop = startstop[1]
		scores = accessibility_table[start:stop]
		score = sum(scores) / len(scores)
		accessibility_scores.append(score)
		accessibility_flag.append(1)
	for startstop in startstop_nonepi.get(testid, []):
		start = startstop[0]
		stop = startstop[1]
		scores = accessibility_table[start:stop]
		score = sum(scores) / len(scores)
		accessibility_scores.append(score)
		accessibility_flag.append(0)
accessibility_scores = np.array(accessibility_scores)
accessibility_flag = np.array(accessibility_flag)

# read lbtope
lbtope_results_dict = lbtope_results()
lbtope_scores = []
lbtope_flag = []

# for file in sorted(os.listdir(kfoldtable_dir)):
# 	if file.endswith(f"_test_set.csv") and file.startswith(f"k-fold_run_"):
# 		testproteinIDs_kfold = get_testproteinIDS(f"{kfoldtable_dir}/{file}")
for testid in testproteinIDs:
	lbtope_table = lbtope_results_dict[testid]
	for startstop in startstop_epi.get(testid, []):
		start = startstop[0]
		stop = startstop[1]
		scores = lbtope_table[start:stop]
		score = sum(scores) / len(scores)
		lbtope_scores.append(score)
		lbtope_flag.append(1)
	for startstop in startstop_nonepi.get(testid, []):
		start = startstop[0]
		stop = startstop[1]
		scores = lbtope_table[start:stop]
		score = sum(scores) / len(scores)
		lbtope_scores.append(score)
		lbtope_flag.append(0)

lbtope_scores = np.array(lbtope_scores)
lbtope_flag = np.array(lbtope_flag)

# read deepipred
deepipred_scores = []
deepipred_flag = []
# for testid in testproteinIDs:
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

tprs_prot = []
aucs_prot = []
mean_fpr_prot = np.linspace(0, 1, 100)

for file in sorted(os.listdir(kfoldtable_dir)):
	deepipred_scores_kfold = []
	deepipred_flag_kfold = []
	if file.endswith(f"_test_set.csv") and file.startswith(f"k-fold_run_"):
		testproteinIDs_kfold = get_testproteinIDS(f"{kfoldtable_dir}/{file}")
		for testid in testproteinIDs_kfold:
			epidope_scores_protein = []
			epidope_flag_protein = []
			deepipred_file = f'{os.path.join(deepipred_results_dir,"epidope/")}{testid}.csv'
			deepipred_table = pd.read_csv(deepipred_file, sep="\t", index_col=None).values
			deepipred_table = deepipred_table[:, 1]
			# help_array = 1 - deepipred_table
			# weights = np.std(np.array([help_array,deepipred_table], dtype=np.float), axis=0)
			# deepipred_table = deepipred_table*weights

			for startstop in startstop_epi.get(testid, []):
				start = startstop[0]
				stop = startstop[1]
				# score = deepipred_table[(start+stop)//2]
				# print(score)
				scores = deepipred_table[start:stop]
				score = sum(scores) / len(scores)

				deepipred_scores_kfold.append(score)
				epidope_scores_protein.append(score)

				deepipred_flag_kfold.append(1)
				epidope_flag_protein.append(1)
			for startstop in startstop_nonepi.get(testid, []):
				start = startstop[0]
				stop = startstop[1]
				scores = deepipred_table[start:stop]
				score = sum(scores) / len(scores)

				deepipred_scores_kfold.append(score)
				epidope_scores_protein.append(score)

				deepipred_flag_kfold.append(0)
				epidope_flag_protein.append(0)

			if 0 in epidope_flag_protein and 1 in epidope_flag_protein:
				fpr, tpr, thresholds = metrics.roc_curve(epidope_flag_protein, epidope_scores_protein)
				tprs_prot.append(interp(mean_fpr, fpr, tpr))
				tprs_prot[-1][0] = 0.04
				roc_auc_prot = metrics.auc(fpr, tpr)
				aucs_prot.append(roc_auc_prot)

		deepipred_scores_kfold = np.array(deepipred_scores_kfold)
		deepipred_scores = np.append(deepipred_scores,deepipred_scores_kfold)
		deepipred_flag_kfold = np.array(deepipred_flag_kfold)
		deepipred_flag = np.append(deepipred_flag,deepipred_flag_kfold)

		fpr, tpr, thresholds = metrics.roc_curve(deepipred_flag_kfold, deepipred_scores_kfold)
		tprs.append(interp(mean_fpr, fpr, tpr))
		tprs[-1][0] = 0.0
		# roc_auc = _binary_roc_auc_score(fpr, tpr, thresh)
		roc_auc = metrics.auc(fpr[fpr<thresh], tpr[fpr<thresh])
		aucs.append(roc_auc)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
# mean_auc = _binary_roc_auc_score(mean_fpr, mean_tpr, thresh)
mean_auc = metrics.auc(mean_fpr[mean_fpr<thresh], mean_tpr[mean_fpr<thresh])
std_auc = np.std(aucs)
std_tpr = np.std(tprs, axis=0)

mean_tpr_prot = np.mean(tprs_prot, axis=0)
mean_tpr_prot[-1] = 1.0
mean_auc_prot = metrics.auc(mean_fpr_prot, mean_tpr_prot)
std_auc_prot = np.std(aucs_prot)
std_tpr_prot = np.std(tprs_prot, axis=0)

tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
# print(any(deepipred_flag == bepipred_flag))

# read raptorx
raptorx_scores = []
raptorx_flag = []
iupred_scores = []
iupred_flag = []
for testid in testproteinIDs:
	raptorx_file = f'/home/le86qiz/Documents/Konrad/tool_comparison/comparison3/raptorx/results/flo_files/{testid}.csv'
	try:
		raptorx_table = pd.read_csv(raptorx_file, sep="\t", index_col=None).values
	except:
		continue
	iupred_table = raptorx_table[:, 6]
	structure_table = raptorx_table[:, 2] - raptorx_table[:, 1] - raptorx_table[:, 0]  # coil - helix - sheet
	accessibility_table = raptorx_table[:, 5] - raptorx_table[:, 3]  # exposed - bury
	raptorx_table = structure_table + accessibility_table
	for startstop in startstop_epi.get(testid, []):
		start = startstop[0]
		stop = startstop[1]
		scores = raptorx_table[start:stop]
		score = sum(scores) / len(scores)
		raptorx_scores.append(score)
		raptorx_flag.append(1)
		scores = iupred_table[start:stop]
		score = sum(scores) / len(scores)
		iupred_scores.append(score)
		iupred_flag.append(1)
	for startstop in startstop_nonepi.get(testid, []):
		start = startstop[0]
		stop = startstop[1]
		scores = raptorx_table[start:stop]
		score = sum(scores) / len(scores)
		raptorx_scores.append(score)
		raptorx_flag.append(0)
		scores = iupred_table[start:stop]
		score = sum(scores) / len(scores)
		iupred_scores.append(score)
		iupred_flag.append(0)
raptorx_scores = np.array(raptorx_scores)
raptorx_flag = np.array(raptorx_flag)
iupred_scores = np.array(iupred_scores)
iupred_flag = np.array(iupred_flag)

# calculate roc curve
from sklearn.metrics import roc_curve, auc


fpr = {}
tpr = {}
roc_auc = {}
thresholds = {}

key = 'bepipred'
fpr[key], tpr[key], thresholds[key] = metrics.roc_curve(bepipred_flag, bepipred_scores, pos_label=1)
# roc_auc[key] = metrics.roc_auc_score(bepipred_flag, bepipred_scores, max_fpr=thresh)
roc_auc[key] = metrics.auc(fpr[key][fpr[key]<=thresh], tpr[key][fpr[key]<=thresh])
key = 'antigenicity'
fpr[key], tpr[key], thresholds[key] = metrics.roc_curve(antigenicity_flag, antigenicity_scores, pos_label=1)
# roc_auc[key] = metrics.roc_auc_score(antigenicity_flag, antigenicity_scores, max_fpr=thresh)
roc_auc[key] = metrics.auc(fpr[key][fpr[key]<=thresh], tpr[key][fpr[key]<=thresh])
key = 'hydrophilicity'
fpr[key], tpr[key], thresholds[key] = metrics.roc_curve(hydrophilicity_flag, hydrophilicity_scores, pos_label=1)
# roc_auc[key] = metrics.roc_auc_score(hydrophilicity_flag, hydrophilicity_scores, max_fpr=thresh)
roc_auc[key] = metrics.auc(fpr[key][fpr[key]<=thresh], tpr[key][fpr[key]<=thresh])
key = 'accessibility'
fpr[key], tpr[key], thresholds[key] = metrics.roc_curve(accessibility_flag, accessibility_scores, pos_label=1)
# roc_auc[key] = metrics.roc_auc_score(accessibility_flag, accessibility_scores, max_fpr=thresh)
roc_auc[key] = metrics.auc(fpr[key][fpr[key]<=thresh], tpr[key][fpr[key]<=thresh])
key = 'betaturn'
fpr[key], tpr[key], thresholds[key] = metrics.roc_curve(betaturn_flag, betaturn_scores, pos_label=1)
# roc_auc[key] = metrics.roc_auc_score(betaturn_flag, betaturn_scores, max_fpr=thresh)
roc_auc[key] = metrics.auc(fpr[key][fpr[key]<=thresh], tpr[key][fpr[key]<=thresh])
key = 'deepipred'
fpr[key], tpr[key], thresholds[key] = metrics.roc_curve(deepipred_flag, deepipred_scores, pos_label=1)
# roc_auc[key] = metrics.roc_auc_score(deepipred_flag, deepipred_scores, max_fpr=thresh)
roc_auc[key] = metrics.auc(fpr[key][fpr[key]<=thresh], tpr[key][fpr[key]<=thresh])
key = 'lbtope'
fpr[key], tpr[key], thresholds[key] = metrics.roc_curve(lbtope_flag, lbtope_scores, pos_label=1)
# roc_auc[key] = metrics.roc_auc_score(lbtope_flag, lbtope_scores, max_fpr=thresh)
roc_auc[key] = metrics.auc(fpr[key][fpr[key]<=thresh], tpr[key][fpr[key]<=thresh])
key = 'iupred'
fpr[key], tpr[key], thresholds[key] = metrics.roc_curve(iupred_flag, iupred_scores, pos_label=1)
# roc_auc[key] = metrics.roc_auc_score(iupred_flag, iupred_scores, max_fpr=thresh)
roc_auc[key] = metrics.auc(fpr[key][fpr[key]<=thresh], tpr[key][fpr[key]<=thresh])

# plot
plt.figure(figsize=(6, 6))
lw = 2
for i in fpr:
	fpr[i] = [x for x in fpr[i] if x <= thresh]
	tpr[i] = tpr[i][:len(fpr[i])]

	# youden j score
	interpolated_tpr = np.interp([0.1],fpr[i], tpr[i])[0]
	print(i, interpolated_tpr - 0.1)
maxtpr = 0
for x in tpr:
	maxtpr = max(maxtpr, max(tpr[x]))

COLORS = pickle.load(open('/home/mu42cuq/scripts/mypymo/colordictionary.pydict', 'rb'))

# show std-div
std_roc = plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

epidope_roc, = plt.plot(mean_fpr, mean_tpr, color='green', lw=2, label='Mean ROC EpiDope \n(AUC = %0.4f $\pm$ %0.4f)' % (mean_auc, std_auc))
# plt.plot(mean_fpr_prot, mean_tpr_prot, color='red', lw=2, label=r'Mean ROC EpiDope per protein (AUC = %0.4f $\pm$ %0.4f)' % (mean_auc_prot, std_auc_prot))
# plt.plot(fpr['deepipred'], tpr['deepipred'], color='green', lw=lw,
#          label='EpiDope (area = %0.4f)' % roc_auc['deepipred'])
lbtope_roc, = plt.plot(fpr['lbtope'], tpr['lbtope'], color='orange', linestyle='--', lw=lw,
         label='LBtope (AUC = %0.4f)' % roc_auc['lbtope'])
iupred_roc, = plt.plot(fpr['iupred'], tpr['iupred'], color='lightcoral', linestyle='-.', lw=lw,
         label='IUPred (AUC = %0.4f)' % roc_auc['iupred'])
bepipred_roc, = plt.plot(fpr['bepipred'], tpr['bepipred'], color='goldenrod', lw=lw, label='Bepipred 2 (AUC = %0.4f)' % roc_auc['bepipred'])
antigen_roc, = plt.plot(fpr['antigenicity'], tpr['antigenicity'], color='grey', linestyle=':', lw=lw,
         label='Antigenicity-avg (AUC = %0.4f)' % roc_auc['antigenicity'])
hydro_roc, = plt.plot(fpr['hydrophilicity'], tpr['hydrophilicity'], color='peru', lw=lw, linestyle='--',
         label='Hydrophilicity-avg (AUC = %0.4f)' % roc_auc['hydrophilicity'])
access_roc, = plt.plot(fpr['accessibility'], tpr['accessibility'], color='teal', linestyle='--', lw=lw,
         label='Accessibility-avg (AUC = %0.4f)' % roc_auc['accessibility'])
betaturn_roc, = plt.plot(fpr['betaturn'], tpr['betaturn'], color='lightsteelblue', linestyle='-.', lw=lw,
         label='Betaturn-avg (AUC = %0.4f)' % roc_auc['betaturn'])

random_roc, = plt.plot([0, thresh], [0, thresh], color='navy', lw=lw, linestyle='--', label='random (AUC = 0.50)' )


plt.xlim([0.0, thresh])
plt.ylim([0.0, 1.0 * maxtpr])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(handles=[epidope_roc, betaturn_roc, access_roc, antigen_roc, hydro_roc, iupred_roc, bepipred_roc, std_roc, lbtope_roc, random_roc], loc="lower right", prop={'size': 9})
plt.savefig(os.path.join(deepipred_results_dir, f"ROC_prediction_comparison_{thresh}_auc10%.pdf"), bbox_inches="tight",
            pad_inches=0)
plt.show()
plt.close()

# calculate precision-recall curve
precision = {}
recall = {}
thresholds = {}

key = 'bepipred'
precision[key], recall[key], thresholds[key] = precision_recall_curve(bepipred_flag, bepipred_scores, pos_label=1)
roc_auc[key] = auc(recall[key], precision[key])
key = 'antigenicity'
precision[key], recall[key], thresholds[key] = precision_recall_curve(antigenicity_flag, antigenicity_scores,
                                                                      pos_label=1)
roc_auc[key] = auc(recall[key], precision[key])
key = 'hydrophilicity'
precision[key], recall[key], thresholds[key] = precision_recall_curve(hydrophilicity_flag, hydrophilicity_scores,
                                                                      pos_label=1)
roc_auc[key] = auc(recall[key], precision[key])
key = 'accessibility'
precision[key], recall[key], thresholds[key] = precision_recall_curve(accessibility_flag, accessibility_scores,
                                                                      pos_label=1)
roc_auc[key] = auc(recall[key], precision[key])
key = 'betaturn'
precision[key], recall[key], thresholds[key] = precision_recall_curve(betaturn_flag, betaturn_scores, pos_label=1)
roc_auc[key] = auc(recall[key], precision[key])
key = 'deepipred'
precision[key], recall[key], thresholds[key] = precision_recall_curve(deepipred_flag, deepipred_scores, pos_label=1)
roc_auc[key] = auc(recall[key], precision[key])
key = 'lbtope'
precision[key], recall[key], thresholds[key] = precision_recall_curve(lbtope_flag, lbtope_scores, pos_label=1)
roc_auc[key] = auc(recall[key], precision[key])
key = 'iupred'
precision[key], recall[key], thresholds[key] = precision_recall_curve(iupred_flag, iupred_scores, pos_label=1)
roc_auc[key] = auc(recall[key], precision[key])

# plot
plt.figure(figsize=(6, 6))
lw = 2
thresh = 1
for i in recall:
	recall[i] = [x for x in recall[i] if x <= thresh]
	precision[i] = precision[i][:len(recall[i])]
maxtpr = 0
for x in precision:
	maxtpr = max(maxtpr, max(precision[x]))
epidope_pr, = plt.plot(recall['deepipred'], precision['deepipred'], color='green', lw=lw,
         label='EpiDope (AUC = %0.4f)' % roc_auc['deepipred'])
bepipred_pr, = plt.plot(recall['bepipred'], precision['bepipred'], color='goldenrod', lw=lw,
         label='Bepipred 2 (AUC = %0.4f)' % roc_auc['bepipred'])
antigen_pr, = plt.plot(recall['antigenicity'], precision['antigenicity'], color='grey', linestyle=':', lw=lw,
         label='Antigenicity-avg (AUC = %0.4f)' % roc_auc['antigenicity'])
hydro_pr, = plt.plot(recall['hydrophilicity'], precision['hydrophilicity'], color='peru', lw=lw, linestyle='--',
         label='Hydrophilicity-avg (AUC = %0.4f)' % roc_auc['hydrophilicity'])
access_pr, = plt.plot(recall['accessibility'], precision['accessibility'], color='teal', linestyle='--', lw=lw,
         label='Accessibility-avg (AUC = %0.4f)' % roc_auc['accessibility'])
betaturn_pr, = plt.plot(recall['betaturn'], precision['betaturn'], color='lightsteelblue', linestyle='-.', lw=lw,
         label='Betaturn-avg (AUC = %0.4f)' % roc_auc['betaturn'])
lbtope_pr, = plt.plot(recall['lbtope'], precision['lbtope'], color='orange', linestyle='--', lw=lw,
         label='LBtope (AUC = %0.4f)' % roc_auc['lbtope'])
iupred_pr, = plt.plot(recall['iupred'], precision['iupred'], color='lightcoral', linestyle=':', lw=lw,
         label='IUPred (AUC = %0.4f)' % roc_auc['iupred'])

ratio_true_false = deepipred_flag.sum() / len(deepipred_flag)
random_pr, = plt.plot([0, 1], [ratio_true_false, ratio_true_false], color='navy', linestyle='--',
         label='random (AUC = %0.4f)' % ratio_true_false)
plt.xlim([0.0, thresh])
plt.ylim([0.0, 1.0 * maxtpr])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower right", handles=[epidope_pr, betaturn_pr, access_pr, antigen_pr, hydro_pr, iupred_pr, bepipred_pr, lbtope_pr, random_pr], prop={'size': 9})
plt.savefig(os.path.join(deepipred_results_dir, f"precision_recall_comparison_{thresh}.pdf"), bbox_inches="tight",
            pad_inches=0)
plt.close()

yPred_0 = deepipred_scores[deepipred_flag == 0]
yPred_1 = deepipred_scores[deepipred_flag == 1]
yPred_total = [yPred_0, yPred_1]

plt.hist(yPred_total, bins=20, range=(0, 1), stacked=False, label=['no Epitope', 'true Epitope'], density=True)
plt.legend()
plt.savefig(os.path.join(deepipred_results_dir, f"prediction_distribution.pdf"))
plt.close()
