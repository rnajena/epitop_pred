import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn import metrics


def get_testproteinIDS(testfiletable):
	"""
	get all training proteins
	:param testfiletable:
	:return:
	"""
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


def bepipred_samples():
	"""
	which samples are in the training set
	:return:
	"""
	samples = set()
	with open("iedb_linear_epitopes.fasta", "r") as bepi_samples:
		for line in bepi_samples:
			if line.startswith(">"):
				samples.add(line[1:].strip())
	return samples


dominant = []
non_dominant = []
testfiletable = '/home/go96bix/projects/raw_data/08_allprotein.csv'
testproteinIDs = get_testproteinIDS(testfiletable)

epiID_to_testid = {}

for testid in testproteinIDs:
	"""
	how many potentail non dom proteins are in the training data
	"""
	file = f'/home/le86qiz/Documents/Konrad/tool_comparison/comparison3/bepipred_proteins_with_marking/{testid}.fasta'
	dominant_bool = False
	with open(file) as infile:
		for line in infile:
			line = line[1:].strip().split()
			for epiID in line:

				epiID = epiID.split('_')
				epiID_to_testid.update({f"{epiID[0]}_{epiID[1]}": testid})
				flag = epiID[0]
				# start = int(epiID[2])
				# stop = int(epiID[3])
				if flag == "PositiveID":
					dominant_bool = True
					break
			break
	if dominant_bool:
		dominant.append(testid)
	else:
		non_dominant.append(testid)

print("dominant: ", len(dominant))
print("non_dominant: ", len(non_dominant))

bepi_sampels = bepipred_samples()

prot_dict = {}

with open("iedb_linear_epitopes_27_11_2019.fasta", "r") as fasta:
	"""
	get all validated regions per protein in the eval set
	"""
	# with open("/home/go96bix/projects/raw_data/validation_samples_nov_2019_08_seqID.fasta", "r") as fasta:
	for line in fasta:
		if line.startswith(">"):
			# put_in_dict = True

			line = line[1:].split("|")
			testid = line[0]

			start_stop = line[-2].split('_')

			epiID = line[-1].strip()

			# if already in bepipred samples than dont add to validation set
			# if epiID in bepi_sampels:
			#     put_in_dict = False
			#     continue
			ids = prot_dict.get(testid, [])
			ids.append(epiID)
			prot_dict.update({testid: ids})
	# epiID = epiID.split('_')
	# flag = epiID[0]

dominant = []
non_dominant = []

for testid, values in prot_dict.items():
	dominant_bool = False
	# print(epiID_to_testid)
	testid_training = epiID_to_testid.get(values[0], None)
	# print(values[0])
	# print(testid_training)
	# exit()
	if testid_training != None:
		testid = f"/home/go96bix/projects/raw_data/binary_both_embeddings_0.8_0.5_seqID_benchmark/results/epidope/{testid_training}.csv"
	else:
		path = f"/home/go96bix/projects/raw_data/validation_11_2019/results/epidope/{testid}.csv"
		if os.path.isfile(path):
			testid = path
		else:
			continue
	for epiID in values:
		# if epiID in bepi_sampels:
		#     put_in_dict = False
		#     continue
		epiID = epiID.split('_')
		flag = epiID[0]
		if flag == "PositiveID":
			dominant_bool = True
			break
	if dominant_bool:
		dominant.append(testid)
	elif len(values) < 10:
		pass
	else:
		non_dominant.append(testid)

print("dominant: ", len(dominant))
print("non_dominant: ", len(non_dominant))
# print(non_dominant)

# teil der proteine ist unter anderen namen getestet wurden im trainingset
# teil wurde gar nicht getestet da sie in die Cluster gefallen sind
# teil ist von eval set


fpr = {}
tpr = {}
roc_auc = {}
thresholds = {}
thresh = 1

scores = {}
min_scores = []
max_scores = []
median_scores = []
quantile_scores = []

epidope_flag = []

for i in non_dominant:
	df = pd.read_csv(i, sep='\t')

	score = np.array(df['Deepipred']).mean()
	scores['mean'] = np.append(scores.get('mean', np.array([])), score)

	score = np.array(df['Deepipred']).min()
	scores['min'] = np.append(scores.get('min', np.array([])), score)

	score = np.array(df['Deepipred']).max()
	scores['max'] = np.append(scores.get('max', np.array([])), score)

	score = np.median(np.array(df['Deepipred']))
	scores['median'] = np.append(scores.get('median', np.array([])), score)

	score = np.quantile(np.array(df['Deepipred']), 0.8)
	scores['quantile_high'] = np.append(scores.get('quantile_high', np.array([])), score)

	score = np.quantile(np.array(df['Deepipred']), 0.2)
	scores['quantile_low'] = np.append(scores.get('quantile_low', np.array([])), score)

	epidope_flag.append(0)

for i in dominant:
	df = pd.read_csv(i, sep='\t')

	score = np.array(df['Deepipred']).mean()
	scores['mean'] = np.append(scores.get('mean', np.array([])), score)

	score = np.array(df['Deepipred']).min()
	scores['min'] = np.append(scores.get('min', np.array([])), score)

	score = np.array(df['Deepipred']).max()
	scores['max'] = np.append(scores.get('max', np.array([])), score)

	score = np.median(np.array(df['Deepipred']))
	scores['median'] = np.append(scores.get('median', np.array([])), score)

	score = np.quantile(np.array(df['Deepipred']), 0.8)
	scores['quantile_high'] = np.append(scores.get('quantile_high', np.array([])), score)

	score = np.quantile(np.array(df['Deepipred']), 0.2)
	scores['quantile_low'] = np.append(scores.get('quantile_low', np.array([])), score)

	epidope_flag.append(1)

epidope_flag = np.array(epidope_flag)
"""
calc ROC
"""
for key in ['mean', 'min', 'max', 'median', 'quantile_high', 'quantile_low']:
	fpr[key], tpr[key], thresholds[key] = metrics.roc_curve(epidope_flag, scores[key], pos_label=1)
	roc_auc[key] = metrics.roc_auc_score(epidope_flag, scores[key], max_fpr=thresh)

"""
plotting
"""
outdir = "."

plt.figure(figsize=(6, 6))
lw = 2
maxtpr = 0
for x in tpr:
	maxtpr = max(maxtpr, max(tpr[x]))

mean_roc, = plt.plot(fpr['mean'], tpr['mean'], color='green', lw=lw,
					 label='mean (AUC = %0.4f)' % roc_auc['mean'])
min_roc, = plt.plot(fpr['min'], tpr['min'], color='lightcoral', linestyle='-.', lw=lw,
					label='min (AUC = %0.4f)' % roc_auc['min'])
max_roc, = plt.plot(fpr['max'], tpr['max'], color='goldenrod', lw=lw, label='max (AUC = %0.4f)' % roc_auc['max'])
median_roc, = plt.plot(fpr['median'], tpr['median'], color='grey', linestyle=':', lw=lw,
					   label='median (AUC = %0.4f)' % roc_auc['median'])
quantile_high_roc, = plt.plot(fpr['quantile_high'], tpr['quantile_high'], color='peru', lw=lw, linestyle='--',
						 label='quantile 0.8 (AUC = %0.4f)' % roc_auc['quantile_high'])
quantile_low_roc, = plt.plot(fpr['quantile_low'], tpr['quantile_low'], color='teal', lw=lw, linestyle='--',
						 label='quantile 0.2 (AUC = %0.4f)' % roc_auc['quantile_low'])

random_roc, = plt.plot([0, thresh], [0, thresh], color='navy', lw=lw, linestyle='--', label='random (AUC = 0.50)')

plt.xlim([0.0, thresh])
plt.ylim([0.0, 1.0 * maxtpr])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for imuno dominance')
plt.legend()
plt.legend(
	handles=[mean_roc, min_roc, max_roc, median_roc, quantile_high_roc, quantile_low_roc, random_roc],
	loc="lower right", prop={'size': 9})
plt.savefig(os.path.join(outdir, f"ROC_prediction_imunodominance_{thresh}.pdf"), bbox_inches="tight", pad_inches=0)
plt.show()
plt.close()

"""
calc precision recall
"""
precision = {}
recall = {}
thresholds = {}
for key in ['mean', 'min', 'max', 'median', 'quantile_high', 'quantile_low']:
	precision[key], recall[key], thresholds[key] = metrics.precision_recall_curve(epidope_flag, scores[key], pos_label=1)
	roc_auc[key] = metrics.auc(recall[key], precision[key])

"""
plotting
"""
outdir = "."

plt.figure(figsize=(6, 6))
lw = 2
maxtpr = 0
for x in tpr:
	maxtpr = max(maxtpr, max(tpr[x]))

mean_roc, = plt.plot(recall['mean'], precision['mean'], color='green', lw=lw, label='mean (AUC = %0.4f)' % roc_auc['mean'])
min_roc, = plt.plot(recall['min'], precision['min'], color='lightcoral', linestyle='-.', lw=lw,
					label='min (AUC = %0.4f)' % roc_auc['min'])
max_roc, = plt.plot(recall['max'], precision['max'], color='goldenrod', lw=lw, label='max (AUC = %0.4f)' % roc_auc['max'])
median_roc, = plt.plot(recall['median'], precision['median'], color='grey', linestyle=':', lw=lw,
					   label='median (AUC = %0.4f)' % roc_auc['median'])
quantile_high_roc, = plt.plot(recall['quantile_high'], precision['quantile_high'], color='peru', lw=lw, linestyle='--',
						 label='quantile 0.8 (AUC = %0.4f)' % roc_auc['quantile_high'])
quantile_low_roc, = plt.plot(recall['quantile_low'], precision['quantile_low'], color='teal', lw=lw, linestyle='--',
						 label='quantile 0.2 (AUC = %0.4f)' % roc_auc['quantile_low'])

ratio_true_false = epidope_flag.sum() / len(epidope_flag)
random_pr, = plt.plot([0, 1], [ratio_true_false, ratio_true_false], color='navy', linestyle='--',
					  label='random (AUC = %0.4f)' % ratio_true_false)
plt.xlim([0.0, thresh])
plt.ylim([0.97, 1.0 * maxtpr])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Precision-Recall Curve for imuno dominance')
plt.legend()
plt.legend(handles=[mean_roc, min_roc, max_roc, median_roc, quantile_high_roc, quantile_low_roc, random_pr], loc="lower right",
		   prop={'size': 9})
plt.savefig(os.path.join(outdir, f"precision_recall_imunodominance_{thresh}.pdf"), bbox_inches="tight", pad_inches=0)
plt.show()
plt.close()

Nr = 3
Nc = 2

left = 0.125  # the left side of the subplots of the figure
right = 0.9   # the right side of the subplots of the figure
bottom = 0.1  # the bottom of the subplots of the figure
top = 0.9     # the top of the subplots of the figure
wspace = 0.2  # the amount of width reserved for space between subplots,
              # expressed as a fraction of the average axis width
hspace = 0.2  # the amount of height reserved for space between subplots,
              # expressed as a fraction of the average axis height


fig, axs = plt.subplots(Nr, Nc, sharex=True, sharey=True)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.4)
fig.suptitle('score distribution')
k = 0
for i in range(Nr):
	for j in range(Nc):
		key = ['mean', 'median', 'max', 'min', 'quantile_high', 'quantile_low'][k]
		yPred_0 = scores[key][epidope_flag == 0]
		yPred_1 = scores[key][epidope_flag == 1]
		yPred_total = [yPred_0, yPred_1]

		foo = axs[i,j].hist(yPred_total, bins=20, range=(0, 1), stacked=False, label=['no Epitope', 'true Epitope'], density=True)
		axs[i,j].set_title(key)
		if i == 0 and j == 1:
			axs[i,j].legend(prop={'size': 9})
		# if i < Nr -1:
		# 	axs[i,j].set_title(key)
# axs.legend()
		# plt.savefig(os.path.join(epidope_results_dir, f"prediction_distribution.pdf"))
		# plt.close()
		k += 1

plt.savefig(os.path.join(outdir, f"prediction_distribution.pdf"))


############################################################### MAX DATA
df = pd.read_csv("/home/go96bix/projects/raw_data/imunodominat_prots/uniprot-proteome_UP000000800.csv", "\t")
prot_to_geneName = {}
for i,j in zip(df["Entry"],df["Gene names"]):
	geneName = j.split()[-1]
	geneName = geneName.replace("_","")
	prot_to_geneName.update({i:geneName})

df2 = pd.read_csv("/home/go96bix/projects/raw_data/imunodominat_prots/immunodominant.txt","\t",header=None)
dominant = set(df2[0])


fpr = {}
tpr = {}
roc_auc = {}
thresholds = {}
thresh = 1

scores = {}
min_scores = []
max_scores = []
median_scores = []
quantile_scores = []

epidope_flag = []

with open("/home/go96bix/projects/raw_data/imunodominat_prots/uniprot-proteome_UP000000800.fasta", "r") as fasta:
	"""
	get all validated regions per protein in the eval set
	"""
	# with open("/home/go96bix/projects/raw_data/validation_samples_nov_2019_08_seqID.fasta", "r") as fasta:
	for line in fasta:
		if line.startswith(">"):
			prot_file_name = line[1:].split()[0]
			prot = prot_file_name.split("|")[1]

			geneName = prot_to_geneName[prot]
			if geneName in dominant:
				epidope_flag.append(1)
			else:
				epidope_flag.append(0)

			path = f"/home/go96bix/projects/raw_data/imunodominat_prots/epidope/{prot_file_name}.csv"
			df = pd.read_csv(path, sep='\t')
			score = np.array(df['Deepipred']).mean()
			scores['mean'] = np.append(scores.get('mean', np.array([])), score)

			score = np.array(df['Deepipred']).min()
			scores['min'] = np.append(scores.get('min', np.array([])), score)

			score = np.array(df['Deepipred']).max()
			scores['max'] = np.append(scores.get('max', np.array([])), score)

			score = np.median(np.array(df['Deepipred']))
			scores['median'] = np.append(scores.get('median', np.array([])), score)

			score = np.quantile(np.array(df['Deepipred']), 0.8)
			scores['quantile_high'] = np.append(scores.get('quantile_high', np.array([])), score)

			score = np.quantile(np.array(df['Deepipred']), 0.2)
			scores['quantile_low'] = np.append(scores.get('quantile_low', np.array([])), score)

	scores['dif_max_min'] = scores['max'] - scores['min']

epidope_flag = np.array(epidope_flag)

"""
calc ROC
"""
for key in ['mean', 'min', 'max', 'median', 'quantile_high', 'quantile_low', 'dif_max_min']:
	fpr[key], tpr[key], thresholds[key] = metrics.roc_curve(epidope_flag, scores[key], pos_label=1)
	roc_auc[key] = metrics.roc_auc_score(epidope_flag, scores[key], max_fpr=thresh)

"""
plotting
"""
outdir = "."

plt.figure(figsize=(6, 6))
lw = 2
maxtpr = 0
for x in tpr:
	maxtpr = max(maxtpr, max(tpr[x]))

mean_roc, = plt.plot(fpr['mean'], tpr['mean'], color='green', lw=lw,
					 label='mean (AUC = %0.4f)' % roc_auc['mean'])
min_roc, = plt.plot(fpr['min'], tpr['min'], color='lightcoral', linestyle='-.', lw=lw,
					label='min (AUC = %0.4f)' % roc_auc['min'])
max_roc, = plt.plot(fpr['max'], tpr['max'], color='goldenrod', lw=lw, label='max (AUC = %0.4f)' % roc_auc['max'])
median_roc, = plt.plot(fpr['median'], tpr['median'], color='grey', linestyle=':', lw=lw,
					   label='median (AUC = %0.4f)' % roc_auc['median'])
quantile_high_roc, = plt.plot(fpr['quantile_high'], tpr['quantile_high'], color='peru', lw=lw, linestyle='--',
						 label='quantile 0.8 (AUC = %0.4f)' % roc_auc['quantile_high'])
quantile_low_roc, = plt.plot(fpr['quantile_low'], tpr['quantile_low'], color='teal', lw=lw, linestyle='--',
						 label='quantile 0.2 (AUC = %0.4f)' % roc_auc['quantile_low'])
dif_roc, = plt.plot(fpr['dif_max_min'], tpr['dif_max_min'], color='lightsteelblue', linestyle='-.',
						 label='differenz max min (AUC = %0.4f)' % roc_auc['dif_max_min'])

random_roc, = plt.plot([0, thresh], [0, thresh], color='navy', lw=lw, linestyle='--', label='random (AUC = 0.50)')

plt.xlim([0.0, thresh])
plt.ylim([0.0, 1.0 * maxtpr])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for imuno dominance')
plt.legend()
plt.legend(
	handles=[mean_roc, min_roc, max_roc, median_roc, quantile_high_roc, quantile_low_roc, dif_roc, random_roc],
	loc="lower right", prop={'size': 9})
plt.savefig(os.path.join(outdir, f"ROC_prediction_imunodominance_{thresh}_max_data.pdf"), bbox_inches="tight", pad_inches=0)
plt.show()
plt.close()

"""
calc precision recall
"""
precision = {}
recall = {}
thresholds = {}
for key in ['mean', 'min', 'max', 'median', 'quantile_high', 'quantile_low', 'dif_max_min']:
	precision[key], recall[key], thresholds[key] = metrics.precision_recall_curve(epidope_flag, scores[key], pos_label=1)
	roc_auc[key] = metrics.auc(recall[key], precision[key])

"""
plotting
"""
outdir = "."

plt.figure(figsize=(6, 6))
lw = 2
maxtpr = 0
for x in tpr:
	maxtpr = max(maxtpr, max(tpr[x]))

mean_roc, = plt.plot(recall['mean'], precision['mean'], color='green', lw=lw, label='mean (AUC = %0.4f)' % roc_auc['mean'])
min_roc, = plt.plot(recall['min'], precision['min'], color='lightcoral', linestyle='-.', lw=lw,
					label='min (AUC = %0.4f)' % roc_auc['min'])
max_roc, = plt.plot(recall['max'], precision['max'], color='goldenrod', lw=lw, label='max (AUC = %0.4f)' % roc_auc['max'])
median_roc, = plt.plot(recall['median'], precision['median'], color='grey', linestyle=':', lw=lw,
					   label='median (AUC = %0.4f)' % roc_auc['median'])
quantile_high_roc, = plt.plot(recall['quantile_high'], precision['quantile_high'], color='peru', lw=lw, linestyle='--',
						 label='quantile 0.8 (AUC = %0.4f)' % roc_auc['quantile_high'])
quantile_low_roc, = plt.plot(recall['quantile_low'], precision['quantile_low'], color='teal', lw=lw, linestyle='--',
						 label='quantile 0.2 (AUC = %0.4f)' % roc_auc['quantile_low'])
dif_roc, = plt.plot(recall['dif_max_min'], precision['dif_max_min'], color='lightsteelblue', linestyle='-.',
						 label='dif max min (AUC = %0.4f)' % roc_auc['dif_max_min'])

ratio_true_false = epidope_flag.sum() / len(epidope_flag)
random_pr, = plt.plot([0, 1], [ratio_true_false, ratio_true_false], color='navy', linestyle='--',
					  label='random (AUC = %0.4f)' % ratio_true_false)
plt.xlim([0.0, thresh])
# plt.ylim([0.97, 1.0 * maxtpr])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Precision-Recall Curve for imuno dominance')
plt.legend()
plt.legend(handles=[mean_roc, min_roc, max_roc, median_roc, quantile_high_roc, quantile_low_roc,dif_roc, random_pr], loc="lower right",
		   prop={'size': 9})
plt.savefig(os.path.join(outdir, f"precision_recall_imunodominance_{thresh}_max_data.pdf"), bbox_inches="tight", pad_inches=0)
plt.show()
plt.close()

Nr = 4
Nc = 2

left = 0.125  # the left side of the subplots of the figure
right = 0.9   # the right side of the subplots of the figure
bottom = 0.1  # the bottom of the subplots of the figure
top = 0.9     # the top of the subplots of the figure
wspace = 0.2  # the amount of width reserved for space between subplots,
              # expressed as a fraction of the average axis width
hspace = 0.2  # the amount of height reserved for space between subplots,
              # expressed as a fraction of the average axis height


fig, axs = plt.subplots(Nr, Nc, sharex=True, sharey=True)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.4)
fig.suptitle('score distribution')
k = 0
for i in range(Nr):
	for j in range(Nc):
		key = ['mean', 'median', 'max', 'min', 'quantile_high', 'quantile_low','dif_max_min'][k]
		yPred_0 = scores[key][epidope_flag == 0]
		yPred_1 = scores[key][epidope_flag == 1]
		yPred_total = [yPred_0, yPred_1]

		foo = axs[i,j].hist(yPred_total, bins=20, range=(0, 1), stacked=False, label=['no Epitope', 'true Epitope'], density=True)
		axs[i,j].set_title(key)
		if i == 0 and j == 1:
			axs[i,j].legend(prop={'size': 9})
		# if i < Nr -1:
		# 	axs[i,j].set_title(key)
# axs.legend()
		# plt.savefig(os.path.join(epidope_results_dir, f"prediction_distribution.pdf"))
		# plt.close()
		k += 1
		print(key)
		if k == len(key):
			break

plt.savefig(os.path.join(outdir, f"prediction_distribution_max_data.pdf"))
