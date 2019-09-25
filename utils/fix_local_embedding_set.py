import os
import numpy as np
import pandas as pd

shift = 22

def fill_with_gaps(samples):
	out = []
	for sample in samples:
		seq = np.array(list(sample[0]))
		start = int(sample[1])
		stop = int(sample[2])
		file = sample[3]
		# print(os.path.join("/home/le86qiz/Documents/Konrad/tool_comparison/raptorx/flo_files", f"{file}.csv"))
		with open(os.path.join("/home/go96bix/projects/raw_data/bepipred_sequences", f"{file}.fasta"),"r") as in_file:
			for line in in_file:
				if line.startswith(">") or line == "\n":
					continue
				else:
					line = line.strip()
					seq_len = len(line)
					line = np.array(shift*["-"]+list(line.lower())+shift*["-"])

					if stop>seq_len and file.startswith("nonepi"):
						print("foo")

						#fill end with gaps
						# seq_new = np.array(["-"] * 49)
						# seq_new[0:seq_len - stop:] = line[start:]
					seq_new = line[start+shift:stop+shift]
					assert len(
						seq_new) == 49, f"Wrong len {len(seq_new)} in file {file}, start {start}, stop {stop}"
					seq = seq_new

					# print(seq_new[-20:])
						# print(seq[-20:])
					# if start<0:
					# 	#fill start with gaps
					# 	# seq_new = np.array(["-"]*49)
					# 	# seq_new[(-1*start):] = line[0:stop]
					# 	seq_new = line[]
					# 	assert len(seq_new) ==49, f"Wrong len {len(seq_new)} in file {file}, start {start}, stop {stop}"
					# 	seq = seq_new
					# 	# print(seq_new[0:20])
					# 	# print(seq[0:20])
					seq = "".join(seq)
		out.append([seq,start,stop,file])
	out_df = pd.DataFrame(np.array(out))

	return out_df

directory = "/home/go96bix/projects/epitop_pred/data_generator_bepipred/local_embedding"
for root, dirs, files in os.walk(directory):
	for file in files:
		if file.endswith(".csv") and not file.startswith("Y"):
			sample_csv = pd.read_csv(os.path.join(root, file), delimiter='\t', dtype='str', header=None).values
			# sample_table = np.array([i[1:] for i in sample_csv])
			sample_df = fill_with_gaps(sample_csv)
			sample_df.to_csv(os.path.join(root, file), sep='\t', encoding='utf-8', header=None,
			        index=None)