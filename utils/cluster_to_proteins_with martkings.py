import numpy as np
from shutil import copyfile
import os

directory = "/home/go96bix/projects/raw_data/clustered_protein_seqs/my_cluster/"
directory_fasta = "/home/go96bix/projects/raw_data/bepipred_proteins_with_marking"
directory_out = "/home/go96bix/projects/raw_data/bepipred_proteins_with_marking_"
unique_prot_dict = {}

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

for root, dirs, files in os.walk(directory):
	# for file in glob.glob("/home/go96bix/projects/raw_data/non_binary_250_nodes_1000epochs/results/deepipred/*.csv"):
	for name in files:
		if name.endswith("clstr"):
			name_dir = directory_out + name.split(".fasta")[0]
			if not os.path.isdir(name_dir):
				os.makedirs(name_dir)
			with open(os.path.join(root, name), "r") as infile:
				allLines = infile.read()
				clusters = allLines.split(">Cluster")
				for cluster in clusters:
					if len(cluster) > 0:
						count_validations = 0
						proteins = cluster.strip().split("\n")
						bestProt = ""
						for index, protein in enumerate(proteins):
							if index>0:
								filename = protein.split(" ")[1][1:-3]+".fasta"
								header, seq, values = readFasta_extended(os.path.join(directory_fasta,filename))
								if len(header) > count_validations:
									count_validations = len(header)
									bestProt = filename
							else:
								continue
									# bestProt = open(os.path.join(directory_fasta,filename)).read()

						copyfile(os.path.join(directory_fasta,bestProt),os.path.join(name_dir,bestProt))


# """
# 1: read file
# 2: get postitions of non-/epitope
# 3: save positions and header in dict with seq as key
# 4: merge non conflicting overlapping epitopes/non-epitopes in proteins"""
# with open("/home/go96bix/projects/epitop_pred/utils/iedb_linear_epitopes.fasta") as input_file:
# 	for line in input_file:
# 		if line.startswith(">"):
# 			header = line.strip()
#
# 		else:
# 			seq = line.strip()
#
# 			upper_pos = [i for i, c in enumerate(seq) if c.isupper()]
# 			if len(upper_pos) > 1:
# 				epitopes = []
# 				start = upper_pos[0]
# 				stop = upper_pos[0]
# 				for i in range(1, len(upper_pos)):
# 					if upper_pos[i] == stop + 1:
# 						stop = upper_pos[i]
# 					else:
# 						if stop > start:
# 							epitopes.append((start, stop))
# 						start = upper_pos[i]
# 						stop = upper_pos[i]
#
# 				epitopes.append((start, stop + 1, header[1:]))
#
# 			seq_lower = seq.lower()
# 			old_entry = unique_prot_dict.get(seq_lower, [])
# 			old_entry.extend(epitopes)
# 			unique_prot_dict.update({seq_lower: old_entry})
#
# print(len(unique_prot_dict.values()))
# number_conflicts = 0
# protein_counter = 0
#
#
#
# for protein, hits in unique_prot_dict.items():
# 	epitope_arr = np.zeros(len(protein))
# 	non_epitope_arr = np.zeros(len(protein))
# 	header_long =[]
# 	mask = np.array([-1] * len(protein))
# 	for index, marked_area in enumerate(hits):
# 		# 	solve merging
# 		start, stop, header = marked_area
# 		header_long.append(f"{header}_{start}_{stop}")
# 		# conflict = (any(mask[start: stop] == 0) and header.startswith("Positive")) or (any(mask[start:stop] == 1) and
# 		#                                                                   header.startswith("Negative"))
# 		# if conflict:
# 		# 	number_conflicts += 1
# 		# 	print(f"{number_conflicts} conflicts: {mask[start: stop]} is in conflict with {header}")
# 		if header.startswith("Positive"):
# 			epitope_arr[start:stop] += 1
# 		else:
# 			non_epitope_arr[start:stop] += 1
# 		if any(mask[start: stop] == 0) and header.startswith("Positive"):
# 			mask[start: stop] = 1
# 		elif any(mask[start:stop] == 1) and header.startswith("Negative"):
# 			# mark as non epitope all aa's which are not labeled as epitope or non epitope
# 			mask[start:stop][mask[start:stop] == -1] = 0
# 		else:
# 			if header.startswith("Positive"):
# 				mask[start: stop] = 1
# 			else:
# 				mask[start: stop] = 0
#
# 	quantity = []
# 	for i in range(len(protein)):
# 		epi_count = epitope_arr[i]
# 		non_epi_count = non_epitope_arr[i]
# 		if epi_count+non_epi_count == 0:
# 			quantity.append("-")
# 		else:
# 			quantity.append(str(epi_count/(epi_count+non_epi_count)))
#
# 	quantity_str = "\t".join(quantity)
# 	mask_str = ["-" if i == -1 else str(i) for i in list(mask)]
# 	mask_str = "\t".join(mask_str)
# 	header_long_str = "\t".join(header_long)
#
# 	with open(f"/home/go96bix/projects/raw_data/bepipred_proteins_with_marking/protein_{protein_counter}.fasta", "w") as out_fasta:
# 		out_fasta.write(f">{header_long_str}\n")
# 		out_fasta.write(f"{protein.upper()}\n")
# 		out_fasta.write(f"{mask_str}\n")
# 		out_fasta.write(f"{quantity_str}\n")
#
# 	protein_counter += 1
