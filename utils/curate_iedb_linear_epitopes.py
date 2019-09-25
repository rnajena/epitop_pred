import numpy as np

unique_prot_dict = {}
"""
1: read file 
2: get postitions of non-/epitope
3: save positions and header in dict with seq as key
4: merge non conflicting overlapping epitopes/non-epitopes in proteins"""
with open("/home/go96bix/projects/epitop_pred/utils/iedb_linear_epitopes.fasta") as input_file:
	for line in input_file:
		if line.startswith(">"):
			header = line.strip()

		else:
			seq = line.strip()

			upper_pos = [i for i, c in enumerate(seq) if c.isupper()]
			if len(upper_pos) > 1:
				epitopes = []
				start = upper_pos[0]
				stop = upper_pos[0]
				for i in range(1, len(upper_pos)):
					if upper_pos[i] == stop + 1:
						stop = upper_pos[i]
					else:
						if stop > start:
							epitopes.append((start, stop))
						start = upper_pos[i]
						stop = upper_pos[i]

				epitopes.append((start, stop + 1, header[1:]))

			seq_lower = seq.lower()
			old_entry = unique_prot_dict.get(seq_lower, [])
			old_entry.extend(epitopes)
			unique_prot_dict.update({seq_lower: old_entry})

print(len(unique_prot_dict.values()))
number_conflicts = 0
protein_counter = 0

for protein, hits in unique_prot_dict.items():
	epitope_arr = np.zeros(len(protein))
	non_epitope_arr = np.zeros(len(protein))
	header_long = []
	mask = np.array([-1] * len(protein))
	for index, marked_area in enumerate(hits):
		# 	solve merging
		start, stop, header = marked_area
		header_long.append(f"{header}_{start}_{stop}")
		if header.startswith("Positive"):
			epitope_arr[start:stop] += 1
		else:
			non_epitope_arr[start:stop] += 1
		if any(mask[start: stop] == 0) and header.startswith("Positive"):
			mask[start: stop] = 1
		elif any(mask[start:stop] == 1) and header.startswith("Negative"):
			# mark as non epitope all aa's which are not labeled as epitope or non epitope
			mask[start:stop][mask[start:stop] == -1] = 0
		else:
			if header.startswith("Positive"):
				mask[start: stop] = 1
			else:
				mask[start: stop] = 0

	quantity = []
	for i in range(len(protein)):
		epi_count = epitope_arr[i]
		non_epi_count = non_epitope_arr[i]
		if epi_count + non_epi_count == 0:
			quantity.append("-")
		else:
			quantity.append(str(epi_count / (epi_count + non_epi_count)))

	quantity_str = "\t".join(quantity)
	mask_str = ["-" if i == -1 else str(i) for i in list(mask)]
	mask_str = "\t".join(mask_str)
	header_long_str = "\t".join(header_long)

	with open(f"/home/go96bix/projects/raw_data/bepipred_proteins_with_marking/protein_{protein_counter}.fasta",
	          "w") as out_fasta:
		out_fasta.write(f">{header_long_str}\n")
		out_fasta.write(f"{protein.upper()}\n")
		out_fasta.write(f"{mask_str}\n")
		out_fasta.write(f"{quantity_str}\n")

	protein_counter += 1
