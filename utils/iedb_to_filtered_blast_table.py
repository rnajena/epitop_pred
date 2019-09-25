slicesize=49
shift_needed=0
with open("/home/go96bix/projects/epitop_pred/utils/iedb_linear_epitopes.fasta") as input_file:
	with open("/home/go96bix/projects/epitop_pred/utils/bepipred_samples_like_filtered_blast_table.tsv","w") as output_file:
		output_file.write("#qseqid\tsseqid\tqstart\tsstart\tsend\n")
		non_epi_counter = 0
		true_epi_counter = 0
		for line in input_file:
			if line.startswith(">"):
				header = line.strip()
				sseqid = header[1:]
				if sseqid.startswith("Negative"):
					non_epi_counter += 1
					qseqid = f"nonepi_{non_epi_counter}"
				elif sseqid.startswith("Positive"):
					true_epi_counter += 1
					qseqid = f"epi_{true_epi_counter}"
				else:
					print(f"error: header {header} not in positive or negative set")
					exit()

			else:
				seq = line

				with open(f"/home/go96bix/projects/epitop_pred/bepipred_sequences/{qseqid}_{sseqid}.fasta","w") as out_fasta:
					out_fasta.write(f"{header}\n")
					out_fasta.write(f"{seq}\n")

				upper_pos = [i for i, c in enumerate(seq) if c.isupper()]
				if len(upper_pos)>1:
					epitopes = []
					start=upper_pos[0]
					stop=upper_pos[0]
					for i in range(1,len(upper_pos)):
						if upper_pos[i] == stop+1:
							stop=upper_pos[i]
						else:
							if stop>start:
								epitopes.append((start,stop))
							start=upper_pos[i]
							stop=upper_pos[i]
					epitopes.append((start,stop))
				for hit in epitopes:
					mean_pos = (hit[0]+hit[1])/2
					start = int(mean_pos - slicesize/2)
					stop = int(mean_pos + slicesize/2)
					if start * -1 > shift_needed:
						shift_needed = start*-1
					if stop-len(seq)>shift_needed:
						shift_needed = stop-len(seq)
					output_file.write(f"{qseqid}\t|{sseqid}|\t{1}\t{start+1}\t{stop+1}\n")
print(f"shift needed:{shift_needed}")