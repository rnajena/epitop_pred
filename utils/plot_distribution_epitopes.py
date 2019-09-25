import matplotlib.pyplot as plt

non_epi_pos = []
true_epi_pos = []
non_epi = True

with open("/home/go96bix/projects/epitop_pred/utils/iedb_linear_epitopes.fasta") as input_file:
	for line in input_file:
		if line.startswith(">"):
			header = line.strip()
			sseqid = header[1:]
			if sseqid.startswith("Negative"):
				non_epi = True
			elif sseqid.startswith("Positive"):
				non_epi = False
			else:
				print(f"error: header {header} not in positive or negative set")
				exit()

		else:
			seq = line

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
				epitopes.append((start, stop))
			for hit in epitopes:
				mean_pos = (hit[0] + hit[1]) / 2
				start = hit[0]
				stop = hit[1]
				rel_start = round((start / len(seq)) * 100)
				rel_stop = round((stop / len(seq)) * 100)
				for i in range(rel_start, rel_stop + 1):
					if non_epi:
						non_epi_pos.append(i)
					else:
						true_epi_pos.append(i)

plt.hist(true_epi_pos, 100)
plt.show()
plt.hist(non_epi_pos, 100)
plt.show()
