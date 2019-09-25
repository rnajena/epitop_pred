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


testfiletable = '/home/go96bix/projects/epitop_pred/data_generator_bepipred_non_binary_0.5_seqID/samples_for_ROC.csv'
# testfiletable = '/home/le86qiz/Documents/Konrad/tool_comparison/comparison3/samples_for_ROC.csv'
testproteinIDs = []
with open(testfiletable) as infile:
	for line in infile:
		file = line.strip().rsplit('/', 1)[1]
		testproteinIDs.append(file[:-6])

out_path = "/home/go96bix/projects/raw_data/bepipred_sequences_test_non_binary_0.5_seqID.fasta"
with open(out_path, "a") as outfile:
	for testid in testproteinIDs:
		file = f'/home/le86qiz/Documents/Konrad/tool_comparison/comparison3/bepipred_proteins_with_marking/{testid}.fasta'
		header, seq_local, values = readFasta_extended(file)
		header_long_str = "\t".join(header)
		outfile.write(f'>{testid}\n')
		outfile.write(f'{seq_local}\n')
