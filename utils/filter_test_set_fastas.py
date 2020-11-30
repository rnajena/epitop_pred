import sys
import os

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

if len(sys.argv)==1:
	# testfiletable = '/home/go96bix/projects/epitop_pred/data_generator_bepipred_binary_0.5_seqID/samples_for_ROC.csv'
	testfiletable = '/home/go96bix/projects/raw_data/allprotein.csv'
	out_path = "/home/go96bix/projects/raw_data/bepipred_sequences_allProteins.fasta"
else:
	testfiletable = sys.argv[1]
	print(f"testfiletable = {sys.argv[1]}")
	out_path = sys.argv[2]
	print(f"out_path = {sys.argv[2]}")
	cluster_dict = cluster_to_dict()

testproteinIDs = []
with open(testfiletable) as infile:
	for line in infile:
		if line.startswith("/"):
			file = line.strip().rsplit('/', 1)[1]
			testproteinIDs.append(file[:-6])
		elif line.startswith("Cluster"):
			files = cluster_dict[line.strip()]
			for file in files:
				file = file.strip().rsplit('/', 1)[-1]
				testproteinIDs.append(file[:-6])
		else:
			print("Error: input test set csv should contain either a path to a protein or a name of a Cluster "
			      f"but contained {line}")
			exit()

if os.path.isfile(out_path):
	os.remove(out_path)
with open(out_path, "a") as outfile:
	for testid in testproteinIDs:
		file = f'/home/le86qiz/Documents/Konrad/tool_comparison/comparison3/bepipred_proteins_with_marking/{testid}.fasta'
		header, seq_local, values = readFasta_extended(file)
		header_long_str = "\t".join(header)
		outfile.write(f'>{testid}\n')
		outfile.write(f'{seq_local}\n')
