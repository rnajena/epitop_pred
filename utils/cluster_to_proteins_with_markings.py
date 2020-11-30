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
							if index > 0:
								filename = protein.split(" ")[1][1:-3] + ".fasta"
								header, seq, values = readFasta_extended(os.path.join(directory_fasta, filename))
								if len(header) > count_validations:
									count_validations = len(header)
									bestProt = filename
							else:
								continue

						copyfile(os.path.join(directory_fasta, bestProt), os.path.join(name_dir, bestProt))
