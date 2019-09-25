import matplotlib.pyplot as plt
import os
directory = "/home/go96bix/projects/raw_data/clustered_protein_seqs/my_cluster/"

for root, dirs, files in os.walk(directory):
	# for file in glob.glob("/home/go96bix/projects/raw_data/non_binary_250_nodes_1000epochs/results/deepipred/*.csv"):
	for name in files:
		if name.endswith("clstr"):
			with open(os.path.join(root, name), "r") as infile:
				count_proteins = []
				allLines = infile.read()
				clusters = allLines.split(">Cluster")
				for cluster in clusters:
					if len(cluster) > 0:
						count_validations = []
						proteins = cluster.strip().split("\n")
						for index, protein in enumerate(proteins):
							if index>0:
								pass
							pass
						count_proteins.append(index)

				plt.hist(count_proteins,bins=20, log=True)
				plt.xlabel("number proteins clustered together")
				plt.ylabel("number of clusters")
				figname = name.split("_")[0] + "hist.pdf"
				plt.savefig(os.path.join(root, figname))
				plt.close()