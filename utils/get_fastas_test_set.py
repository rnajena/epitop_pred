import os
import glob
import re

table = '/home/go96bix/projects/epitop_pred/data_generator_bepipred_final/local_embedding/X_test.csv'

ids = []
with open(table) as infile:
	for line in infile:
		ids.append(line.strip().split('\t')[3])


for x in ids:
	os.system(f'cp /home/go96bix/projects/raw_data/bepipred_sequences/{x}.fasta /home/go96bix/projects/raw_data/bepipred_sequences_test')

fastas = glob.glob('/home/go96bix/projects/raw_data/bepipred_sequences_test/*.fasta')

remove_lower = lambda text: re.sub('[a-z]+', '\n', text)

epitopelist = '/home/go96bix/projects/raw_data/epitopes.csv'
multifasta = '/home/go96bix/projects/raw_data/bepipred_sequences_test.fasta'
with open(multifasta, 'w') as outfile:
	with open(epitopelist, 'w') as epiout:
		for fasta in fastas:
			with open(fasta) as infile:
				for line in infile:
					if line.startswith('>'):
						outfile.write('>' + fasta.rsplit('/',1)[1][:-6] + '\n')
					elif line.strip():
						outfile.write(line.strip().upper() + '\n')
						epiout.write(remove_lower(line.strip()) + '\n')
