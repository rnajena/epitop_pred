import numpy as np
import pandas as pd
import urllib.request

"""
previously 
1. download home/go96bix/projects/raw_data/bcell_full_v3.csv
2. Download "all" entries of emble with download_from_iedb.py
now 
"""


df = pd.read_csv("/home/go96bix/projects/raw_data/bcell_full_v3.csv", ",", skiprows=1)
df = df.drop_duplicates("Epitope IRI")
df = df.rename(columns={"Epitope IRI": "epitope"})


def check_ncbi(hit, old_url, old_page_source):
	# print("try ncbi")
	url = hit["Antigen IRI"].values[0]
	protein_name = ""

	if url == old_url:
		protein_name = url.split("/")[-1]
		return old_page_source, url, protein_name

	if type(url) != str:
		print("did not work")
		page_source = np.nan

	elif "ncbi" in url:
		protein_name = url.split("/")[-1]
		try:
			response = urllib.request.urlopen(
				f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=protein&id={protein_name}&rettype=fasta&retmode=text",
				timeout=1)
			data = response.read()
			assert len(data) > 0, "no answer"
			page_source = data.decode("utf-8")

		except:
			print("did not work")
			page_source = np.nan
	else:
		print("did not work")
		page_source = np.nan

	return page_source, url, protein_name

def check_uniprot(hit, old_url, old_page_source):
	print("try uniprot")
	url = hit['Parent Protein IRI'].values[0]
	protein_name = ""

	if url == old_url:
		protein_name = url.split("/")[-1]
		return old_page_source, url, protein_name

	if type(url) != str:
		print("did not work")
		page_source = np.nan

	elif "uniprot" in url:
		protein_name = url.split("/")[-1]
		try:
			if url == old_url:
				return old_page_source, start, stop, epi_seq, url, protein_name

			response = urllib.request.urlopen(url + ".fasta", timeout=1)
			data = response.read()  # a `bytes` object
			assert len(data) > 0, "no answer"
			page_source = data.decode("utf-8")

		except:
			print("did not work")
			page_source = np.nan
	else:
		print("did not work")
		page_source = np.nan

	return page_source, url, protein_name

def get_protein(id_epi, old_url, old_page_source):
	hit = df.query(f'epitope=="http://www.iedb.org/epitope/{id_epi}"')

	start = hit["Starting Position"].values[0]-1
	stop = hit["Ending Position"].values[0]

	epi_seq = hit["Description"].values[0]

	if np.isnan(start) or np.isnan(stop):
		start = -1
		stop = -1
	else:
		start = int(start)
		stop = int(stop)


	page_source, url, protein_name = check_ncbi(hit, old_url, old_page_source)
	if type(page_source) != str:
		page_source, url, protein_name = check_uniprot(hit, old_url, old_page_source)

	return page_source, start, stop, epi_seq, url, protein_name


old_line = ""
old_url = ""
page_source = ""
with open("iedb_linear_epitopes_27_11_2019_version3.fasta", "w") as out_fasta:
	with open("negative_samples.txt", "r") as input_negativ:
		with open("positive_samples.txt", "r") as input_positiv:
			for i, samples in enumerate([input_negativ, input_positiv]):
				epi_bool = i
				# for line in samples:
				for line in np.unique(list(samples)):
					if line == old_line:
						continue
					else:
						old_line = line
						id_epi = line.strip()
						page_source, start, stop, epi_seq, old_url, protein_name = get_protein(id_epi, old_url, page_source)
						if type(page_source) != str:
							continue
						seq = ""
						lines = page_source.split("\n")
						header = ""
						for index, line in enumerate(lines):
							if index == 0:
								header = f">{protein_name}"
							else:
								seq += line

						if start == -1 or stop == -1:
							start = seq.find(epi_seq)
							stop = start + len(epi_seq)
							print(f"sequence slice: {seq[start:stop]}, from table: {epi_seq}")
							if start == -1:
								continue
						else:
							if seq[start:stop] != epi_seq:
								print(f"sequence slice: {seq[start:stop]}, from table: {epi_seq}")
								continue

						if epi_bool:
							out_fasta.write(header + f"|{start}_{stop}|PositiveID_{id_epi}\n")
						else:
							out_fasta.write(header + f"|{start}_{stop}|NegativeID_{id_epi}\n")
						out_fasta.write(seq + "\n")
						print(header)
			# print(seq[start-1:stop])
			# print(epi_seq)
