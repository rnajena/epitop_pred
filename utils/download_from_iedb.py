import urllib.request
import pickle
import pandas as pd

"""
Download "all" entries of emble
"""


# IDArray = get_accessions()
length = 492000
negative_samples = []
positive_samples = []
not_working = []

samples = {}
df = pd.read_csv("/home/go96bix/projects/raw_data/bcell_full_v3.csv",",",skiprows=1)

for i, line in df.iterrows():
	if line["Object Type"] != "Linear peptide":
		continue
	else:
		# url = f'http://www.iedb.org/epitope/{i}'
		url = line["Epitope IRI"]
		j = url.split("/")[-1]
		response = urllib.request.urlopen(url)
		data = response.read()  # a `bytes` object
		page_source = str(data)

		if '404 Error' in page_source:
			not_working.append(j)
			continue

		# find b_cell_assays part
		start = page_source.find('"type":"bcell"')
		stop = page_source[start::].find("]")
		b_cell_assays = page_source[start:start + stop]
		parameters = b_cell_assays.split(",")

		# count positive assays
		pos_counts = sum([int(i.split(":")[1].strip('"')) for i in parameters if i.startswith('"positive_count"')])
		# count negative assays
		total_counts = sum([int(i.split(":")[1].strip('}').strip('"')) for i in parameters if i.startswith('"total_count"')])

		samples.update({int(j):(pos_counts,total_counts)})

		if total_counts < 2:
			continue
		elif pos_counts == 0:
			negative_samples.append(str(j)+"\n")
		elif pos_counts >= 2:
			positive_samples.append(str(j)+"\n")
		else:
			print(f"{pos_counts}/{total_counts}")

		print(f"\t===== {i+1} / {length} -- {int(((i+1) / length)*100)}% =====") #, end='\r')

pickle.dump(samples, open("all_samples.pkl","wb"))

with open("negative_samples.txt", "w") as out:
	out.writelines(negative_samples)

with open("positive_samples.txt", "w") as out:
	out.writelines(positive_samples)

print(f"not working samples {not_working}")