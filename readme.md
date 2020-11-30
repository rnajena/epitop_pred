## EpiDope creation space

We have numerous scripts with specific niche functionality. Therefore, we expect that most of the scripts are not of high interest for most users. 
Because of this, we only rudimentarily polished most of the code and do not guaranty it's functionality.

### possible usage

###### get the raw data

download from
`http://www.iedb.org/bcelldetails_v3.php`  
export to csv file

###### download positive and negative samples
`utils/download_from_iedb.py`  
(change local variables like line 17 (path to csv file))

`utils/download_proteins_from_epitopeNumber.py` 
(change local variables like line 13 (path to csv file))

###### curate data
`curate_iedb_linear_epitopes.py`  
(again changing input path)

###### make cluster by different sequence identity:

`cd previous_output_dir`  
`cat * >> protein_all.fasta`

`cd-hit -i protein_all.fasta  -c 1 -o 1_seqID.fasta`  
`cd-hit -i protein_all.fasta  -c 0.9 -o 0.9_seqID.fasta`  
`cd-hit -i protein_all.fasta  -c 0.8 -o 0.8_seqID.fasta`  
`cd-hit -i protein_all.fasta  -c 0.7 -o 0.7_seqID.fasta`  
`cd-hit -i protein_all.fasta  -n 4 -c 0.6 -o 0.6_seqID.fasta`  
`cd-hit -i protein_all.fasta  -n 3 -c 0.5 -o 0.5_seqID.fasta`

select proteins with most verified regions  
`utils/cluster_to_proteins_with_markings.py`

###### generate training, test, val set
simple clustered  
`generate_binary_clustered_training_sets.py`

more complex, if your data is clustered twice (like in the paper explained), to reduce bias of similar sequences in the test set.   
`generate_binary_double_clustered_training_sets.py`

###### train and test the models
`train_DL.py` trains multiple neural networks on your training data  
`epidope.py` testing suit for trained your models 

###### further
make multi fasta file with only test set entries  
`utils/filter_test_set_fastas.py`

get the ROC precision-recall und distribution of predictions:  
`utils/make_ROC_curves.py`

get plots with only the parts marked which are part of ROC  
`utils/plots_test_set.py`