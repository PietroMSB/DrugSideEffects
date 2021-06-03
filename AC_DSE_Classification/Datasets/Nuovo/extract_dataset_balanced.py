#coding=utf-8
import sys
import os
import numpy as np
import pandas
import pickle

#parameters
side_effects_occurrences_minimum = 100
term_type = "PT" #can be either primary (PT) or lowest level (LLT)
drugs_without_pubchem_features = [143, 146, 1125, 2022, 2094, 2182, 2818, 3043, 3405, 3454, 4585, 4725, 5212, 5647, 60843, 64147, 104758, 110634, 151165, 153941, 219090, 5281007, 5353894, 5353980, 5361912, 5362070, 5381226, 5462337, 5487301, 5488383, 5493381, 6323497, 6918366, 6918430, 9825285, 9846180, 11238823, 11505907, 25880656, 25880664, 40468184, 51508717, 51601240]

#acquire data
ppi_a = pandas.read_table("PPI_HuRI/HI-union.tsv")
dpi_a = pandas.read_table("STITCH/dpi_preprocessed.tsv", dtype=np.unicode_)
dsa_aaa = pandas.read_table("SIDER/meddra_all_se.tsv")

#preprocess dsa table
print("Preprocessing DSA table: selecting "+term_type+" entries and deleting duplicates")
row_ind_a = dsa_aaa.values[:,3]==term_type
dsa_aa = dsa_aaa[row_ind_a]
#delete duplicate entries
row_ind_aa = list()
dsa_dict = dict()
for i in range(dsa_aa.values.shape[0]):
	if dsa_aa.values[i][0] not in dsa_dict.keys():
		dsa_dict[dsa_aa.values[i][0]] = [dsa_aa.values[i][4]]
		row_ind_aa.append(True)
	elif dsa_aa.values[i][4] in dsa_dict[dsa_aa.values[i][0]]:
		row_ind_aa.append(False)
	else:
		dsa_dict[dsa_aa.values[i][0]].append(dsa_aa.values[i][4])
		row_ind_aa.append(True)
dsa_a = dsa_aa[row_ind_aa]
print("Removed "+str(len(row_ind_aa)-sum(row_ind_aa))+" duplicate entries")

#print table shapes
print("PPI table shape: "+str(ppi_a.values.shape))
print("DPI table shape: "+str(dpi_a.values.shape))
print("DSA table shape: "+str(dsa_a.values.shape))

#retrieve unique protein drug and side effect ids
unique_proteins = np.unique(dpi_a.values[:,1])
unique_drugs_dpi = np.unique(dpi_a.values[:,0])
unique_drugs_dsa = np.unique(dsa_a.values[:,0])
unique_se, counts = np.unique(dsa_a.values[:,4], return_counts=True)

### DPI PREPROCESSING START ###
'''
#load translator table
tt = pandas.read_table("STITCH/mart_export.txt")
translator = dict()
for i in range(tt.values.shape[0]):
	pp = "ENSP"+tt.values[i][1][4:]
	translator[pp] = tt.values[i][0]

#convert dpi table from ENSP to ENSG and from CID to PubChem ID
dpi_b = list()
for i in dpi_a.index:
	print("Analyzing DPI "+str(i+1)+" of "+(str(len(dpi_a.index))), end='\r')
	if dpi_a.at[i,'protein'][5:] in translator.keys():
		dpi_b.append([dpi_a.at[i,'chemical'][4:], translator[dpi_a.at[i,'protein'][5:]], dpi_a.at[i,'combined_score']])
dpi_b = np.array(dpi_b)
print("")

#save new version of dpi_a
print("Saving "+str(dpi_b.shape[0])+" preproceseed drug protein interactions")
np.savetxt("STITCH/dpi_preprocessed.tsv", dpi_b, fmt='%s', delimiter='\t', header="chemical\tprotein\tcombined_score")
sys.exit()
'''
### DPI PREPROCESSING STOP ###

### DEBUG DPI DRUG ID START ###
'''
print(dpi_a.values[8000000][0])
print(str(dpi_a.values[8000000][0]))
sys.exit()
'''
### DEBUG DPI DRUG ID STOP ###

#print number of unique entities in each table
print("Unique proteins: "+str(len(unique_proteins)))
print("Unique side-effects: "+str(len(unique_se)))
print("Unique drugs in dpi: "+str(len(unique_drugs_dpi)))
print("Unique drugs in dsa: "+str(len(unique_drugs_dsa)))

#select side effects with drug occurrences greater than or equal to <side_effects_occurrences_minimum>
print("Selecting side-effects with a minimum of "+str(side_effects_occurrences_minimum)+" occurrences")
discarded = 0
final_se = []
for i in range(len(unique_se)):
	if counts[i] >= side_effects_occurrences_minimum:
		final_se.append(unique_se[i])
	else:
		discarded += 1
print("Removed "+str(discarded)+" side-effects")

#remove dsa of the se removed due to insufficient occurrences
print("Removing drug-Side effect associations of removed side-effects")
del_indices = []
for i in range(dsa_a.values.shape[0]):
	if dsa_a.values[i][4] not in final_se:
		del_indices.append(i)
dsa_b = np.delete(dsa_a.values, del_indices, axis=0)
print("Removed "+str(len(del_indices))+" drug-side-effect associations")

#select drugs with at least one side effect and at least one dpi
print("Selecting drugs with at least one side effect after filtering, and at least one DPI")
final_drugs = []
discarded = 0
for d in unique_drugs_dsa:
	if d[4:] not in unique_drugs_dpi:
		discarded += 1
		continue
	if int(d[4:]) in drugs_without_pubchem_features:
		discarded += 1
		continue
	stop = False
	i = 0
	while i < dsa_b.shape[0] and not stop:
		if d == dsa_b[i][0]:
			stop = True
			final_drugs.append(d)
		i = i+1
	if not stop:
		discarded += 1
print("Removed "+str(discarded)+" drugs")

#select dsa of the drugs in <final_drugs>
print("Removing drug-Side effect associations of removed drugs")
del_indices = []
for i in range(dsa_b.shape[0]):
	if (dsa_b[i][0] not in final_drugs):
		del_indices.append(i)
final_dsa = np.delete(dsa_b, del_indices, axis=0)
print("Removed "+str(len(del_indices))+" drug-side-effect associations")

#filter dpi list using only drugs in <final_drugs>
print("Removing DPIs of molecules that are outside of this study or that were excluded by filters")
del_indices = []
for i in range(dpi_a.values.shape[0]):
	print("Processing DPI "+str(i+1)+" of "+str(dpi_a.values.shape[0]), end = '\r')
	if ("CID1"+dpi_a.values[i][0] not in final_drugs):
		del_indices.append(i)
final_dpi = np.delete(dpi_a.values, del_indices, axis=0)
print("")
print("Removed "+str(len(del_indices))+" DPIs")

#select proteins with at least one dpi
print("Selecting proteins with at least one DPI")
final_proteins = []
discarded = 0
for p in unique_proteins:
	stop = False
	i = 0
	while i < final_dpi.shape[0] and not stop:
		if p == final_dpi[i][1]:
			stop = True
			final_proteins.append(p)
		i = i+1
	if not stop:
		discarded += 1
print("Removed "+str(discarded)+" proteins")

#obtain filtered ppi list
("Removing PPIs of removed proteins")
del_indices = []
for i in range(ppi_a.values.shape[0]):
	print("Processing PPI "+str(i+1)+" of "+str(ppi_a.values.shape[0]), end = '\r')
	if (ppi_a.values[i][0] not in final_proteins) or (ppi_a.values[i][1] not in final_proteins):
		del_indices.append(i)
final_ppi =  np.delete(ppi_a.values, del_indices, axis=0)
print("")
print("Removed "+str(len(del_indices))+" PPIs")

#list numbers of instances
print("")
print("Drugs: "+str(len(final_drugs)))
print("Proteins: "+str(len(final_proteins)))
print("Side Effects: "+str(len(final_se)))
print("Dpi: "+str(final_dpi.shape[0]))
print("Ppi: "+str(final_ppi.shape[0]))
print("Drug-SE links: "+str(final_dsa.shape[0]))
print("Total predictions: "+str(len(final_drugs)*len(final_se)))

#save output files
out_file = open("Output/drugs.pkl", 'wb')
pickle.dump(final_drugs, out_file)
out_file.close()
out_file = open("Output/genes.pkl", 'wb')
pickle.dump(final_proteins, out_file)
out_file.close()
out_file = open("Output/side_effects.pkl", 'wb')
pickle.dump(final_se, out_file)
out_file.close()
out_file = open("Output/drug_gene_links.pkl", 'wb')
pickle.dump(final_dpi, out_file)
out_file.close()
out_file = open("Output/gene_gene_links.pkl", 'wb')
pickle.dump(final_ppi, out_file)
out_file.close()
out_file = open("Output/drug_side_effect_links.pkl", 'wb')
pickle.dump(final_dsa, out_file)
out_file.close()
