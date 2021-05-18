#coding=utf-8

import sys
import os
import math
import pickle
import numpy as np
import pandas
import tensorflow as tf
import scipy
import pickle

from sklearn import preprocessing as skl_pre
from GNNv3.GNN.CGNN.CGNN import *
from GNNv3.GNN.CGNN.composite_graph_class import CompositeGraphObject
from GNNv3.GNN import GNN_utils as utils

#network parameters
TRAINING_BATCHES = 10        #number of batches in which the training set should be split

#script parameters
path_data = "Datasets/Nuovo/Output/Soglia_100/"
path_results = "Datasets/Nuovo/Transduction/Soglia_100/"
splitting_seed = 920305
validation_share = 0.1
test_share = 0.1
atomic_number = {'Li':3, 'B':5, 'C':6, 'N':7, 'O':8, 'F':9, 'Mg':12, 'Al':13, 'P':15, 'S':16, 'Cl':17, 'K':19, 'Ca':20, 'Fe':26, 'Co':27, 'As':33, 'Br':35, 'I':53, 'Au':79 }
atomic_label = {3:'Li', 5:'B', 6:'C', 7:'N', 8:'O', 9:'F', 12:'Mg', 13:'Al', 15:'P', 16:'S', 17:'Cl', 19:'K', 20:'Ca', 26:'Fe', 27:'Co', 33:'As', 35:'Br', 53:'I', 79:'Au' }
label_translator = {'C':1, 'N':2, 'O':3, 'S':4, 'F':5, 'P':6, 'Cl':7, 'I':7, 'Br':7, 'Ca':8, 'Mg':8, 'K':8, 'Li':8, 'Co':8, 'As':8, 'B':8, 'Al':8, 'Au':8, 'Fe':8}

#batch building function
def BuildBatch(links_dg, links_gg, drugs, side_effects, pos_boolean_array, node_number, nodes, type_mask, known_indices, prediction_indices):
	#calculate number of edges
	n_edges = 2*links_dg.shape[0]+2*links_gg.shape[0]+2*len(np.where(pos_boolean_array[known_indices]==1)[0])+len(prediction_indices)
	#build edge mask
	edge_mask = np.concatenate((np.zeros(2*links_dg.shape[0]+2*links_gg.shape[0]+2*len(np.where(pos_boolean_array[known_indices]==1)[0]), dtype=int), np.ones(len(prediction_indices), dtype=int)), axis=0)
	#build target tensor
	targets = np.concatenate( ( np.zeros((len(prediction_indices), 1)), np.ones((len(prediction_indices), 1)) ), axis=1) # [1,0] -> positive ; [0,1] -> negative (creates a vector of default negative targets, then modify positive ones only)
	for i in range(len(prediction_indices)):
		if pos_boolean_array[prediction_indices[i]] == 1:
			targets[i][0] = 1
			targets[i][1] = 0
	#build arcs tensor
	arcs = np.zeros((n_edges,2), dtype=int)
	l = 0
	#add drug-gene edges
	for i in range(links_dg.shape[0]):
		arcs[l][:] = [node_number[str(int(links_dg[i][0]))],node_number[str(links_dg[i][1])]]
		arcs[l+1][:] = [node_number[str(links_dg[i][1])],node_number[str(int(links_dg[i][0]))]]
		l = l+2
	#add gene-gene edges
	for i in range(links_gg.shape[0]):
		arcs[l][:] = [node_number[str(links_gg[i][0])],node_number[str(links_gg[i][1])]]
		arcs[l+1][:] = [node_number[str(links_gg[i][1])],node_number[str(links_gg[i][0])]]
		l = l+2
	#add drug-se edges
	for k in known_indices:
		#only process positive examples in the known subset
		if pos_boolean_array[k] == 0: continue
		#calculate drug index (i) and side effect index (j) 
		i = int(k/len(side_effects))
		j = k - i*len(side_effects) 
		#add an edge in both directions
		arcs[l][:] = [node_number[str(side_effects[j])], node_number[str(drugs[i])]]
		arcs[l+1][:] = [node_number[str(drugs[i])], node_number[str(side_effects[j])]]
		l = l+2
	#add edges to predict
	for k in prediction_indices:
		#calculate drug index (i) and side effect index (j) 
		i = int(k/len(side_effects))
		j = k - i*len(side_effects) 
		#add edge
		arcs[l][:] = [node_number[str(drugs[i])], node_number[str(side_effects[j])]]
		l = l+1
	arcs = np.array(arcs)
	#build set mask
	set_mask = np.ones(len(prediction_indices), dtype=int)
	#concatenate a zero mask for all the non-targeted edges to each set mask
	set_mask = np.concatenate((np.zeros(n_edges-len(prediction_indices), dtype=int), set_mask), axis=0)
	#concatenate a placeholder [0,0]xN vector of targets for non-output nodes
	expanded_targets = np.concatenate((np.zeros((n_edges-len(prediction_indices), 2), dtype=int), targets), axis=0)
	#build CompositeGraphObject
	return CompositeGraphObject(arcs, nodes, expanded_targets, 'a', set_mask, edge_mask, type_mask, node_aggregation='average')

#load side-effect data
print("Loading side-effect data")
in_file = open(path_data+"side_effects.pkl", 'rb')
side_effects = pickle.load(in_file)
in_file.close()
#load gene data
print("Loading gene data")
in_file = open(path_data+"genes.pkl", 'rb')
genes = pickle.load(in_file)
in_file.close()
#load drug data
print("Loading drug data")
in_file = open(path_data+"drugs.pkl", 'rb')
drugs_pre = pickle.load(in_file)
in_file.close()
#load drug-side effect links
print("Loading drug-side-effect associations")
in_file = open(path_data+"drug_side_effect_links.pkl", 'rb')
links_dse = pickle.load(in_file)
in_file.close()
#load gene-gene links
print("Loading gene interaction network")
in_file = open(path_data+"gene_gene_links.pkl", 'rb')
links_gg = pickle.load(in_file)
in_file.close()
#load drug-gene links
print("Loading drug target interactions")
in_file = open(path_data+"drug_gene_links.pkl", 'rb')
links_dg = pickle.load(in_file)
in_file.close()
#load drug features
print("Loading drug features")
pubchem_data = pandas.read_csv(path_data+"pubchem_output.csv")

#preprocess drug ids
print("Preprocessing drug IDs")
drugs = list()
for i in range(len(drugs_pre)):
	drugs.append(str(int(drugs_pre[i][4:])))

#determine graph dimensions
print("Determining graph dimensions")
n_nodes = len(drugs)+len(side_effects)+len(genes)
n_edges = 2*links_dg.shape[0]+2*links_gg.shape[0]+2*len(drugs)*len(side_effects)
dim_node_label = 7
type_mask = np.zeros((n_nodes,3), dtype=int)

#build id -> node number mappings and node number -> id mappings
print("Mapping entity IDs to node numbers")
node_number = dict()
reverse_id = dict()
for i in range(len(drugs)):
	node_number[str(drugs[i])] = i
	reverse_id[i] = str(drugs[i])
	type_mask[i][0] = 1
for i in range(len(side_effects)):
	node_number[str(side_effects[i])] = i + len(drugs)
	reverse_id[i+len(drugs)] = str(side_effects[i]) 
	type_mask[i + len(drugs)][1] = 1
for i in range(len(genes)):
	node_number[str(genes[i])] = i + len(drugs) + len(side_effects)
	reverse_id[i+len(drugs)+len(side_effects)] = str(genes[i])
	type_mask[i + len(drugs) + len(side_effects)][2] = 1

#build list of positive examples
print("Building list of positive link prediction examples")
positive_dsa_list = list()
for i in range(links_dse.shape[0]):
	if str(int(links_dse[i][0][4:])) in node_number.keys():
		positive_dsa_list.append((node_number[str(int(links_dse[i][0][4:]))],node_number[links_dse[i][4]]))
	else:
		sys.exit("ERROR: drug-side-effect link pointing to incorrect drug id")

#build node feature matrix
print("Processing node features")
nodes = np.zeros((n_nodes, dim_node_label))
for i in pubchem_data.index:
	#skip drugs that have been eliminated
	if str(pubchem_data.at[i,'cid']) not in node_number.keys():
		continue
	nn = node_number[str(pubchem_data.at[i,'cid'])]
	nodes[nn][0] = float(pubchem_data.at[i,'mw'])#molecular weight
	nodes[nn][1] = float(pubchem_data.at[i,'polararea'])#polar area
	nodes[nn][2] = float(pubchem_data.at[i,'xlogp'])#log octanal/water partition coefficient
	nodes[nn][3] = float(pubchem_data.at[i,'heavycnt'])#heavy atom count
	nodes[nn][4] = float(pubchem_data.at[i,'hbonddonor'])#hydrogen bond donors
	nodes[nn][5] = float(pubchem_data.at[i,'hbondacc'])#hydrogen bond acceptors
	nodes[nn][6] = float(pubchem_data.at[i,'rotbonds'])#number of rotatable bonds
#normalize features
scaler = skl_pre.MinMaxScaler()
scaler.fit(nodes[:len(drugs),3:])
nodes[:len(drugs),3:] = scaler.transform(nodes[:len(drugs),3:])

#calculate dataset dimensions
print("Calculating dataset dimensions")
total_examples = len(drugs)*len(side_effects)
positive_examples = len(positive_dsa_list)
negative_examples = total_examples - positive_examples

#split the dataset
print("Splitting the dataset")
validation_size_pos = int(validation_share*positive_examples)
validation_size_neg = int(validation_share*negative_examples)
validation_size = validation_size_pos + validation_size_neg
test_size_pos = int(test_share*positive_examples)
test_size_neg = int(test_share*negative_examples)
test_size = test_size_pos + test_size_neg
#define global index
index = np.array(list(range(total_examples)))
#extract positive and negative indices from the global index, by looking in the positive_dsa_list
index_pos = []
index_neg = list(index)
pos_boolean_array = np.zeros(total_examples)
i = 0
for p in positive_dsa_list:
	#each drug d has a block of len(side_effects) indices starting at (node_number[d]*len(side_effects)). The single side effect s has an offset equal to node_number[s]-len(drugs) inside this block (len(drugs) is subtracted because side effect node numbers are after drug node numbers, so that side effect #3 will have node number equal to len(drugs)+3)
	k = p[0]*len(side_effects) + p[1]-len(drugs)
	print("Processing DSA "+str(i+1)+" of "+str(len(positive_dsa_list)), end='\r')
	i=i+1
	### DEBUG START ###
	'''
	print("")
	print(k)
	print(str(p[0])+"   "+reverse_id[p[0]])
	print(str(p[1])+"   "+reverse_id[p[1]])
	'''
	### DEBUG STOP ###
	index_pos.append(k)
	index_neg.remove(k)
	pos_boolean_array[k] = 1
print("")
#check correctness of results
if len(index_pos) != positive_examples:
	sys.exit("ERROR : found "+str(len(index_pos))+" positive examples, but expected "+str(positive_examples))
if len(index_neg) != negative_examples:
	sys.exit("ERROR : found "+str(len(index_neg))+" negative examples, but expected "+str(negative_examples))
#shuffle indices
index_pos = np.array(index_pos)
index_neg = np.array(index_neg)
np.random.shuffle(index_pos)
np.random.shuffle(index_neg)
test_index = np.concatenate( (index_pos[:test_size_pos], index_neg[:test_size_neg]) )
validation_index = np.concatenate( (index_pos[test_size_pos:test_size_pos+validation_size_pos], index_neg[test_size_neg:test_size_neg+validation_size_neg]) )
#split the training set
training_index_pos = index_pos[test_size_pos+validation_size_pos:]
training_index_neg = index_neg[test_size_neg+validation_size_neg:]
tr_batch_sizes_pos = [int(float(len(training_index_pos))/TRAINING_BATCHES) for i in range(TRAINING_BATCHES)]
tr_batch_sizes_neg = [int(float(len(training_index_neg))/TRAINING_BATCHES) for i in range(TRAINING_BATCHES)]
tr_batch_sizes = [tr_batch_sizes_pos[i]+tr_batch_sizes_neg[i] for i in range(TRAINING_BATCHES)]
i = 0
while sum(tr_batch_sizes_pos)<len(training_index_pos):
	tr_batch_sizes_pos[i] = tr_batch_sizes_pos[i] + 1
	i = i + 1
i = 0
while sum(tr_batch_sizes_neg)<len(training_index_neg):
	tr_batch_sizes_neg[i] = tr_batch_sizes_neg[i] + 1
	i = i + 1
training_batch_indices_pos = [[training_index_pos[j+sum(tr_batch_sizes_pos[:i])] for j in range(tr_batch_sizes_pos[i])] for i in range(TRAINING_BATCHES)]
training_batch_indices_neg = [[training_index_neg[j+sum(tr_batch_sizes_neg[:i])] for j in range(tr_batch_sizes_neg[i])] for i in range(TRAINING_BATCHES)]
training_batch_indices = [np.concatenate( (training_batch_indices_pos[i], training_batch_indices_neg[i]) ) for i in range(TRAINING_BATCHES)]

#build validation batch
print("Building validation batch")
validation_batch = BuildBatch(links_dg, links_gg, drugs, side_effects, pos_boolean_array, node_number, nodes, type_mask, np.concatenate(tuple(training_batch_indices)), validation_index)
#save validation batch
out_file = open(path_results+"batch_validation.pkl", 'wb')
pickle.dump(validation_batch, out_file)
out_file.close()
del validation_batch

#build test batch
print("Building test batch")
test_batch = BuildBatch(links_dg, links_gg, drugs, side_effects, pos_boolean_array, node_number, nodes, type_mask, np.concatenate ((np.concatenate(tuple(training_batch_indices)), validation_index)), test_index)
#save test batch
out_file = open(path_results+"batch_test.pkl", 'wb')
pickle.dump(test_batch, out_file)
out_file.close()
del test_batch

#build training batches
for i in range(TRAINING_BATCHES):
	print("Building training batch "+str(i+1)+" of "+str(TRAINING_BATCHES), end="\r")
	#add the list of indices for each training batch but the i-th one to the list of known indices
	known_indices = np.zeros((0,), dtype=int)
	for j in range(TRAINING_BATCHES):
		if i == j: continue
		known_indices = np.concatenate((known_indices, training_batch_indices[j]))
	#build i-th training batch
	ith_training_batch = BuildBatch(links_dg, links_gg, drugs, side_effects, pos_boolean_array, node_number, nodes, type_mask, known_indices, training_batch_indices[i])
	#save i-th training batch
	out_file = open(path_results+"batch_training_"+str(i)+".pkl", 'wb')
	pickle.dump(ith_training_batch, out_file)
	out_file.close()
print("")	

#terminate execution
print("Execution completed correctly")

