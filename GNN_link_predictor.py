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

from GNNv3.GNN.CGNN.CGNN import *
from GNNv3.GNN.CGNN.composite_graph_class import CompositeGraphObject

#network parameters
CLASSES = 1					#number of outputs
EPOCHS = 500                #number of training epochs
STATE_DIM = 10				#node state dimension
LR = 0.001					#learning rate
THRESHOLD = 0.001			#state convergence threshold, in terms of relative state difference
MAX_ITER = 6				#maximum number of state convergence iterations
VALIDATION_INTERVAL = 10	#interval between two validation checks, in training epochs
TRAINING_BATCHES = 1        #number of batches in which the training set should be split

#script parameters
run_id = sys.argv[1]
path_data = "Datasets/Nuovo/Output/Soglia_100/"
path_results = "Results/Nuovo/LinkPredictor/"+run_id+".txt"
splitting_seed = 920305
validation_share = 0.1
test_share = 0.1
atomic_number = { 'Li':3, 'B':5, 'C':6, 'N':7, 'O':8, 'F':9, 'Mg':12, 'Al':13, 'P':15, 'S':16, 'Cl':17, 'K':19, 'Ca':20, 'Fe':26, 'Co':27, 'As':33, 'Br':35, 'I':53, 'Au':79 }
atomic_label = { 3:'Li', 5:'B', 6:'C', 7:'N', 8:'O', 9:'F', 12:'Mg', 13:'Al', 15:'P', 16:'S', 17:'Cl', 19:'K' ,20:'Ca', 26:'Fe', 27:'Co', 33:'As', 35:'Br', 53:'I', 79:'Au' }
label_translator = {'C':1, 'N':2, 'O':3, 'S':4, 'F':5, 'P':6, 'Cl':7, 'I':7, 'Br':7, 'Ca':8, 'Mg':8, 'K':8, 'Li':8, 'Co':8, 'As':8, 'B':8, 'Al':8, 'Au':8, 'Fe':8}

#function that translates a nx_graph into a graph_object
def NXtoGO(nx_graph, target):
	nodes = np.zeros((len(nx_graph.nodes), 8))
	arcs = list()
	targets = np.reshape(target, (1,10))
	for i in range(len(nx_graph.nodes)):
		nodes[i][label_translator[nx_graph.nodes[i]['info']]-1] = 1
	for n1,n2 in nx_graph.edges:
		label = [n1, n2, 0, 0, 0, 0]
		label[nx_graph.edges[n1,n2]['info']+1] = 1
		arcs.append(label)
		label = [n2, n1, 0, 0, 0, 0]
		label[nx_graph.edges[n1,n2]['info']+1] = 1
		arcs.append(label)
	#skip graphs without edges
	if not arcs:
		return None
	arcs = np.array(arcs)
	node_graph = np.ones((nodes.shape[0],1))
	return GraphObject(arcs,nodes,targets,'g',NodeGraph=node_graph)

#load side-effect data
in_file = open(path_data+"side_effects.pkl", 'rb')
side_effects = pickle.load(in_file)
in_file.close()
#load gene data
in_file = open(path_data+"genes.pkl", 'rb')
genes = pickle.load(in_file)
in_file.close()
#load drug data
in_file = open(path_data+"drugs.pkl", 'rb')
drugs_pre = pickle.load(in_file)
in_file.close()
#load drug-side effect links
in_file = open(path_data+"drug_side_effect_links.pkl", 'rb')
links_dse = pickle.load(in_file)
in_file.close()
#load gene-gene links
in_file = open(path_data+"gene_gene_links.pkl", 'rb')
links_gg = pickle.load(in_file)
in_file.close()
#load drug-gene links
in_file = open(path_data+"drug_gene_links.pkl", 'rb')
links_dg = pickle.load(in_file)
in_file.close()
#load drug features
pubchem_data = pandas.read_csv(path_data+"pubchem_output.csv")

#preprocess drug ids
drugs = list()
for i in range(len(drugs_pre)):
	drugs.append(str(int(drugs_pre[i][4:])))

#determine graph dimensions
n_nodes = len(drugs)+len(side_effects)+len(genes)
n_edges = 2*links_dg.shape[0]+2*links_gg.shape[0]+2*len(drugs)*len(side_effects)
dim_node_label = 7
type_mask = np.zeros((n_nodes,3), dtype=int)
#build id -> node number mappings
node_number = dict()
for i in range(len(drugs)):
	node_number[str(drugs[i])] = i
	type_mask[i][0] = 1
for i in range(len(side_effects)):
	node_number[str(side_effects[i])] = i + len(drugs) 
	type_mask[i + len(drugs)][1] = 1
for i in range(len(genes)):
	node_number[str(genes[i])] = i + len(drugs) + len(side_effects)
	type_mask[i + len(drugs) + len(side_effects)][2] = 1

#build list of positive examples
positive_dsa_list = list()
for i in range(links_dse.shape[0]):
	if str(int(links_dse[i][0][4:])) in node_number.keys():
		positive_dsa_list.append((node_number[str(int(links_dse[i][0][4:]))],node_number[links_dse[i][2]]))
	else:
		sys.exit("ERROR: drug-side-effect link pointing to incorrect drug id")

#build node feature matrix
nodes = np.zeros((n_nodes, dim_node_label))
for i in pubchem_data.index:
	nn = node_number[str(pubchem_data.at[i,'cid'])]
	nodes[nn][0] = float(pubchem_data.at[i,'mw'])#molecular weight
	nodes[nn][1] = float(pubchem_data.at[i,'polararea'])#polar area
	nodes[nn][2] = float(pubchem_data.at[i,'xlogp'])#log octanal/water partition coefficient
	nodes[nn][3] = float(pubchem_data.at[i,'heavycnt'])#heavy atom count
	nodes[nn][4] = float(pubchem_data.at[i,'hbonddonor'])#hydrogen bond donors
	nodes[nn][5] = float(pubchem_data.at[i,'hbondacc'])#hydrogen bond acceptors
	nodes[nn][6] = float(pubchem_data.at[i,'rotbonds'])#number of rotatable bonds

#build adjacency matrix, edge mask, output edges, and edge targets
edge_mask = np.concatenate((np.zeros(2*links_dg.shape[0]+2*links_gg.shape[0]+len(drugs)*len(side_effects), dtype=int), np.ones(len(drugs)*len(side_effects), dtype=int)), axis=0)
arcs = list()
targets = np.zeros(len(drugs)*len(side_effects))
#add drug-gene edges
for i in range(links_dg.shape[0]):
	arcs.append([node_number[str(int(links_dg[i][0]))],node_number[str(links_dg[i][1])]])
	arcs.append([node_number[str(links_dg[i][1])],node_number[str(int(links_dg[i][0]))]])
#add gene-gene edges
for i in range(links_gg.shape[0]):
	arcs.append([node_number[str(links_gg[i][0])],node_number[str(links_gg[i][1])]])
	arcs.append([node_number[str(links_gg[i][1])],node_number[str(links_gg[i][0])]])
#add drug-se edges
for i in range(len(drugs)):
	for j in range(len(side_effects)):
		#side-effect-drug links only exist for message passing
		arcs.append([node_number[str(side_effects[j])], node_number[str(drugs[i])]])
k = 0
for i in range(len(drugs)):
	for j in range(len(side_effects)):
		print("Preprocessing Possible Drug-Side Effect Link "+str(i*len(drugs)+j+1)+" of "+str(len(drugs)*len(side_effects)), end='\r')
		#drug-side-effect links are the ones on which the prediction is carried out
		arcs.append([node_number[str(drugs[i])],node_number[str(side_effects[j])]])
		if (i,j) in positive_dsa_list:
			targets[k] = 1
		k+=1
print("")
arcs = np.array(arcs)
	
#split the dataset
validation_size = int(validation_share*len(targets))
test_size = int(test_share*len(targets))
index = np.array(list(range(len(targets))))
np.random.shuffle(index)
test_index = index[:test_size]
validation_index = index[test_size:test_size+validation_size]
training_index = index[test_size+validation_size:]
te_mask = np.zeros(len(targets), dtype=int)
va_mask = np.zeros(len(targets), dtype=int)
tr_mask = np.zeros(len(targets), dtype=int)
for i in test_index:
	te_mask[i] = 1
for i in validation_index:
	va_mask[i] = 1
for i in training_index:
	tr_mask[i] = 1

#build CompositeGraphObject
tr_graph = CompositeGraphObject(arcs, nodes, targets, 'a', tr_mask, edge_mask, type_mask, node_aggregation='average')
va_graph = CompositeGraphObject(arcs, nodes, targets, 'a', va_mask, edge_mask, type_mask, node_aggregation='average')
te_graph = CompositeGraphObject(arcs, nodes, targets, 'a', te_mask, edge_mask, type_mask, node_aggregation='average')

#build network
netSt_drugs = utils.MLP(input_dim=STATE_DIM+nodes.shape[1], layers=[10, 15], activations=['relu', 'relu'],
                     kernel_initializer=[tf.keras.initializers.GlorotNormal() for i in range(2)],
                     bias_initializer=[tf.keras.initializers.GlorotNormal() for i in range(2)],
                     dropout_percs=[0.2, 0],
                     dropout_pos=[0, 0])
netSt_genes = utils.MLP(input_dim=STATE_DIM+nodes.shape[1], layers=[10, 15], activations=['relu', 'relu'],
                     kernel_initializer=[tf.keras.initializers.GlorotNormal() for i in range(2)],
                     bias_initializer=[tf.keras.initializers.GlorotNormal() for i in range(2)],
                     dropout_percs=[0.2, 0],
                     dropout_pos=[0, 0])
netSt_sideeffects = utils.MLP(input_dim=STATE_DIM+nodes.shape[1], layers=[10, 15], activations=['relu', 'relu'],
                     kernel_initializer=[tf.keras.initializers.GlorotNormal() for i in range(2)],
                     bias_initializer=[tf.keras.initializers.GlorotNormal() for i in range(2)],
                     dropout_percs=[0.2, 0],
                     dropout_pos=[0, 0])
netOut = utils.MLP(input_dim=2*STATE_DIM, layers=[20,1], activations=['relu', 'linear'],
                      kernel_initializer=[tf.keras.initializers.GlorotNormal() for i in range(2)],
                      bias_initializer=[tf.keras.initializers.GlorotNormal() for i in range(2)],
                      dropout_percs=[0, 0],
                      dropout_pos=[0, 0])
model = GNNedgeBased([netSt_drugs, netSt_sideeffects, netSt_genes], netOut, optimizer = tf.keras.optimizers.Adam(LR), loss_function = tf.nn.binary_cross_entropy, state_vect_dim = 0, max_iteration=MAX_ITER, threshold=THRESHOLD, addressed_problem='c', loss_arguments=None, output_activation=tf.math.sigmoid)

#train the network
model.fit(tr_graph, EPOCHS, va_graph)

#evaluate the network
iterations, loss, targets, outputs = model.evaluate_single_graph(te_graph, class_weights=[1 for i in range(CLASSES)], training=False)

#calculate results
TP = [0 for j in range(CLASSES)]
TN = [0 for j in range(CLASSES)]
FP = [0 for j in range(CLASSES)]
FN = [0 for j in range(CLASSES)]
for i in range(targets.shape[0]):
	for j in range(CLASSES):
		if targets[i][j] > 0.5:
			if outputs[i][j] > 0.5: TP[j] += 1
			else: FN[j] += 1
		else:
			if outputs[i][j] > 0.5: FP[j] += 1
			else: TN[j] += 1
accuracy = [ float(TP[j]+TN[j])/float(TP[j]+TN[j]+FP[j]+FN[j])  for j in range(CLASSES)]
precision = [ float(TP[j])/float(TP[j]+FP[j]) if TP[j]+FP[j] > 0 else 0.0 for j in range(CLASSES)]
recall = [ float(TP[j])/float(TP[j]+FN[j]) if TP[j]+FP[j] > 0 else 0.0 for j in range(CLASSES)]
global_accuracy = float(sum(TP)+sum(TN))/float(sum(TP)+sum(TN)+sum(FP)+sum(FN))

print("Class Precision:")
print(precision)
print("")

print("Class Recall:")
print(recall)
print("")

print("Class Accuracy:")
print(accuracy)
print("")

print("Global Accuracy:\n"+str(global_accuracy))


