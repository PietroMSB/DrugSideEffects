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
from GNNv3.GNN import GNN_utils as utils

#network parameters
CLASSES = 2					#number of outputs
EPOCHS = 500                #number of training epochs
STATE_DIM = 10				#node state dimension
STATE_INIT_STDEV = 0.1		#standard deviation of random state initialization
LR = 0.01					#learning rate
MAX_ITER = 6				#maximum number of state convergence iterations
VALIDATION_INTERVAL = 10	#interval between two validation checks, in training epochs
TRAINING_BATCHES = 1        #number of batches in which the training set should be split

#gpu parameters
use_gpu = True
target_gpu = "1"

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
chromosome_dict = {'MT':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, '10':10, '11':11, '12':12, '13':13, '14':14, '15':15, '16':16, '17':17, '18':18, '19':19, '20':20, '21':21, '22':22, 'X':23, 'Y':24}
class_weights = [0.8, 0.2]

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

#set target gpu as the only visible device
if use_gpu:
	os.environ["CUDA_VISIBLE_DEVICES"]=target_gpu

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
#load gene features
in_file = open(path_data+"Gene_Features/mmvv.pkl", 'rb')
gene_data = pickle.load(in_file)
in_file.close()

#preprocess drug ids
drugs = list()
for i in range(len(drugs_pre)):
	drugs.append(str(int(drugs_pre[i][4:])))

#determine graph dimensions
n_nodes = len(drugs)+len(genes)
n_edges = 2*links_dg.shape[0]+2*links_gg.shape[0]
dim_node_label = 27
type_mask = np.zeros((n_nodes,2), dtype=int)
#build id -> node number mappings
node_number = dict()
for i in range(len(drugs)):
	node_number[str(drugs[i])] = i
	type_mask[i][0] = 1
for i in range(len(genes)):
	node_number[str(genes[i])] = i + len(drugs)
	type_mask[i + len(drugs)][1] = 1

#build list of positive examples
positive_dsa_list = list()
for i in range(links_dse.shape[0]):
	if str(int(links_dse[i][0][4:])) in node_number.keys():
		positive_dsa_list.append((node_number[str(int(links_dse[i][0][4:]))],node_number[links_dse[i][2]]))
	else:
		sys.exit("ERROR: drug-side-effect link pointing to incorrect drug id")

#build node feature matrix
nodes = np.zeros((n_nodes, dim_node_label))
#build drug features
for i in pubchem_data.index:
	nn = node_number[str(pubchem_data.at[i,'cid'])]
	nodes[nn][0] = float(pubchem_data.at[i,'mw'])#molecular weight
	nodes[nn][1] = float(pubchem_data.at[i,'polararea'])#polar area
	nodes[nn][2] = float(pubchem_data.at[i,'xlogp'])#log octanal/water partition coefficient
	nodes[nn][3] = float(pubchem_data.at[i,'heavycnt'])#heavy atom count
	nodes[nn][4] = float(pubchem_data.at[i,'hbonddonor'])#hydrogen bond donors
	nodes[nn][5] = float(pubchem_data.at[i,'hbondacc'])#hydrogen bond acceptors
	nodes[nn][6] = float(pubchem_data.at[i,'rotbonds'])#number of rotatable bonds
#build gene features
for i in range(gene_data.shape[0]):
	nn = node_number[gene_data[i,0]]
	nodes[nn][0] = float(gene_data[i,1])#dna strand (-1 or +1)
	nodes[nn][1] = float(gene_data[i,2])#percent GC content (real value in [0,1])
	nodes[nn][2+chromosome_dict[gene_data[i,4]]] = float(1)#one-hot encoding of chromosome
	
#build target tensor
targets = np.concatenate( ( np.zeros((len(drugs)*len(side_effects), 1)), np.ones((len(drugs)*len(side_effects), 1)) ), axis=1) # [1,0] -> positive ; [0,1] -> negative (creates a vector of default negative targets, then modify positive ones only)
for p in positive_dsa_list:
	#each drug d has a block of len(side_effects) indices starting at (node_number[d]*len(side_effects)). The single side effect s has an offset equal to node_number[s]-len(drugs) inside this block (len(drugs) is subtracted because side effect node numbers are after drug node numbers, so that side effect #3 will have node number equal to len(drugs)+3)
	k = p[0]*len(side_effects) + p[1]-len(drugs)
	targets[k][0] = 1
	targets[k][1] = 0
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
for i in range(len(drugs)):
	for j in range(len(side_effects)):
		#side-effect-drug links only exist for message passing
		arcs[l][:] = [node_number[str(side_effects[j])], node_number[str(drugs[i])]]
		l = l+1
for i in range(len(drugs)):
	for j in range(len(side_effects)):
		print("Preprocessing Possible Drug-Side Effect Link "+str(i*len(side_effects)+j+1)+" of "+str(len(drugs)*len(side_effects)), end='\r')
		#drug-side-effect links are the ones on which the prediction is carried out
		arcs[l][:] = [node_number[str(drugs[i])],node_number[str(side_effects[j])]]
		l=l+1
print("")
arcs = np.array(arcs)
	
#split the dataset
validation_size = int(validation_share*targets.shape[0])
test_size = int(test_share*targets.shape[0])
index = np.array(list(range(targets.shape[0])))
np.random.shuffle(index)
test_index = index[:test_size]
validation_index = index[test_size:test_size+validation_size]
training_index = index[test_size+validation_size:]
#build set masks
te_mask = np.zeros(targets.shape[0], dtype=int)
va_mask = np.zeros(targets.shape[0], dtype=int)
tr_mask = np.zeros(targets.shape[0], dtype=int)
for i in test_index:
	te_mask[i] = 1
for i in validation_index:
	va_mask[i] = 1
for i in training_index:
	tr_mask[i] = 1
#concatenate a zero mask for all the non-targeted edges to each set mask
te_mask = np.concatenate((np.zeros(2*links_dg.shape[0]+2*links_gg.shape[0]+len(drugs)*len(side_effects), dtype=int), te_mask), axis=0)
va_mask = np.concatenate((np.zeros(2*links_dg.shape[0]+2*links_gg.shape[0]+len(drugs)*len(side_effects), dtype=int), va_mask), axis=0)
tr_mask = np.concatenate((np.zeros(2*links_dg.shape[0]+2*links_gg.shape[0]+len(drugs)*len(side_effects), dtype=int), tr_mask), axis=0)
#concatenate a placeholder [0,0]xN vector of targets for non-output nodes
expanded_targets = np.concatenate((np.zeros((2*links_dg.shape[0]+2*links_gg.shape[0]+len(drugs)*len(side_effects) , 2), dtype=int), targets), axis=0)

### DEBUG START ###
'''
print(nodes.shape)
print(arcs.shape)
print(expanded_targets.shape)
print(tr_mask.shape)
print(va_mask.shape)
print(te_mask.shape)
print(edge_mask.shape)
print(type_mask.shape)
sys.exit()
'''
### DEBUG STOP ###

#build CompositeGraphObject
tr_graph = CompositeGraphObject(arcs, nodes, expanded_targets, 'a', tr_mask, edge_mask, type_mask, node_aggregation='average')
va_graph = CompositeGraphObject(arcs, nodes, expanded_targets, 'a', va_mask, edge_mask, type_mask, node_aggregation='average')
te_graph = CompositeGraphObject(arcs, nodes, expanded_targets, 'a', te_mask, edge_mask, type_mask, node_aggregation='average')

#build network
netSt_drugs = utils.MLP(input_dim=2*STATE_DIM+nodes.shape[1], layers=[15, 10], activations=['relu', 'relu'],
                     kernel_initializer=[tf.keras.initializers.GlorotNormal() for i in range(2)],
                     bias_initializer=[tf.keras.initializers.GlorotNormal() for i in range(2)],
                     dropout_percs=[0.2, 0],
                     dropout_pos=[0, 0])
netSt_genes = utils.MLP(input_dim=2*STATE_DIM+nodes.shape[1], layers=[15, 10], activations=['relu', 'relu'],
                     kernel_initializer=[tf.keras.initializers.GlorotNormal() for i in range(2)],
                     bias_initializer=[tf.keras.initializers.GlorotNormal() for i in range(2)],
                     dropout_percs=[0.2, 0],
                     dropout_pos=[0, 0])
netSt_sideeffects = utils.MLP(input_dim=2*STATE_DIM+nodes.shape[1], layers=[15, 10], activations=['relu', 'relu'],
                     kernel_initializer=[tf.keras.initializers.GlorotNormal() for i in range(2)],
                     bias_initializer=[tf.keras.initializers.GlorotNormal() for i in range(2)],
                     dropout_percs=[0.2, 0],
                     dropout_pos=[0, 0])
netOut = utils.MLP(input_dim=2*STATE_DIM+2*nodes.shape[1], layers=[15,2], activations=['relu', 'linear'],
                      kernel_initializer=[tf.keras.initializers.GlorotNormal() for i in range(2)],
                      bias_initializer=[tf.keras.initializers.GlorotNormal() for i in range(2)],
                      dropout_percs=[0, 0],
                      dropout_pos=[0, 0])
model = CGNNedgeBased([netSt_drugs, netSt_sideeffects, netSt_genes], netOut, optimizer = tf.keras.optimizers.Adam(LR), loss_function = tf.nn.softmax_cross_entropy_with_logits, state_vect_dim = 10, type_label_lengths=np.array([7,0,0,0]),state_init_stdev=STATE_INIT_STDEV, max_iteration=MAX_ITER, addressed_problem='c', loss_arguments=None)

#train the network
model.train(tr_graph, EPOCHS, va_graph, class_weights=class_weights)

#evaluate the network
iterations, loss, targets, outputs = model.evaluate_single_graph(te_graph, class_weights=class_weights, training=False)

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
recall = [ float(TP[j])/float(TP[j]+FN[j]) if TP[j]+FN[j] > 0 else 0.0 for j in range(CLASSES)]
global_accuracy = float(sum(TP)+sum(TN))/float(sum(TP)+sum(TN)+sum(FP)+sum(FN))

print("TP = "+str(TP[0])+" , TN = "+str(TN[0]))
print("FP = "+str(FP[0])+" , FN = "+str(FN[0]))

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


