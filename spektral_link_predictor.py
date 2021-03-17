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
from itertools import product

import spektral
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from spektral.transforms import GCNFilter
from spektral.layers import GCNConv
from spektral.data import Graph
from spektral.data import Dataset
from spektral.data.loaders import SingleLoader
from spektral.utils import gcn_filter
from tensorflow import gather 

#network parameters
EPOCHS = 500                #number of training epochs
LR = 0.001					#learning rate
THRESHOLD = 0.001			#state convergence threshold, in terms of relative state difference
MAX_ITER = 6				#maximum number of state convergence iterations
VALIDATION_INTERVAL = 10	#interval between two validation checks, in training epochs
TRAINING_BATCHES = 1        #number of batches in which the training set should be split

#script parameters
run_id = sys.argv[1]
path_data = "Datasets/Nuovo/Output/Soglia_100/"
path_preprocessed = "Datasets/Nuovo/Output/Spektral_Preprocessed/Soglia_100/"
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

#definition of GNN Model Class
class LinkPredictor(Model):

	#constructor
	def __init__(self, list_hidden, units_dense):
		#Keras.Model class constructor
		super().__init__()
		#define list of convolutional layers
		self.graph_conv = list()
		for h in list_hidden:
			self.graph_conv.append(GCNConv(h))
		self.dense = Dense(units_dense,"relu")
		self.output_layer = Dense(1,'sigmoid')

	#call predictor on input data
	def call(self, inputs):
		node_state = inputs[0][0] #input node features
		adjacency = inputs[0][1] #input adjacency tensor (previously transformed with GCNFilter())
		out_edges = inputs[0][2] #output edges
		set_mask = inputs[0][3] #training/validation/test mask
		#call every convolutional layers
		for gc in self.graph_conv:
			node_state = gc(node_state, adjacency)
		#transform node states to edge states
		edge_state = tf.concat((gather(node_state, out_edges[:,0]),gather(node_state, out_edges[:,1])), axis = 1)
		#apply set mask
		edge_state_set = tf.boolean_mask(edge_state, set_mask)
		#apply dense layer
		out = self.dense(edge_state_set)
		out = self.output_layer(out)
		return out

#custom dataset class
class CustomDataset(Dataset):

	def __init__(self, adjacency, nodes, arcs, targets, out_edges, **kwargs):
		self.adjacency = adjacency
		self.nodes = nodes
		self.arcs = arcs
		self.targets = targets
		self.out_edges = out_edges
		super().__init__(**kwargs)

	def read(self):
		g = Graph(a=self.adjacency, e=self.arcs, x=self.nodes, y=self.targets)
		g.out_edges = out_edges
		return [g]

#custom loader class
class CustomLoader(SingleLoader):
	
	def __init__(self, dataset, epochs, sample_weights, set_mask):
		self.set_mask = set_mask
		super().__init__(dataset, epochs, sample_weights)
	
	def collate(self, batch):
		return (batch[0].x, tf.sparse.SparseTensor(batch[0].a), batch[0].out_edges, self.set_mask), batch[0].y

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
#build id -> node number mappings
node_number = dict()
for i in range(len(drugs)):
	node_number[str(drugs[i])] = i
for i in range(len(side_effects)):
	node_number[str(side_effects[i])] = i + len(drugs) 
for i in range(len(genes)):
	node_number[str(genes[i])] = i + len(drugs) + len(side_effects)

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

#build edge mask
edge_mask = np.concatenate((np.zeros(2*links_dg.shape[0]+2*links_gg.shape[0]+len(drugs)*len(side_effects), dtype=int), np.ones(len(drugs)*len(side_effects), dtype=int)), axis=0)
#build target tensor
targets = np.zeros(len(drugs)*len(side_effects))
for p in positive_dsa_list:
	#each drug d has a block of len(side_effects) indices starting at (node_number[d]*len(side_effects)). The single side effect s has an offset equal to node_number[s]-len(drugs) inside this block (len(drugs) is subtracted because side effect node numbers are after drug node numbers, so that side effect #3 will have node number equal to len(drugs)+3)
	k = p[0]*len(side_effects) + p[1]-len(drugs)
	targets[k] = 1
#build adjacency matrix and list of output edges
out_edges = list()
adj_data = np.ones(n_edges, dtype=int)
adj_row = list()
adj_col = list()
#add drug-gene edges
for i in range(links_dg.shape[0]):
	adj_row.append(node_number[str(int(links_dg[i][0]))])
	adj_col.append(node_number[str(links_dg[i][1])])
	adj_row.append(node_number[str(links_dg[i][1])])
	adj_col.append(node_number[str(int(links_dg[i][0]))])
#add gene-gene edges
for i in range(links_gg.shape[0]):
	adj_row.append(node_number[str(links_gg[i][0])])
	adj_col.append(node_number[str(links_gg[i][1])])
	adj_row.append(node_number[str(links_gg[i][1])])
	adj_col.append(node_number[str(links_gg[i][0])])
#add drug-se edges
for i in range(len(drugs)):
	for j in range(len(side_effects)):
		#side-effect-drug links only exist for message passing
		adj_row.append(node_number[str(side_effects[j])])
		adj_col.append(node_number[str(drugs[i])])
for i in range(len(drugs)):
	for j in range(len(side_effects)):
		#drug-side-effect links are the ones on which the prediction is carried out
		print("Preprocessing Possible Drug-Side Effect Link "+str(i*len(side_effects)+j+1)+" of "+str(len(drugs)*len(side_effects)), end='\r')
		adj_row.append(node_number[str(drugs[i])])
		adj_col.append(node_number[str(side_effects[j])])
		out_edges.append([node_number[str(drugs[i])], node_number[str(side_effects[j])]])
print("")
out_edges = np.array(out_edges)
adjacency = scipy.sparse.coo_matrix((adj_data, (adj_row, adj_col)))

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


#create spektral dataset object
dataset = CustomDataset(adjacency, nodes, None, targets, out_edges)
dataset.apply(GCNFilter())
#create loader objects
tr_loader = CustomLoader(dataset, EPOCHS, None, tr_mask)
va_loader = CustomLoader(dataset, 1, None, va_mask)
te_loader = CustomLoader(dataset, 1, None, te_mask)

#build network
model = LinkPredictor([20,20,20], 30)
model.compile("Adam","binary_crossentropy")

#train network
model.fit(tr_loader.load(), EPOCHS, va_loader.load())

#evaluate the network
iterations, loss, targets, outputs = model.evaluate_single_graph(te_loader.load(), class_weights=[1 for i in range(CLASSES)], training=False)

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


