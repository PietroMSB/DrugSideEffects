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

from CompositeGNN.composite_graph_class import CompositeGraphObject

import spektral
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Multiply
from spektral.layers import GCNConv
from spektral.transforms import GCNFilter
from spektral.data import Graph
from spektral.data import Dataset
from spektral.data.loaders import SingleLoader
from spektral.data.loaders import BatchLoader
from spektral.utils import gcn_filter 

#network parameters
EPOCHS = 500                #number of training epochs
LR = 0.001					#learning rate
MAX_ITER = 4				#maximum number of state convergence iterations
VALIDATION_INTERVAL = 10	#interval between two validation checks, in training epochs
TRAINING_BATCHES = 1        #number of batches in which the training set should be split
LABEL_DIM = [7, 27]

#gpu parameters
use_gpu = True
target_gpu = "1"

#script parameters
run_id = sys.argv[1]
path_data = "Datasets/Nuovo/Output/Soglia_100/"
#path_data = "Datasets/Artificiale/Output/"
path_results = "Results/Nuovo/LinkPredictor/"+run_id+".txt"
splitting_seed = 920305
validation_share = 0.1
test_share = 0.1
atomic_number = { 'Li':3, 'B':5, 'C':6, 'N':7, 'O':8, 'F':9, 'Mg':12, 'Al':13, 'P':15, 'S':16, 'Cl':17, 'K':19, 'Ca':20, 'Fe':26, 'Co':27, 'As':33, 'Br':35, 'I':53, 'Au':79 }
atomic_label = { 3:'Li', 5:'B', 6:'C', 7:'N', 8:'O', 9:'F', 12:'Mg', 13:'Al', 15:'P', 16:'S', 17:'Cl', 19:'K' ,20:'Ca', 26:'Fe', 27:'Co', 33:'As', 35:'Br', 53:'I', 79:'Au' }
label_translator = {'C':1, 'N':2, 'O':3, 'S':4, 'F':5, 'P':6, 'Cl':7, 'I':7, 'Br':7, 'Ca':8, 'Mg':8, 'K':8, 'Li':8, 'Co':8, 'As':8, 'B':8, 'Al':8, 'Au':8, 'Fe':8}
chromosome_dict = {'MT':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, '10':10, '11':11, '12':12, '13':13, '14':14, '15':15, '16':16, '17':17, '18':18, '19':19, '20':20, '21':21, '22':22, 'X':23, 'Y':24}

#definition of GNN Model Class
class DrugClassifier(Model):

	#constructor
	def __init__(self, list_hidden, units_dense, classes):
		#Keras.Model class constructor
		super().__init__()
		#define list of convolutional layers
		self.graph_conv = list()
		for h in list_hidden:
			self.graph_conv.append(GCNConv(h, activation="relu"))
		self.dense = Dense(units_dense,"relu")
		self.output_layer = Dense(classes,'softmax')

	#call predictor on input data
	def call(self, inputs):
		node_state = inputs[0][0] #input node features
		adjacency = inputs[1][0] #input adjacency tensor (previously transformed with GCNFilter())
		set_mask = inputs[2] #training/validation/test mask
		#call every convolutional layer
		for gc in self.graph_conv:
			node_state = gc((node_state, adjacency))
		#apply set mask
		node_state_set = tf.boolean_mask(node_state,tf.reshape(set_mask,set_mask.shape[0]))
		#apply dense layer
		out = self.dense(node_state_set)
		out = self.output_layer(out)
		return out

#custom dataset class
class CustomDataset(Dataset):

	def __init__(self, graph_objects, **kwargs):
		self.num_batches = len(graph_objects)
		self.adjacency = [go.getAdjacency().todense() for go in graph_objects]
		self.nodes = [go.getNodes() for go in graph_objects]
		self.arcs = [go.getArcs() for go in graph_objects]
		self.targets = [go.getTargets() for go in graph_objects]
		self.set_mask = [go.getSetMask() for go in graph_objects]
		super().__init__(**kwargs)

	def read(self):
		graphs = list()
		for i in range(self.num_batches):
			g = Graph(a=self.adjacency[i], e=self.arcs[i], x=self.nodes[i], y=self.targets[i])
			g.set_mask = self.set_mask[i]
			graphs.append(g)
		return graphs

#custom loader class
class CustomLoader(BatchLoader):
		
	def collate(self, batch):
		return ([batch[0].x], [batch[0].a], batch[0].set_mask), batch[0].y

#custom loss function for multilabel classification
'''
def multilabel_crossentropy_loss(y, x):
	element_loss = np.zeros_like(y)
	for i in range(y.shape[1]):
		element_loss[:,i] = tf.keras.losses.MSE(y[:,i], x[:,i])
	return np.sum(element_loss, axis=1)
'''

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
in_file = open(path_data+"gene_features.pkl", 'rb')
gene_data = pickle.load(in_file)
in_file.close()

#preprocess drug ids
drugs = list()
for i in range(len(drugs_pre)):
	drugs.append(str(int(drugs_pre[i][4:])))

#determine graph dimensions
CLASSES = len(side_effects)		#number of outputs
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
#build id -> class number mappings
class_number = dict()
for i in range(len(side_effects)):
	class_number[side_effects[i]] = i

#build output mask
output_mask = np.concatenate((np.ones(len(drugs)), np.zeros(len(genes))))

#build list of positive examples
positive_dsa_list = list()
for i in range(links_dse.shape[0]):
	if str(int(links_dse[i][0][4:])) in node_number.keys():
		#skip side-effects which were filtered out of the dataset
		if links_dse[i][2] not in side_effects:
			continue
		positive_dsa_list.append((node_number[str(int(links_dse[i][0][4:]))],class_number[links_dse[i][2]]))
	else:
		sys.exit("ERROR: drug-side-effect link pointing to incorrect drug id")

#build node feature matrix
nodes = np.zeros((n_nodes, dim_node_label))
#build drug features
for i in pubchem_data.index:
	#skip drugs which were filtered out of the dataset
	if pubchem_data.at[i,'cid'] not in node_number.keys():
		continue
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
	#skip genes which were filtered out of the dataset
	if gene_data[i,0] not in node_number.keys():
		continue
	nn = node_number[gene_data[i,0]]
	nodes[nn][0] = float(gene_data[i,1])#dna strand (-1 or +1)
	nodes[nn][1] = float(gene_data[i,2])#percent GC content (real value in [0,1])
	nodes[nn][2+chromosome_dict[gene_data[i,4]]] = float(1)#one-hot encoding of chromosome
	
#build target tensor
targets = np.zeros((len(drugs),len(side_effects)))
for p in positive_dsa_list:	
	targets[p[0]][p[1]] = 1

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
#split targets
te_targets = targets[ te_mask.astype(bool), : ]
va_targets = targets[ va_mask.astype(bool), : ]
tr_targets = targets[ tr_mask.astype(bool), : ]
#concatenate all-zero set mask extensions for gene nodes
te_mask = np.concatenate((te_mask,np.zeros(len(genes), dtype=int)))
va_mask = np.concatenate((va_mask,np.zeros(len(genes), dtype=int)))
tr_mask = np.concatenate((tr_mask,np.zeros(len(genes), dtype=int)))


### DEBUG START ###
'''
print(nodes.shape)
print(arcs.shape)
print(expanded_targets.shape)
print(tr_mask.shape)
print(va_mask.shape)
print(te_mask.shape)
print(type_mask.shape)
sys.exit()
'''
### DEBUG STOP ###

#build CompositeGraphObject
tr_graph = CompositeGraphObject(arcs, nodes, tr_targets, type_mask, [7,27], 'n', tr_mask, output_mask, aggregation_mode='average')
va_graph = CompositeGraphObject(arcs, nodes, va_targets, type_mask, [7,27], 'n', va_mask, output_mask, aggregation_mode='average')
te_graph = CompositeGraphObject(arcs, nodes, te_targets, type_mask, [7,27], 'n', te_mask, output_mask, aggregation_mode='average')

#create spektral dataset objects
print("Packing data")
tr_dataset = CustomDataset([tr_graph])
va_dataset = CustomDataset([va_graph])
te_dataset = CustomDataset([te_graph])
tr_dataset.apply(GCNFilter())
va_dataset.apply(GCNFilter())
te_dataset.apply(GCNFilter())
#create loader objects
tr_loader = CustomLoader(tr_dataset, batch_size=1, epochs=None, shuffle=False)
va_loader = CustomLoader(va_dataset, batch_size=1, epochs=None, shuffle=False)
te_loader = CustomLoader(te_dataset, batch_size=1, epochs=None, shuffle=False)

#build network
print("Building the network")
model = DrugClassifier([50, 50], 100, classes = CLASSES)
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.binary_crossentropy, metrics=[tf.keras.metrics.Accuracy()], loss_weights=None, weighted_metrics=None, run_eagerly=True)

#train the network
model.fit(tr_loader.load(), steps_per_epoch=tr_loader.steps_per_epoch, epochs=EPOCHS, validation_data=va_loader.load(), validation_steps=1)

#evaluate the network
outputs = model.predict(te_loader.load(), steps=1)

#calculate results
TP = [0 for j in range(CLASSES)]
TN = [0 for j in range(CLASSES)]
FP = [0 for j in range(CLASSES)]
FN = [0 for j in range(CLASSES)]
for i in range(te_targets.shape[0]):
	for j in range(CLASSES):
		if te_targets[i][j] > 0.5:
			if outputs[i][j] > 0.5: TP[j] += 1
			else: FN[j] += 1
		else:
			if outputs[i][j] > 0.5: FP[j] += 1
			else: TN[j] += 1
accuracy = [ float(TP[j]+TN[j])/float(TP[j]+TN[j]+FP[j]+FN[j])  for j in range(CLASSES)]
precision = [ float(TP[j])/float(TP[j]+FP[j]) if TP[j]+FP[j] > 0 else 0.0 for j in range(CLASSES)]
recall = [ float(TP[j])/float(TP[j]+FN[j]) if TP[j]+FN[j] > 0 else 0.0 for j in range(CLASSES)]
global_accuracy = float(sum(TP)+sum(TN))/float(sum(TP)+sum(TN)+sum(FP)+sum(FN))
global_sensitivity = float(sum(TP))/float(sum(TP)+sum(FN))
global_specificity = float(sum(TN))/float(sum(FP)+sum(TN))
global_balanced_accuracy = float(global_specificity+global_sensitivity)/2

print("TP = "+str(sum(TP))+" , TN = "+str(sum(TN)))
print("FP = "+str(sum(FP))+" , FN = "+str(sum(FN)))

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
print("\nGlobal Balanced Accuracy:\n"+str(global_balanced_accuracy))

'''
for i in range(10):
	print(targets[i,:].astype(int))
	print(np.greater(outputs[i,:],0.5).astype(int))
'''
