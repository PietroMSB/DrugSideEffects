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
from sklearn.preprocessing import MinMaxScaler

import dgl


import spektral
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Multiply
from spektral.layers import GCNConv
from spektral.transforms import GCNFilter
from spektral.data import Graph
from spektral.data import Dataset
from spektral.data.loaders import BatchLoader
from spektral.utils import gcn_filter
from tensorflow import gather 

#network parameters
EPOCHS = 50                	#number of training epochs
LR = 0.001					#learning rate
THRESHOLD = 0.001			#state convergence threshold, in terms of relative state difference
MAX_ITER = 6				#maximum number of state convergence iterations
VALIDATION_INTERVAL = 10	#interval between two validation checks, in training epochs
TRAINING_BATCHES = 7        #number of batches in which the training set should be split
CLASSES = 2					#number of output classes

#gpu parameters
use_gpu = True
target_gpu = "1"
dgl_device = "cuda:"+target_gpu

#script parameters
run_id = sys.argv[1]
path_data = "Datasets/D1/Transduction/"
path_results = "Results/D1/Spektral_"+run_id+".txt"
splitting_seed = 920305
validation_share = 0.1
test_share = 0.1
atomic_number = { 'Li':3, 'B':5, 'C':6, 'N':7, 'O':8, 'F':9, 'Mg':12, 'Al':13, 'P':15, 'S':16, 'Cl':17, 'K':19, 'Ca':20, 'Fe':26, 'Co':27, 'As':33, 'Br':35, 'I':53, 'Au':79 }
atomic_label = { 3:'Li', 5:'B', 6:'C', 7:'N', 8:'O', 9:'F', 12:'Mg', 13:'Al', 15:'P', 16:'S', 17:'Cl', 19:'K' ,20:'Ca', 26:'Fe', 27:'Co', 33:'As', 35:'Br', 53:'I', 79:'Au' }
label_translator = {'C':1, 'N':2, 'O':3, 'S':4, 'F':5, 'P':6, 'Cl':7, 'I':7, 'Br':7, 'Ca':8, 'Mg':8, 'K':8, 'Li':8, 'Co':8, 'As':8, 'B':8, 'Al':8, 'Au':8, 'Fe':8}

#definition of GNN Model Class
class LinkPredictor(Model):

	#constructor
	def __init__(self, input_dim, list_hidden, units_dense):
		#Keras.Model class constructor
		super().__init__()
		#define list of convolutional layers
		self.graph_conv = list()
		in_features = input_dim
		for h in list_hidden:
			self.graph_conv.append(dgl.nn.GraphConv(in_features, h))
			in_features = h
		self.dense = Dense(units_dense,"relu")
		self.output_layer = Dense(2,'softmax')

	#call predictor on input data
	def call(self, inputs):
		node_state = inputs[0][0] #input node features
		adjacency = inputs[1][0] #input adjacency tensor (previously transformed with GCNFilter())
		out_edges = inputs[2] #output edges
		set_mask = inputs[3] #training/validation/test mask
		#transform adjacency matrix into a sparse tensor
		#adjacency = tf.sparse.from_dense(adjacency)
		print(node_state)
		print(node_state.shape)
		#call every convolutional layer
		for gc in self.graph_conv:
			node_state = gc((node_state, adjacency))
			print(node_state)
			print(node_state.shape)
		#transform node states to edge states
		edge_state = tf.concat((tf.gather(node_state, out_edges[:][0]),tf.gather(node_state, out_edges[:][1])), axis=0)
		#edge_state = tf.concat((tf.gather(node_state[0], out_edges[:,0]),tf.gather(node_state[0], out_edges[:,1])), axis = 1)
		print(edge_state)
		#apply set mask
		edge_state_set = tf.boolean_mask(edge_state,set_mask)
		print(edge_state_set)
		#apply dense layer
		out = self.dense(edge_state_set)
		print(out)
		out = self.output_layer(out)
		print(out)
		return out

#custom dataset class
def GOtoDGL(graph_objects):
	graphs = list()
	targets = list()
	set_masks = list()
	out_edges = list()
	for go in graph_objects:
		g = dgl.graph((go.getArcs()[:][0], go.getArcs()[:][1]), num_nodes = go.getNodes().shape[0])
		g.nodes.data = go.getNodes()
		graphs.append(g)
		out_edges.append(go.getArcs()[go.getOutputMask().astype(bool)])
		set_masks.append(go.getSetMask()[go.getOutputMask().astype(bool)])
		targets.append(go.getTargets())
	return (graphs, out_edges, set_masks, targets)

#custom loader class
class CustomLoader(BatchLoader):
		
	def collate(self, batch):
		return ([batch[0].x], [batch[0].a], batch[0].out_edges, batch[0].set_mask), batch[0].y

#set target gpu as the only visible device
if use_gpu:
	os.environ["CUDA_VISIBLE_DEVICES"]=target_gpu

#load data batches
in_file = open(path_data+"batch_validation.pkl", 'rb')
batch_validation = pickle.load(in_file)
in_file.close()
in_file = open(path_data+"batch_test.pkl", 'rb')
batch_test = pickle.load(in_file)
in_file.close()
batch_list_training = list()
for i in range(TRAINING_BATCHES):
	in_file = open(path_data+"batch_training_"+str(i)+".pkl", 'rb')
	batch_list_training.append(pickle.load(in_file))
	in_file.close()

#create dgl graph objects
print("Packing data")
tr_dataset = GOtoDGL(batch_list_training)
va_dataset = GOtoDGL([batch_validation])
te_dataset = GOtoDGL([batch_test])

#build network
print("Building the network")
model = LinkPredictor([20,20,20], 30)
model.compile( optimizer=tf.keras.optimizers.Adam(), loss=tf.nn.softmax_cross_entropy_with_logits, metrics=[tf.keras.metrics.Accuracy()], loss_weights=None, weighted_metrics=None, run_eagerly=True)

#train network
print("Training the network")
model.fit(tr_loader.load(), steps_per_epoch=tr_loader.steps_per_epoch, epochs=EPOCHS, validation_data=va_loader.load(), validation_steps=1)

#evaluate the network
outputs = model.predict(te_loader.load(), steps=1)

#calculate results
test_targets = targets[te_mask]
TP = [0 for j in range(CLASSES)]
TN = [0 for j in range(CLASSES)]
FP = [0 for j in range(CLASSES)]
FN = [0 for j in range(CLASSES)]
for i in range(test_targets.shape[0]):
	for j in range(CLASSES):
		if test_targets[i][j] > 0.5:
			if outputs[i][j] > 0.5: TP[j] += 1
			else: FN[j] += 1
		else:
			if outputs[i][j] > 0.5: FP[j] += 1
			else: TN[j] += 1
accuracy = [ float(TP[j]+TN[j])/float(TP[j]+TN[j]+FP[j]+FN[j])  for j in range(CLASSES)]
precision = [ float(TP[j])/float(TP[j]+FP[j]) if TP[j]+FP[j] > 0 else 0.0 for j in range(CLASSES)]
recall = [ float(TP[j])/float(TP[j]+FN[j]) if TP[j]+FN[j] > 0 else 0.0 for j in range(CLASSES)]
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

