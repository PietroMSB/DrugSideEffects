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
STATE_DIM = 5				#node state dimension
STATE_INIT_STDEV = 0.1		#standard deviation of random state initialization
LR = 0.01					#learning rate
MAX_ITER = 3				#maximum number of state convergence iterations
VALIDATION_INTERVAL = 10	#interval between two validation checks, in training epochs
TRAINING_BATCHES = 1        #number of batches in which the training set should be split

#gpu parameters
use_gpu = True
target_gpu = "1"

#script parameters
run_id = sys.argv[1]
path_data = "Datasets/Nuovo/Transduction/Soglia_100/"
path_results = "Results/Nuovo/LinkPredictor/"+run_id+".txt"
splitting_seed = 920305
validation_share = 0.1
test_share = 0.1
atomic_number = { 'Li':3, 'B':5, 'C':6, 'N':7, 'O':8, 'F':9, 'Mg':12, 'Al':13, 'P':15, 'S':16, 'Cl':17, 'K':19, 'Ca':20, 'Fe':26, 'Co':27, 'As':33, 'Br':35, 'I':53, 'Au':79 }
atomic_label = { 3:'Li', 5:'B', 6:'C', 7:'N', 8:'O', 9:'F', 12:'Mg', 13:'Al', 15:'P', 16:'S', 17:'Cl', 19:'K' ,20:'Ca', 26:'Fe', 27:'Co', 33:'As', 35:'Br', 53:'I', 79:'Au' }
label_translator = {'C':1, 'N':2, 'O':3, 'S':4, 'F':5, 'P':6, 'Cl':7, 'I':7, 'Br':7, 'Ca':8, 'Mg':8, 'K':8, 'Li':8, 'Co':8, 'As':8, 'B':8, 'Al':8, 'Au':8, 'Fe':8}
class_weights = [0.80, 0.20]

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

#build network
node_label_dim = batch_validation.getNodes().shape[1]
netSt_drugs = utils.MLP(input_dim=2*STATE_DIM+node_label_dim, layers=[STATE_DIM], activations=['selu'],
                     kernel_initializer=[tf.keras.initializers.GlorotNormal() for i in range(1)],
                     bias_initializer=[tf.keras.initializers.GlorotNormal() for i in range(1)],
                     dropout_percs=[0.2, 0],
                     dropout_pos=[0, 0])
netSt_genes = utils.MLP(input_dim=2*STATE_DIM+node_label_dim, layers=[STATE_DIM], activations=['selu'],
                     kernel_initializer=[tf.keras.initializers.GlorotNormal() for i in range(1)],
                     bias_initializer=[tf.keras.initializers.GlorotNormal() for i in range(1)],
                     dropout_percs=[0.2, 0],
                     dropout_pos=[0, 0])
netSt_sideeffects = utils.MLP(input_dim=2*STATE_DIM+node_label_dim, layers=[STATE_DIM], activations=['selu'],
                     kernel_initializer=[tf.keras.initializers.GlorotNormal() for i in range(1)],
                     bias_initializer=[tf.keras.initializers.GlorotNormal() for i in range(1)],
                     dropout_percs=[0.2, 0],
                     dropout_pos=[0, 0])
netOut = utils.MLP(input_dim=2*STATE_DIM+2*node_label_dim, layers=[10, 2], activations=['selu', 'softmax'],
                      kernel_initializer=[tf.keras.initializers.GlorotNormal() for i in range(2)],
                      bias_initializer=[tf.keras.initializers.GlorotNormal() for i in range(2)],
                      dropout_percs=[0, 0],
                      dropout_pos=[0, 0])
model = CGNNedgeBased([netSt_drugs, netSt_sideeffects, netSt_genes], netOut, optimizer = tf.keras.optimizers.Adam(LR), loss_function = tf.nn.softmax_cross_entropy_with_logits, state_vect_dim = STATE_DIM, type_label_lengths=np.array([7,0,0,0]),state_init_stdev=STATE_INIT_STDEV, max_iteration=MAX_ITER, addressed_problem='c', loss_arguments=None)

#train the network
model.train(batch_list_training, EPOCHS, batch_validation, class_weights=class_weights)

#evaluate the network
iterations, loss, targets, outputs = model.evaluate_single_graph(batch_test, class_weights=class_weights, training=False)

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

