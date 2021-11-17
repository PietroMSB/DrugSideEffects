#coding=utf-8

import sys
import os
import math
import pickle
import numpy as np
import pandas
import scipy
import pickle
import rdkit
from rdkit import Chem
from rdkit.Chem import FunctionalGroups
from rdkit import DataStructs

from sklearn import neural_network as sklnn

#network parameters
EPOCHS = 300                #number of training epochs
HIDDEN_UNITS = (10)		#number of units in each hidden layer
LR = 0.001					#learning rate
VALIDATION_INTERVAL = 10	#interval between two validation checks, in training epochs
DROPOUT_RATE = 0.0			#dropout rate for MLPs
ACTIVATION = "tanh"			#activation function
LABEL_DIM = 135				#dimension of feature vector

#gpu parameters
use_gpu = True
target_gpu = "1"

#script parameters
run_id = sys.argv[1]
path_data = "Datasets/Nuovo/Output/Soglia_100/"
path_results = "Results/Nuovo/LinkPredictor/"+run_id+".txt"
feature_fingerprint_size = 128
splitting_seed = 3
validation_share = 0.1
test_share = 0.1
atomic_number = { 'Li':3, 'B':5, 'C':6, 'N':7, 'O':8, 'F':9, 'Mg':12, 'Al':13, 'P':15, 'S':16, 'Cl':17, 'K':19, 'Ca':20, 'Fe':26, 'Co':27, 'As':33, 'Br':35, 'I':53, 'Au':79 }
atomic_label = { 3:'Li', 5:'B', 6:'C', 7:'N', 8:'O', 9:'F', 12:'Mg', 13:'Al', 15:'P', 16:'S', 17:'Cl', 19:'K' ,20:'Ca', 26:'Fe', 27:'Co', 33:'As', 35:'Br', 53:'I', 79:'Au' }
label_translator = {'C':1, 'N':2, 'O':3, 'S':4, 'F':5, 'P':6, 'Cl':7, 'I':7, 'Br':7, 'Ca':8, 'Mg':8, 'K':8, 'Li':8, 'Co':8, 'As':8, 'B':8, 'Al':8, 'Au':8, 'Fe':8}

#function that adjusts NaN features
def CheckedFeature(feature):
	if feature is None:
		return 0.0
	if np.isnan(feature):
		return 0.0
	return feature

#function that binarizes a fingerprint
def BinarizedFingerprint(fp):
	bfp = [0 for i in range(len(fp))]
	for i in range(len(fp)):
		if fp[i] > 0:
			bfp[i] = 1
	return bfp

#set target gpu as the only visible device
if use_gpu:
	os.environ["CUDA_VISIBLE_DEVICES"]=target_gpu

#load side-effect data
print("Loading side-effects")
in_file = open(path_data+"side_effects.pkl", 'rb')
side_effects = pickle.load(in_file)
in_file.close()
#load drug data
print("Loading drugs")
in_file = open(path_data+"drugs.pkl", 'rb')
drugs_pre = pickle.load(in_file)
in_file.close()
#load drug-side effect links
print("Loading drug - side-effects associations")
in_file = open(path_data+"drug_side_effect_links.pkl", 'rb')
links_dse = pickle.load(in_file)
in_file.close()
#load drug features
print("Loading drug features")
pubchem_data = pandas.read_csv(path_data+"pubchem_output.csv")

#preprocess drug ids
print("Preprocessing drug identifiers")
drugs = list()
for i in range(len(drugs_pre)):
	drugs.append(str(int(drugs_pre[i][4:])))

#determine dataset dimensions
print("Calculating graph dimensions")
CLASSES = len(side_effects)		#number of outputs
n_examples = len(drugs)
#build id -> example number mappings
example_number = dict()
for i in range(len(drugs)):
	example_number[str(drugs[i])] = i
#build id -> class number mappings
class_number = dict()
for i in range(len(side_effects)):
	class_number[side_effects[i]] = i

#build list of positive examples
print("Building list of positive examples")
positive_dsa_list = list()
for i in range(links_dse.shape[0]):
	if str(int(links_dse[i][0][4:])) in example_number.keys():
		#skip side-effects which were filtered out of the dataset
		if links_dse[i][2] not in side_effects:
			continue
		positive_dsa_list.append((example_number[str(int(links_dse[i][0][4:]))],class_number[links_dse[i][2]]))
	else:
		sys.exit("ERROR: drug-side-effect link pointing to incorrect drug id")

#build drug feature matrix
print("Building drug feature matrix")
X = np.zeros((n_examples, LABEL_DIM))
#build drug features
for i in pubchem_data.index:
	#skip drugs which were filtered out of the dataset
	if str(pubchem_data.at[i,'cid']) not in example_number.keys():
		continue
	en = example_number[str(pubchem_data.at[i,'cid'])]
	X[en][0] = CheckedFeature(float(pubchem_data.at[i,'mw']))#molecular weight
	X[en][1] = CheckedFeature(float(pubchem_data.at[i,'polararea']))#polar area
	X[en][2] = CheckedFeature(float(pubchem_data.at[i,'xlogp']))#log octanal/water partition coefficient
	X[en][3] = CheckedFeature(float(pubchem_data.at[i,'heavycnt']))#heavy atom count
	X[en][4] = CheckedFeature(float(pubchem_data.at[i,'hbonddonor']))#hydrogen bond donors
	X[en][5] = CheckedFeature(float(pubchem_data.at[i,'hbondacc']))#hydrogen bond acceptors
	X[en][6] = CheckedFeature(float(pubchem_data.at[i,'rotbonds']))#number of rotatable bonds

#normalize drug features
print("Normalizing drug features")
for i in range(LABEL_DIM):
	col_min = None
	col_max = None
	for j in range(n_examples):
		#skip zeros
		if X[j][i] == 0:
			continue
		if col_min is None:
			col_min = X[j][i]
		if col_max is None:
			col_max = X[j][i]
		if X[j][i] < col_min:
			col_min = X[j][i]
		if X[j][i] > col_max:
			col_max = X[j][i]
	#do not normalize zero columns
	if col_min is None or col_max is None:
		continue
	for j in range(X.shape[0]):
		#do not normalize zeros
		if X[j][i] == 0:
			continue
		X[j][i] = float(X[j][i] - col_min) / float(col_max - col_min)

#build dict of molecular structures
molecule_dict = dict()
for i in pubchem_data.index:
	#skip drugs which were filtered out of the dataset
	if str(pubchem_data.at[i,'cid']) not in example_number.keys():
		continue
	en = example_number[str(pubchem_data.at[i,'cid'])]
	molecule_dict[en] = rdkit.Chem.MolFromSmiles(pubchem_data.at[i,'isosmiles'])
#build dicts of fingerprints
feature_fingerprint_dict = dict()
for k in molecule_dict.keys():
	feature_fingerprint_dict[k] = Chem.RDKFingerprint(molecule_dict[k], fpSize=feature_fingerprint_size)

#add fingerprints to drug features
for i in pubchem_data.index:
	#skip drugs which were filtered out of the dataset
	if str(pubchem_data.at[i,'cid']) not in example_number.keys():
		continue
	en = example_number[str(pubchem_data.at[i,'cid'])]
	#get fingerprint from dictionary and convert it to numpy array
	fingerprint = np.array((1,))
	rdkit.DataStructs.cDataStructs.ConvertToNumpyArray(feature_fingerprint_dict[en], fingerprint)
	#add fingerprint
	X[en][-feature_fingerprint_size:] = fingerprint

#build target tensor
print("Building target tensor")
targets = np.zeros((len(drugs),len(side_effects)))
for p in positive_dsa_list:	
	targets[p[0]][p[1]] = 1

#split the dataset
print("Splitting the dataset")
validation_size = int(validation_share*targets.shape[0])
test_size = int(test_share*targets.shape[0])
index = np.array(list(range(targets.shape[0])))
np.random.seed(splitting_seed)
np.random.shuffle(index)
test_index = index[:test_size]
validation_index = index[test_size:test_size+validation_size]
training_index = index[test_size+validation_size:]
#build sets
X_tr = X[training_index, :]
X_va = X[validation_index, :]
X_te = X[test_index, :]
Y_tr = targets[training_index, :]
Y_va = targets[validation_index, :]
Y_te = targets[test_index, :]

#build the network
model = sklnn.MLPClassifier(HIDDEN_UNITS, ACTIVATION, 'adam', batch_size = X_tr.shape[0], learning_rate_init=LR, max_iter=EPOCHS)

#train the network
model = model.fit(X_tr, Y_tr)

#test the network
outputs = model.predict(X_te)

#calculate results
targets = Y_te
TP = [0 for j in range(CLASSES)]
TN = [0 for j in range(CLASSES)]
FP = [0 for j in range(CLASSES)]
FN = [0 for j in range(CLASSES)]
exact_matches = 0
for i in range(targets.shape[0]):
	exact_match = True
	for j in range(CLASSES):
		if targets[i][j] > 0.5:
			if outputs[i][j] > 0.5: 
				TP[j] += 1
			else: 
				FN[j] += 1
				exact_match = False
		else:
			if outputs[i][j] > 0.5: 
				FP[j] += 1
				exact_match = False
			else:
				TN[j] += 1
	if exact_match:
		exact_matches += 1
accuracy = [ float(TP[j]+TN[j])/float(TP[j]+TN[j]+FP[j]+FN[j])  for j in range(CLASSES)]
precision = [ float(TP[j])/float(TP[j]+FP[j]) if TP[j]+FP[j] > 0 else 0.0 for j in range(CLASSES)]
recall = [ float(TP[j])/float(TP[j]+FN[j]) if TP[j]+FN[j] > 0 else 0.0 for j in range(CLASSES)]

#calculate global metrics
EMR = float(exact_matches) / targets.shape[0]
global_accuracy = float(sum(TP)+sum(TN))/float(sum(TP)+sum(TN)+sum(FP)+sum(FN))
global_sensitivity = 0.0
if float(sum(TP)+sum(FN)) > 0: global_sensitivity = float(sum(TP))/float(sum(TP)+sum(FN))
global_specificity = 0.0
if float(sum(FP)+sum(TN)) > 0: global_specificity = float(sum(TN))/float(sum(FP)+sum(TN))
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
print("\nExact Match Ratio:\n"+str(EMR))
 
