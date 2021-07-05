#coding=utf-8
import sys
import os
import subprocess

#input parameters
model = sys.argv[1] #can either be "gnn", "gcn" or "gsage"
run_id = sys.argv[2]

#determine path
path_dict = {	"gnn": "GNN_node_classifier.py",
				"gcn": "spektral_node_classifier.py",
				"gsage": "graphsage_node_classifier.py"
}
if model not in ["gnn", "gcn", "gsage"]:
	sys.exit("Unknown model : "+model)
path = path_dict[model]

#launch multiple runs
for i in range(10):
	subprocess.run(["python", path, run_id+"_"+str(i+1)])
