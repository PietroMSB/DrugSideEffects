# DrugSideEffects

This project aims at predicting side effects of drugs based on their chemical and structural features, their interactions with the (metabolic) network of genes, and their (structure) similarity relationships. Side effects, drugs, and genes are represented as node of a heterogeneous graph. Gene-gene interactions, drug-gene interactions, and drug-drug similarity relationships are represented as distinct sets of edges in the graph. The associations between drugs and side-effects are determined through a link prediction between the set of drug nodes and the set of side-effect nodes.

# Dependencies

* Graph Neural Networks are implemented with the GNN software described here: 
https://github.com/NickDrake117/GNN_tf_2.x

Niccolò Pancino et al. Graph Neural Networks for the Prediction of Protein-Protein Interfaces, 28th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning.


 * Graph Convolution Networks are implemented with Spektral:
https://github.com/danielegrattarola/spektral

Thomas Kipf, Max Welling. Semi-supervised classification with Graph Convolutional Networks, International Conference on Learning Representations, 2017.

Daniele Grattarola, Cesare Alippi. Graph Neural Networks in TensorFlow and Keras with Spektral. International Conference on Machine Learning, Graph Representation Learning workshop, 2020.


* Both packages require Tensorflow to run:
https://www.tensorflow.org/

Martín Abadi et al. TensorFlow: Large-scale machine learning on heterogeneous systems, 2015. Software available from tensorflow.org.


# Data Sources

The graph was built according to data coming from multiple sources. 

Gene-Gene links were determined according to the HUman Reference Interactome (HURI):
[cita]
[link]

Drug-Gene links were extracted from STITCH:
[cita]
[link]

Drug-Side-Effect associations were downloaded from SIDER:
[cita]
[link]

Chemical descriptors of drugs were retrieved on PubChem:
[cita]
[link]
