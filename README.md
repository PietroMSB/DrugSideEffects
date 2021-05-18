# DrugSideEffects

This project aims at predicting side effects of drugs based on their chemical and structural features, their interactions with the (metabolic) network of genes, and their (structure) similarity relationships. Side effects, drugs, and genes are represented as node of a heterogeneous graph. Gene-gene interactions, drug-gene interactions, and drug-drug similarity relationships are represented as distinct sets of edges in the graph. The associations between drugs and side-effects are determined through a link prediction between the set of drug nodes and the set of side-effect nodes.

# Dependencies

Graph Neural Networks are implemented with the GNN software described here: 
[cita]
[link]

Graph Convolution Networks are implemented with Spektral:
[cita]
[link]

Both packages require Tensorflow to run:
[cita]
[link]

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
