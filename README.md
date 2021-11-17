# DrugSideEffects

This project aims at predicting side effects of drugs based on their chemical and structural features, their interactions with the (metabolic) network of genes, and their (structure) similarity relationships. Drugs and genes are represented as nodes of a heterogeneous graph. Gene-gene interactions, drug-gene interactions, and drug-drug similarity relationships are represented as distinct sets of edges in the graph. The associations between drugs and side-effects are determined through a node multilabel classification task, where drug nodes are classified over the set of side-effects they are associated to. Drug features include chemical descriptors and molecular fingerprints (vectors describing the functional groups that appear in each drug). Gene features include chromosome and strand information and the molecular function ontology. Transductive learning experiments were carried out: in this framework, side-effect supervisions were exploited as transductive features, in order to better train the network for the task it has to solve (finding side-effects of new drugs, given side-effects of drugs which were observed in the past). 

# Dependencies

* Graph Neural Networks are implemented with the GNN software described here: 
https://github.com/NickDrake117/GNN_tf_2.x

Niccolò Pancino et al. Graph Neural Networks for the Prediction of Protein-Protein Interfaces, 28th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning.

* Composite Graph Neural Networks are implemented with this repository:
https://github.com/NickDrake117/CompositeGNN

Work in progress

* Graph Convolution Networks are implemented with Spektral:
https://github.com/danielegrattarola/spektral

Thomas Kipf, Max Welling. Semi-supervised classification with Graph Convolutional Networks, International Conference on Learning Representations, 2017.

Daniele Grattarola, Cesare Alippi. Graph Neural Networks in TensorFlow and Keras with Spektral. International Conference on Machine Learning, Graph Representation Learning workshop, 2020.


* All the packages listed above require Tensorflow to run:
https://www.tensorflow.org/

Martín Abadi et al. TensorFlow: Large-scale machine learning on heterogeneous systems, 2015. Software available from tensorflow.org.


# Data Sources

The graph was built according to data coming from multiple sources. 

* Gene-Gene links were determined according to the HUman Reference Interactome (HURI):
http://www.interactome-atlas.org/

Katja Luck et al. A reference map of the human binary protein interactome. Nature 580: 402-408, 2020.

* Gene Features were obtained from BioMart:
https://www.ensembl.org/biomart
http://www.biomart.org/

Damian Smedley et al. The BioMart community portal: an innovative alternative to large centralized data repositories. Nucleic Acids Research 43(W1):W589-W598, 2015.

* Drug-Gene links were extracted from STITCH:
http://stitch.embl.de/

Michael Kuhn et al. Interaction networks of chemical and proteins. Nucleic Acids Research 36(Suppl_1):D684-D688, 2008.

* Drug-Side-Effect associations were downloaded from SIDER:
http://sideeffects.embl.de/

Michael Kuhn et al. The SIDER database of drugs and side effects. Nucleic Acids Research 44(D1):D1075-D1079, 2016.

* Chemical descriptors of drugs were retrieved on PubChem:
https://pubchem.ncbi.nlm.nih.gov/

Sunghwan Kim et al. PubChem in 2021: new data content and improved web interfaces. Nucleic Acids Research 49(D1):D1388–D1395, 2021. doi:10.1093/nar/gkaa971

* Drug-drug links were based on the Tanimoto similarity of chemical fingerprints of drugs, obtained with RdKit. Fingerprints were also used as additional drug features:
https://www.rdkit.org/

* Molecular function features, based on the Gene Ontology, were extracted with DAVID:
http://geneontology.org/
https://david.ncifcrf.gov/tools.jsp

Michael Ashburner et al. Gene ontology: tool for the unification of biology. Nature Genetics 25(1):25-9, 2000.

Glynn Dennis Jr. et al. DAVID: Database for Annotation, Visualization, and Integrated Discovery. Genome Biology 4:R60, 2003.

