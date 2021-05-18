from __future__ import annotations

from typing import Union, Optional

import numpy as np

from GNN.graph_class import GraphObject


#######################################################################################################################
# FUNCTIONS ###########################################################################################################
#######################################################################################################################

# ---------------------------------------------------------------------------------------------------------------------
def randomGraph(nodes_number: int, dim_node_label: int, dim_arc_label: int, dim_target: int, density: float,
                *, normalize_features: bool = False, aggregation: str = 'average',
                problem_based: str = 'n') -> GraphObject:
    """ Create randoms nodes and arcs matrices, such that label of arc (i,j) == (j,i)
    :param nodes_number: number of nodes belonging to the graph
    :param dim_node_label: number of components for a generic node's label
    :param dim_arc_label: number of components for a generic arc's label
    :param dim_target: number of components for a generic target 1-hot
    :param density: define the "max" density for the graph
    :param normalize_features: (bool) if True normalize the column of the labels, otherwise raw data will be considered
    :param aggregation: (str) in ['average','normalized','sum']. Default 'average'. See GraphObject.ArcNode for details
    :param problem_based: (str) in ['n','a','g']: 'n'-nodeBased; 'a'-arcBased; 'g'-graphBased
    :return: GraphObject
    """
    # NODES
    nodes_ids = range(nodes_number)
    nodes = 2 * np.random.random((nodes_number, dim_node_label)) - 1

    # ARCS
    arcs_number = round(density * nodes_number * (nodes_number - 1) / 2)
    sources = np.random.choice(nodes_ids[:-1], arcs_number // 2)
    # max_diff is the maximum id for picking the destination node when random.random() is called
    max_diff = np.ones_like(sources) * nodes_number - sources - 1
    # destination obtained by adding a random id from 'source' to nodes_number
    destination = sources + np.ceil(max_diff * np.random.random(len(sources)))
    # arcs id node1 - id node2
    arcs_ascend = np.zeros((arcs_number // 2, 2))
    arcs_ascend[:, 0] = sources
    arcs_ascend[:, 1] = destination
    arcs_ascend = np.unique(arcs_ascend, axis=0)
    arcs_descend = np.flip(arcs_ascend, axis=1)
    # arc labels
    arcs_ids = np.concatenate((arcs_ascend, arcs_descend))
    arcs_label = 2 * np.random.random((arcs_ascend.shape[0], dim_arc_label)) - 1
    arcs_label = np.concatenate((arcs_label, arcs_label))
    arcs = np.concatenate((arcs_ids, arcs_label), axis=1)
    arcs = np.unique(arcs, axis=0)

    # TARGETS - 1-HOT
    tn = {'n': nodes.shape[0], 'a': arcs.shape[0], 'g': 1}
    assert problem_based in tn.keys()
    target_number = tn[problem_based]
    targs = np.zeros((target_number, dim_target))
    # clusters
    if problem_based in ['a', 'n']:
        from sklearn.cluster import AgglomerativeClustering
        j = AgglomerativeClustering(n_clusters=dim_target).fit(arcs[:, 2:] if problem_based == 'a' else nodes)
        i = range(target_number)
        targs[i, j.labels_] = 1
    else: targs[0, np.random.choice(range(targs.shape[1]))] = 1

    # OUTPUT MASK
    output_mask = np.ones(arcs.shape[0] if problem_based == 'a' else nodes.shape[0])

    # NORMALIZE FEATURES
    if normalize_features:
        nodes = nodes / np.max(nodes, axis=0)
        arcs[:, 2:] = arcs[:, 2:] / np.max(arcs[:, 2:], axis=0)

    # RETURN GRAPH
    return GraphObject(arcs=arcs, nodes=nodes, targets=targs, problem_based=problem_based,
                       output_mask=output_mask, node_aggregation=aggregation)


# ---------------------------------------------------------------------------------------------------------------------
def simple_graph(problem_based: str, aggregation: str = 'average') -> GraphObject:
    """ return a single simple GraphObject for debugging purpose """
    nodes = np.array([[11, 21], [12, 22], [13, 23], [14, 24]])
    arcs = np.array([[0, 1, 10], [0, 2, 40], [1, 0, 10], [1, 2, 20], [2, 0, 40], [2, 1, 20], [2, 3, 30], [3, 2, 30]])
    tn = {'n': nodes.shape[0], 'a': arcs.shape[0], 'g': 1}
    target_number = tn[problem_based]
    targs = np.zeros((target_number, 2))
    if problem_based in ['a', 'n']:
        from sklearn.cluster import AgglomerativeClustering
        j = AgglomerativeClustering(n_clusters=2).fit(arcs[:, 2:] if problem_based == 'a' else nodes)
        i = range(target_number)
        targs[i, j.labels_] = 1
    else: targs[0, 1] = 1
    return GraphObject(arcs=arcs, nodes=nodes, targets=targs, problem_based=problem_based, node_aggregation=aggregation)


# ---------------------------------------------------------------------------------------------------------------------
def progressbar(percent: float, width: int = 30) -> None:
    """ Print a progressbar, given a percentage in [0,100] and a fixed length """
    left = round(width * percent / 100)
    right = int(width - left)
    print('\r[', '#' * left, ' ' * right, ']', f' {percent:.1f}%', sep='', end='', flush=True)


# ---------------------------------------------------------------------------------------------------------------------
def getindices(len_dataset: int, perc_Train: float = 0.7, perc_Valid: float = 0.1, seed=None) -> Union[
    tuple[list[int], list[int]], tuple[list[int], list[int], list[int]]]:
    """ Divide the dataset into Train/Test or Train/Validation/Test
    :param len_dataset: length of the dataset
    :param perc_Train: (float) in [0,1]
    :param perc_Valid: (float) in [0,1]
    :param seed: (float/None/False) Fixed shuffle mode / random shuffle mode / no shuffle performed
    :return: 2 or 3 list of indices
    """
    if perc_Train < 0 or perc_Valid < 0 or perc_Train + perc_Valid > 1:
        raise ValueError('Error - percentage must stay in [0-1] and their sum must be <= 1')
    
    # shuffle elements
    idx = list(range(len_dataset))
    if seed: np.random.seed(seed)
    if seed is not False: np.random.shuffle(idx)
    
    # samples
    perc_Test = 1 - perc_Train - perc_Valid
    sampleTest = round(len_dataset * perc_Test)
    sampleValid = round(len_dataset * perc_Valid)
    
    # test indices
    test_idx = idx[:sampleTest]
    
    # validation indices
    valid_idx = idx[sampleTest:sampleTest + sampleValid]
    
    # train indices (usually the longest set)
    train_idx = idx[sampleTest + sampleValid:]
    return (train_idx, test_idx, valid_idx)


# ---------------------------------------------------------------------------------------------------------------------
def getSet(glist: list[GraphObject], set_indices: list[int], problem_based: str, node_aggregation: str,
           verbose: bool = False) -> list[GraphObject]:
    """ get the Set from a dataset given its set of indices
    :param glist: (list of GraphObject or str) dataset from which the set is picked
    :param set_indices: (list of int) indices of the elements belonging to the Set
    :param problem_based: (str) in ['n','a','g'] defining the problem to be faced: [node, arcs, graph]-based
    :param verbose: (bool) if True print the progressbar, else silent mode
    :return: list of GraphObject, composing the Set
    """
    # check type
    if not (type(glist) == list and all(isinstance(x, str) for x in glist)):
        raise TypeError('type of param <glist> must be list of str \'path-like\' or GraphObjects')
    
    # get set
    length, setlist = len(set_indices), list()
    for i, elem in enumerate(set_indices):
        setlist.append(glist[elem])
        if verbose: progressbar((i + 1) * 100 / length)
    setlist = [GraphObject.load(i, problem_based=problem_based, node_aggregation=node_aggregation) for i in setlist]
    return setlist


# ---------------------------------------------------------------------------------------------------------------------
def getbatches(glist: list[GraphObject], node_aggregation: str, batch_size: int = 32, number_of_batches=None,
               one_graph_per_batch=True) -> Union[list[GraphObject], list[list[GraphObject]]]:
    """ Divide the Set into batches, in which every batch is a GraphObject or a list of GraphObject
    :param glist: (list of GraphObject) to be split into batches
    :param batch_size: (int) specify the size of a normal batch. Note: len(batches[-1])<=batch_size
    :param number of batches: (int) specify in how many batches glist will be partitioned.
                                > Default value is None; if given, param <batch_size> will be ignored.
    :param one_graph_per_batch: (bool) if True, all graphs belonging to a batch are merged to form a single GraphObject
    :return: a list of batches
    """
    if number_of_batches is None: batches = [glist[i:i + batch_size] for i in range(0, len(glist), batch_size)]
    else: batches = [list(i) for i in np.array_split(glist, number_of_batches)]
    if one_graph_per_batch: batches = [GraphObject.merge(i, node_aggregation=node_aggregation) for i in batches]
    return batches


# ---------------------------------------------------------------------------------------------------------------------
def normalize_graphs(gTr: Union[GraphObject, list[GraphObject]], gVa: Optional[Union[GraphObject, list[GraphObject]]],
                     gTe: Optional[Union[GraphObject, list[GraphObject]]], based_on: str = 'gTr',
                     norm_rangeN: Optional[Union[tuple[float, float]]] = None,
                     norm_rangeA: Optional[Union[tuple[float, float], None]] = None) -> None:
    """ Normalize graph by using gTr or gTr+gVa+gTe information
    :param gTr: (GraphObject or list of GraphObjects) for Training Set
    :param gVa: (GraphObject or list of GraphObjects or None) for Validation Set
    :param gTe: (GraphObject or list of GraphObjects or None) for Test Set
    :param based_on: (str) in ['gTr','all']. If 'gTr', ony info on gTr are used; if 'all', entire dataset info are used
    """

    def checktype(g: Union[list[GraphObject], None], name: str):
        """ check g: it must be a GraphObect or a list of GraphObjects """
        if g is None: return list()
        if not (type(g) == GraphObject or (type(g) == list and all(isinstance(x, GraphObject) for x in g))):
            raise TypeError('type of param <{}> must be GraphObject or list of Graphobjects'.format(name))
        return g if type(g) == list else [g]

    # check if inputs are GraphObject OR list(s) of GraphObject(s) and the normalization method
    gTr, gVa, gTe = map(checktype, [gTr, gVa, gTe], ['gTr', 'gVa', 'gTe'])
    if based_on not in ['gTr', 'all']: raise ValueError('param <based_on> must be \'gTr\' or \'all\'')
    
    # create list of graphs to merge
    g2merge = gTr[:]
    if based_on == 'all': g2merge += gVa[:] + gTe[:]
    
    # merge all the graphs into a single one
    G = GraphObject.merge(g2merge, node_aggregation='average')
   
    # G nodes label max and min
    LabelNodesMAX, LabelNodesMIN = np.max(G.nodes, axis=0), np.min(G.nodes, axis=0)
    LabelNodesMAX[LabelNodesMAX == 0] = 1  # s.t. a 0 element does not force NaN values
    
    # G arcs label max and min
    LabelArcsMAX, LabelArcsMIN = np.max(G.arcs[:, 2:], axis=0), np.min(G.arcs[:, 2:], axis=0)
    LabelArcsMAX[LabelArcsMAX == 0] = 1  # s.t. a 0 element does not force NaN values
    
    # normalize graphs
    for i in gTr + gVa + gTe:
        i.nodes = i.nodes / LabelNodesMAX if norm_rangeN is None else \
            norm_rangeN[0] + (norm_rangeN[1] - norm_rangeN[0]) * (i.nodes - LabelNodesMIN) / (LabelNodesMAX - LabelNodesMIN)
        i.arcs[:, 2:] = i.arcs[:, 2:] / LabelArcsMAX if norm_rangeA is None else \
            norm_rangeA[0] + (norm_rangeA[1] - norm_rangeA[0]) * (i.arcs[:, 2:] - LabelArcsMIN) / (LabelArcsMAX - LabelArcsMIN)


# ---------------------------------------------------------------------------------------------------------------------
def MLP(input_dim: int, layers: list[int], activations, kernel_initializer, bias_initializer,
        dropout_percs: Union[list[float], float, None] = None, dropout_pos: Optional[Union[list[int], int]] = None,
        alphadropout: bool = False):
    """ Quick building function for MLP model. All lists must have the same length
    :param input_dim: (int) specify the input dimension for the model
    :param layers: (int or list of int) specify the number of units in every layers
    :param activations: (functions or list of functions)
    :param kernel_initializer: (initializers or list of initializers) for weights initialization (NOT biases)
    :param bias_initializer: (initializers or list of initializers) for biases initialization (NOT weights)
    :return: Sequential (MLP) model
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, AlphaDropout

    # check type
    if type(activations) != list: activations = [activations for i in layers]
    if type(kernel_initializer) != list: kernel_initializer = [kernel_initializer for i in layers]
    if type(bias_initializer) != list: bias_initializer = [bias_initializer for i in layers]
    if type(dropout_pos) == int:  dropout_pos = [dropout_pos]
    if type(dropout_percs) == float: dropout_percs = [dropout_percs for i in dropout_pos]
    if dropout_percs == None or dropout_pos == None: dropout_percs, dropout_pos = list(), list()

    # check lengths
    if not (len(activations) == len(kernel_initializer) == len(bias_initializer) == len(layers)):
        raise ValueError('Dense parameters must have the same length to be correctly processed')
    if len(dropout_percs) != len(dropout_pos):
        raise ValueError('Dropout parameters must have the same length to be correctly processed')

    # Dense layers
    keys = ['units', 'activation', 'kernel_initializer', 'bias_initializer']
    vals = [[layers[i], activations[i], kernel_initializer[i], bias_initializer[i]] for i in range(len(layers))]
    params = [dict(zip(keys, i)) for i in vals]

    # Dropout layers
    if dropout_percs and dropout_pos:
        dropout_pos = list(np.array(dropout_pos) + np.arange(len(dropout_pos)))
        for i, elem in enumerate(dropout_percs): params.insert(dropout_pos[i], {'rate': elem})

    # set input shape for first layer
    params[0]['input_shape'] = (input_dim,)

    # return MLP model
    dropout = AlphaDropout if alphadropout else Dropout
    mlp_layers = [Dense(**i) if 'units' in i else dropout(**i) for i in params]
    return Sequential(mlp_layers)


# ---------------------------------------------------------------------------------------------------------------------
def get_inout_dims(net_name: str, dim_node_label: int, dim_arc_label: int, dim_target: int, problem: str, dim_state: int,
                   hidden_units: Union[None, int, list[int]],
                   *, layer: int = 0, get_state: bool = False, get_output: bool = False) -> tuple[int, list[int]]:
    """ Calculate input and output dimension for the MLP of state and output
    :param g: (GraphObject) generic graph of the dataset calculations are based on
    :param problem: (str) s.t. len(problem)=3 [{'c','r'} | {'a','n','g'} | {'1','2'}]
    :param net_name: (str) in ['state','output']
    :param dim_state: (int)>=0 for state dimension paramenter of the gnn
    :param hidden_units: (int or list of int) for specifying units on hidden layers
    :param layer: (int) LGNN USE: get the dims at gnn of the layer <layer>, from graph dims on layer 0. Default is 0, since GNN==LGNN in this case
    :param get_state: (bool) LGNN USE: set accordingly to LGNN behaviour, if gnns get state, output or both from previous layer
    :param get_output: (bool) LGNN USE: set accordingly to LGNN behaviour, if gnns get state, output or both from previous layer
    """
    assert net_name in ['state', 'output']
    assert layer >= 0
    if len(problem) == 1: problem += '1'
    DS = dim_state
    NL, AL, T = dim_node_label, dim_arc_label, dim_target

    # if LGNN, get MLPs layers for gnn in layer 2+
    if layer > 0:
        GS, GO, P = get_state, get_output, problem[0]
        if DS != 0: NL, AL = NL + DS * GS + T * (P != 'a') * GO, AL + T * (P == 'a') * GO
        else: NL, AL = NL + layer * NL * GS + ((layer - 1) * GS + 1) * T * (P != 'a') * GO, AL + T * (P == 'a') * GO

    # MLP output
    if net_name == 'output':
        input_shape = (problem[0] == 'a') * (NL + AL + DS) + NL + dim_state
        output_shape = T

    # MLP state
    else:
        input_shape = AL + 2 * NL + DS * (1 + (problem[1] == '1'))
        input_shape += NL * (DS == 0) * (problem[1] == '2')
        output_shape = DS if DS else NL

    # hidden part
    if hidden_units is None or type(hidden_units) == int and hidden_units <= 0: hidden_units = []
    if type(hidden_units) == list: layers = hidden_units + [output_shape]
    else: layers = [hidden_units, output_shape]
    return input_shape, layers
