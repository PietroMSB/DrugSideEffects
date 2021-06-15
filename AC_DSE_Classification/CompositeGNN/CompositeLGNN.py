from __future__ import annotations

from typing import Union, Optional

import tensorflow as tf
from numpy import array

from CompositeGNN.CompositeGNN import CompositeGNNnodeBased, CompositeGNNgraphBased, CompositeGNNedgeBased
from CompositeGNN.CompositeBaseClass import CompositeBaseClass
from CompositeGNN.composite_graph_class import CompositeGraphObject, CompositeGraphTensor


class CompositeLGNN(CompositeBaseClass):
    ## CONSTRUCTORS METHODS ###########################################################################################
    def __init__(self,
                 gnns: list[CompositeGNNnodeBased, CompositeGNNedgeBased, CompositeGNNgraphBased],
                 get_state: bool,
                 get_output: bool,
                 optimizer: tf.keras.optimizers.Optimizer,
                 loss_function: tf.keras.losses.Loss,
                 loss_arguments: Optional[dict],
                 addressed_problem: str,
                 extra_metrics: Optional[dict] = None,
                 extra_metrics_arguments: Optional[dict[str, dict]] = None,
                 path_writer: str = 'writer/',
                 namespace: str = 'LGNN') -> None:
        """ CONSTRUCTOR

        :param gnns: (list) of instances of type GNN representing LGNN layers, initialized externally.
        :param get_state: (bool) if True node_state are propagated through LGNN layers.
        :param get_output: (bool) if True gnn_outputs on nodes/arcs are propagated through LGNN layers.
        :param optimizer: (tf.keras.optimizers) for gradient application, initialized externally.
        :param loss_function: (tf.keras.losses) for the loss computation.
        :param loss_arguments: (dict) with some {'argument':values} one could pass to loss when computed.
        :param addressed_problem: (str) in ['r','c'], 'r':regression, 'c':classification for the addressed problem.
        :param extra_metrics: None or dict {'name':function} for metrics to be watched during training/validation/test procedures.
        :param extra_metrics_arguments: None or dict {'name':{'argument':value}} for arguments passed to extra_metrics['name'].
        :param path_writer: (str) path for saving TensorBoard objects in training procedure. If folder is not empty, all files are removed.
        :param namespace: (str) namespace for tensorboard visualization.
        """

        GNNS_TYPE = set([type(i) for i in gnns])
        if len(GNNS_TYPE) != 1: raise TypeError('parameter <gnn> must contain gnns of the same type')

        # BaseGNN constructor - number of nodes' type == len(gnn[0].net_state), as all GNNs layer process the same number of nodes' type
        super().__init__(len(gnns[0].net_state), optimizer, loss_function, loss_arguments, addressed_problem, extra_metrics, extra_metrics_arguments,
                         path_writer, namespace)

        ### LGNNs parameter
        self.get_state = get_state
        self.get_output = get_output
        self.gnns = gnns
        self.LAYERS = len(gnns)
        self.GNNS_TYPE = list(GNNS_TYPE)[0]
        self.namespace = [f'{namespace} - GNN{i}' for i in range(self.LAYERS)]
        self.training_mode = None

        # Change namespace for self.gnns
        for gnn, name in zip(self.gnns, self.namespace):
            gnn.namespace = [name]
            gnn.path_writer = f'{self.path_writer}{name}/'

    # -----------------------------------------------------------------------------------------------------------------
    def copy(self, *, path_writer: str = '', namespace: str = '', copy_weights: bool = True) -> 'LGNN':
        """ COPY METHOD

        :param path_writer: None or (str), to save copied lgnn tensorboard writer. Default in the same folder + '_copied'.
        :param namespace: (str) for tensorboard visualization in model training procedure.
        :param copy_weights: (bool) True: state and output weights of gnns are copied in new lgnn, otherwise they are re-initialized.
        :return: a Deep Copy of the LGNN instance.
        """
        return self.__class__(gnns=[i.copy(copy_weights=copy_weights) for i in self.gnns], get_state=self.get_state,
                              get_output=self.get_output,
                              optimizer=self.optimizer.__class__(**self.optimizer.get_config()), loss_function=self.loss_function,
                              loss_arguments=self.loss_args, addressed_problem=self.addressed_problem, extra_metrics=self.extra_metrics,
                              extra_metrics_arguments=self.mt_args,
                              path_writer=path_writer if path_writer else self.path_writer + '_copied/',
                              namespace=namespace if namespace else 'LGNN')

    ## SAVE AND LOAD METHODs ##########################################################################################
    def save(self, path: str):
        """ Save model to folder <path> """
        from json import dump

        # check paths
        if path[-1] != '/': path += '/'

        # save GNNs
        for i, gnn in enumerate(self.gnns): gnn.save(f'{path}GNN{i}/')

        # save configuration file in json format
        gnns_type = {CompositeGNNnodeBased: 'n', CompositeGNNedgeBased: 'a', CompositeGNNgraphBased: 'g'}
        config = {'get_state': self.get_state, 'get_output': self.get_output,
                  'loss_function': tf.keras.losses.serialize(self.loss_function), 'loss_arguments': self.loss_args,
                  'optimizer': str(tf.keras.optimizers.serialize(self.optimizer)),
                  'addressed_problem': self.addressed_problem, 'gnns_type': gnns_type[self.GNNS_TYPE]}

        with open(f'{path}config.json', 'w') as json_file:
            dump(config, json_file)

    # -----------------------------------------------------------------------------------------------------------------
    @classmethod
    def load(self, path: str, path_writer: str, namespace: str = 'LGNN',
             extra_metrics: Optional[dict] = None, extra_metrics_arguments: Optional[dict[str, dict]] = None):
        """ Load model from folder <path>.

        Only Loss is considered as metrics after loading process.
        To use more metrics, set :param extra_metrics: and :param extra_metrics_arguments:

        :param path: (str) folder path containing all useful files to load the model.
        :param path_writer: (str) path for writer folder. !!! Constructor method deletes a non-empty folder and makes a new empty one.
        :param namespace: (str) namespace for tensorboard visualization in model training procedure.
        :return: the loaded lgnn model.
        """
        from json import loads
        from os import listdir
        from os.path import isdir

        # check paths
        if path[-1] != '/': path += '/'

        # load configuration file
        with open(f'{path}config.json', 'r') as read_file:
            config = loads(read_file.read())

        # load GNNs
        gnns_type = {'n': CompositeGNNnodeBased, 'a': CompositeGNNedgeBased, 'g': CompositeGNNgraphBased}
        gnns_type = gnns_type[config.pop('gnns_type')]
        gnns = [gnns_type.load(f'{path}{i}', path_writer=f'{path_writer}{namespace} - GNN{i}/', namespace='GNN')
                for i in listdir(path) if isdir(f'{path}{i}')]

        # get optimizer, loss function
        optz = tf.keras.optimizers.deserialize(eval(config.pop('optimizer')))
        loss = tf.keras.losses.deserialize(config.pop('loss_function'))

        return self(gnns=gnns, optimizer=optz, loss_function=loss,
                    extra_metrics=extra_metrics, extra_metrics_arguments=extra_metrics_arguments,
                    path_writer=path_writer, namespace=namespace, **config)

    ## GETTERS AND SETTERS METHODs ####################################################################################
    def get_dense_layers(self) -> list[tf.keras.layers.Layer]:
        """ Get dense layer for the application of regularizers in training time """
        return [layer for gnn in self.gnns for layer in gnn.get_dense_layers()]

    def trainable_variables(self) -> tuple[list[list[tf.Tensor]], list[list[tf.Tensor]]]:
        """ Get tensor weights for net_state and net_output for each gnn layer """
        #return [i.net_state.trainable_variables for i in self.gnns], [i.net_output.trainable_variables for i in self.gnns]
        return [[n.trainable_variables for n in i.net_state] for i in self.gnns], [i.net_output.trainable_variables for i in self.gnns]

    # -----------------------------------------------------------------------------------------------------------------
    def get_weights(self) -> tuple[list[list[array]], list[list[array]]]:
        """ Get array weights for net_state and net_output for each gnn layer """
        #return [i.net_state.get_weights() for i in self.gnns], [i.net_output.get_weights() for i in self.gnns]
        return [[n.get_weights() for n in i.net_state] for i in self.gnns], [i.net_output.get_weights() for i in self.gnns]

    # -----------------------------------------------------------------------------------------------------------------
    def set_weights(self, weights_state: list[list[array]], weights_output: list[list[array]]) -> None:
        """ Set weights for net_state and net_output for each gnn layer """
        assert len(weights_state) == len(weights_output) == self.LAYERS
        for gnn, wst, wout in zip(self.gnns, weights_state, weights_output):
            #gnn.net_state.set_weights(wst)
            for n, w in zip(gnn.net_state, wst): n.set_weights(w)
            gnn.net_output.set_weights(wout)

    ## CALL/PREDICT METHOD ############################################################################################
    def __call__(self, g: Union[CompositeGraphObject, CompositeGraphTensor]) -> tf.Tensor:
        """ Return ONLY the LGNN output for graph g of type CompositeGraphObject """
        out = self.Loop(g, training=False)[-1]
        return out[-1]

    # -----------------------------------------------------------------------------------------------------------------
    def predict(self, g: Union[CompositeGraphObject, CompositeGraphTensor], idx: Union[int, list[int], range] = -1) -> Union[tf.Tensor, list[tf.Tensor]]:
        """ Get LGNN one or more output(s) in test mode (training == False) for graph g of type CompositeGraphObject
        :param g: (CompositeGraphObject) single CompositeGraphObject element
        :param idx: set the layer whose output is wanted to be returned.
                    More than one layer output can be returned, setting idx as ordered list/range
        :return: a list of output(s) of the model processing graph g
        """
        if type(idx) == int:
            assert idx in range(-self.LAYERS, self.LAYERS)
        elif isinstance(idx, (list, range)):
            assert all(i in range(-self.LAYERS, self.LAYERS) for i in idx) and list(idx) == sorted(idx)
        else:
            raise ValueError('param <idx> must be int or list of int in range(-self.LAYERS, self.LAYERS)')

        # transform CompositeGraphObject in CompositeGraphTensor
        if isinstance(g, CompositeGraphObject): g = CompositeGraphTensor.fromGraphObject(g)

        # get only outputs, without iteration and states
        out = self.Loop(g, training=False)[-1]
        return out[idx] if type(idx) == int else [out[i] for i in idx]

    ## EVALUATE METHODS ###############################################################################################
    def evaluate_single_graph(self, g: Union[CompositeGraphObject, CompositeGraphTensor], training: bool) -> tuple:
        """ Evaluate single CompositeGraphObject element g in test mode (training == False)

        :param g: (CompositeGraphObject or CompositeGraphTensor) single CompositeGraphObject element
        :param training: (bool) set internal models behavior, s.t. they work in training or testing mode
        :return: (tuple) convergence iteration (int), loss value (matrix), target and output (matrices) of the model
        """
        # transform CompositeGraphObject in CompositeGraphTensor
        if isinstance(g, CompositeGraphObject): g = CompositeGraphTensor.fromGraphObject(g)

        # get targets and loss_weights
        targs = self.GNNS_TYPE.get_filtered_tensor(g, g.targets)
        loss_weights = self.GNNS_TYPE.get_filtered_tensor(g, g.sample_weights)

        # graph processing
        it, _, out = self.Loop(g, training=training)

        # loss computation
        if training and self.training_mode == 'residual':
            loss = self.loss_function(targs, tf.reduce_mean(out, axis=0), **self.loss_args) * loss_weights
        else:
            loss = tf.reduce_mean([self.loss_function(targs, o, **self.loss_args) * loss_weights for o in out], axis=0)

        return it, loss, targs, out[-1]

    ## LOOP METHODS ###################################################################################################
    def update_graph(self, g: CompositeGraphTensor, state: Union[tf.Tensor, array], output: Union[tf.Tensor, array]) -> CompositeGraphObject:
        """
        :param g: (CompositeGraphTensor) single CompositeGraphTensor element the update process is based on
        :param state: (tensor) output of the net_state model of a single gnn layer
        :param output: (tensor) output of the net_output model of a single gnn layer
        :return: (CompositeGraphTensor) a new CompositeGraphTensor where actual state and/or output are integrated in nodes/arcs label
        """
        # copy graph to preserve original graph data
        g = g.copy()

        # define tensors with shape[1]==0 so that it can be concatenate with tf.concat()
        nodeplus = tf.zeros((g.nodes.shape[0], 0), dtype='float32')
        arcplus = tf.zeros((g.arcs.shape[0], 0), dtype='float32')

        # check state
        if self.get_state: nodeplus = tf.concat([nodeplus, state], axis=1)

        # check output
        if self.get_output:
            # process output to make it shape compatible.
            # Note that what is concatenated is not nodeplus/arcplus, but out, as it has the same length of nodes/arcs
            mask = tf.logical_and(g.set_mask, g.output_mask)

            # scatter_nd creates a zeros matrix 'node or arcs-compatible' with the elements of output located in mask==True
            out = tf.scatter_nd(tf.where(mask), output, shape=(len(mask), output.shape[1]))

            if self.GNNS_TYPE == CompositeGNNedgeBased:
                arcplus = tf.concat([arcplus, out], axis=1)
            else:
                nodeplus = tf.concat([nodeplus, out], axis=1)

        g.nodes = tf.concat([nodeplus, g.nodes], axis=1)
        g.arcs = tf.concat([arcplus, g.arcs], axis=1)
        g.DIM_NODE_LABELS = g.DIM_NODE_LABELS + nodeplus.shape[1]
        return g

    # -----------------------------------------------------------------------------------------------------------------
    def Loop(self, g: Union[CompositeGraphObject, CompositeGraphTensor], *, training: bool = False) -> tuple[list[tf.Tensor], tf.Tensor, list[tf.Tensor]]:
        """ Process a single CompositeGraphObject/CompositeGraphTensor element g, returning iteration, states and output """

        # transform CompositeGraphObject in CompositeGraphTensor
        if isinstance(g, CompositeGraphObject): g = CompositeGraphTensor.fromGraphObject(g)

        # copy graph, to preserve original one by the state/output integrating process
        gtmp = g.copy()

        # forward pass
        K, outs = list(), list()
        for idx, gnn in enumerate(self.gnns[:-1]):

            if type(gnn) == CompositeGNNgraphBased:
                k, state, out = super(CompositeGNNgraphBased, gnn).Loop(gtmp, training=training)
                outs.append(tf.matmul(gtmp.NodeGraph, out, transpose_a=True))

            else:
                k, state, out = gnn.Loop(gtmp, training=training)
                outs.append(out)

            K.append(k)

            # update graph with state and output of the current GNN layer, to feed next GNN layer
            gtmp = self.update_graph(g, state, out)

        k, state, out = self.gnns[-1].Loop(gtmp, training=training)
        return K + [k], state, outs + [out]

    ## TRAINING METHOD ################################################################################################
    def train(self, gTr: Union[CompositeGraphObject, CompositeGraphTensor, list[CompositeGraphObject, CompositeGraphTensor]], epochs: int,
              gVa: Union[CompositeGraphObject, CompositeGraphTensor, list[CompositeGraphObject, CompositeGraphTensor]] = None,
              update_freq: int = 10, max_fails: int = 10, observed_metric: str = 'Loss', policy='min',
              *, mean: bool = True, training_mode: str = 'parallel', verbose: int = 3) -> None:
        """ LEARNING PROCEDURE

        :param gTr: element/list of GraphsObjects/GraphTensors used for the learning procedure.
        :param epochs: (int) the max number of epochs for the learning procedure.
        :param gVa: element/list of GraphsObjects/GraphTensors for early stopping. Default None, no early stopping performed.
        :param update_freq: (int) how many epochs must be completed before evaluating gVa and gTr and/or print learning progress. Default 10.
        :param max_fails: (int) specifies the max number of failures in gVa improvement loss evaluation before early sopping. Default 10.
        :param training_mode: (str) in ['serial','parallel','residual']. Default set to 'parallel'
            > 'serial' - GNNs are trained separately, from layer 0 to layer N
            > 'parallel' - GNNs are trained all together, from loss = mean(Loss_Function( t, Oi)) where Oi is the output of GNNi
            > 'residual' - GNNs are trained all together, from loss = Loss_Function(t, mean(Oi)), where Oi is the output of GNNi
        :param mean: (bool) if False the applied gradients are computed as the sum of every iteration, otherwise as the mean. Default True.
        :param verbose: (int) 0: silent mode; 1: print history; 2: print epochs/batches, 3: history + epochs/batches. Default 3.
        :return: None
        """

        assert training_mode in ['parallel', 'serial', 'residual']
        self.training_mode = training_mode

        # Checking type for gTr and gVa + Initialization of Validation parameters
        # check type - new gTr is a list of GraphTensors. gVa is in [None, list[CompositeGraphTensor]].
        # All GraphObjects are now GraphTensors, to speed up the learning procedure
        gTr = self.checktype(gTr)
        gVa = self.checktype(gVa)

        # SERIAL TRAINING
        if training_mode == 'serial':
            gTr1 = [i.copy() for i in gTr]
            gVa1 = [i.copy() for i in gVa] if gVa is not None else None

            for idx, gnn in enumerate(self.gnns):
                if verbose in [1, 3]: print(f'\n\n------------------- GNN{idx} -------------------\n')

                # train the idx-th gnn
                gnn.train(gTr1, epochs, gVa1, update_freq, max_fails, observed_metric, policy, mean=mean, verbose=verbose)

                # extrapolate state and output to update labels
                _, sTr, oTr = zip(*[super(CompositeGNNgraphBased, gnn).Loop(i) if type(gnn) == CompositeGNNgraphBased else gnn.Loop(i) for i in gTr1])
                gTr1 = [self.update_graph(i, s, o) for i, s, o in zip(gTr, sTr, oTr)]
                if gVa:
                    _, sVa, oVa = zip(*[super(CompositeGNNgraphBased, gnn).Loop(i) if type(gnn) == CompositeGNNgraphBased else gnn.Loop(i) for i in gVa1])
                    gVa1 = [self.update_graph(i, s, o) for i, s, o in zip(gVa, sVa, oVa)]

        # RESIDUAL OR PARALLEL TRAINING
        else:
            super().train(gTr, epochs, gVa, update_freq, max_fails, observed_metric, policy, mean=mean, verbose=verbose)
