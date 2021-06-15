from __future__ import annotations

from typing import Optional, Union

import tensorflow as tf
from numpy import array
from pandas import options

from CompositeGNN.CompositeBaseClass import CompositeBaseClass
from CompositeGNN.composite_graph_class import CompositeGraphObject, CompositeGraphTensor

options.display.max_rows = 15


#######################################################################################################################
### CLASS GNN - NODE BASED ############################################################################################
#######################################################################################################################
class CompositeGNNnodeBased(CompositeBaseClass):
    """ GNN for node-based problem """

    ## CONSTRUCTORS METHODS ###########################################################################################
    def __init__(self,
                 net_state: list[tf.keras.models.Sequential],
                 net_output: tf.keras.models.Sequential,
                 optimizer: tf.keras.optimizers.Optimizer,
                 loss_function: tf.keras.losses.Loss,
                 loss_arguments: Optional[dict],
                 state_vect_dim: int,
                 max_iteration: int,
                 threshold: float,
                 addressed_problem: str,
                 extra_metrics: Optional[dict] = None,
                 extra_metrics_arguments: Optional[dict[str, dict]] = None,
                 path_writer: str = 'writer/',
                 namespace: str = 'CGNN') -> None:
        """ CONSTRUCTOR

        :param net_state: list of (tf.keras.model.Sequential) MLPs for the state network, one for each node type, initialized externally.
        :param net_output: (tf.keras.model.Sequential) MLP for the output network, initialized externally.
        :param optimizer: (tf.keras.optimizers) for gradient application, initialized externally.
        :param loss_function: (tf.keras.losses) for the loss computation.
        :param loss_arguments: (dict) with some {'argument':values} one could pass to loss when computed.
        :param state_vect_dim: None or (int)>=0, vector dim for a GNN which does not initialize states with node labels.
        :param max_iteration: (int) max number of iteration for the unfolding procedure (to reach convergence).
        :param threshold: threshold for specifying if convergence is reached or not.
        :param addressed_problem: (str) in ['r','c'], 'r':regression, 'c':classification for the addressed problem.
        :param extra_metrics: None or dict {'name':function} for metrics to be watched during training/validation/test procedures.
        :param extra_metrics_arguments: None or dict {'name':{'argument':value}} for arguments passed to extra_metrics['name'].
        :param path_writer: (str) path for saving TensorBoard objects in training procedure. If folder is not empty, all files are removed.
        :param namespace: (str) namespace for tensorboard visualization.
        """
        # Check arguments
        if state_vect_dim <= 0: raise TypeError('param <state_vect_dim> must be int>0')

        # BaseGNN constructor
        super().__init__(len(net_state), optimizer, loss_function, loss_arguments, addressed_problem, extra_metrics, extra_metrics_arguments,
                         path_writer, namespace)

        ### GNN parameter
        self.net_state = net_state
        self.net_output = net_output
        self.max_iteration = max_iteration
        self.state_threshold = threshold
        self.state_vect_dim = state_vect_dim

    # -----------------------------------------------------------------------------------------------------------------
    def copy(self, *, path_writer: str = '', namespace: str = '', copy_weights: bool = True):
        """ COPY METHOD

        :param path_writer: None or (str), to save copied gnn tensorboard writer. Default in the same folder + '_copied'.
        :param namespace: (str) for tensorboard visualization in model training procedure.
        :param copy_weights: (bool) True: state and output weights are copied in new gnn, otherwise they are re-initialized.
        :return: a Deep Copy of the GNN instance.
        """
        netS = [tf.keras.models.clone_model(i) for i in self.net_state]
        netO = tf.keras.models.clone_model(self.net_output)
        if copy_weights:
            for i,j in zip(netS, self.net_state): i.set_weights(j.get_weights())
            netO.set_weights(self.net_output.get_weights())

        return self.__class__(net_state=netS, net_output=netO, optimizer=self.optimizer.__class__(**self.optimizer.get_config()),
                              loss_function=self.loss_function, loss_arguments=self.loss_args, max_iteration=self.max_iteration,
                              threshold=self.state_threshold, addressed_problem=self.addressed_problem, extra_metrics=self.extra_metrics,
                              extra_metrics_arguments=self.mt_args, state_vect_dim=self.state_vect_dim,
                              path_writer=path_writer if path_writer else self.path_writer + '_copied/',
                              namespace=namespace if namespace else 'GNN')

    ## SAVE AND LOAD METHODs ##########################################################################################
    def save(self, path: str):
        """ Save model to folder <path>, without extra_metrics info """
        from json import dump

        # check path
        if path[-1] != '/': path += '/'

        # save net_state and net_output
        for i, elem in enumerate(self.net_state): tf.keras.models.save_model(elem, f'{path}net_state_{i}/')
        tf.keras.models.save_model(self.net_output, f'{path}net_output/')

        # save configuration file in json format
        config = {'loss_function': tf.keras.losses.serialize(self.loss_function), 'loss_arguments': self.loss_args,
                  'optimizer': str(tf.keras.optimizers.serialize(self.optimizer)),
                  'max_iteration': self.max_iteration, 'threshold': self.state_threshold,
                  'addressed_problem': self.addressed_problem, 'state_vect_dim': self.state_vect_dim}

        with open(f'{path}config.json', 'w') as json_file:
            dump(config, json_file)

    # -----------------------------------------------------------------------------------------------------------------
    @classmethod
    def load(self, path: str, path_writer: str, namespace: str = 'GNN',
             extra_metrics: Optional[dict] = None, extra_metrics_arguments: Optional[dict[str, dict]] = None):
        """ Load model from folder <path>.

        Only Loss is considered as metrics after loading process.
        To use more metrics, set :param extra_metrics: and :param extra_metrics_arguments:

        :param path: (str) folder path containing all useful files to load the model.
        :param path_writer: (str) path for writer folder. !!! Constructor method deletes a non-empty folder and makes a new empty one.
        :param namespace: (str) namespace for tensorboard visualization in model training procedure.
        :param extra_metrics: None or dict {'name':function} for metrics to be watched during training/validation/test procedures.
        :param extra_metrics_arguments: None or dict {'name':{'argument':value}} for arguments passed to extra_metrics['name'].
        :return: the loaded gnn model. GNN type depends on class which call load method.
        """
        from json import loads
        from os import listdir

        # check path
        if path[-1] != '/': path += '/'

        # load configuration file
        with open(f'{path}config.json', 'r') as read_file:
            config = loads(read_file.read())

        # get optimizer, loss function
        optz = tf.keras.optimizers.deserialize(eval(config.pop('optimizer')))
        loss = tf.keras.losses.deserialize(config.pop('loss_function'))

        # load net_state and net_output
        net_state_dirs = [f'{path}{i}/' for i in listdir(path) if 'net_state' in i]
        netS = [tf.keras.models.load_model(i, compile=False) for i in net_state_dirs]
        netO = tf.keras.models.load_model(f'{path}net_output/', compile=False)

        return self(net_state=netS, net_output=netO, optimizer=optz, loss_function=loss,
                    extra_metrics=extra_metrics, extra_metrics_arguments=extra_metrics_arguments,
                    path_writer=path_writer, namespace=namespace, **config)

    ## GETTERS AND SETTERS METHODs ####################################################################################
    def get_dense_layers(self) -> list[tf.keras.layers.Layer]:
        """ Get dense layer for the application of regularizers in training time """
        #netSt_dense_layers = [i for i in self.net_state.layers if isinstance(i, tf.keras.layers.Dense)]
        netSt_dense_layers = [i for j in self.net_state for i in j.layers if isinstance(i, tf.keras.layers.Dense)]
        netOut_dense_layers = [i for i in self.net_output.layers if isinstance(i, tf.keras.layers.Dense)]
        return netSt_dense_layers + netOut_dense_layers

    # -----------------------------------------------------------------------------------------------------------------
    def trainable_variables(self) -> tuple[list[list[tf.Tensor]], list[list[tf.Tensor]]]:
        """ Get tensor weights for net_state and net_output """
        #return [self.net_state.trainable_variables], [self.net_output.trainable_variables]
        return [[i.trainable_variables for i in self.net_state]], [self.net_output.trainable_variables]

    # -----------------------------------------------------------------------------------------------------------------
    def get_weights(self) -> tuple[list[list[array]], list[list[array]]]:
        """ Get array weights for net_state and net_output """
        #return [self.net_state.get_weights()], [self.net_output.get_weights()]
        return [[i.get_weights() for i in self.net_state]], [self.net_output.get_weights()]

    # -----------------------------------------------------------------------------------------------------------------
    def set_weights(self, weights_state: list[list[array]], weights_output: list[list[array]]) -> None:
        """ Set weights for net_state and net_output """
        #assert len(weights_state) == len(weights_output) == 1
        for i,j in zip(self.net_state, weights_state[0]): i.set_weights(j)
        #self.net_state.set_weights(weights_state[0])
        self.net_output.set_weights(weights_output[0])

    ## CALL/PREDICT METHOD ############################################################################################
    def __call__(self, g: Union[CompositeGraphObject, CompositeGraphTensor]) -> tf.Tensor:
        """ Return ONLY the GNN output in test mode (training == False) for graph g of type CompositeGraphObject/CompositeGraphTensor """
        return self.Loop(g, training=False)[-1]

    ## EVALUATE METHODS ###############################################################################################
    def evaluate_single_graph(self, g: Union[CompositeGraphObject, CompositeGraphTensor], training: bool) -> tuple:
        """ Evaluate single GraphObject element g in test mode (training == False)

        :param g: (CompositeGraphObject or GraphTensor) single graph element
        :param training: (bool) set internal models behavior, s.t. they work in training or testing mode
        :return: (tuple) convergence iteration (int), loss value (matrix), target and output  (matrices) of the model
        """
        # transform CompositeGraphObject in CompositeGraphTensor
        if isinstance(g, CompositeGraphObject): g = CompositeGraphTensor.fromGraphObject(g)

        # get targets
        targs = self.get_filtered_tensor(g, g.targets)
        loss_weights = self.get_filtered_tensor(g, g.sample_weights)

        # graph processing
        it, _, out = self.Loop(g, training=training)

        # loss computation
        loss = self.loss_function(targs, out, **self.loss_args) * loss_weights

        return it, loss, targs, out

    ## LOOP METHODS ###################################################################################################
    def condition(self, k, state, state_old, *args) -> tf.bool:
        """ Boolean function condition for tf.while_loop correct processing graphs """

        # distance_vector is the Euclidean Distance: √ Σ(xi-yi)² between current state xi and past state yi
        outDistance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(state, state_old)), axis=1))

        # state_norm is the norm of state_old, defined by ||state_old|| = √ Σxi²
        state_norm = tf.sqrt(tf.reduce_sum(tf.square(state_old), axis=1))

        # boolean vector that stores the "convergence reached" flag for each node
        scaled_state_norm = tf.math.scalar_mul(self.state_threshold, state_norm)

        # check whether global convergence and/or the maximum number of iterations have been reached
        checkDistanceVec = tf.greater(outDistance, scaled_state_norm)

        # compute boolean
        c1 = tf.reduce_any(checkDistanceVec)
        c2 = tf.less(k, self.max_iteration)
        return tf.logical_and(c1, c2)

    # -----------------------------------------------------------------------------------------------------------------
    def convergence(self, k, state, state_old, nodes, type_mask, dim_node_labels, adjacency, aggregated_component, training) -> tuple:
        """ Compute new state for the nodes graph """

        # aggregated_states is the aggregation of ONLY neighbors' states.
        # NOTE: if state_vect_dim != 0, neighbors' label are considered using :param aggregated_nodes: since it is constant
        aggregated_states = tf.sparse.sparse_dense_matmul(adjacency, state)

        # concatenate the destination node 'old' states to the incoming message, to obtain the input to net_state
        state_new = list()
        for d,m,net in zip(dim_node_labels, type_mask, self.net_state):
            inp_state = tf.concat([nodes[:,:d], state, aggregated_states, aggregated_component], axis=1)
            inp_state = tf.boolean_mask(inp_state, m)

            # compute new state and update step iteration counter
            ###debug
            #state_new.append(inp_state)
            state_new.append(net(inp_state, training=training))

        # reorder state based on nodes' ordering
        state_new = [tf.scatter_nd(tf.where(m), s, (len(m), s.shape[1])) for m,s in zip(type_mask, state_new)]
        state_new = tf.reduce_sum(state_new, axis=0)

        return k + 1, state_new, state, nodes, type_mask, dim_node_labels, adjacency, aggregated_component, training

    # -----------------------------------------------------------------------------------------------------------------
    def apply_filters(self, state_converged, nodes, adjacency, arcs_label, mask) -> tf.Tensor:
        """ Takes only nodes' [states] or [states|labels] for those with output_mask==1 AND belonging to set """
        #if self.state_vect_dim: state_converged = tf.concat([state_converged, nodes], axis=1)
        return tf.boolean_mask(state_converged, mask)

    # -----------------------------------------------------------------------------------------------------------------
    def Loop(self, g: Union[CompositeGraphObject, CompositeGraphTensor], *, training: bool = False) -> tuple[int, tf.Tensor, tf.Tensor]:
        """ Process a single CompositeGraphObject/CompositeGraphTensor element g, returning iteration, states and output """

        # transform CompositeGraphObject in CompositeGraphTensor
        if isinstance(g, CompositeGraphObject): g = CompositeGraphTensor.fromGraphObject(g)

        # initialize states and iters for convergence loop
        # including aggregated neighbors' label and aggregated incoming arcs' label
        aggregated_arcs = tf.sparse.sparse_dense_matmul(g.ArcNode, g.arcs[:, 2:])
        aggregated_nodes = tf.concat([tf.sparse.sparse_dense_matmul(a, g.nodes[:,:d]) for a,d in zip(g.CompositeAdjacencies, g.DIM_NODE_LABELS)], axis=1)
        aggregated_component = tf.concat([aggregated_nodes, aggregated_arcs], axis=1)

        # new values for Loop
        state = tf.random.normal((g.nodes.shape[0], self.state_vect_dim), stddev=0.1, dtype='float32')
        state_old = tf.ones_like(state, dtype='float32')
        k = tf.constant(0, dtype='float32')
        training = tf.constant(training, dtype=bool)

        # loop until convergence is reached
        k, state, state_old, *_ = tf.while_loop(self.condition, self.convergence,
                                                [k, state, state_old, g.nodes, g.type_mask, g.DIM_NODE_LABELS, g.Adjacency, aggregated_component, training])

        # out_st is the converged state for the filtered nodes, depending on g.set_mask
        mask = tf.logical_and(g.set_mask, g.output_mask)
        input_to_net_output = self.apply_filters(state, g.nodes, g.Adjacency, g.arcs[:, 2:], mask)

        # compute the output of the gnn network
        out = self.net_output(input_to_net_output, training=training)
        return k, state, out


#######################################################################################################################
### CLASS GNN - EDGE BASED ############################################################################################
#######################################################################################################################
class CompositeGNNedgeBased(CompositeGNNnodeBased):
    """ GNN for edge-based problem """

    def apply_filters(self, state_converged, nodes, adjacency, arcs_label, mask) -> tf.Tensor:
        """ Takes only nodes' [states] or [states|labels] for those with output_mask==1 AND belonging to set """
        #if self.state_vect_dim: state_converged = tf.concat([state_converged, nodes], axis=1)

        # gather source nodes' and destination nodes' state
        states = tf.gather(state_converged, adjacency.indices)
        states = tf.reshape(states, shape=(arcs_label.shape[0], 2 * state_converged.shape[1]))
        states = tf.cast(states, tf.float32)

        # concatenate source and destination states (and labels) to arc labels
        arc_state = tf.concat([states, arcs_label], axis=1)

        # takes only arcs states for those with output_mask==1 AND belonging to the set (in case Dataset == 1 Graph)
        return tf.boolean_mask(arc_state, mask)


#######################################################################################################################
### CLASS GNN - GRAPH BASED ###########################################################################################
#######################################################################################################################
class CompositeGNNgraphBased(CompositeGNNnodeBased):
    """ GNN for graph-based problem """

    # -----------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_filtered_tensor(g: CompositeGraphTensor, inp: tf.Tensor):
        """ Get inp [targets or sample_weights] for graph based problems -> nodes states are not filtered by set_mask and output_mask """
        return tf.constant(inp, dtype='float32')

    # -----------------------------------------------------------------------------------------------------------------
    def Loop(self, g: Union[CompositeGraphObject, CompositeGraphTensor], *, training: bool = False) -> tuple[int, tf.Tensor, tf.Tensor]:
        """ Process a single graph, returning iteration, states and output. Output of graph-based problem is the averaged nodes output """

        # transform CompositeGraphObject in CompositeGraphTensor
        if isinstance(g, CompositeGraphObject): g = CompositeGraphTensor.fromGraphObject(g)

        # get iter, states and output of every nodes from GNNnodeBased
        iter, state_nodes, out_nodes = super().Loop(g, training=training)

        # obtain a single output for each graph, by using nodegraph matrix to the output of all of its nodes
        nodegraph = tf.constant(g.NodeGraph, dtype='float32')
        out_gnn = tf.matmul(nodegraph, out_nodes, transpose_a=True)
        return iter, state_nodes, out_gnn
