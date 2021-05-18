from __future__ import annotations

from typing import Optional, Union

import tensorflow as tf
from numpy import array
from pandas import options

from GNN.GNN_BaseClass import BaseGNN
from GNN.graph_class import GraphObject

options.display.max_rows = 15


#######################################################################################################################
### CLASS GNN - NODE BASED ############################################################################################
#######################################################################################################################
class GNNnodeBased(BaseGNN):

    ## CONSTRUCTORS METHODS ###########################################################################################
    def __init__(self,
                 net_state: tf.keras.models.Sequential,
                 net_output: tf.keras.models.Sequential,
                 output_activation: Optional[tf.function],
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
                 namespace: str = 'GNN') -> None:
        """ CONSTRUCTOR
        :param net_state: (tf.keras.model.Sequential) MLP for the state network, initialized externally
        :param net_output: (tf.keras.model.Sequential) MLP for the output network, initialized externally
        :param optimizer: (tf.keras.optimizers) for gradient application, initialized externally
        :param loss_function: (tf.keras.losses) or (tf.function) for the loss computation
        :param loss_arguments: (dict) with some {'argument':values} one could pass to loss when computed
        :param output_activation: (tf.keras.activation) function in case net_output.layers[-1] is 'linear'
        :param max_iteration: (int) max number of iteration for the unfolding procedure (to reach convergence)
        :param threshold: threshold for specifying if convergence is reached or not
        :param path_writer: (str) path for saving TensorBoard objects
        :param addressed_problem: (str) in ['r','c'], 'r':regression, 'c':classification for the addressed problem
        :param extra_metrics: None or dict {'name':function} for metrics to be watched during training/validaion/test
        :param metrics_arguments: None or dict {'name':{'argument':value}} for arguments to be passed to extra_metrics
        :param state_vect_dim: None or (int)>=0, vector dim for a GNN which does not initialize states with node labels
        """
        # Check arguments
        if type(state_vect_dim) != int or state_vect_dim < 0: raise TypeError('param <state_vect_dim> must be int>=0')
        
        # parameters and hyperparameters
        super().__init__(optimizer, loss_function, loss_arguments, addressed_problem, extra_metrics, extra_metrics_arguments, path_writer, namespace)
        self.net_state = net_state
        self.net_output = net_output
        self.max_iteration = max_iteration
        self.state_threshold = threshold
        self.state_vect_dim = state_vect_dim
        
        # check last activation function: in case loss works with logits, it is set by <output_activation> parameter
        self.output_activation = tf.keras.activations.linear if output_activation is None else output_activation

    # -----------------------------------------------------------------------------------------------------------------
    def copy(self, *, path_writer: str = '', namespace: str = '', copy_weights: bool = True) -> Union['GNN', 'GNNedgeBased', 'GNNgraphBased', 'GNN2']:
        """ COPY METHOD
        :param path_writer: None or (str), to save copied gnn writer. Default is in the same folder + '_copied'
        :param copy_weights: (bool) True: copied_gnn.nets==self.nets; False: state and output are re-initialized
        :return: a Deep Copy of the GNN instance.
        """
        netS = tf.keras.models.clone_model(self.net_state)
        netO = tf.keras.models.clone_model(self.net_output)
        if copy_weights:
            netS.set_weights(self.net_state.get_weights())
            netO.set_weights(self.net_output.get_weights())
        return self.__class__(net_state=netS, net_output=netO, optimizer=self.optimizer.__class__(**self.optimizer.get_config()),
                              loss_function=self.loss_function, loss_arguments=self.loss_args, output_activation=self.output_activation,
                              max_iteration=self.max_iteration, threshold=self.state_threshold, addressed_problem=self.addressed_problem,
                              extra_metrics=self.extra_metrics, extra_metrics_arguments=self.mt_args, state_vect_dim=self.state_vect_dim,
                              path_writer=path_writer if path_writer else self.path_writer + '_copied/',
                              namespace=namespace if namespace else 'GNN')


    ## GETTERS AND SETTERS METHODs ####################################################################################
    def trainable_variables(self) -> tuple[list[list[tf.Tensor]], list[list[tf.Tensor]]]:
        return [self.net_state.trainable_variables], [self.net_output.trainable_variables]

    # -----------------------------------------------------------------------------------------------------------------
    def get_weights(self) -> tuple[list[list[array]], list[list[array]]]:
        return [self.net_state.get_weights()], [self.net_output.get_weights()]

    # -----------------------------------------------------------------------------------------------------------------
    def set_weights(self, weights_state: list[list[array]], weights_output: list[list[array]]) -> None:
        assert len(weights_state) == len(weights_output) == 1
        self.net_state.set_weights(weights_state[0])
        self.net_output.set_weights(weights_output[0])


    ## CALL/PREDICT METHOD ############################################################################################
    def __call__(self, g: GraphObject) -> tf.Tensor:
        """ return ONLY the GNN output in testo mode (training == False) for graph g of type GraphObject """
        out = self.Loop(g, training=False)[-1]
        return self.output_activation(out)


    ## LOOP METHODS ###################################################################################################
    # @tf.function
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
    # @tf.function
    def convergence(self, k, state, state_old, nodes, nodes_index, arcs_label, arcnode, training) -> tuple:
        # compute the incoming message for each node: shape == (len(source_nodes_index, Num state components)
        source_state = tf.gather(state, nodes_index[:, 0])
        
        # concatenate the gathered source node states with the corresponding arc labels
        arc_message = tf.concat([source_state, arcs_label], axis=1)
        if self.state_vect_dim:
            source_label = tf.gather(nodes, nodes_index[:, 0])
            arc_message = tf.concat([source_label, arc_message], axis=1)
        
        # multiply by ArcNode matrix to get the incoming average/total/normalized messages on each node
        message = tf.sparse.sparse_dense_matmul(arcnode, arc_message)
        
        # concatenate the destination node 'old' states to the incoming messages
        inp_state = tf.concat((nodes, state, message) if self.state_vect_dim else (state, message), axis=1)
        
        # compute new state and update step iteration counter
        state_new = self.net_state(inp_state, training=training)
        return k + 1, state_new, state, nodes, nodes_index, arcs_label, arcnode, training

    # -----------------------------------------------------------------------------------------------------------------
    # @tf.function
    def apply_filters(self, state_converged, nodes, nodes_index, arcs_label, mask) -> tf.Tensor:
        """ takes only nodes states for those with output_mask==1 AND belonging to set (in case Dataset == 1 Graph) """
        if self.state_vect_dim: state_converged = tf.concat((nodes, state_converged), axis=1)
        return tf.boolean_mask(state_converged, mask)

    # -----------------------------------------------------------------------------------------------------------------
    def Loop(self, g: GraphObject, *, training: bool = False, nodeplus: Optional[tf.Tensor] = None, arcplus: Optional[tf.Tensor] = None) \
            -> tuple[int, tf.Tensor, tf.Tensor]:
        """ process a single graph, returning iteration, states and output """
        
        # retrieve quantities from graph f
        nodes = tf.constant(g.getNodes(), dtype=tf.float32)
        nodes_index = tf.constant(g.getArcs()[:, :2], dtype=tf.int32)
        arcs_label = tf.constant(g.getArcs()[:, 2:], dtype=tf.float32)
        arcnode = self.ArcNode2SparseTensor(g.getArcNode())
        mask = tf.logical_and(g.getSetMask(), g.getOutputMask())
        
        # concatenate other quantities, if LGNN 2+
        if nodeplus is not None: nodes = tf.concat([nodes, nodeplus], axis=1)
        if arcplus is not None: arcs_label = tf.concat([arcs_label, arcplus], axis=1)
        
        # initialize all the useful variables for convergence loop
        k = tf.constant(0, dtype=tf.float32)
        state = tf.zeros((nodes.shape[0], self.state_vect_dim), dtype=tf.float32) if self.state_vect_dim > 0 else tf.constant(nodes, dtype=tf.float32)
        state_old = tf.ones_like(state, dtype=tf.float32)
        training = tf.constant(training)
        
        # loop until convergence is reached
        k, state, state_old, *_ = tf.while_loop(self.condition, self.convergence, [k, state, state_old, nodes, nodes_index, arcs_label, arcnode, training])
        
        # out_st is the converged state for the filtered nodes, depending on g.set_mask
        input_to_net_output = self.apply_filters(state, nodes, nodes_index, arcs_label, mask)
        
        # compute the output of the gnn network
        out = self.net_output(input_to_net_output, training=training)
        return k, state, out


    ## EVALUATE METHODs ###############################################################################################
    def evaluate_single_graph(self, g: GraphObject, class_weights: Union[int, float, list[float]], training: bool) -> tuple:
        iter, loss, targs, out = super().evaluate_single_graph(g, class_weights=class_weights, training=training)
        return iter, loss, targs, self.output_activation(out)


#######################################################################################################################
### CLASS GNN - GRAPH BASED ###########################################################################################
#######################################################################################################################
class GNNgraphBased(GNNnodeBased):
    def Loop(self, g: GraphObject, *, nodeplus=None, arcplus=None, training: bool = False) -> tuple[int, tf.Tensor, tf.Tensor]:
        iter, state_nodes, out_nodes = super().Loop(g, nodeplus=nodeplus, arcplus=arcplus, training=training)
        
        # obtain a single output for each graph, by using nodegraph matrix to the output of all of its nodes
        nodegraph = tf.constant(g.getNodeGraph(), dtype=tf.float32)
        out_gnn = tf.matmul(nodegraph, out_nodes, transpose_a=True)
        return iter, state_nodes, out_gnn


#######################################################################################################################
### CLASS GNN - EDGE BASED ############################################################################################
#######################################################################################################################
class GNNedgeBased(GNNnodeBased):
    # @tf.function
    def apply_filters(self, state_converged, nodes, nodes_index, arcs_label, mask) -> tf.Tensor:
        """ takes only arcs info of those with output_mask==1 AND belonging to set (in case Dataset == 1 Graph) """
        if self.state_vect_dim: state_converged = tf.concat((nodes, state_converged), axis=1)
        
        # gather source nodes state
        source_state = tf.gather(state_converged, nodes_index[:, 0])
        source_state = tf.cast(source_state, tf.float32)
        
        # gather destination nodes state
        destination_state = tf.gather(state_converged, nodes_index[:, 1])
        destination_state = tf.cast(destination_state, tf.float32)
        
        # concatenate source and destination states to arc labels
        arc_state = tf.concat([source_state, destination_state, arcs_label], axis=1)
        
        # takes only arcs states for those with output_mask==1 AND belonging to the set (in case Dataset == 1 Graph)
        return tf.boolean_mask(arc_state, mask)


#######################################################################################################################
### CLASS GNN - NODE BASED ## First MLP, then sum-up for states #######################################################
#######################################################################################################################
## GNN v1 by A.Rossi and M.Tiezzi
class GNN2(GNNnodeBased):
    # @tf.function
    def convergence(self, k, state, state_old, nodes, nodes_index, arcs_label, arcnode, training):
        # gather source nodes label
        source_label = tf.gather(nodes, nodes_index[:, 0])
        source_label = tf.cast(source_label, tf.float32)
        
        # gather destination nodes label
        destination_label = tf.gather(nodes, nodes_index[:, 1])
        destination_label = tf.cast(destination_label, tf.float32)
        
        # gather destination nodes state
        destination_state = tf.gather(state, nodes_index[:, 1])
        destination_state = tf.cast(destination_state, tf.float32)
        
        # concatenate: source_node_label, destination_node_label, edge_attributes and destinatino_node_state
        arc_message = tf.concat([source_label, destination_label, arcs_label, destination_state], axis=1)
        arc_message = tf.cast(arc_message, tf.float32)  # re-cast to be computable, CAN BE OMITTED
        
        # compute the incoming message for each node with MLP: shape == (len(source_nodes_index, Num state components)
        messages = self.net_state(arc_message, training=training)

        # multiply by ArcNode matrix to get the incoming average/total/normalized messages on each node and
        # compute new state and update step iteration counter
        state_new = tf.sparse.sparse_dense_matmul(arcnode, messages)
        state_new = tf.cast(state_new, tf.float32)
        k = k + 1
        return k, state_new, state, nodes, nodes_index, arcs_label, arcnode, training
