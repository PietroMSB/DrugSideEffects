from __future__ import annotations

from typing import Optional, Union

import tensorflow as tf
import numpy as np
from pandas import options
import sys

from GNNv3.GNN.CGNN.CGNN_BaseClass import BaseCGNN
from GNNv3.GNN.graph_class import GraphObject

options.display.max_rows = 15


#######################################################################################################################
### CLASS CGNN - NODE BASED ############################################################################################
#######################################################################################################################
class CGNNnodeBased(BaseCGNN):

	## CONSTRUCTORS METHODS ###########################################################################################
	def __init__(self,
				 net_state_list: list[tf.keras.models.Sequential],
				 net_output: tf.keras.models.Sequential,
				 optimizer: tf.keras.optimizers.Optimizer,
				 loss_function: tf.keras.losses.Loss,
				 loss_arguments: Optional[dict],
				 state_vect_dim: int,
				 type_label_lengths: np.array,
				 state_init_stdev: float,
				 max_iteration: int,
				 addressed_problem: str,
				 extra_metrics: Optional[dict] = None,
				 extra_metrics_arguments: Optional[dict[str, dict]] = None,
				 path_writer: str = 'writer/',
				 namespace: str = 'GNN') -> None:
		""" CONSTRUCTOR
		:param net_state_list: (list) MLPs for the state networks, initialized externally
		:param net_output: (tf.keras.model.Sequential) MLP for the output network, initialized externally
		:param optimizer: (tf.keras.optimizers) for gradient application, initialized externally
		:param loss_function: (tf.keras.losses) or (tf.function) for the loss computation
		:param loss_arguments: (dict) with some {'argument':values} one could pass to loss when computed
		:param max_iteration: (int) max number of iteration for the unfolding procedure (to reach convergence)
		:param path_writer: (str) path for saving TensorBoard objects
		:param addressed_problem: (str) in ['r','c'], 'r':regression, 'c':classification for the addressed problem
		:param extra_metrics: None or dict {'name':function} for metrics to be watched during training/validaion/test
		:param metrics_arguments: None or dict {'name':{'argument':value}} for arguments to be passed to extra_metrics
		:param state_vect_dim: (int)>0, vector dim for a GNN which does not initialize states with node labels
		:param state_init_stdev: (float)>=0 standard deviation of normal distribution (mean=0) for state tensor initialization
		:param type_label_lengths: (np.array) array storing the label length of each node type
		"""
		# Check arguments
		if type(state_vect_dim) != int or state_vect_dim <= 0: raise TypeError('param <state_vect_dim> must be int>0')
		if type(state_init_stdev) != float or state_init_stdev < 0: raise TypeError('param <state_init_stdev> must be float>=0.0')
		
		# parameters and hyperparameters
		super().__init__(optimizer, loss_function, loss_arguments, addressed_problem, extra_metrics, extra_metrics_arguments, path_writer, namespace)
		self.net_state_list = net_state_list
		self.net_output = net_output
		self.max_iteration = max_iteration
		self.state_vect_dim = state_vect_dim
		self.state_init_stdev = state_init_stdev
		self.type_label_lengths = type_label_lengths
		self.type_offsets = tf.constant(np.subtract(self.type_label_lengths, np.max(self.type_label_lengths)))

	# -----------------------------------------------------------------------------------------------------------------
	def copy(self, *, path_writer: str = '', namespace: str = '', copy_weights: bool = True) -> Union['GNN', 'GNNedgeBased', 'GNNgraphBased', 'GNN2']:
		""" COPY METHOD
		:param path_writer: None or (str), to save copied gnn writer. Default is in the same folder + '_copied'
		:param copy_weights: (bool) True: copied_gnn.nets==self.nets; False: state and output are re-initialized
		:return: a Deep Copy of the GNN instance.
		"""
		netS = list()
		for net_state in self.net_state_list:
			netS.append(tf.keras.models.clone_model(net_state))
		netO = tf.keras.models.clone_model(self.net_output)
		if copy_weights:
			for i in range(len(netS)):
				netS[i].set_weights(self.net_state_list[i].get_weights())
			netO.set_weights(self.net_output.get_weights())
		return self.__class__(net_state_list=netS, net_output=netO, optimizer=self.optimizer.__class__(**self.optimizer.get_config()),
							  loss_function=self.loss_function, loss_arguments=self.loss_args,
							  max_iteration=self.max_iteration, addressed_problem=self.addressed_problem,
							  extra_metrics=self.extra_metrics, extra_metrics_arguments=self.mt_args, 
							  state_vect_dim=self.state_vect_dim, type_label_lengths=self.type_label_lengths,
							  path_writer=path_writer if path_writer else self.path_writer + '_copied/',
							  namespace=namespace if namespace else 'GNN')


	## GETTER AND SETTER METHODs ####################################################################################
	def trainable_variables(self) -> tuple[list[list[list[tf.Tensor]]], list[list[tf.Tensor]]]:
		return [[net_state.trainable_variables] for net_state in self.net_state_list], [self.net_output.trainable_variables]

	# -----------------------------------------------------------------------------------------------------------------
	def get_weights(self) -> tuple[list[list[list[array]]], list[list[array]]]:
		return [[net_state.get_weights()] for net_state in self.net_state_list], [self.net_output.get_weights()]

	# -----------------------------------------------------------------------------------------------------------------
	def set_weights(self, weights_state: list[list[list[array]]], weights_output: list[list[array]]) -> None:
		assert len(weights_state) == len(self.net_state_list)
		assert len(weights_output) == 1
		for i in range(len(self.net_state_list)):
			self.net_state_list[i].set_weights(weights_state[i][0])
		self.net_output.set_weights(weights_output[0])


	## CALL/PREDICT METHOD ############################################################################################
	def __call__(self, g: GraphObject) -> tf.Tensor:
		""" return ONLY the GNN output in test mode (training == False) for graph g of type GraphObject """
		return self.Loop(g, training=False)[-1]


	## LOOP METHODS ###################################################################################################
	# @tf.function
	def type_loop_condition(self, i, *args) -> tf.bool:
		""" Boolean function for type sub-loop condition """
		#check the number of types processed
		return tf.less(i, len(self.net_state_list))

	# -----------------------------------------------------------------------------------------------------------------
	# @tf.function
	def type_loop_body(self, i, out_state, out_index, type_mask, inp_state, inp_index, training) -> tuple:
		""" Loop body function for type sub-loop """
		# apply i-th column of type_mask to input
		ith_input = tf.boolean_mask(inp_state, type_mask[:,i], axis=0)
		# apply i-th column of type_mask to index
		ith_index = tf.boolean_mask(inp_index, type_mask[:,i], axis=0)
		# trim i-th input tensor in accordance to the length of the label of that type of node (state len + message len + type label len)
		ith_input = ith_input[:,:self.type_offsets[i]]
		# call i-th state network
		ith_output = self.net_state_list[i](inp_state, training=training)
		# concatenate i-th output to <out_state>
		out_state = tf.concat((out_state, ith_output), axis=0)
		# concatenate i-th index to <out_index>
		out_index = tf.concat((out_index, ith_index), axis=0)
		# return
		return i+1, out_state, out_index, type_mask, inp_state, inp_index, training 
	
	# -----------------------------------------------------------------------------------------------------------------
	# @tf.function
	def condition(self, k, *args) -> tf.bool:
		""" Boolean function condition for tf.while_loop correct processing graphs """
		# check if the maximum number of iterations has been reached
		return tf.less(k, self.max_iteration)

	# -----------------------------------------------------------------------------------------------------------------
	# @tf.function
	def convergence(self, k, state, state_old, nodes, type_mask, nodes_index, arcs_label, arcnode, training) -> tuple:

		# compute the incoming message for each node: shape == (len(source_nodes_index, Num state components))
		source_state = tf.gather(state, nodes_index[:, 0])
		
		# concatenate the gathered source node states with the corresponding arc labels
		arc_message = tf.concat([source_state, arcs_label], axis=1)
		
		# multiply by ArcNode matrix to get the incoming average/total/normalized messages on each node
		message = tf.sparse.sparse_dense_matmul(arcnode, arc_message)
		
		# concatenate the destination node 'old' states to the incoming messages
		inp_state = tf.concat((state, message, nodes), axis=1)
		# define index vector to allow the reconstruction of the state tensor
		inp_index = tf.transpose(tf.range(inp_state.shape[0]))
		
		# pre-build output tensor for state calculation
		out_state = tf.zeros((0, self.state_vect_dim), dtype=tf.float32)
		# pre-build output index for state reconstruction
		out_index = tf.zeros((0), dtype=tf.int32)
		
		# initialize iteration counter
		i = tf.constant(0, dtype=tf.int32)

		# compute new state and update iteration counter
		i, out_state, out_index, *_ = tf.while_loop(self.type_loop_condition, self.type_loop_body, [i, out_state, out_index, type_mask, inp_state, inp_index, training])

		#reconstruct state tensor
		inverted_index = tf.scalar_mul(-1, out_index)
		sorted_index = tf.math.top_k(inverted_index, tf.shape(inverted_index)[0], True).indices
		state_new = tf.gather(out_state, sorted_index, axis=0)

		#return
		return k + 1, state_new, state, nodes, type_mask, nodes_index, arcs_label, arcnode, training

	# -----------------------------------------------------------------------------------------------------------------
	# @tf.function
	def apply_filters(self, state_converged, nodes, nodes_index, arcs_label, mask) -> tf.Tensor:
		""" takes only nodes states for those with output_mask==1 AND belonging to set (in case Dataset == 1 Graph) """
		if self.state_vect_dim: state_converged = tf.concat((nodes, state_converged), axis=1)
		return tf.boolean_mask(state_converged, mask)

	# -----------------------------------------------------------------------------------------------------------------
	def Loop(self, g: GraphObject, *, training: bool = False) \
			-> tuple[int, tf.Tensor, tf.Tensor]:
		""" process a single graph, returning iteration, states and output """
		
		# retrieve quantities from graph f
		nodes = tf.constant(g.getNodes(), dtype=tf.float32)
		nodes_index = tf.constant(g.getArcs()[:, :2], dtype=tf.int32)
		arcs_label = tf.constant(g.getArcs()[:, 2:], dtype=tf.float32)
		arcnode = self.ArcNode2SparseTensor(g.getArcNode())
		mask = tf.logical_and(g.getSetMask(), g.getOutputMask())
		type_mask = tf.constant(g.getTypeMask(), dtype=tf.int32)
		
		# initialize all the useful variables for convergence loop
		k = tf.constant(0, dtype=tf.float32)
		state = tf.zeros((nodes.shape[0], self.state_vect_dim), dtype=tf.float32)
		state_old = tf.random.normal((nodes.shape[0], self.state_vect_dim), mean=0.0, stddev=self.state_init_stdev, dtype=tf.float32, seed=None, name=None)
		print(state_old)
		sys.exit()
		training = tf.constant(training)
		
		# loop until convergence is reached
		k, state, state_old, *_ = tf.while_loop(self.condition, self.convergence, [k, state, state_old, nodes, type_mask, nodes_index, arcs_label, arcnode, training])
		
		### DEBUG START ###
		print(state[:100,:])
		### DEBUG STOP ###
		
		# out_st is the converged state for the filtered nodes, depending on g.set_mask
		input_to_net_output = self.apply_filters(state, nodes, nodes_index, arcs_label, mask)
		
		# compute the output of the gnn network
		out = self.net_output(input_to_net_output, training=training)
		return k, state, out
   

#######################################################################################################################
### CLASS CGNN - GRAPH BASED ###########################################################################################
#######################################################################################################################
class CGNNgraphBased(CGNNnodeBased):
	def Loop(self, g: GraphObject, *, training: bool = False) -> tuple[int, tf.Tensor, tf.Tensor]:
		iter, state_nodes, out_nodes = super().Loop(g, training=training)
		
		# obtain a single output for each graph, by using nodegraph matrix to the output of all of its nodes
		nodegraph = tf.constant(g.getNodeGraph(), dtype=tf.float32)
		out_gnn = tf.matmul(nodegraph, out_nodes, transpose_a=True)
		return iter, state_nodes, out_gnn


#######################################################################################################################
### CLASS CGNN - EDGE BASED ############################################################################################
#######################################################################################################################
class CGNNedgeBased(CGNNnodeBased):
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



