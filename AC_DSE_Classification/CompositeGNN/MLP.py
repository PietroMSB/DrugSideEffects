from __future__ import annotations

from typing import Union, Optional

from numpy import array, arange
from tensorflow.keras.layers import Dense, Dropout, AlphaDropout
from tensorflow.keras.models import Sequential


# ---------------------------------------------------------------------------------------------------------------------
def MLP(input_dim: int, layers: list[int], activations, kernel_initializer, bias_initializer, kernel_regularizer=None,
        bias_regularizer=None, dropout_rate: Union[list[float], float, None] = None,
        dropout_pos: Optional[Union[list[int], int]] = None, alphadropout: bool = False):
    """ Quick building function for MLP model. All lists must have the same length

    :param input_dim: (int) specify the input dimension for the model
    :param layers: (int or list of int) specify the number of units in every layers
    :param activations: (functions or list of functions)
    :param kernel_initializer: (initializers or list of initializers) for weights initialization (NOT biases)
    :param bias_initializer: (initializers or list of initializers) for biases initialization (NOT weights)
    :param kernel_regularizer: (regularizer or list of regularizers) for weight regularization (NOT biases)
    :param bias_regularizer: (regularizer or list of regularizers) for biases regularization (NOT weights)
    :param dropout_rate: (float) s.t. 0 <= dropout_percs <= 1 for dropout rate
    :param dropout_pos: int or list of int describing dropout layers position
    :param alphadropout: (bool) for dropout type, if any
    :return: Sequential (MLP) model
    """
    # check type
    if dropout_rate == None or dropout_pos == None: dropout_rate, dropout_pos = list(), list()

    # build lists
    if type(activations) != list: activations = [activations for _ in layers]
    if type(kernel_initializer) != list: kernel_initializer = [kernel_initializer for _ in layers]
    if type(bias_initializer) != list: bias_initializer = [bias_initializer for _ in layers]
    if type(kernel_regularizer) != list: kernel_regularizer = [kernel_regularizer for _ in layers]
    if type(bias_regularizer) != list: bias_regularizer = [bias_regularizer for _ in layers]
    if type(dropout_pos) == int:  dropout_pos = [dropout_pos]
    if type(dropout_rate) == float: dropout_rate = [dropout_rate for _ in dropout_pos]

    # check lengths
    if len(set(map(len, [activations, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, layers]))) > 1:
        raise ValueError('Dense parameters must have the same length to be correctly processed')
    if len(dropout_rate) != len(dropout_pos):
        raise ValueError('Dropout parameters must have the same length to be correctly processed')

    # Dense layers
    keys = ['units', 'activation', 'kernel_initializer', 'bias_initializer', 'kernel_regularizer', 'bias_regularizer']
    vals = zip(layers, activations, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer)
    params = [dict(zip(keys, i)) for i in vals]

    # Dropout layers
    if dropout_rate and dropout_pos:
        dropout_pos = list(array(dropout_pos) + arange(len(dropout_pos)))
        for i, elem in enumerate(dropout_rate): params.insert(dropout_pos[i], {'rate': elem})

    # set input shape for first layer
    params[0]['input_shape'] = (input_dim,)

    # return MLP model
    dropout = AlphaDropout if alphadropout else Dropout
    mlp_layers = [Dense(**i) if 'units' in i else dropout(**i) for i in params]
    return Sequential(mlp_layers)


# ---------------------------------------------------------------------------------------------------------------------
def get_inout_dims(net_name: str, dim_node_label: int, dim_arc_label: int, dim_target: int, problem_based: str, dim_state: int,
                   hidden_units: Union[None, int, list[int], tuple[int]],
                   *, layer: int = 0, get_state: bool = False, get_output: bool = False) -> tuple[int, list[int]]:
    """ Calculate input and output dimension for the MLP of state and output

    :param net_name: (str) in ['state','output']
    :param dim_node_label: (int) dimension of node label
    :param dim_arc_label: (int) dimension of arc label
    :param dim_target: (int) dimension of target
    :param problem_based: (str) s.t. len(problem_based) in [1,2] -> [{'a','n','g'} | {'1','2'}]
    :param dim_state: (int)>=0 for state dimension paramenter of the gnn
    :param hidden_units: (int or list of int) for specifying units on hidden layers
    :param layer: (int) LGNN USE: get the dims at gnn of the layer <layer>, from graph dims on layer 0. Default is 0, since GNN==LGNN in this case
    :param get_state: (bool) LGNN USE: set accordingly to LGNN behaviour, if gnns get state, output or both from previous layer
    :param get_output: (bool) LGNN USE: set accordingly to LGNN behaviour, if gnns get state, output or both from previous layer
    :return: (tuple) (input_shape, layers) s.t. input_shape (int) is the input shape for mlp, layers (list of ints) defines hidden+output layers
    """
    assert layer >= 0
    assert problem_based in ['a', 'n', 'g']
    assert dim_state >= 0
    assert isinstance(hidden_units, (int, list, tuple, type(None)))

    DS = dim_state
    NL, AL, T = array(dim_node_label), dim_arc_label, dim_target

    # if LGNN, get MLPs layers for gnn in layer 2+ - DA CONTROLLARE PER COMPOSITE LGNN
    if layer > 0:
        GS, GO = get_state, get_output
        if DS != 0:
            NL = NL + DS * GS + T * (problem_based != 'a') * GO
            AL = AL + T * (problem_based == 'a') * GO
        else:
            NL = NL + layer * NL * GS + ((layer - 1) * GS + 1) * T * (problem_based != 'a') * GO
            AL = AL + T * (problem_based == 'a') * GO

        # MLP state - one for each node's type
    if net_name == 'state':
        NLgen = sum(NL)
        input_shape = tuple(NL + NLgen + AL + 2 * DS)  # [i + NLgen + AL + 2 * DS for i in NL]  # AL + 2 * (NL + DS)
        output_shape = DS

    # MLP output
    elif net_name == 'output':
        input_shape = DS + (problem_based == 'a') * (AL + DS)
        output_shape = T

    # possible values for net_name in ['state','output'], otherwise raise error
    else: raise ValueError(':param net_name: not in [\'state\', \'output\']')

    # hidden layers
    if hidden_units is None:
        hidden_units = []
    elif isinstance(hidden_units, int):
        hidden_units = [hidden_units]
    layers = list(hidden_units) + [output_shape]

    return input_shape, layers
