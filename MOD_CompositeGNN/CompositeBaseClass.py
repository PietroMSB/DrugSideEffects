from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from typing import Union, Optional

import numpy as np
import tensorflow as tf
from numpy import array
from pandas import DataFrame

import MOD_CompositeGNN.GNN_metrics as mt
from MOD_CompositeGNN.composite_graph_class import CompositeGraphObject, CompositeGraphTensor


class CompositeBaseClass(ABC):
    ## CONSTRUCTORS METHODS ###########################################################################################
    def __init__(self,
                 node_type_number: int,
                 optimizer: tf.keras.optimizers.Optimizer,
                 loss_function: tf.keras.losses.Loss,
                 loss_arguments: Optional[dict],
                 addressed_problem: str,
                 extra_metrics: Optional[dict] = None,
                 extra_metrics_arguments: Optional[dict[str, dict]] = None,
                 path_writer: str = 'writer/',
                 namespace='CGNN') -> None:
        """ CONSTRUCTOR - Other attributes must be defined in inheriting class
        
        :param node_type_number: (int) specifying the number of nodes' type the model has to deal with.
        :param optimizer: (tf.keras.optimizers) for gradient application, initialized externally.
        :param loss_function: (tf.keras.losses) for the loss computation.
        :param loss_arguments: (dict) with some {'argument':values} one could pass to loss when computed.
        :param addressed_problem: (str) in ['r','c'], 'r':regression, 'c':classification for the addressed problem.
        :param extra_metrics: None or dict {'name':function} for metrics to be watched during training/validation/test procedures.
        :param extra_metrics_arguments: None or dict {'name':{'argument':value}} for arguments passed to extra_metrics['name'].
        :param path_writer: (str) path for saving TensorBoard objects in training procedure. If folder is not empty, all files are removed.
        :param namespace: (str) namespace for tensorboard visualization.
        """
        # check types and values
        if addressed_problem not in ['c', 'r']: raise ValueError('param <addressed_problem> not in [\'c\',\'r\']')
        if not isinstance(extra_metrics, (dict, type(None))): raise TypeError('type of param <extra_metrics> must be None or dict')

        # set attributes
        self.node_type_number = node_type_number
        self.loss_function = loss_function
        self.loss_args = dict() if loss_arguments is None else loss_arguments
        self.optimizer = optimizer

        # Problem type: c: Classification | r: Regression
        self.addressed_problem = addressed_problem

        # Metrics to be evaluated during training process
        self.extra_metrics = dict() if extra_metrics is None else extra_metrics
        self.mt_args = dict() if extra_metrics_arguments is None else extra_metrics_arguments

        # Writer and Namespace for Tensorboard - Nets histograms and Distributions
        if path_writer[-1] != '/': path_writer += '/'
        if type(namespace) != list: namespace = [namespace]
        if os.path.exists(path_writer): shutil.rmtree(path_writer)
        self.path_writer = path_writer
        self.namespace = namespace

        # history object (dict) - to summarize the training process, initialized as empty dict
        self.history = dict()

    ## ABSTRACT METHODS ###############################################################################################
    @abstractmethod
    def copy(self, *, path_writer: str = '', namespace: str = '', copy_weights: bool = True):
        """ COPY METHOD - defined in inheriting class

        :param path_writer: None or (str), to save copied gnn tensorboard writer.
        :param namespace: (str) for tensorboard visualization in model training procedure.
        :param copy_weights: (bool) True: state and output weights are copied in new model, otherwise they are re-initialized.
        :return: a Deep Copy of the GNN instance.
        """
        pass

    # -----------------------------------------------------------------------------------------------------------------
    @abstractmethod
    def save(self, path: str) -> None:
        """ Save model to folder <path> """
        pass

    # -----------------------------------------------------------------------------------------------------------------
    @classmethod
    @abstractmethod
    def load(self, path: str, path_writer: str, namespace: str,
             extra_metrics: Optional[dict] = None, extra_metrics_arguments: Optional[dict[str, dict]] = None):
        """ Load model from folder

        Only Loss is considered as metrics after loading process.
        To use more metrics, set :param extra_metrics: and :param extra_metrics_arguments:

        :param path: (str) folder path containing all useful files to load the model
        :param path_writer: (str) path for writer folder. !!! Constructor method makes delete a non-empty folder and makes a new empty one.
        :param namespace: (str) namespace for tensorboard visualization of the model in training procedure.
        :param extra_metrics: None or dict {'name':function} for metrics to be watched during training/validation/test procedures.
        :param extra_metrics_arguments: None or dict {'name':{'argument':value}} for arguments passed to extra_metrics['name'].
        :return: the model
        """
        pass

    # -----------------------------------------------------------------------------------------------------------------
    @abstractmethod
    def get_dense_layers(self) -> list[tf.keras.layers.Layer]:
        """ Get dense layer for the application of regularizers in training time """
        pass

    # -----------------------------------------------------------------------------------------------------------------
    @abstractmethod
    def trainable_variables(self) -> tuple[list[list[tf.Tensor]], list[list[tf.Tensor]]]:
        """ Get tensor weights for net_state and net_output for each gnn layer """
        pass

    # -----------------------------------------------------------------------------------------------------------------
    @abstractmethod
    def get_weights(self) -> tuple[list[list[array]], list[list[array]]]:
        """ Get array weights for net_state and net_output for each gnn layer """
        pass

    # -----------------------------------------------------------------------------------------------------------------
    @abstractmethod
    def set_weights(self, weights_state: Union[list[array], list[list[array]]],
                    weights_output: Union[list[array], list[list[array]]]) -> None:
        """ Set weights for net_state and net_output """
        pass

    # -----------------------------------------------------------------------------------------------------------------
    @abstractmethod
    def Loop(self, g: Union[CompositeGraphObject, CompositeGraphTensor], *, training: bool = False) -> tuple[int, tf.Tensor, tf.Tensor]:
        """ Process a single CompositeGraphObject element g, returning iteration, states and output """
        pass

    # -----------------------------------------------------------------------------------------------------------------
    @abstractmethod
    def __call__(self, g: Union[CompositeGraphObject, CompositeGraphTensor]):
        """ Return the model output in test mode (training == False) for graph g of type CompositeGraphObject """
        pass

    ## HISTORY METHOD #################################################################################################
    def printHistory(self) -> None:
        """ Print self.history as a pd.Dataframe. Pandas automatically detects terminal width """
        print('\n', DataFrame(self.history), end='\n\n')

    # -----------------------------------------------------------------------------------------------------------------
    def saveHistory_csv(self, path) -> None:
        """ Save history attribute to .csv file. Extension can be omitted from path string """
        if path[-3:] != '.csv': path += '.csv'
        df = DataFrame(self.history)
        df.to_csv(path, index=False)

    # -----------------------------------------------------------------------------------------------------------------
    def saveHistory_txt(self, path) -> None:
        """ Save history attribute to .txt file. Extension can be omitted from path string """
        if path[-3:] != '.txt': path += '.txt'
        df = DataFrame(self.history)
        with open(path, 'w') as txt:
            txt.write(df.to_string(index=False))

    ## EVALUATE METHODs ###############################################################################################
    def evaluate_single_graph(self, g: Union[CompositeGraphObject, CompositeGraphTensor], training: bool) -> tuple:
        """ Evaluate method for evaluating one CompositeGraphObject or CompositeGraphTensor element g. Returns iteration, loss, targets and outputs """
        pass

    # -----------------------------------------------------------------------------------------------------------------
    def evaluate(self, g: Union[CompositeGraphObject, CompositeGraphTensor, list[CompositeGraphObject, CompositeGraphTensor]]) -> tuple:
        """ Return metrics in self.extra_metrics + Iter & Loss for a CompositeGraphObject/CompositeGraphTensor or a list of them in test mode

        :param g: element/list of CompositeGraphObject/CompositeGraphTensor to be evaluated.
        :return: metrics, target_labels, prediction_labels, targets_raw and prediction_raw.
        """
        # check type - new g is a list of GraphTensors
        g = self.checktype(g)

        # process input data
        iters, losses, targets, outs = zip(*[self.evaluate_single_graph(i, training=False) for i in g])

        # concatenate all the values from every graph and take clas labels or values
        loss = tf.concat(losses, axis=0)
        targets = tf.concat(targets, axis=0)
        y_score = tf.concat(outs, axis=0)
        y_true = tf.argmax(targets, axis=1) if self.addressed_problem == 'c' else targets
        y_pred = tf.argmax(y_score, axis=1) if self.addressed_problem == 'c' else y_score

        # evaluate metrics
        metrics = {k: self.extra_metrics[k](y_true, y_pred, **self.mt_args.get(k, dict())) for k in self.extra_metrics}
        metrics = {k: float(tf.reduce_mean(metrics[k])) for k in metrics}
        metrics['It'] = int(tf.reduce_mean(iters))
        metrics['Loss'] = float(tf.reduce_mean(loss))
        return metrics, y_true, y_pred, targets, y_score

    ## TRAINING METHOD ################################################################################################
    def train(self, gTr: Union[CompositeGraphObject, CompositeGraphTensor, list[CompositeGraphObject, CompositeGraphTensor]], epochs: int,
              gVa: Union[CompositeGraphObject, CompositeGraphTensor, list[CompositeGraphObject, CompositeGraphTensor]] = None,
              update_freq: int = 10, max_fails: int = 10,
              observed_metric='Loss', policy='min',
              *, mean: bool = True, verbose: int = 3) -> None:
        """ TRAINING PROCEDURE

        :param gTr: element/list of GraphsObjects/GraphTensors used for the learning procedure.
        :param epochs: (int) the max number of epochs for the learning procedure.
        :param gVa: element/list of GraphsObjects/GraphTensors for early stopping. Default None, no early stopping performed.
        :param update_freq: (int) how many epochs must be completed before evaluating gVa and gTr and/or print learning progress. Default 10.
        :param max_fails: (int) specifies the max number of failures in gVa improvement loss evaluation before early sopping. Default 10.
        :param observed_metric: (str) key of the metric to be observed for early sopping
        :param policy: (str) possible choices: ['min','max'], to minimize/maximize :param observed_metric: during learning procedure
        :param mean: (bool) if False the applied gradients are computed as the sum of every iteration, otherwise as the mean. Default True.
        :param verbose: (int) 0: silent mode; 1: print history; 2: print epochs/batches, 3: history + epochs/batches. Default 3.
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        def update_history(name: str, val: dict[str, float], e=None) -> None:
            """ update self.history with a dict s.t. val.keys()==self.history.keys()^{'Epoch','Best Loss Va'} """
            # name must be 'Tr' or 'Va', to update correctly training or validation history
            if name not in ['Tr', 'Va']: raise TypeError('param <name> must be \'Tr\' or \'Va\'')
            for key in val: self.history[f'{key} {name}'].append(val[key])
            if e is not None: self.history['Epoch'].append(e)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        def reset_validation(new_valid_best_metric_value: float) -> tuple[float, int, list[list[array]], list[list[array]]]:
            """ reset the validation check parameters and to save the 'best weights until now' """
            wst, wout = self.get_weights()
            return new_valid_best_metric_value, 0, wst, wout

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        def regularizer_terms():
            extra_loss = 0
            for layer in self.get_dense_layers():
                if layer.kernel_regularizer is not None: extra_loss += layer.kernel_regularizer(layer.kernel)
                if layer.bias_regularizer is not None: extra_loss += layer.bias_regularizer(layer.bias)
            return extra_loss

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        def training_step(gTr: CompositeGraphTensor, mean: bool) -> None:
            """ compute the gradients and apply them """
            with tf.GradientTape(persistent=True) as tape:
                iter, loss, *_ = self.evaluate_single_graph(gTr, training=True)
                loss = tf.reduce_sum(loss) + regularizer_terms()
            wS, wO = self.trainable_variables()
            dwbS, dwbO = tape.gradient(loss, [wS, wO])
            # average net_state dw and db w.r.t. the number of iteration.
            if not isinstance(iter, list): iter = [iter]
            assert len(dwbS) == len(dwbO) == len(wS) == len(wO) == len(iter)
            if mean: dwbS = [[[elem / it for elem in type] for type in layer] for it, layer in zip(iter, dwbS)]
            assert len(dwbS) == len(dwbO) == len(wS) == len(wO) == len(iter)

            # apply gradients
            # zipped1 = zip([i for j in dwbS + dwbO for i in j], [i for j in wS + wO for i in j])
            list_dwb = [elem for layer in dwbS for type in layer for elem in type] + [elem for layer in dwbO for elem in layer]
            list_wb = [elem for layer in wS for type in layer for elem in type] + [elem for layer in wO for elem in layer]
            zipped = zip(list_dwb, list_wb)
            self.optimizer.apply_gradients(zipped)

        ### TRAINING FUNCTION -----------------------------------------------------------------------------------------
        if verbose not in range(4): raise ValueError('param <verbose> not in [0,1,2,3]')

        # Checking type for gTr and gVa + Initialization of Validation parameters
        # check type - new gTr is a list of GraphTensors. gVa is in [None, list[CompositeGraphTensor]].
        # All GraphObjects are now GraphTensors, to speed up the learning procedure
        gTr = self.checktype(gTr)
        gVa = self.checktype(gVa)

        # initialize history attribute and writer directory
        if not self.history:
            keys = ['Epoch'] + [i + j for i in ['It', 'Loss'] + list(self.extra_metrics) for j in ([' Tr', ' Va'] if gVa else [' Tr'])]
            if gVa: keys += ['Fail', f'Best {observed_metric} Va']
            self.history.update({i: list() for i in keys})
            os.makedirs(self.path_writer)

        # Writers: Training, Validation (scalars) + Net_state, Net_output (histogram for weights/biases)
        netS_writer = [tf.summary.create_file_writer(f'{self.path_writer}Net - State - T{i}') for i in range(self.node_type_number)]
        netO_writer = tf.summary.create_file_writer(f'{self.path_writer}Net - Output')
        trainining_writer = tf.summary.create_file_writer(f'{self.path_writer}Training')
        if gVa:
            assert policy in ['min', 'max']
            best_valid_key = f'Best {observed_metric} Va'
            policy_function, valid_new_metric_value = (np.less, float(1e30)) if policy == 'min' else (np.greater, float(-1e30))
            if self.history[best_valid_key]: valid_new_metric_value = self.history[best_valid_key][-1]
            valid_best_metric, valid_fails, ws, wo = reset_validation(valid_new_metric_value)
            validation_writer = tf.summary.create_file_writer(self.path_writer + 'Validation')

        # pre-Training procedure: check if it's the first learning time to correctly update tensorboard
        initial_epoch = self.history['Epoch'][-1] + 1 if self.history['Epoch'] else 0
        epochs += initial_epoch

        ### TRAINING PROCEDURE
        for e in range(initial_epoch, epochs):

            # TRAINING STEP
            for i, elem in enumerate(gTr):
                training_step(elem, mean=mean)
                if verbose > 2: print(f' > Epoch {e:4d}/{epochs} \t\t> Batch {i + 1:4d}/{len(gTr)}', end='\r')

            # TRAINING EVALUATION STEP
            if e % update_freq == 0:
                metricsTr, *_ = self.evaluate(gTr)
                # History Update
                update_history('Tr', metricsTr, e)
                # TensorBoard Update Tr: Losses, Interation@Convergence, Accuracies + histograms of weights
                self.write_scalars(trainining_writer, metricsTr, e)
                for i, j, namespace in zip(*self.get_weights(), self.namespace):
                    for t, (w, stw) in enumerate(zip(i, netS_writer)): self.write_net_weights(stw, namespace, f'N1 T{t}', w, e)
                    self.write_net_weights(netO_writer, namespace, 'N2', j, e)

            # VALIDATION STEP
            if (e % update_freq == 0) and gVa:
                metricsVa, *_ = self.evaluate(gVa)
                valid_new_metric_value = metricsVa[observed_metric]
                # Validation check
                # if valid_new_metric_value < valid_best_metric:
                if policy_function(valid_new_metric_value, valid_best_metric):
                    valid_best_metric, valid_fails, ws, wo = reset_validation(valid_new_metric_value)
                else:
                    valid_fails += 1
                # History Update
                self.history[best_valid_key].append(valid_best_metric)
                self.history['Fail'].append(valid_fails)
                update_history('Va', metricsVa)
                # TensorBoard Update Va: Losses, Interation@Convergence, Accuracies + histograms of weights
                self.write_scalars(validation_writer, metricsVa, e)
                # Early Stoping - reached max_fails for validation set
                if valid_fails >= max_fails:
                    if verbose in [1, 3]: self.printHistory()
                    print('\r Validation Stop')
                    break

            # PRINT HISTORY
            if (e % update_freq == 0) and verbose in [1, 3]: self.printHistory()

        else:
            print('\r End of Epochs Stop')

        # if end of epochs is reached and you're using early stopping, take weights of the overall best epochs
        if gVa: self.set_weights(ws, wo)

        # Tensorboard Update FINAL: write BEST WEIGHTS + BIASES
        for i, j, namespace in zip(*self.get_weights(), self.namespace):
            for t, (w, stw) in enumerate(zip(i, netS_writer)): self.write_net_weights(stw, namespace, f'N1 T{t}', w, e)
            self.write_net_weights(netO_writer, namespace, 'N2', j, e)

    ## TEST METHOD ####################################################################################################
    def test(self, gTe: Union[CompositeGraphObject, CompositeGraphTensor, list[CompositeGraphObject, CompositeGraphTensor]],
             *, rocdir: str = '', micro_and_macro: bool = False, prisofsdir: str = '', pos_label=0) -> dict[str, list[float]]:
        """ TEST PROCEDURE

        :param gTe: element/list of GraphObjects for testing procedure.
        :param acc_classes: (bool) if True print accuracy for each class, in classification problems.
        :param rocdir: (str) path for saving ROC images file.
        :param micro_and_macro: (bool) for computing micro and macro average quantities in roc curve.
        :param prisofsdir: (str) path for saving Precision-Recall curve with ISO F-Score images file.
        :param pos_label: (int) for classification problems, identify the positive class.
        :return: metrics for gTe.
        """
        # check type - new g is a list of GraphTensors
        gTe = self.checktype(gTe)

        # Evaluate all the metrics in gnn.extra_metrics + Iter and Loss
        metricsTe, y_true, y_pred, targets, y_score = self.evaluate(gTe)

        # ROC e PR curves
        if rocdir: mt.ROC(targets, y_score, rocdir, micro_and_macro, pos_label=pos_label)
        if prisofsdir: mt.PRISOFS(targets, y_score, prisofsdir, pos_label=pos_label)
        return metricsTe

    ## K-FOLD CROSS VALIDATION METHOD #################################################################################
    def LKO(self, batches: tuple[Union[list[CompositeGraphTensor], list[list[CompositeGraphTensor]]], list[CompositeGraphTensor], Optional[
        list[CompositeGraphTensor]]],
            epochs: int = 500, training_mode=None, update_freq: int = 10, max_fails: int = 10,
            observed_metric: str = 'Loss', policy='min',
            mean: bool = True, verbose: int = 3) -> dict[str, list[float]]:
        """ LEAVE K OUT CROSS VALIDATION PROCEDURE

        :param batches: (tuple) s.t. batches[0]:=training, [1]:=test, [2]:=validation. It is the output of prepare_LKO_data function
        :param epochs: (int) number of epochs for training <gnn>, the gnn will be trained for all the epochs.
        :param training_mode: (str) for lgnn cross validation procedure. LGNN.train() for details.
        :param update_freq: (int) specifies how many epochs must be completed before evaluating gVa and gTr.
        :param max_fails: (int) specifies the max number of failures before early sopping.
        :param observed_metric: (str) key of the metric to be observed for early sopping
        :param policy: (str) possible choices: ['min','max'], to minimize/maximize :param observed_metric: during learning procedure
        :param mean: (bool) if False the applied gradients are computed as the sum of every iteration, else as the mean.
        :param verbose: (int) 0: silent mode; 1: print history; 2: print epochs/batches, 3: history + epochs/batches. Default 3.
        :return: a dict containing all the considered metrics in <gnn>.history.
        """

        # initialize results
        metrics = {i: list() for i in list(self.extra_metrics) + ['It', 'Loss']}

        # If model is LGNN, integrate training_mode in training procedure as kwargs dict
        kwargs = dict()
        if training_mode: kwargs['training_mode'] = training_mode

        # LKO PROCEDURE
        number_of_batches = len(batches[0])
        for i, (gTr, gTe, gVa) in enumerate(zip(*batches)):

            # model creation, learning and test
            print(f'\nBATCH K-OUT {i + 1}/{number_of_batches}')
            temp = self.copy(copy_weights=False, path_writer=self.path_writer + str(i), namespace=f'Batch {i + 1}-{number_of_batches}')
            temp.train(gTr, epochs, gVa, update_freq, max_fails, observed_metric, policy, mean=mean, verbose=verbose, **kwargs)
            res = temp.test(gTe)

            # evaluate metrics
            for m in res: metrics[m].append(res[m])

            # verbose option
            if verbose > 1: print(f'\nRESULTS BATCH {i + 1}/{number_of_batches}\n', DataFrame(res, index=['res']).transpose())
        return metrics

    ## STATIC METHODs #################################################################################################
    @staticmethod
    def get_filtered_tensor(g: CompositeGraphTensor, inp: tf.Tensor):
        """ Get inp [targets or sample_weights] for graph based problems -> nodes states are not filtered by set_mask and output_mask """
        mask = tf.boolean_mask(g.set_mask, g.output_mask)
        return tf.boolean_mask(inp, mask)

    # -----------------------------------------------------------------------------------------------------------------
    @staticmethod
    def checktype(elem: Optional[Union[CompositeGraphObject, CompositeGraphTensor, list[CompositeGraphObject, CompositeGraphTensor]]]) -> \
            list[CompositeGraphTensor]:
        """ check if type(elem) is correct. If so, return None or a list og GraphObjects """
        if elem is None:
            pass
        elif isinstance(elem, CompositeGraphTensor):
            elem = [elem]
        elif isinstance(elem, CompositeGraphObject):
            elem = [CompositeGraphTensor.fromGraphObject(elem)]
        elif isinstance(elem, (list, tuple)) and all(isinstance(g, (CompositeGraphObject, CompositeGraphTensor)) for g in elem):
            elem = [CompositeGraphTensor.fromGraphObject(g) if isinstance(g, CompositeGraphObject) else g for g in elem]
        else:
            raise TypeError(
                'Error - <gTr> and/or <gVa> are not CompositeGraphObject/CompositeGraphTensor or LIST/TUPLE of GraphObjects/GraphTensors')
        return elem

    # -----------------------------------------------------------------------------------------------------------------
    @staticmethod
    def write_scalars(writer: tf.summary.SummaryWriter, metrics: dict[str, float], epoch: int) -> None:
        """ TENSORBOARD METHOD: writes scalars values of the metrics """
        if type(metrics) != dict: raise TypeError('type of param <metrics> must be dict')
        names = {'Acc': 'Accuracy', 'Bacc': 'Balanced Accuracy', 'Ck': 'Cohen\'s Kappa', 'Js': 'Jaccard Score',
                 'Fs': 'F1-Score', 'Prec': 'Precision Score', 'Rec': 'Recall Score', 'Tpr': 'TPR', 'Tnr': 'TNR',
                 'Fpr': 'FPR', 'Fnr': 'FNR', 'Loss': 'Loss', 'It': 'Iteration @ Convergence'}

        namescopes = {**{i: 'Accuracy & Loss' for i in ['Acc', 'Bacc', 'It', 'Loss']},
                      **{i: 'F-Score, Precision and Recall' for i in ['Fs', 'Prec', 'Rec']},
                      **{i: 'Positive and Negative Rates' for i in ['Tpr', 'Tnr', 'Fpr', 'Fnr']},
                      **{i: 'Other Scores' for i in ['Ck', 'Js']}}

        with writer.as_default():
            for i in metrics:
                name = names.get(i, i)
                with tf.name_scope(namescopes.get(i, 'Other Scores')):
                    tf.summary.scalar(name, metrics[i], step=epoch, description=name)

    # -----------------------------------------------------------------------------------------------------------------
    @staticmethod
    def write_net_weights(writer: tf.summary.SummaryWriter, namespace: str, net_name: str, val_list: list[array], epoch: int) -> None:
        """ TENSORBOARD METHOD: writes histograms of the nets weights """
        W, B, names_layers = val_list[0::2], val_list[1::2], [f'{net_name} L{i}' for i in range(len(val_list) // 2)]
        assert len(names_layers) == len(W) == len(B)

        with writer.as_default():
            for n, w, b in zip(names_layers, W, B):
                with tf.name_scope(f'{namespace}: Weights'):
                    tf.summary.histogram(n, w, step=epoch)
                with tf.name_scope(f'{namespace}: Biases'):
                    tf.summary.histogram(n, b, step=epoch)
