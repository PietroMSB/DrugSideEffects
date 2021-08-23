# coding=utf-8
import numpy as np
from sklearn import metrics as mt
from tensorflow.keras.backend import flatten


#######################################################################################################################
### METRICS ###########################################################################################################
#######################################################################################################################
# FOR DETAILS <https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
# Hp: the target is one-hot encoded
# targs	  = targets, as it is, with shape (n_samples, dim_target)
# y_score = raw output of the model with shape (n_samples, dim_output)
# y_true  = argmax(targets, axis=1)
# y_pred  = argmax(output,  axis=1)


# ---------------------------------------------------------------------------------------------------------------------
def TPR(y_true, y_pred):
    return mt.recall_score(y_true=y_true, y_pred=y_pred)


# ---------------------------------------------------------------------------------------------------------------------
def TNR(y_true, y_pred):
    return 2 * mt.balanced_accuracy_score(y_true=y_true, y_pred=y_pred) - mt.recall_score(y_true=y_true, y_pred=y_pred)


# ---------------------------------------------------------------------------------------------------------------------
def FPR(y_true, y_pred):
    return 1 - TNR(y_true=y_true, y_pred=y_pred)


# ---------------------------------------------------------------------------------------------------------------------
def FNR(y_true, y_pred):
    return 1 - TPR(y_true=y_true, y_pred=y_pred)


# ---------------------------------------------------------------------------------------------------------------------
def accuracy_per_class(y_true, y_pred, class_label: int = None):
    mat = mt.confusion_matrix(y_true=y_true, y_pred=y_pred)
    class_accuracy = np.diag(mat) / np.sum(mat, axis=1)
    if class_label is not None:
        class_accuracy = class_accuracy[class_label]
    return class_accuracy


# ---------------------------------------------------------------------------------------------------------------------
def plot_roc(fpr, tpr, roc_auc, savedir, *, line_width=1.5, cmap='Set2'):
    """ Plot ROC curve. Show figure with plot_figure, save figure with savedir==path """
    import matplotlib.pyplot as plt
    from matplotlib import cm

    color_map = cm.get_cmap(cmap)
    plt.figure()

    for i, key in enumerate(fpr):
        plt.plot(fpr[key], tpr[key], color=color_map(i), lw=line_width + 1 if key in ['macro', 'micro'] else line_width,
                 label='ROC curve - class {0} (area = {1:0.2f})'.format(key, roc_auc[key]))

    plt.plot([0, 1], [0, 1], color='navy', lw=line_width, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC - Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    if savedir: plt.savefig(savedir)


# ---------------------------------------------------------------------------------------------------------------------
def plot_prisofs(recall, precision, avg_precision, savedir, *, line_width=1.5, cmap='Set2'):
    """ Plot Precision-Recall Cure. Show figure with plot_figure, save figure with savedir==path """
    import matplotlib.pyplot as plt
    from matplotlib import cm

    color_map = cm.get_cmap(cmap)
    plt.figure()

    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []

    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')

    for i, key in enumerate(precision):
        l, = plt.plot(recall[key], precision[key], color=color_map(i), lw=line_width)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.2f})'.format(i, avg_precision[i]))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve to multi-class with iso-Fscore curves')
    plt.legend(lines, labels, loc="lower center")
    if savedir: plt.savefig(savedir)


# ---------------------------------------------------------------------------------------------------------------------
def ROC(y_test, y_score, savedir='', macro_and_micro: bool = False, pos_label=0):
    """ Receiver Operating Characteristic curve: process and plot/save """
    n_classes = y_test.shape[1]
    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = mt.roc_curve(y_test[:, i], y_score[:, i], pos_label=pos_label)
        roc_auc[i] = mt.auc(fpr[i], tpr[i])

    if macro_and_micro == True:
        # flatten the matrices to obtain a single array
        y_test = flatten(y_test)
        y_score = flatten(y_score)

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = mt.roc_curve(y_test, y_score, pos_label=pos_label)
        roc_auc["micro"] = mt.auc(fpr["micro"], tpr["micro"])

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes): mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = mt.auc(fpr["macro"], tpr["macro"])

    plot_roc(fpr, tpr, roc_auc, savedir)


# ---------------------------------------------------------------------------------------------------------------------
def PRISOFS(targs, y_score, savedir='', pos_label=0):
    """ Precision, Recall and ISO FScore curve: process and plot/save """
    precision, recall, average_precision_score = dict(), dict(), dict()
    for i in range(targs.shape[1]):
        precision[i], recall[i], _ = mt.precision_recall_curve(targs[:, i], y_score[:, i], pos_label=pos_label)
        average_precision_score[i] = mt.average_precision_score(targs[:, i], y_score[:, i], pos_label=pos_label)
    plot_prisofs(recall, precision, average_precision_score, savedir)


######
Metrics = {'Acc': mt.accuracy_score, 'Bacc': mt.balanced_accuracy_score, 'Js': mt.jaccard_score,
           'Ck': mt.cohen_kappa_score, 'Prec': mt.precision_score, 'Rec': mt.recall_score,
           'Fs': mt.f1_score, 'Tpr': TPR, 'Tnr': TNR, 'Fpr': FPR, 'Fnr': FNR,
           'Cl0': accuracy_per_class, 'Cl1':accuracy_per_class}
