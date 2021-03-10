# coding=utf-8
import numpy as np
import tensorflow as tf
from sklearn import metrics as mt


#######################################################################################################################
### METRICS ###########################################################################################################
#######################################################################################################################
# FOR DETAILS <https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
# Hp: the target is one-hot encoded
# targs	  = targets, as it is, with shape (n_samples, dim_target)
# y_score = raw output of the model with shape (n_samples, dim_output)
# y_true  = argmax(targets, axis= embedding axis)
# y_pred  = argmax(output,  axis= embedding axis)

# ---------------------------------------------------------------------------------------------------------------------
def accuracy_score(y_true, y_pred, norm=True):
    return mt.accuracy_score(y_true=y_true, y_pred=y_pred, normalize=norm)


# ---------------------------------------------------------------------------------------------------------------------
def auc(fpr, tpr):
    return mt.auc(fpr, tpr)


# ---------------------------------------------------------------------------------------------------------------------
def balanced_accuracy_score(y_true, y_pred):
    return mt.balanced_accuracy_score(y_true=y_true, y_pred=y_pred)


# ---------------------------------------------------------------------------------------------------------------------
def cohen_kappa_score(y_true, y_pred):
    return mt.cohen_kappa_score(y1=y_true, y2=y_pred)


# ---------------------------------------------------------------------------------------------------------------------
def confusion_matrix(y_true, y_pred):
    return mt.confusion_matrix(y_true=y_true, y_pred=y_pred)


# ---------------------------------------------------------------------------------------------------------------------
def fbscore(y_true, y_pred, b=1, avg='binary', pos_label=0):
    return mt.fbeta_score(y_true=y_true, y_pred=y_pred, beta=b, average=avg, zero_division=0, pos_label=pos_label)


# ---------------------------------------------------------------------------------------------------------------------
def jaccard_score(y_true, y_pred, avg='binary', pos_label=0):
    return mt.jaccard_score(y_true=y_true, y_pred=y_pred, average=avg, pos_label=pos_label)


# ---------------------------------------------------------------------------------------------------------------------
def precision_recall_curve(y_test_col, y_score_col, pos_label=0):
    return mt.precision_recall_curve(y_test_col, y_score_col, pos_label=pos_label)


# ---------------------------------------------------------------------------------------------------------------------
def recall_score(y_true, y_pred, avg='binary', pos_label=0):
    return mt.recall_score(y_true=y_true, y_pred=y_pred, average=avg, zero_division=0, pos_label=pos_label)


# ---------------------------------------------------------------------------------------------------------------------
def roc_curve(y_test_col, y_score_col):
    return mt.roc_curve(y_test_col, y_score_col)


# ---------------------------------------------------------------------------------------------------------------------
def precision_score(y_test, y_pred, avg='binary', pos_label=0):
    return mt.precision_score(y_test, y_pred, average=avg, zero_division=0, pos_label=pos_label)


# ---------------------------------------------------------------------------------------------------------------------
def avg_precision_score(y_test, y_score, avg='binary', pos_label=0):
    return mt.average_precision_score(y_true=y_test, y_score=y_score, average=avg, pos_label=pos_label)


# ---------------------------------------------------------------------------------------------------------------------
def TPR(y_true, y_pred):
    return recall_score(y_true=y_true, y_pred=y_pred)


# ---------------------------------------------------------------------------------------------------------------------
def TNR(y_true, y_pred):
    return 2 * balanced_accuracy_score(y_true=y_true, y_pred=y_pred) - recall_score(y_true=y_true, y_pred=y_pred)


# ---------------------------------------------------------------------------------------------------------------------
def FPR(y_true, y_pred):
    return 1 - TNR(y_true=y_true, y_pred=y_pred)


# ---------------------------------------------------------------------------------------------------------------------
def FNR(y_true, y_pred):
    return 1 - TPR(y_true=y_true, y_pred=y_pred)


# ---------------------------------------------------------------------------------------------------------------------
def accuracy_per_class(y_true, y_pred):
    mat = confusion_matrix(y_true=y_true, y_pred=y_pred)
    return np.diag(mat) / np.sum(mat, axis=1)


# ---------------------------------------------------------------------------------------------------------------------
def plot_roc(fpr, tpr, roc_auc, savedir, *, line_width=1.5, cmap='Set2'):
    """ plot ROC curve. Show figure with plot_figure, save figure with savedir==path """
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
    """ plot Precision-Recall Cure. Show figure with plot_figure, save figure with savedir==path """
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
def ROC(y_test, y_score, savedir='', macro_and_micro: bool = False):
    """ Receiver Operating Characteristic curve: process and plot/save """
    n_classes = y_test.shape[1]
    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    if macro_and_micro == True:
        # flatten the matrices to obtain a single array
        y_test = tf.keras.backend.flatten(y_test)
        y_score = tf.keras.backend.flatten(y_score)
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test, y_score)
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes): mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        # Finally average it and compute AUC
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    plot_roc(fpr, tpr, roc_auc, savedir)


# ---------------------------------------------------------------------------------------------------------------------
def PRISOFS(targs, y_score, savedir=''):
    """ Precision, Recall and ISO FScore curve: process and plot/save """
    precision, recall, average_precision_score = dict(), dict(), dict()
    for i in range(targs.shape[1]):
        precision[i], recall[i], _ = precision_recall_curve(targs[:, i], y_score[:, i])
        average_precision_score[i] = avg_precision_score(targs[:, i], y_score[:, i], avg=None)
    plot_prisofs(recall, precision, average_precision_score, savedir)
