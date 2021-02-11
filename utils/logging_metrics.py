import itertools

import matplotlib.pyplot as plt
import numpy as np
from neptune import log_image, log_metric
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             explained_variance_score, mean_absolute_error,
                             mean_squared_error,
                             precision_recall_fscore_support, precision_score,
                             r2_score, recall_score)


def logging_classification(y_true, y_pred, name=None):
    """ logging_classification logging metrics for classification

    - metrics: accuracy, precision, recall
    - figure: confusion matrix
    - score by metrics
    Parameters
    ----------
    y_true : 1d array-like
        Ground truth target values
    y_pred : 1d or 2d array-like
        Estimated targets

    Returns
    -------
    None
    """

    is_multiclass = False
    if len(set(y_true)) > 2:
        is_multiclass = True

    # if prediction values are  probability, choose the maximum index.
    if y_pred.ndim == 2:
        y_pred = np.argmax(y_pred, axis=1)

    # accuracy
    acc = accuracy_score(y_true, y_pred)
    log_metric('Accuracy', acc)

    # recall
    if is_multiclass:
        recall = recall_score(y_true, y_pred, average='micro')
        log_metric('Recall(micro)', recall)
    else:
        recall = recall_score(y_true, y_pred)
        log_metric('Recall', recall)

    # precision
    if is_multiclass:
        precision = precision_score(y_true, y_pred, average='micro')
        log_metric('Precision(micro)', precision)
    else:
        precision = precision_score(y_true, y_pred)
        log_metric('Recall', precision)

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig = plot_confusion_matrix(cm)
    log_image('performance charts', fig)

    # other metrics
    for metric_name, values in zip(
        ['precision', 'recall', 'fbeta_score', 'support'],
            precision_recall_fscore_support(y_true, y_pred)):
        for i, value in enumerate(values):
            log_metric('{}_class_{}_sklearn'.format(metric_name, i), value)
    return None


def logging_regression(y_true, y_pred):
    """logging_regression logging metrics for regression problem

    - rmse
    - mae
    - r2
    - explained variance
    - yyplot

    Parameters
    ----------
    y_true : 1d array like
        ground truth target value
    y_pred : 1d array like
        estimated target value

    Returns
    -------
    None

    """
    # rmse
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    log_metric('RMSE', rmse)

    # mae
    mae = mean_absolute_error(y_true, y_pred)
    log_metric('MAE', mae)

    # r2
    r2 = r2_score(y_true, y_pred)
    log_metric('R2', r2)

    # explained variance
    evs = explained_variance_score(y_true, y_pred)
    log_metric('Explained Variance', evs)

    # 相関
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    log_metric('corr', corr)

    # scatter plot
    fig = yyplot(y_true, y_pred)
    log_image('performance charts', fig)

    return None


def plot_confusion_matrix(cm,
                          target_names=None,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=False):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j,
                     i,
                     "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j,
                     i,
                     "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(
        accuracy, misclass))

    # plt.show()
    return fig


def yyplot(y_true, y_pred):
    yvalues = np.concatenate([y_true.flatten(), y_pred.flatten()])
    ymin, ymax, yrange = np.amin(yvalues), np.amax(yvalues), np.ptp(yvalues)
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred)
    plt.plot([ymin - yrange * 0.01, ymax + yrange * 0.01],
             [ymin - yrange * 0.01, ymax + yrange * 0.01])
    plt.xlim(ymin - yrange * 0.01, ymax + yrange * 0.01)
    plt.ylim(ymin - yrange * 0.01, ymax + yrange * 0.01)
    plt.xlabel('y_true', fontsize=16)
    plt.ylabel('y_pred', fontsize=16)
    plt.title('Ground truth vs Prediction', fontsize=16)
    plt.tick_params(labelsize=16)
    # plt.show()

    return fig
