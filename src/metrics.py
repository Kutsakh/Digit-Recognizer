from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_curve
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(y_test, y_pred, fig_size):
    cf_matrix = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=fig_size)
    sns.heatmap(cf_matrix / np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues', ax=ax)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    pred_true = np.sum(np.array(cf_matrix).diagonal())
    pred_num = np.sum(cf_matrix)
    accuracy = pred_true / pred_num
    return accuracy


def print_precision_recall(y_test, y_pred):
    print(f'Precision {precision_score(y_test, y_pred)}')
    print(f'Recall {recall_score(y_test, y_pred)}')


def plot_roc_curve(y_test, y_proba):
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--')  # dashed diagonal
    plt.grid()
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate (recall)')
    plt.show()
