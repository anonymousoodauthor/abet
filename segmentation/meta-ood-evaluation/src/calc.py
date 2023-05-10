import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve

from src.helper import counts_array_to_data_list


def calc_precision_recall(data, balance=False):
    if balance:
        x1 = counts_array_to_data_list(np.array(data["in"]), 1e5)
        x2 = counts_array_to_data_list(np.array(data["out"]), 1e5)
    else:
        ratio_in = np.sum(data["in"]) / (np.sum(data["in"]) + np.sum(data["out"]))
        ratio_out = 1 - ratio_in
        x1 = counts_array_to_data_list(np.array(data["in"]), 1e7 * ratio_in)
        x2 = counts_array_to_data_list(np.array(data["out"]), 1e7 * ratio_out)
    probas_pred1 = np.array(x1) / 100
    probas_pred2 = np.array(x2) / 100
    y_true = np.concatenate((np.zeros(len(probas_pred1)), np.ones(len(probas_pred2))))
    y_scores = np.concatenate((probas_pred1, probas_pred2))
    return precision_recall_curve(y_true, y_scores) + (average_precision_score(y_true, y_scores),)


def get_tpr95_ind(fpr, tpr):
    ind = (np.abs(tpr - 0.95)).argmin()
    fpr95 = fpr[ind]
    tpr95 = tpr[ind]
    if tpr95 == 1.0:
        ind -= 1
        fpr95 = fpr[ind]
        tpr95 = tpr[ind]
    return ind, tpr95, fpr95


def calc_sensitivity_specificity(data, balance=False):

    if balance:
        x1 = counts_array_to_data_list(np.array(data["in"]), max_size=1e5)
        x2 = counts_array_to_data_list(np.array(data["out"]), max_size=1e5)
    else:
        x1 = counts_array_to_data_list(np.array(data["in"]))
        x2 = counts_array_to_data_list(np.array(data["out"]))

    probas_pred1 = np.array(x1) / 100
    probas_pred2 = np.array(x2) / 100
    y_true = np.concatenate((np.zeros(len(probas_pred1)), np.ones(len(probas_pred2)))).astype(
        "uint8"
    )
    y_scores = np.concatenate((probas_pred1, probas_pred2))
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    return fpr, tpr, thresholds, auc(fpr, tpr)
