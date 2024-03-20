import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score, precision_score, recall_score, \
    f1_score


def cal_roc_auc(pred, label):
    """
    计算AUC@ROC
    :param pred:
    :param label:
    :return:
    """
    fpr, tpr, _ = roc_curve(label, pred)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc


def cal_pr_auc(pred, label):
    """
    计算AUC@PR
    :param pred:
    :param label:
    :return:
    """
    precision, recall, _ = precision_recall_curve(label, pred)
    pr_auc = auc(recall, precision)
    return precision, recall, pr_auc


def cal_f1_acc(pred, label):
    """
    计算f1
    :param pred:
    :param label :
    :return:
    """
    threshold = np.percentile(pred, 80)
    pred = (pred > threshold).astype(int)

    acc = accuracy_score(label, pred)

    f1 = f1_score(label, pred, average='binary')

    precision = precision_score(label, pred, average='binary')
    recall = recall_score(label, pred, average='binary')

    return acc, f1, precision, recall
