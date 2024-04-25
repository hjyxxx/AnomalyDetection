import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score, precision_score, recall_score, \
    f1_score, confusion_matrix


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


def cal_f1_acc(pred, label, threshold):
    """

    :param pred:
    :param label:
    :param threshold:
    :return:
    """
    pred = (pred > threshold).astype(int)
    f1 = f1_score(label, pred, average='binary')
    acc = accuracy_score(label, pred)
    precision = precision_score(label, pred, average='binary')
    recall = recall_score(label, pred, average='binary')
    conf_matrix = confusion_matrix(label, pred)
    TP = conf_matrix[1, 1]
    FP = conf_matrix[0, 1]
    TN = conf_matrix[0, 0]
    FN = conf_matrix[1, 0]
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)

    return f1, acc, precision, recall, TP, FP, TN, FN, TPR, FPR


def cal_f1_acc_by_percentile(pred, label):
    """
    计算f1
    :param pred:
    :param label:
    :return:
    """
    threshold = np.percentile(pred, 80)

    f1, acc, precision, recall, TP, FP, TN, FN, TPR, FPR = cal_f1_acc(pred, label, threshold)

    return threshold, f1, acc, precision, recall, TP, FP, TN, FN, TPR, FPR


def cal_f1_acc_by_best(pred, label):
    """
    :param pred:
    :param label:
    :return:
    """
    thresholds = pred

    f1s = [f1_score(label, (pred > threshold).astype(int)) for threshold in thresholds]
    max_index = f1s.index(max(f1s))
    threshold = thresholds[max_index]
    pred = (pred > threshold).astype(int)
    f1, acc, precision, recall, TP, FP, TN, FN, TPR, FPR = cal_f1_acc(pred, label, threshold)

    return threshold, f1, acc, precision, recall, TP, FP, TN, FN, TPR, FPR
