import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve


def cal_roc_auc(pred, label):
    fpr, tpr, _ = roc_curve(label, pred)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc


def cal_pr_auc(pred, label):
    p, r, _ = precision_recall_curve(label, pred)
    pr_auc = np.trapz(p, r)
    return p, r, pr_auc