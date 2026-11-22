"""Metrics and threshold helpers ported from notebook cell 84, plus policy helpers."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)

MAX_FPR = 0.05


def tpr_at_fpr(y_true, scores, max_fpr=MAX_FPR):
    """Largest TPR reachable while FPR <= max_fpr (ranking quality)."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(tpr[np.where(fpr <= max_fpr)[0][-1]])


def threshold_at_fpr(y_true, scores, max_fpr=MAX_FPR):
    """Score threshold at the last ROC point with FPR <= max_fpr (as in the notebook)."""
    fpr, _, thresholds = roc_curve(y_true, scores)
    return float(thresholds[np.where(fpr <= max_fpr)[0][-1]])


def evaluate(y_true, scores, thr):
    y_pred = (scores >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "TPR@FPR5% (xếp hạng)": tpr_at_fpr(y_true, scores),
        "TPR@ngưỡng": float(tp / (tp + fn)),
        "FPR": float(fp / (fp + tn)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_true, scores)),
        "PR-AUC": float(average_precision_score(y_true, scores)),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }


def confusion_at(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(bool)
    tp = int(np.sum(y_pred & (y_true == 1)))
    fp = int(np.sum(y_pred & (y_true == 0)))
    fn = int(np.sum(~y_pred & (y_true == 1)))
    tn = int(np.sum(~y_pred & (y_true == 0)))
    return tp, fp, fn, tn


def rates(tp, fp, fn, tn):
    return {
        "TPR": tp / (tp + fn) if tp + fn else float("nan"),
        "FPR": fp / (fp + tn) if fp + tn else float("nan"),
        "Precision": tp / (tp + fp) if tp + fp else float("nan"),
    }
