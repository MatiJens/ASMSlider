import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    recall_score,
    roc_curve,
)


def tpr_at_fpr(y_true, y_score, target_fpr):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = max(np.searchsorted(fpr, target_fpr, side="right") - 1, 0)
    return float(tpr[idx])


def best_f1(y_true, y_score):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    p, r = precision[:-1], recall[:-1]
    f1 = np.where((p + r) > 0, 2 * p * r / (p + r), 0.0)
    idx = int(np.argmax(f1))
    return float(thresholds[idx]), float(f1[idx]), float(p[idx]), float(r[idx])


def classifier_metrics(probs, targets, threshold=0.7):
    if np.isnan(probs).any():
        return {"mcc": 0.0, "ap": 0.0, "recall": 0.0, "f1": 0.0}
    preds = (probs >= threshold).astype(int)
    return {
        "mcc": matthews_corrcoef(targets, preds),
        "ap": average_precision_score(targets, probs),
        "recall": recall_score(targets, preds, zero_division=0),
        "f1": f1_score(targets, preds, zero_division=0),
    }
