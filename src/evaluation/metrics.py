import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)

def find_best_threshold(y_true, y_prob):
    # Grid search to find threshold that maximizes F1
    thresholds = np.linspace(0.01, 0.99, 99)
    best_threshold = 0.5
    best_f1 = 0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    return best_threshold

def compute_metrics(y_true, y_prob, threshold=0.5):
    # Convert probabilities to binary predictions
    y_pred = (y_prob >= threshold).astype(int)
    results = {}

    # Threshold-independent metrics
    results["roc_auc"] = roc_auc_score(y_true, y_prob)
    results["pr_auc"] = average_precision_score(y_true, y_prob)
    
    # Threshold-dependent metrics
    results["f1"] = f1_score(y_true, y_pred)
    results["precision"] = precision_score(y_true, y_pred)
    results["recall"] = recall_score(y_true, y_pred)

    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    results["tp"] = tp
    results["tn"] = tn
    results["fp"] = fp
    results["fn"] = fn
    results["threshold"] = threshold

    return results