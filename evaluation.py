"""
evaluation.py
-------------
Shared evaluation metrics for the credit card fraud detection project.
All metrics implemented from scratch using NumPy.

Used by: train_logistic.py, train_knn.py, app.py
"""

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute 2x2 confusion matrix.

    Returns
    -------
    cm : np.ndarray, shape (2, 2)
        [[TN, FP],
         [FN, TP]]
    """
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    TP = np.sum((y_true == 1) & (y_pred == 1))
    return np.array([[TN, FP], [FN, TP]])


def precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Precision = TP / (TP + FP)"""
    cm = confusion_matrix(y_true, y_pred)
    TP, FP = cm[1, 1], cm[0, 1]
    return TP / (TP + FP) if (TP + FP) > 0 else 0.0


def recall_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Recall = TP / (TP + FN)"""
    cm = confusion_matrix(y_true, y_pred)
    TP, FN = cm[1, 1], cm[1, 0]
    return TP / (TP + FN) if (TP + FN) > 0 else 0.0


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F1 = 2 * (Precision * Recall) / (Precision + Recall)"""
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def auc_roc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Compute Area Under the ROC Curve using the trapezoidal rule.

    Parameters
    ----------
    y_true   : binary labels {0, 1}
    y_scores : predicted probabilities for class 1
    """
    desc_order = np.argsort(-y_scores)
    y_sorted = y_true[desc_order]
    scores_sorted = y_scores[desc_order]

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    if n_pos == 0 or n_neg == 0:
        return 0.0

    tpr_list = [0.0]
    fpr_list = [0.0]
    tp = 0
    fp = 0

    for i in range(len(y_sorted)):
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1

        if i == len(y_sorted) - 1 or scores_sorted[i] != scores_sorted[i + 1]:
            tpr_list.append(tp / n_pos)
            fpr_list.append(fp / n_neg)

    tpr_arr = np.array(tpr_list)
    fpr_arr = np.array(fpr_list)

    _trapz = getattr(np, "trapezoid", None) or np.trapz
    return _trapz(tpr_arr, fpr_arr)


def roc_curve_data(y_true: np.ndarray, y_scores: np.ndarray) -> tuple:
    """
    Compute FPR and TPR arrays for plotting the ROC curve.

    Returns
    -------
    fpr, tpr, thresholds : np.ndarray
    """
    desc_order = np.argsort(-y_scores)
    y_sorted = y_true[desc_order]
    scores_sorted = y_scores[desc_order]

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    tpr_list = [0.0]
    fpr_list = [0.0]
    thresh_list = [scores_sorted[0] + 1e-6]

    tp = 0
    fp = 0

    for i in range(len(y_sorted)):
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1

        if i == len(y_sorted) - 1 or scores_sorted[i] != scores_sorted[i + 1]:
            tpr_list.append(tp / n_pos)
            fpr_list.append(fp / n_neg)
            thresh_list.append(scores_sorted[i])

    return np.array(fpr_list), np.array(tpr_list), np.array(thresh_list)


def compute_all_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba_class1: np.ndarray
) -> dict:
    """Compute all four evaluation metrics at once."""
    return {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auc_roc": auc_roc(y_true, y_proba_class1),
    }


def print_metrics(metrics: dict, header: str = "") -> None:
    """Pretty-print evaluation metrics."""
    if header:
        print(f"\n  {header}")
        print("  " + "-" * 40)
    for name, val in metrics.items():
        print(f"    {name:12s}: {val:.4f}")
