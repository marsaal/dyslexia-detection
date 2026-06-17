"""
utils/metrics.py
Càlcul i agregació de les mètriques principals utilitzades.
"""

import json
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    f1_score,
    fbeta_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    accuracy_score,
    confusion_matrix,
)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Calcula totes les mètriques principals per a un sol fold o avaluació.

    Parameters
    ----------
    y_true : array de 0/1
    y_prob : probabilitats de la classe positiva (dislèxia)
    threshold : llindar de decisió (defecte 0.5)

    Returns
    -------
    dict amb: f1, f2, recall, specificity, roc_auc, pr_auc,
              balanced_acc, accuracy
    """
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # AUC necessita almenys una mostra de cada classe
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
    except ValueError:
        roc_auc = float("nan")
        pr_auc = float("nan")

    return {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def aggregate_metrics(metrics_list: list[dict]) -> dict:
    """
    Retorna la mitjana i la desviació estàndard per a cada mètrica.

    Returns
    -------
    dict {metric_name: {"mean": float, "std": float}}
    """
    keys = metrics_list[0].keys()
    result = {}
    for k in keys:
        vals = np.array([m[k] for m in metrics_list], dtype=float)
        result[k] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}
    return result


def format_metrics(agg: dict) -> dict:
    """Retorna strings amb el format '0.842 ± 0.037' per a la memòria."""
    return {k: f"{v['mean']:.3f} ± {v['std']:.3f}" for k, v in agg.items()}