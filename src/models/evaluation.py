import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

from src.utils.config import BOOTSTRAP_N_ITERS, BOOTSTRAP_CI, RANDOM_SEED


def compute_binary_metrics(y_true, y_pred_proba, threshold: float = 0.5) -> dict:
    """
    Compute standard binary classification metrics from probabilities.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred_proba = np.asarray(y_pred_proba).astype(float)

    if y_true.shape[0] != y_pred_proba.shape[0]:
        raise ValueError("y_true and y_pred_proba must have the same length")

    y_pred = (y_pred_proba >= threshold).astype(int)

    metrics = {}
    # AUC can fail if only one class is present. Guard against that explicitly.
    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_pred_proba))
    except ValueError:
        metrics["auc"] = float("nan")

    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

    return metrics


def _bootstrap_once(y_true, y_pred_proba, rng, threshold: float):
    n = len(y_true)
    idx = rng.integers(0, n, size=n)
    return compute_binary_metrics(y_true[idx], y_pred_proba[idx], threshold=threshold)


def bootstrap_confidence_intervals(
    y_true,
    y_pred_proba,
    n_iters: int = BOOTSTRAP_N_ITERS,
    ci_percent: int = BOOTSTRAP_CI,
    threshold: float = 0.5,
    seed: int = RANDOM_SEED,
) -> dict:
    """
    Bootstrap test metrics to get mean and confidence intervals.

    Returns a dict of the form:
        {
            'auc': {'mean': ..., 'lower': ..., 'upper': ...},
            'f1':  {...},
            ...
        }
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred_proba = np.asarray(y_pred_proba).astype(float)

    rng = np.random.default_rng(seed)

    metric_names = ["auc", "f1", "precision", "recall", "accuracy"]
    collected = {m: [] for m in metric_names}

    for _ in range(n_iters):
        m = _bootstrap_once(y_true, y_pred_proba, rng, threshold=threshold)
        for name in metric_names:
            collected[name].append(m[name])

    lower_q = (100 - ci_percent) / 2
    upper_q = 100 - lower_q

    summary = {}
    for name in metric_names:
        vals = np.asarray(collected[name], dtype=float)
        summary[name] = {
            "mean": float(np.nanmean(vals)),
            "lower": float(np.nanpercentile(vals, lower_q)),
            "upper": float(np.nanpercentile(vals, upper_q)),
        }

    return summary
