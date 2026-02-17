# src/evaluation.py
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import cross_validate, StratifiedKFold
from typing import Dict, Any, Tuple, Union
import warnings

# Suppress only specific sklearn warnings we know about
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


def compute_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: np.ndarray,
    y_prob: np.ndarray
) -> Dict[str, float]:
    """
    Compute AUC-PR, F1-score, and confusion matrix components for imbalanced classification.
    Handles edge cases (all negative, all positive, perfect prediction) safely.

    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted binary labels (0 or 1)
        y_prob: Predicted probabilities for positive class

    Returns:
        Dictionary with AUC-PR, F1, and TN/FP/FN/TP counts
    """
    metrics: Dict[str, float] = {}

    # AUC-PR (should work even with no positives)
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        metrics['auc_pr'] = average_precision_score(y_true, y_prob)
    except ValueError:
        metrics['auc_pr'] = 0.0  # no positives → no meaningful PR curve

    # F1-score – safe with zero_division
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)

    # Confusion matrix – force 2x2 shape even if one class is missing
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics['tn'] = float(tn)
    metrics['fp'] = float(fp)
    metrics['fn'] = float(fn)
    metrics['tp'] = float(tp)

    return metrics


def cross_validate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    scoring: list[str] = ['average_precision', 'f1']
) -> Dict[str, float]:
    """
    Perform Stratified K-Fold cross-validation and return mean/std of key metrics.

    Args:
        model: Scikit-learn compatible classifier
        X: Features
        y: Target
        cv: Number of folds
        scoring: Metrics to compute

    Returns:
        Dictionary with mean and std for each score
    """
    cv_results = cross_validate(
        model,
        X, y,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )

    return {
        'auc_pr_mean': float(cv_results['test_average_precision'].mean()),
        'auc_pr_std': float(cv_results['test_average_precision'].std()),
        'f1_mean': float(cv_results['test_f1'].mean()),
        'f1_std': float(cv_results['test_f1'].std()),
    }