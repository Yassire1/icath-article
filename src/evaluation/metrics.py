"""
Evaluation metrics for TSFM benchmarking
Tasks: Forecasting, Anomaly Detection, RUL Prediction
"""

import numpy as np
from typing import Dict, Optional, Union
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve,
    auc
)
from scipy import stats


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error"""
    return mean_absolute_error(y_true.flatten(), y_pred.flatten())


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error"""
    return mean_squared_error(y_true.flatten(), y_pred.flatten())


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error"""
    return np.sqrt(mse(y_true, y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """Mean Absolute Percentage Error"""
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100


def crps(y_true: np.ndarray, y_pred_samples: np.ndarray) -> float:
    """Continuous Ranked Probability Score"""
    if y_pred_samples.ndim == 2:
        return mae(y_true, y_pred_samples)

    y_pred_sorted = np.sort(y_pred_samples, axis=1)
    n_samples = y_pred_samples.shape[1]

    crps_scores = []
    for i in range(len(y_true)):
        diff = y_pred_sorted[i] - y_true[i]
        crps_i = np.mean(np.abs(diff)) - 0.5 * np.mean(
            np.abs(y_pred_sorted[i][:, None] - y_pred_sorted[i][None, :])
        )
        crps_scores.append(crps_i)

    return np.mean(crps_scores)


def f1_at_threshold(y_true: np.ndarray, y_scores: np.ndarray, threshold: float = 0.5) -> float:
    """F1 score at given threshold"""
    y_pred = (y_scores > threshold).astype(int)
    return f1_score(y_true.flatten(), y_pred.flatten(), zero_division=0)


def auc_roc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Area Under ROC Curve"""
    try:
        return roc_auc_score(y_true.flatten(), y_scores.flatten())
    except ValueError:
        return 0.5


def auc_pr(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Area Under Precision-Recall Curve"""
    precision, recall, _ = precision_recall_curve(y_true.flatten(), y_scores.flatten())
    return auc(recall, precision)


def concordance_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Concordance Index (C-Index) for RUL prediction"""
    n = len(y_true)
    concordant = 0
    discordant = 0
    tied = 0

    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] != y_true[j]:
                if (y_true[i] < y_true[j] and y_pred[i] < y_pred[j]) or \
                   (y_true[i] > y_true[j] and y_pred[i] > y_pred[j]):
                    concordant += 1
                elif y_pred[i] == y_pred[j]:
                    tied += 1
                else:
                    discordant += 1

    total = concordant + discordant + tied
    if total == 0:
        return 0.5

    return (concordant + 0.5 * tied) / total


def rul_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """NASA scoring function for RUL - asymmetric penalty"""
    diff = y_pred - y_true
    scores = np.where(
        diff < 0,
        np.exp(-diff / 13) - 1,
        np.exp(diff / 10) - 1
    )
    return np.sum(scores)


def compute_forecasting_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_samples: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Compute all forecasting metrics"""
    metrics = {
        'mae': mae(y_true, y_pred),
        'mse': mse(y_true, y_pred),
        'rmse': rmse(y_true, y_pred),
        'mape': mape(y_true, y_pred)
    }

    if y_pred_samples is not None:
        metrics['crps'] = crps(y_true, y_pred_samples)

    return metrics


def compute_anomaly_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """Compute all anomaly detection metrics"""
    y_pred = (y_scores > threshold).astype(int)

    return {
        'f1': f1_score(y_true.flatten(), y_pred.flatten(), zero_division=0),
        'precision': precision_score(y_true.flatten(), y_pred.flatten(), zero_division=0),
        'recall': recall_score(y_true.flatten(), y_pred.flatten(), zero_division=0),
        'auc_roc': auc_roc(y_true, y_scores),
        'auc_pr': auc_pr(y_true, y_scores)
    }


def compute_rul_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute all RUL prediction metrics"""
    return {
        'mae': mae(y_true, y_pred),
        'rmse': rmse(y_true, y_pred),
        'c_index': concordance_index(y_true, y_pred),
        'rul_score': rul_score(y_true, y_pred)
    }


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: str = "forecasting",
    **kwargs
) -> Dict[str, float]:
    """Compute metrics based on task type"""
    if task == "forecasting":
        return compute_forecasting_metrics(y_true, y_pred, **kwargs)
    elif task == "anomaly_detection":
        return compute_anomaly_metrics(y_true, y_pred, **kwargs)
    elif task == "rul_prediction":
        return compute_rul_metrics(y_true, y_pred)
    else:
        raise ValueError(f"Unknown task: {task}")
