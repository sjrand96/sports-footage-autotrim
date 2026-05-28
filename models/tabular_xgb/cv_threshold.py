"""Clip-grouped CV and probability threshold tuning for frame classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import ClassifierMixin, clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold


@dataclass(frozen=True)
class ThresholdTuneResult:
    """Outcome of OOF threshold search on train clips."""

    best_threshold: float
    best_fbeta_oof: float
    f_beta: float
    n_cv_folds: int
    threshold_grid: tuple[float, ...]
    fbeta_on_grid: tuple[float, ...]


def apply_threshold(proba: np.ndarray, threshold: float) -> np.ndarray:
    """Binary predictions from P(class=1) and a decision threshold."""
    return (np.asarray(proba, dtype=np.float64) >= threshold).astype(np.int32)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    beta: float,
) -> dict[str, float | list[list[int]]]:
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        f"f{beta:g}": float(fbeta_score(y_true, y_pred, beta=beta, zero_division=0)),
        "confusion_matrix": {
            "labels": [0, 1],
            "matrix": cm.astype(int).tolist(),
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
    }


def sweep_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    beta: float = 2.0,
    n_threshold_steps: int = 91,
) -> ThresholdTuneResult:
    """Pick threshold maximizing F-beta on fixed (y, proba) pairs."""
    y_true = np.asarray(y_true, dtype=np.int32)
    proba = np.asarray(proba, dtype=np.float64)
    if len(y_true) != len(proba):
        raise ValueError("y_true and proba length mismatch")

    thresholds = np.linspace(0.05, 0.95, n_threshold_steps)
    scores: list[float] = []
    for t in thresholds:
        pred = apply_threshold(proba, float(t))
        scores.append(float(fbeta_score(y_true, pred, beta=beta, zero_division=0)))

    best_idx = int(np.argmax(scores))
    return ThresholdTuneResult(
        best_threshold=float(thresholds[best_idx]),
        best_fbeta_oof=float(scores[best_idx]),
        f_beta=beta,
        n_cv_folds=0,
        threshold_grid=tuple(float(t) for t in thresholds),
        fbeta_on_grid=tuple(scores),
    )


def out_of_fold_proba_clip_cv(
    estimator: ClassifierMixin,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, int]:
    """Out-of-fold P(class=1) with StratifiedGroupKFold (groups = clip ids)."""
    X = np.asarray(X)
    y = np.asarray(y, dtype=np.int32)
    groups = np.asarray(groups)
    if len(X) != len(y) or len(y) != len(groups):
        raise ValueError("X, y, and groups must have the same length")

    unique_groups = np.unique(groups)
    n_splits = min(n_splits, len(unique_groups))
    if n_splits < 2:
        raise ValueError(f"need at least 2 clips for CV, got {len(unique_groups)}")

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_proba = np.zeros(len(y), dtype=np.float64)

    for train_idx, val_idx in cv.split(X, y, groups=groups):
        fold_est = clone(estimator)
        fold_est.fit(X[train_idx], y[train_idx])
        oof_proba[val_idx] = fold_est.predict_proba(X[val_idx])[:, 1]

    return oof_proba, n_splits


def tune_threshold_clip_cv(
    estimator: ClassifierMixin,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    beta: float = 2.0,
    n_splits: int = 5,
    random_state: int = 42,
    n_threshold_steps: int = 91,
) -> ThresholdTuneResult:
    """Tune decision threshold on train clips via grouped OOF probabilities."""
    oof_proba, used_folds = out_of_fold_proba_clip_cv(
        estimator,
        X,
        y,
        groups,
        n_splits=n_splits,
        random_state=random_state,
    )
    result = sweep_threshold(y, oof_proba, beta=beta, n_threshold_steps=n_threshold_steps)
    return ThresholdTuneResult(
        best_threshold=result.best_threshold,
        best_fbeta_oof=result.best_fbeta_oof,
        f_beta=result.f_beta,
        n_cv_folds=used_folds,
        threshold_grid=result.threshold_grid,
        fbeta_on_grid=result.fbeta_on_grid,
    )


def threshold_tune_to_dict(result: ThresholdTuneResult) -> dict[str, Any]:
    """JSON-serializable summary (full sweep grid for W&B / offline plots)."""
    return {
        "best_threshold": result.best_threshold,
        "oof_fbeta": result.best_fbeta_oof,
        "f_beta": result.f_beta,
        "n_cv_folds": result.n_cv_folds,
        "threshold_sweep": [
            {"threshold": t, "fbeta": s}
            for t, s in zip(result.threshold_grid, result.fbeta_on_grid, strict=True)
        ],
    }
