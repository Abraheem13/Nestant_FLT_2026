"""
metrics.py — Evaluation metrics.

Per-target regression metrics (R2, RMSE, MAE) and continual-learning
forgetting metrics (Backward Transfer, Forgetting Measure).

Continual learning metrics follow the standard definitions of
  - Lopez-Paz & Ranzato, "Gradient Episodic Memory for Continual Learning",
    NeurIPS 2017 (BWT).
  - Chaudhry et al., "Riemannian Walk for Incremental Learning", ECCV 2018
    (Forgetting Measure).

For a regression surrogate, we use the per-task R2 as the accuracy measure
(a_{ij} = R2 on task j after finishing training on task i). Higher R2 is
better, so the sign conventions match the classification versions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------
# Basic regression metrics
# ---------------------------------------------------------------------
def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean(axis=0)) ** 2).sum())
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(((y_true - y_pred) ** 2).mean()))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.abs(y_true - y_pred).mean())


@dataclass
class EvalRow:
    model: str
    protocol: str          # "joint" or "continual"
    task: str              # "1.5GHz", "2.4GHz", "5.8GHz", or "ALL"
    target: str            # target column name
    r2: float
    rmse: float
    mae: float


def per_task_per_target(
    model_name: str,
    protocol: str,
    y_true_std: np.ndarray,
    y_pred_std: np.ndarray,
    tasks: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    target_names: list[str],
    task_labels: dict[int, str],
) -> list[EvalRow]:
    """Evaluate a model per (task, target) pair on physical-unit values.

    y_true_std / y_pred_std are z-scored; we invert with y_mean, y_std so
    that RMSE and MAE are reported in physical units (dB, Hz, ...)."""
    y_true = y_true_std * y_std + y_mean
    y_pred = y_pred_std * y_std + y_mean

    rows: list[EvalRow] = []
    for t_id, t_lab in task_labels.items():
        mask = tasks == t_id
        if not mask.any():
            continue
        for j, tname in enumerate(target_names):
            rows.append(EvalRow(
                model=model_name, protocol=protocol, task=t_lab, target=tname,
                r2=r2_score(y_true[mask, j:j+1], y_pred[mask, j:j+1]),
                rmse=rmse(y_true[mask, j:j+1], y_pred[mask, j:j+1]),
                mae=mae(y_true[mask, j:j+1], y_pred[mask, j:j+1]),
            ))
    # ALL (pooled)
    for j, tname in enumerate(target_names):
        rows.append(EvalRow(
            model=model_name, protocol=protocol, task="ALL", target=tname,
            r2=r2_score(y_true[:, j:j+1], y_pred[:, j:j+1]),
            rmse=rmse(y_true[:, j:j+1], y_pred[:, j:j+1]),
            mae=mae(y_true[:, j:j+1], y_pred[:, j:j+1]),
        ))
    return rows


# ---------------------------------------------------------------------
# Continual-learning forgetting metrics
# ---------------------------------------------------------------------
def backward_transfer(A: np.ndarray) -> float:
    """BWT = (1/(T-1)) * sum_{i<T} (A[T-1, i] - A[i, i]).

    A is an upper-triangular/full TxT matrix of accuracies, where A[i, j]
    is the R2 on task j after finishing training on task i (only i >= j
    entries are populated; we use A[T-1, i] and A[i, i]).

    Positive BWT means learning later tasks *helped* earlier ones;
    negative BWT is forgetting."""
    T = A.shape[0]
    if T < 2:
        return 0.0
    diag = np.array([A[i, i] for i in range(T - 1)])
    final = A[T - 1, :T - 1]
    return float((final - diag).mean())


def forgetting_measure(A: np.ndarray) -> float:
    """FM = (1/(T-1)) * sum_{i<T-1} max_{t in [i, T-2]} (A[t, i] - A[T-1, i]).

    Per Chaudhry et al.: for each past task i, how much did performance on
    it drop from its best observed value to the final? Higher FM = more
    forgetting."""
    T = A.shape[0]
    if T < 2:
        return 0.0
    out = []
    for i in range(T - 1):
        # A[t, i] for t in [i, T-2]; then compare to A[T-1, i]
        best = max(A[t, i] for t in range(i, T - 1))
        out.append(best - A[T - 1, i])
    return float(np.mean(out))
