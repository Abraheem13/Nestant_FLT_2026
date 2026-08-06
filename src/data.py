"""
data.py — Dataset loading, preprocessing, and task construction.

Dataset: AntennaCAT microstrip patch antenna CST simulations, released as
colab_dataset_microstrip.csv under GPL-2.0 with the URSI NRSM 2024 paper of
Linkous & Topsakal. 27,026 rows, 13 columns, three target frequencies
(1.5 / 2.4 / 5.8 GHz).

Inputs (geometric features, X):
    width         patch width [mm]
    length        patch length [mm]
    ground_plane  square ground plane side [mm]
    x_0           feed-point notch depth [mm]
    target_freq   design target frequency [Hz] -- this is the "task id" in
                  the continual-learning protocol; as a raw feature it is
                  log-scaled.

Note: `depth` is constant at 1.6 mm across the entire dataset so it carries
no information and is dropped.

Targets (y):
    s11_at_target       S11 at the design frequency [dB]   -- primary target
    max_total_gain      maximum total gain [dB]
    local_resonance_f   frequency of closest-to-target resonance [Hz]
                        (log-scaled for learning)

For continual learning the three target_freq classes define three sequential
"tasks" T1 (1.5 GHz) -> T2 (2.4 GHz) -> T3 (5.8 GHz). The order is fixed
(ascending frequency) and reflects a realistic lab workflow in which the
engineer acquires data for the lowest band first.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd


FEATURES_GEOM = ["width", "length", "ground_plane", "x_0"]
FEATURE_BAND = "target_freq_log"     # log10 of target_freq in Hz
TARGETS = ["s11_at_target", "max_total_gain", "local_resonance_f_log"]
TASK_FREQS_HZ = [1_500_000_000, 2_400_000_000, 5_800_000_000]


# ------------------------------------------------------------------
# Loading & preprocessing
# ------------------------------------------------------------------
def load_raw(csv_path: str | Path) -> pd.DataFrame:
    """Load the raw CSV, strip BOM and `$` prefixes from column names."""
    df = pd.read_csv(csv_path)
    df.columns = [c.replace("\ufeff", "").replace("$", "") for c in df.columns]
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns used as features or targets; drop redundant ones.

    - depth is constant (1.6 mm) -> dropped.
    - target_freq -> target_freq_log = log10(target_freq) so the model
      sees a smooth band coordinate rather than a categorical jump.
    - local_resonance_f -> local_resonance_f_log = log10(f).
    """
    df = df.copy()
    df["target_freq_log"] = np.log10(df["target_freq"].astype(float))
    df["local_resonance_f_log"] = np.log10(df["local_resonance_f"].astype(float))
    if "depth" in df.columns and df["depth"].nunique() <= 1:
        df = df.drop(columns=["depth"])
    return df


# ------------------------------------------------------------------
# Splits
# ------------------------------------------------------------------
@dataclass
class SplitArrays:
    """Container holding standardised numpy arrays and the scalers that
    produced them, so we can invert predictions back to physical units."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    task_train: np.ndarray   # integer task id per row (0,1,2) on train
    task_test: np.ndarray    # integer task id per row (0,1,2) on test
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    feature_names: list[str]
    target_names: list[str]

    def inv_y(self, y_std: np.ndarray) -> np.ndarray:
        return y_std * self.y_std + self.y_mean


def _freq_to_task_id(freqs_hz: np.ndarray) -> np.ndarray:
    """Map the Hz value of target_freq to {0,1,2}."""
    out = np.full(len(freqs_hz), -1, dtype=int)
    for i, f in enumerate(TASK_FREQS_HZ):
        out[freqs_hz == f] = i
    if (out < 0).any():
        raise ValueError("Found a target_freq not in the expected set")
    return out


def make_splits(
    df: pd.DataFrame,
    test_frac: float = 0.20,
    seed: int = 0,
) -> SplitArrays:
    """Stratified random split that preserves the per-band proportions.

    Features: FEATURES_GEOM + FEATURE_BAND.
    Targets:  TARGETS.
    Everything is z-scored using training-set statistics only.
    """
    rng = np.random.default_rng(seed)
    df = df.reset_index(drop=True)

    # Stratify by target_freq to keep band proportions in train and test.
    test_mask = np.zeros(len(df), dtype=bool)
    for f in TASK_FREQS_HZ:
        idx = np.where(df["target_freq"].values == f)[0]
        rng.shuffle(idx)
        n_test = int(round(test_frac * len(idx)))
        test_mask[idx[:n_test]] = True

    feat_cols = FEATURES_GEOM + [FEATURE_BAND]
    X_all = df[feat_cols].to_numpy(dtype=float)
    y_all = df[TARGETS].to_numpy(dtype=float)
    tasks_all = _freq_to_task_id(df["target_freq"].to_numpy())

    X_train = X_all[~test_mask]
    X_test = X_all[test_mask]
    y_train = y_all[~test_mask]
    y_test = y_all[test_mask]
    task_train = tasks_all[~test_mask]
    task_test = tasks_all[test_mask]

    x_mean = X_train.mean(0)
    x_std = X_train.std(0) + 1e-8
    y_mean = y_train.mean(0)
    y_std = y_train.std(0) + 1e-8

    X_train_std = (X_train - x_mean) / x_std
    X_test_std = (X_test - x_mean) / x_std
    y_train_std = (y_train - y_mean) / y_std
    y_test_std = (y_test - y_mean) / y_std

    return SplitArrays(
        X_train=X_train_std,
        y_train=y_train_std,
        X_test=X_test_std,
        y_test=y_test_std,
        task_train=task_train,
        task_test=task_test,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        feature_names=feat_cols,
        target_names=TARGETS,
    )


def task_slice(arr: np.ndarray, task_ids: np.ndarray, task: int) -> np.ndarray:
    """Return the rows belonging to a single task."""
    return arr[task_ids == task]
