"""
data5.py -- 5-task continual benchmark.

Original 3-band split is augmented by splitting the 1.5 GHz and 5.8 GHz
bands by patch-length median (the dominant geometric scale), giving five
ordered tasks:

    T1: 1.5 GHz, length <  47 mm (small patches at 1.5 GHz)  [n=4800]
    T2: 1.5 GHz, length >= 47 mm (large patches at 1.5 GHz)  [n=5280]
    T3: 2.4 GHz, all lengths                                  [n=6158]
    T4: 5.8 GHz, length <  11 mm (small patches at 5.8 GHz)  [n=4799]
    T5: 5.8 GHz, length >= 11 mm (large patches at 5.8 GHz)  [n=5989]

This task ordering reflects a realistic design workflow that sweeps one
geometric corner at a time (small patches first within a band, then large)
and progresses across frequency bands. Each transition involves either a
geometric shift, a frequency shift, or both -- so any method that overfits
to a single distribution will forget across transitions.

For consistency with the 3-task experiments, the band feature in the input
remains the original target_freq_log (we don't add a sixth coordinate for
small/large -- that information is implicit in the geometry). This means
methods like NestAnt-CMS need a 5-band fast head table, but the slow prior
sees the same input features as before.

We define a NEW 'task_id' separate from the band id used for the per-band
fast heads. NestAnt-CMS in the 5-task setting routes to fast head
band_id = task_id (i.e. each task has its own fast head; bands sharing
target_freq still get separate heads).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data import (FEATURES_GEOM, FEATURE_BAND, TARGETS, SplitArrays,
                  load_raw, preprocess)


# Task definitions: (target_freq_hz, length_predicate, label)
# length_predicate(length) -> bool tells you whether a row belongs to this task.
TASK5_DEFS = [
    (1_500_000_000, lambda L: L <  47.0, "1.5GHz_S"),
    (1_500_000_000, lambda L: L >= 47.0, "1.5GHz_L"),
    (2_400_000_000, lambda L: True,       "2.4GHz"  ),
    (5_800_000_000, lambda L: L <  11.0, "5.8GHz_S"),
    (5_800_000_000, lambda L: L >= 11.0, "5.8GHz_L"),
]
TASK5_LABELS = {i: TASK5_DEFS[i][2] for i in range(5)}
N_TASKS5 = 5


def _row_to_task5(target_freq_hz, length):
    for i, (f, pred, _) in enumerate(TASK5_DEFS):
        if target_freq_hz == f and pred(length):
            return i
    return -1


def make_5task_splits(df: pd.DataFrame, test_frac: float = 0.20,
                       seed: int = 0) -> SplitArrays:
    """Stratified 80/20 split that preserves per-task proportions."""
    rng = np.random.default_rng(seed)
    df = df.reset_index(drop=True).copy()

    tasks_all = np.array([_row_to_task5(int(df.target_freq.iloc[i]),
                                         float(df.length.iloc[i]))
                          for i in range(len(df))])
    if (tasks_all < 0).any():
        raise ValueError("some rows did not match any 5-task bin")

    test_mask = np.zeros(len(df), dtype=bool)
    for t in range(N_TASKS5):
        idx = np.where(tasks_all == t)[0]
        rng.shuffle(idx)
        n_test = int(round(test_frac * len(idx)))
        test_mask[idx[:n_test]] = True

    feat_cols = FEATURES_GEOM + [FEATURE_BAND]
    X_all = df[feat_cols].to_numpy(dtype=float)
    y_all = df[TARGETS].to_numpy(dtype=float)

    X_train = X_all[~test_mask]; X_test = X_all[test_mask]
    y_train = y_all[~test_mask]; y_test = y_all[test_mask]
    task_train = tasks_all[~test_mask]; task_test = tasks_all[test_mask]

    x_mean = X_train.mean(0); x_std = X_train.std(0) + 1e-8
    y_mean = y_train.mean(0); y_std = y_train.std(0) + 1e-8

    X_train_std = (X_train - x_mean) / x_std
    X_test_std = (X_test - x_mean) / x_std
    y_train_std = (y_train - y_mean) / y_std
    y_test_std = (y_test - y_mean) / y_std

    return SplitArrays(
        X_train=X_train_std, y_train=y_train_std,
        X_test=X_test_std, y_test=y_test_std,
        task_train=task_train, task_test=task_test,
        x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std,
        feature_names=feat_cols, target_names=TARGETS,
    )


if __name__ == "__main__":
    df = preprocess(load_raw("/home/claude/project/nestant/data/colab_dataset_microstrip.csv"))
    s = make_5task_splits(df)
    print(f"train {s.X_train.shape}  test {s.X_test.shape}")
    for t in range(N_TASKS5):
        n_tr = (s.task_train == t).sum()
        n_te = (s.task_test == t).sum()
        print(f"  T{t+1} ({TASK5_LABELS[t]}): train={n_tr}, test={n_te}")
