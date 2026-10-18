"""
data.py -- loading, task definitions and geometry-grouped train/test splits.

Dataset: colab_dataset_microstrip.csv from the AntennaCAT project
(Linkous & Topsakal, USNC-URSI NRSM 2024; GPL-2.0). 27,026 rows, 13 columns,
three design frequencies (1.5 / 2.4 / 5.8 GHz). Substrate depth is constant
(1.6 mm) and is dropped.

Inputs  : width, length, ground_plane, x_0 [mm] and log10(target_freq [Hz]).
Targets : s11_at_target [dB], max_total_gain [dB], log10(local_resonance_f [Hz]).

Replicates. The release contains 6,848 distinct input geometries; every
geometry appears 2-5 times (mostly 3 or 5) with different recorded outputs.
A row-level random split would therefore place replicates of most test
geometries in the training set. All splits below are grouped by geometry:
every replicate of a geometry falls on the same side of the split.

Tasks. T=5 splits the 1.5 and 5.8 GHz bands at the patch-length median;
T=10 refines each T=5 task further by patch length, and T=20 splits every
T=10 task in two (all coarser boundaries are kept). Task order is ascending frequency, then ascending length.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "colab_dataset_microstrip.csv"

FEATURES = ["width", "length", "ground_plane", "x_0", "target_freq_log"]
TARGETS = ["s11_at_target", "max_total_gain", "local_resonance_f_log"]
GEOM_KEY = ["target_freq", "width", "length", "ground_plane", "x_0"]

# (frequency in Hz, inclusive lower length bound, exclusive upper bound, label)
TASKS = {
    5: [(1.5e9, 37.0, 47.0, "1.5S"), (1.5e9, 47.0, 99.0, "1.5L"),
        (2.4e9, 0.0, 99.0, "2.4"),
        (5.8e9, 7.0, 11.0, "5.8S"), (5.8e9, 11.0, 99.0, "5.8L")],
    10: [(1.5e9, 37.0, 42.0, "1.5a"), (1.5e9, 42.0, 47.0, "1.5b"),
         (1.5e9, 47.0, 52.0, "1.5c"), (1.5e9, 52.0, 99.0, "1.5d"),
         (2.4e9, 0.0, 30.0, "2.4a"), (2.4e9, 30.0, 99.0, "2.4b"),
         (5.8e9, 7.0, 9.0, "5.8a"), (5.8e9, 9.0, 11.0, "5.8b"),
         (5.8e9, 11.0, 13.5, "5.8c"), (5.8e9, 13.5, 99.0, "5.8d")],
    # every T=10 task split in two by patch length
    20: [(1.5e9, 37.0, 39.0, "1.5-1"), (1.5e9, 39.0, 42.0, "1.5-2"),
         (1.5e9, 42.0, 44.0, "1.5-3"), (1.5e9, 44.0, 47.0, "1.5-4"),
         (1.5e9, 47.0, 49.0, "1.5-5"), (1.5e9, 49.0, 52.0, "1.5-6"),
         (1.5e9, 52.0, 55.0, "1.5-7"), (1.5e9, 55.0, 99.0, "1.5-8"),
         (2.4e9, 0.0, 27.0, "2.4-1"), (2.4e9, 27.0, 30.0, "2.4-2"),
         (2.4e9, 30.0, 33.0, "2.4-3"), (2.4e9, 33.0, 99.0, "2.4-4"),
         (5.8e9, 7.0, 8.0, "5.8-1"), (5.8e9, 8.0, 9.0, "5.8-2"),
         (5.8e9, 9.0, 10.0, "5.8-3"), (5.8e9, 10.0, 11.0, "5.8-4"),
         (5.8e9, 11.0, 12.0, "5.8-5"), (5.8e9, 12.0, 13.5, "5.8-6"),
         (5.8e9, 13.5, 14.5, "5.8-7"), (5.8e9, 14.5, 99.0, "5.8-8")],
}


def load(csv_path: str | Path = CSV) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.replace("﻿", "").replace("$", "") for c in df.columns]
    assert df["depth"].nunique() == 1
    df = df.drop(columns=["depth"])
    df["target_freq_log"] = np.log10(df["target_freq"].astype(float))
    df["local_resonance_f_log"] = np.log10(df["local_resonance_f"].astype(float))
    return df


def task_ids(df: pd.DataFrame, n_tasks: int) -> np.ndarray:
    out = np.full(len(df), -1, dtype=int)
    f = df["target_freq"].to_numpy(dtype=float)
    L = df["length"].to_numpy(dtype=float)
    for t, (freq, lo, hi, _) in enumerate(TASKS[n_tasks]):
        out[(f == freq) & (L >= lo) & (L < hi)] = t
    assert (out >= 0).all(), "a row matched no task"
    return out


def labels(n_tasks: int) -> list[str]:
    return [t[3] for t in TASKS[n_tasks]]


@dataclass
class Split:
    X_train: np.ndarray
    y_train: np.ndarray
    t_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    t_test: np.ndarray
    g_test: np.ndarray          # geometry id of every test row
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    n_tasks: int

    def y_phys(self, y_std_units: np.ndarray) -> np.ndarray:
        return y_std_units * self.y_std + self.y_mean


def make_split(df: pd.DataFrame, n_tasks: int, seed: int,
               test_frac: float = 0.20) -> Split:
    """Geometry-grouped 80/20 split, stratified by task, drawn with `seed`.

    Within each task the distinct geometries are shuffled and 20 % of them
    (rounded) go to the test side together with all their replicate rows.
    Inputs and targets are z-scored with statistics of the pooled training
    partition."""
    tids = task_ids(df, n_tasks)
    gid = df.groupby(GEOM_KEY, sort=True).ngroup().to_numpy()
    rng = np.random.default_rng([seed, 7])
    test = np.zeros(len(df), dtype=bool)
    for t in range(n_tasks):
        geoms = np.unique(gid[tids == t])
        rng.shuffle(geoms)
        n_test = int(round(test_frac * len(geoms)))
        test[np.isin(gid, geoms[:n_test])] = True
    X = df[FEATURES].to_numpy(dtype=float)
    y = df[TARGETS].to_numpy(dtype=float)
    xm, xs = X[~test].mean(0), X[~test].std(0)
    ym, ys = y[~test].mean(0), y[~test].std(0)
    return Split(
        X_train=(X[~test] - xm) / xs, y_train=(y[~test] - ym) / ys, t_train=tids[~test],
        X_test=(X[test] - xm) / xs, y_test=(y[test] - ym) / ys, t_test=tids[test],
        g_test=gid[test], x_mean=xm, x_std=xs, y_mean=ym, y_std=ys, n_tasks=n_tasks)


if __name__ == "__main__":
    df = load()
    gid = df.groupby(GEOM_KEY).ngroup()
    print(f"rows {len(df)}, distinct geometries {gid.nunique()}")
    print("replicates per geometry:", gid.value_counts().value_counts().sort_index().to_dict())
    for T in (5, 10, 20):
        s = make_split(df, T, seed=0)
        print(f"T={T}")
        for t, lab in enumerate(labels(T)):
            print(f"  {lab:>5}: train rows {(s.t_train == t).sum():5d}  test rows {(s.t_test == t).sum():5d}")
