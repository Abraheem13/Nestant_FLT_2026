"""
progressive_nets.py -- Progressive Networks (Rusu et al. 2016) for the
5-task continual benchmark.

A separate "column" (2-layer MLP) is allocated per task. Column t reads
the input plus lateral connections from the (frozen) embed-layer activations
of all previous columns 0..t-1. Lateral connections are linear projections
gated by tanh, in the spirit of the original paper. When training task t,
columns 0..t-1 are frozen; only column t and its lateral projections U_{0..t-1}
update. Predictions for task k use column k (frozen after task k finished).
By construction zero forgetting on prior tasks.
"""
from __future__ import annotations

import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from pathlib import Path

from data import load_raw, preprocess, task_slice
from data5 import make_5task_splits, TASK5_LABELS, N_TASKS5
from metrics import r2_score, backward_transfer, forgetting_measure
from cl_baselines import _init_mlp, _gelu, _gelu_grad


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results"
TASK_ORDER5 = list(range(N_TASKS5))


def _init_column(rng, n_in, hidden, embed, n_out, n_prev):
    p = _init_mlp(rng, n_in, hidden, embed, n_out)
    p["U"] = []
    for _ in range(n_prev):
        lim = np.sqrt(6.0 / (embed + embed))
        p["U"].append(rng.uniform(-lim, lim, (embed, embed)))
    return p


def _column_forward(col, X, prev_h2_list):
    h1p = X @ col["W1"] + col["b1"]; h1 = _gelu(h1p)
    h2p = h1 @ col["W2"] + col["b2"]
    if prev_h2_list:
        for U, prev_h2 in zip(col["U"], prev_h2_list):
            h2p = h2p + np.tanh(prev_h2) @ U
    h2 = _gelu(h2p)
    yhat = h2 @ col["W3"] + col["b3"]
    cache = dict(X=X, h1p=h1p, h1=h1, h2p=h2p, h2=h2,
                 prev_h2_list=prev_h2_list)
    return yhat, cache


def _column_backward(col, cache, y, yhat):
    N, K = y.shape
    dy = (2.0/(N*K)) * (yhat - y)
    h2 = cache["h2"]; h1 = cache["h1"]; X = cache["X"]
    dW3 = h2.T @ dy; db3 = dy.sum(0)
    dh2 = dy @ col["W3"].T
    dh2p = dh2 * _gelu_grad(cache["h2p"])
    dU = []
    for prev_h2 in cache["prev_h2_list"]:
        dU.append(np.tanh(prev_h2).T @ dh2p)
    dW2 = h1.T @ dh2p; db2 = dh2p.sum(0)
    dh1 = dh2p @ col["W2"].T
    dh1p = dh1 * _gelu_grad(cache["h1p"])
    dW1 = X.T @ dh1p; db1 = dh1p.sum(0)
    return dict(W1=dW1, b1=db1, W2=dW2, b2=db2, W3=dW3, b3=db3, U=dU)


def _eval_prev_h2_for(prev_columns, X):
    so_far = []
    for c in prev_columns:
        h1 = _gelu(X @ c["W1"] + c["b1"])
        h2p = h1 @ c["W2"] + c["b2"]
        for U, prev_h2 in zip(c["U"], so_far):
            h2p = h2p + np.tanh(prev_h2) @ U
        h2 = _gelu(h2p)
        so_far.append(h2)
    return so_far


def _init_adam_pn(col):
    m = {}; v = {}
    for k, val in col.items():
        if k == "U":
            m[k] = [np.zeros_like(u) for u in val]
            v[k] = [np.zeros_like(u) for u in val]
        else:
            m[k] = np.zeros_like(val); v[k] = np.zeros_like(val)
    return dict(t=0, m=m, v=v)


def _adam_step_pn(col, grads, st, lr=1.5e-3, weight_decay=1e-5,
                   beta1=0.9, beta2=0.999, eps=1e-8):
    st["t"] += 1
    for k, g in grads.items():
        if k == "U":
            for i, gi in enumerate(g):
                st["m"][k][i] = beta1*st["m"][k][i] + (1-beta1)*gi
                st["v"][k][i] = beta2*st["v"][k][i] + (1-beta2)*gi*gi
                mh = st["m"][k][i] / (1 - beta1**st["t"])
                vh = st["v"][k][i] / (1 - beta2**st["t"])
                col[k][i] = col[k][i] - lr * mh / (np.sqrt(vh) + eps)
        else:
            if "W" in k: g = g + weight_decay * col[k]
            st["m"][k] = beta1*st["m"][k] + (1-beta1)*g
            st["v"][k] = beta2*st["v"][k] + (1-beta2)*g*g
            mh = st["m"][k] / (1 - beta1**st["t"])
            vh = st["v"][k] / (1 - beta2**st["t"])
            col[k] = col[k] - lr * mh / (np.sqrt(vh) + eps)


def run_prognet5(split, seed, epochs=30):
    n_in = split.X_train.shape[1]; n_out = split.y_train.shape[1]
    columns = []
    A = np.full((N_TASKS5, N_TASKS5), np.nan)
    s11_idx = split.target_names.index("s11_at_target")
    seed_path = RES / f"forgetting5_prognet_seed{seed}_v3.csv"
    for stage, tid in enumerate(TASK_ORDER5):
        col = _init_column(np.random.default_rng(seed + 1000*tid),
                            n_in, 128, 80, n_out, n_prev=stage)
        opt = _init_adam_pn(col)
        Xt = task_slice(split.X_train, split.task_train, tid)
        yt = task_slice(split.y_train, split.task_train, tid)
        prev_h2_full = _eval_prev_h2_for(columns, Xt)
        rng2 = np.random.default_rng(seed + 100*stage)
        for ep in range(epochs):
            perm = rng2.permutation(Xt.shape[0])
            for s in range(0, len(perm), 1024):
                b = perm[s:s+1024]
                Xb, yb = Xt[b], yt[b]
                prev_h2_b = [h[b] for h in prev_h2_full]
                yhat, cache = _column_forward(col, Xb, prev_h2_b)
                grads = _column_backward(col, cache, yb, yhat)
                _adam_step_pn(col, grads, opt, lr=1.5e-3)
        columns.append(col)
        y_true = split.y_test * split.y_std + split.y_mean
        for jb in TASK_ORDER5:
            mask = split.task_test == jb
            if jb >= len(columns):
                A[stage, jb] = 0.0
                continue
            X_eval = split.X_test[mask]
            prev_h2_eval = _eval_prev_h2_for(columns[:jb], X_eval)
            yhat, _ = _column_forward(columns[jb], X_eval, prev_h2_eval)
            yp_phys = yhat * split.y_std + split.y_mean
            A[stage, jb] = r2_score(y_true[mask, s11_idx:s11_idx+1],
                                     yp_phys[:, s11_idx:s11_idx+1])
        pd.DataFrame(A,
                     index=[f"after_{TASK5_LABELS[i]}" for i in range(N_TASKS5)],
                     columns=[f"eval_{TASK5_LABELS[j]}" for j in range(N_TASKS5)]
                     ).to_csv(seed_path)
    return A


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    df = preprocess(load_raw(ROOT / "data" / "colab_dataset_microstrip.csv"))
    split = make_5task_splits(df, test_frac=0.20, seed=0)
    t0 = time.time()
    A = run_prognet5(split, args.seed, args.epochs)
    rt = time.time() - t0
    bwt = backward_transfer(A); fm = forgetting_measure(A)
    print(f"prognet seed={args.seed} {rt:.1f}s  "
          f"BWT={bwt:+.4f}  FM={fm:+.4f}  "
          f"final={A[-1].mean():+.4f}  diag={[round(A[i,i],3) for i in range(N_TASKS5)]}")
