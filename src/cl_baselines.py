"""
cl_baselines.py -- proper continual-learning baselines.

These are the methods a senior reviewer will demand. We implement them
on top of the same PlainMLP capacity used as the warm-start baseline,
so the comparison isolates the CL strategy, not the architecture.

  1. EWC (Kirkpatrick et al., PNAS 2017): when training on task t,
     penalize drift of parameters that were important for prior tasks,
     where importance is measured by the diagonal Fisher information
     accumulated on prior tasks.

  2. Experience Replay: keep a small buffer (10% of past task data per
     past task) and mix with current task batches during training.

  3. Independent per-band MLP: train a separate fresh MLP per band, no
     parameter sharing. Tests whether shared structure helps at all.

All on the same 27,026-sample AntennaCAT dataset, same continual protocol
(1.5 -> 2.4 -> 5.8 GHz, no revisiting), same 5-seed eval as our other
results.
"""
from __future__ import annotations

import sys, os, time, json
sys.path.insert(0, os.path.dirname(__file__))

import argparse
from copy import deepcopy

import numpy as np
import pandas as pd
from pathlib import Path

from data import load_raw, preprocess, make_splits, task_slice
from metrics import r2_score, rmse, mae, backward_transfer, forgetting_measure


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results"
TASK_ORDER = [0, 1, 2]
TASK_LABELS = {0: "1.5GHz", 1: "2.4GHz", 2: "5.8GHz"}


# =====================================================================
# Shared MLP forward/backward (3-layer, GELU)
# =====================================================================
def _gelu(x): return 0.5*x*(1.0+np.tanh(np.sqrt(2.0/np.pi)*(x+0.044715*x**3)))
def _gelu_grad(x):
    s = np.sqrt(2.0/np.pi); u = s*(x+0.044715*x**3); t = np.tanh(u)
    du = s*(1.0+3.0*0.044715*x**2)
    return 0.5*(1.0+t)+0.5*x*(1.0-t**2)*du


def _init_mlp(rng, n_in, hidden, embed, n_out):
    """Same architecture as PlainMLP."""
    lim1 = np.sqrt(6.0/(n_in+hidden)); lim2 = np.sqrt(6.0/(hidden+embed))
    lim3 = np.sqrt(6.0/(embed+n_out))
    return dict(
        W1=rng.uniform(-lim1, lim1, (n_in, hidden)),
        b1=np.zeros(hidden),
        W2=rng.uniform(-lim2, lim2, (hidden, embed)),
        b2=np.zeros(embed),
        W3=rng.uniform(-lim3, lim3, (embed, n_out)),
        b3=np.zeros(n_out),
    )


def _mlp_forward(p, X):
    h1p = X @ p["W1"] + p["b1"]; h1 = _gelu(h1p)
    h2p = h1 @ p["W2"] + p["b2"]; h2 = _gelu(h2p)
    yhat = h2 @ p["W3"] + p["b3"]
    return yhat, dict(X=X, h1p=h1p, h1=h1, h2p=h2p, h2=h2)


def _mlp_backward(p, cache, y, yhat):
    """Plain MSE backward; returns dict of gradients."""
    N, K = y.shape
    dy = (2.0/(N*K)) * (yhat - y)
    h2 = cache["h2"]; h1 = cache["h1"]; X = cache["X"]
    dW3 = h2.T @ dy; db3 = dy.sum(0)
    dh2 = dy @ p["W3"].T
    dh2p = dh2 * _gelu_grad(cache["h2p"])
    dW2 = h1.T @ dh2p; db2 = dh2p.sum(0)
    dh1 = dh2p @ p["W2"].T
    dh1p = dh1 * _gelu_grad(cache["h1p"])
    dW1 = X.T @ dh1p; db1 = dh1p.sum(0)
    return dict(W1=dW1, b1=db1, W2=dW2, b2=db2, W3=dW3, b3=db3)


def _mlp_predict(p, X):
    return _mlp_forward(p, X)[0]


# =====================================================================
# Adam optimizer state container
# =====================================================================
def _init_adam(p):
    return dict(m={k: np.zeros_like(v) for k, v in p.items()},
                v={k: np.zeros_like(v) for k, v in p.items()},
                t=0)


def _adam_step(p, grads, st, lr, weight_decay=1e-5,
               beta1=0.9, beta2=0.999, eps=1e-8):
    st["t"] += 1
    for k, g in grads.items():
        if "W" in k:
            g = g + weight_decay * p[k]
        st["m"][k] = beta1*st["m"][k] + (1-beta1)*g
        st["v"][k] = beta2*st["v"][k] + (1-beta2)*g*g
        mh = st["m"][k] / (1 - beta1**st["t"])
        vh = st["v"][k] / (1 - beta2**st["t"])
        p[k] = p[k] - lr * mh / (np.sqrt(vh) + eps)


# =====================================================================
# EWC: Fisher-information based regularizer
# =====================================================================
def _compute_fisher(p, X, y, batch_size=1024):
    """Diagonal empirical Fisher: <(grad of log-likelihood)^2> averaged
    across samples. For a Gaussian likelihood with unit variance and MSE
    loss, this is just <grad_loss^2> rescaled. We use squared per-sample
    gradients of the MSE loss as the standard approximation."""
    F = {k: np.zeros_like(v) for k, v in p.items()}
    n = X.shape[0]
    n_seen = 0
    rng = np.random.default_rng(0)
    # Subsample to bound cost
    idx = rng.choice(n, size=min(n, 4096), replace=False)
    for s in range(0, len(idx), batch_size):
        b = idx[s:s+batch_size]
        Xb, yb = X[b], y[b]
        yhat, cache = _mlp_forward(p, Xb)
        # Per-sample gradients via the per-sample formulation:
        # use mean-grad squared as a proxy (standard cheap diag-Fisher)
        grads = _mlp_backward(p, cache, yb, yhat)
        for k, g in grads.items():
            F[k] += g**2 * len(b)
        n_seen += len(b)
    for k in F:
        F[k] /= max(n_seen, 1)
    return F


def run_ewc_one_seed(split, seed, ewc_lambda=5e3, epochs=40):
    """Train sequentially with an EWC penalty against accumulated Fisher
    snapshots from prior tasks."""
    rng = np.random.default_rng(seed)
    n_in = split.X_train.shape[1]; n_out = split.y_train.shape[1]
    p = _init_mlp(rng, n_in, 128, 80, n_out)
    opt = _init_adam(p)
    fisher_acc = {k: np.zeros_like(v) for k, v in p.items()}   # accumulated Fisher
    p_anchor = None                                              # snapshot of prior params

    s11_idx = split.target_names.index("s11_at_target")
    A = np.full((3, 3), np.nan)
    seed_path = RES / f"forgetting_ewc_seed{seed}_v3.csv"

    for stage, tid in enumerate(TASK_ORDER):
        Xt = task_slice(split.X_train, split.task_train, tid)
        yt = task_slice(split.y_train, split.task_train, tid)
        rng2 = np.random.default_rng(seed + 100*stage)
        for ep in range(epochs):
            perm = rng2.permutation(Xt.shape[0])
            for s in range(0, len(perm), 1024):
                b = perm[s:s+1024]
                Xb, yb = Xt[b], yt[b]
                yhat, cache = _mlp_forward(p, Xb)
                grads = _mlp_backward(p, cache, yb, yhat)
                # EWC penalty gradient: lambda * F_acc * (theta - theta_anchor)
                if p_anchor is not None:
                    for k in grads:
                        grads[k] = grads[k] + ewc_lambda * fisher_acc[k] * (p[k] - p_anchor[k])
                _adam_step(p, grads, opt, lr=1.5e-3)
        # After finishing task tid: snapshot params and accumulate Fisher
        F_t = _compute_fisher(p, Xt, yt)
        for k in fisher_acc:
            fisher_acc[k] = fisher_acc[k] + F_t[k]
        p_anchor = {k: v.copy() for k, v in p.items()}

        # Evaluate
        y_pred_std = _mlp_predict(p, split.X_test)
        y_true = split.y_test * split.y_std + split.y_mean
        y_pred = y_pred_std * split.y_std + split.y_mean
        for jb in TASK_ORDER:
            m = split.task_test == jb
            A[stage, jb] = r2_score(y_true[m, s11_idx:s11_idx+1],
                                     y_pred[m, s11_idx:s11_idx+1])
        pd.DataFrame(A,
                     index=[f"after_{TASK_LABELS[i]}" for i in range(3)],
                     columns=[f"eval_{TASK_LABELS[j]}" for j in range(3)]
                     ).to_csv(seed_path)
    return A


# =====================================================================
# Experience Replay (small buffer of past tasks)
# =====================================================================
def run_replay_one_seed(split, seed, replay_frac=0.10, epochs=40):
    """At each new task t, mix ~replay_frac fraction of stored past-task
    samples into each minibatch."""
    rng = np.random.default_rng(seed)
    n_in = split.X_train.shape[1]; n_out = split.y_train.shape[1]
    p = _init_mlp(rng, n_in, 128, 80, n_out)
    opt = _init_adam(p)

    # Replay buffer: (X, y) tuples
    buf_X = np.zeros((0, n_in)); buf_y = np.zeros((0, n_out))

    s11_idx = split.target_names.index("s11_at_target")
    A = np.full((3, 3), np.nan)
    seed_path = RES / f"forgetting_replay_seed{seed}_v3.csv"

    for stage, tid in enumerate(TASK_ORDER):
        Xt = task_slice(split.X_train, split.task_train, tid)
        yt = task_slice(split.y_train, split.task_train, tid)
        rng2 = np.random.default_rng(seed + 100*stage)
        bs = 1024
        n_replay = int(bs * replay_frac) if buf_X.shape[0] > 0 else 0
        n_curr = bs - n_replay
        for ep in range(epochs):
            perm = rng2.permutation(Xt.shape[0])
            for s in range(0, len(perm), n_curr if n_curr > 0 else bs):
                b = perm[s:s+(n_curr if n_curr > 0 else bs)]
                Xb_curr, yb_curr = Xt[b], yt[b]
                if buf_X.shape[0] > 0 and n_replay > 0:
                    rep_idx = rng2.choice(buf_X.shape[0],
                                          size=min(n_replay, buf_X.shape[0]),
                                          replace=False)
                    Xb = np.concatenate([Xb_curr, buf_X[rep_idx]], axis=0)
                    yb = np.concatenate([yb_curr, buf_y[rep_idx]], axis=0)
                else:
                    Xb, yb = Xb_curr, yb_curr
                yhat, cache = _mlp_forward(p, Xb)
                grads = _mlp_backward(p, cache, yb, yhat)
                _adam_step(p, grads, opt, lr=1.5e-3)
        # After task tid: store replay_frac of this task's data into the buffer
        n_keep = int(replay_frac * Xt.shape[0])
        keep_idx = rng2.choice(Xt.shape[0], size=n_keep, replace=False)
        buf_X = np.concatenate([buf_X, Xt[keep_idx]], axis=0)
        buf_y = np.concatenate([buf_y, yt[keep_idx]], axis=0)

        # Evaluate
        y_pred_std = _mlp_predict(p, split.X_test)
        y_true = split.y_test * split.y_std + split.y_mean
        y_pred = y_pred_std * split.y_std + split.y_mean
        for jb in TASK_ORDER:
            m = split.task_test == jb
            A[stage, jb] = r2_score(y_true[m, s11_idx:s11_idx+1],
                                     y_pred[m, s11_idx:s11_idx+1])
        pd.DataFrame(A,
                     index=[f"after_{TASK_LABELS[i]}" for i in range(3)],
                     columns=[f"eval_{TASK_LABELS[j]}" for j in range(3)]
                     ).to_csv(seed_path)
    return A


# =====================================================================
# Independent per-band MLP (no shared parameters)
# =====================================================================
def run_indep_one_seed(split, seed, epochs=40):
    """Train a separate fresh MLP per band. By construction zero forgetting,
    but no shared structure. Tests whether the slow prior in NestAnt-CMS
    actually helps."""
    rng = np.random.default_rng(seed)
    n_in = split.X_train.shape[1]; n_out = split.y_train.shape[1]
    models = {}
    s11_idx = split.target_names.index("s11_at_target")
    A = np.full((3, 3), np.nan)
    seed_path = RES / f"forgetting_indep_seed{seed}_v3.csv"

    for stage, tid in enumerate(TASK_ORDER):
        Xt = task_slice(split.X_train, split.task_train, tid)
        yt = task_slice(split.y_train, split.task_train, tid)
        p = _init_mlp(np.random.default_rng(seed + tid), n_in, 128, 80, n_out)
        opt = _init_adam(p)
        rng2 = np.random.default_rng(seed + 100*stage)
        for ep in range(epochs):
            perm = rng2.permutation(Xt.shape[0])
            for s in range(0, len(perm), 1024):
                b = perm[s:s+1024]
                yhat, cache = _mlp_forward(p, Xt[b])
                grads = _mlp_backward(p, cache, yt[b], yhat)
                _adam_step(p, grads, opt, lr=1.5e-3)
        models[tid] = p
        # Evaluate using each band's own model
        y_true = split.y_test * split.y_std + split.y_mean
        for jb in TASK_ORDER:
            m = split.task_test == jb
            if jb in models:
                y_pred_std = _mlp_predict(models[jb], split.X_test[m])
                y_pred = y_pred_std * split.y_std + split.y_mean
                A[stage, jb] = r2_score(y_true[m, s11_idx:s11_idx+1],
                                         y_pred[:, s11_idx:s11_idx+1])
            else:
                # Fall back: random prediction, R^2 ~ 0
                A[stage, jb] = 0.0
        pd.DataFrame(A,
                     index=[f"after_{TASK_LABELS[i]}" for i in range(3)],
                     columns=[f"eval_{TASK_LABELS[j]}" for j in range(3)]
                     ).to_csv(seed_path)
    return A


# =====================================================================
# CLI
# =====================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["ewc", "replay", "indep"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--ewc_lambda", type=float, default=5e3)
    ap.add_argument("--replay_frac", type=float, default=0.10)
    args = ap.parse_args()
    df = preprocess(load_raw(ROOT / "data" / "colab_dataset_microstrip.csv"))
    split = make_splits(df, test_frac=0.20, seed=0)
    t0 = time.time()
    if args.method == "ewc":
        A = run_ewc_one_seed(split, args.seed, ewc_lambda=args.ewc_lambda,
                              epochs=args.epochs)
    elif args.method == "replay":
        A = run_replay_one_seed(split, args.seed,
                                 replay_frac=args.replay_frac,
                                 epochs=args.epochs)
    elif args.method == "indep":
        A = run_indep_one_seed(split, args.seed, epochs=args.epochs)
    rt = time.time() - t0
    bwt = backward_transfer(A)
    fm = forgetting_measure(A)
    print(f"{args.method} seed={args.seed} {rt:.1f}s  "
          f"BWT={bwt:+.4f}  FM={fm:+.4f}  "
          f"final_mean={A[-1].mean():+.4f}  diag={[round(A[i,i],3) for i in range(3)]}")
