"""
continual5_runner.py -- 5-task continual learning for all 6 methods.

Methods supported:
  * cms_freeze  : NestAnt-CMS with strict slow-prior freeze (our method)
  * indep       : Independent per-task MLP (zero-forgetting baseline)
  * replay      : Experience Replay (10% buffer of past tasks)
  * ewc         : Elastic Weight Consolidation (lambda=1e6)
  * xgb         : XGBoost refit-from-scratch on current task only
  * rf          : Random Forest refit-from-scratch on current task only

Per-seed forgetting matrix is 5x5 saved per stage so partial runs survive
timeouts. Final aggregation is done in continual5_aggregate.py.
"""
from __future__ import annotations

import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from pathlib import Path

from data import load_raw, preprocess, task_slice
from data5 import (make_5task_splits, TASK5_LABELS, N_TASKS5)
from metrics import r2_score, backward_transfer, forgetting_measure

from baselines import XGBWrapper, RFWrapper, HAS_XGB
from cl_baselines import (_init_mlp, _mlp_forward, _mlp_backward, _mlp_predict,
                           _init_adam, _adam_step, _compute_fisher)
from nestant_cms import (CMSConfig, init_cms, init_cms_opt,
                          cms_forward, cms_backward, cms_update, cms_predict)
from nestant import iterate_minibatches


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results"
TASK_ORDER5 = list(range(N_TASKS5))   # 0..4


def _r2_per_task(split, y_pred_std, target="s11_at_target"):
    idx = split.target_names.index(target)
    y_true = split.y_test * split.y_std + split.y_mean
    y_pred = y_pred_std * split.y_std + split.y_mean
    return {tid: r2_score(y_true[split.task_test == tid, idx:idx+1],
                          y_pred[split.task_test == tid, idx:idx+1])
            for tid in TASK_ORDER5}


def _save_matrix(name, seed, A):
    out = RES / f"forgetting5_{name}_seed{seed}_v3.csv"
    pd.DataFrame(A,
                 index=[f"after_{TASK5_LABELS[i]}" for i in range(N_TASKS5)],
                 columns=[f"eval_{TASK5_LABELS[j]}" for j in range(N_TASKS5)]
                 ).to_csv(out)


# =====================================================================
# CMS-freeze (our method) — 5-band per-band fast heads
# =====================================================================
def run_cms5(split, seed, epochs=30):
    cfg = CMSConfig(n_features=split.X_train.shape[1],
                    n_targets=split.y_train.shape[1],
                    n_bands=N_TASKS5,           # 5 fast heads
                    hidden=128, embed=64, band_embed=16, fast_hidden=80,
                    slow_lr=1e-3, fast_lr=3e-3,
                    epochs=epochs, batch_size=1024,
                    weight_decay=1e-5, seed=seed)
    p = init_cms(cfg); opt = init_cms_opt(p, N_TASKS5)
    rng = np.random.default_rng(seed)
    A = np.full((N_TASKS5, N_TASKS5), np.nan)
    for stage, tid in enumerate(TASK_ORDER5):
        Xt = task_slice(split.X_train, split.task_train, tid)
        yt = task_slice(split.y_train, split.task_train, tid)
        tt = task_slice(split.task_train, split.task_train, tid)
        slow_lr_here = 0.0 if stage > 0 else cfg.slow_lr
        for ep in range(cfg.epochs):
            for b in iterate_minibatches(rng, Xt.shape[0], cfg.batch_size):
                yhat, cache = cms_forward(p, Xt[b], tt[b])
                grads, _ = cms_backward(p, cache, yt[b], yhat,
                                         cfg.n_bands, cfg.weight_decay)
                cms_update(p, grads, opt, slow_lr_here, cfg.fast_lr,
                           cfg.weight_decay, active_bands=[tid])
        y_pred = cms_predict(p, split.X_test, split.task_test)
        per = _r2_per_task(split, y_pred)
        for jb in TASK_ORDER5:
            A[stage, jb] = per[jb]
        _save_matrix("cms_freeze", seed, A)
    return A


# =====================================================================
# Independent per-task MLP — train one fresh MLP per task
# =====================================================================
def run_indep5(split, seed, epochs=30):
    rng = np.random.default_rng(seed)
    n_in = split.X_train.shape[1]; n_out = split.y_train.shape[1]
    models = {}
    A = np.full((N_TASKS5, N_TASKS5), np.nan)
    for stage, tid in enumerate(TASK_ORDER5):
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
        y_true = split.y_test * split.y_std + split.y_mean
        s11_idx = split.target_names.index("s11_at_target")
        for jb in TASK_ORDER5:
            m = split.task_test == jb
            if jb in models:
                yp = _mlp_predict(models[jb], split.X_test[m])
                yp_phys = yp * split.y_std + split.y_mean
                A[stage, jb] = r2_score(y_true[m, s11_idx:s11_idx+1],
                                         yp_phys[:, s11_idx:s11_idx+1])
            else:
                A[stage, jb] = 0.0
        _save_matrix("indep", seed, A)
    return A


# =====================================================================
# Experience Replay (10% buffer)
# =====================================================================
def run_replay5(split, seed, epochs=30, replay_frac=0.10):
    rng = np.random.default_rng(seed)
    n_in = split.X_train.shape[1]; n_out = split.y_train.shape[1]
    p = _init_mlp(rng, n_in, 128, 80, n_out)
    opt = _init_adam(p)
    buf_X = np.zeros((0, n_in)); buf_y = np.zeros((0, n_out))
    A = np.full((N_TASKS5, N_TASKS5), np.nan)
    s11_idx = split.target_names.index("s11_at_target")
    for stage, tid in enumerate(TASK_ORDER5):
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
        n_keep = int(replay_frac * Xt.shape[0])
        keep_idx = rng2.choice(Xt.shape[0], size=n_keep, replace=False)
        buf_X = np.concatenate([buf_X, Xt[keep_idx]], axis=0)
        buf_y = np.concatenate([buf_y, yt[keep_idx]], axis=0)

        y_pred = _mlp_predict(p, split.X_test)
        per = _r2_per_task(split, y_pred)
        for jb in TASK_ORDER5:
            A[stage, jb] = per[jb]
        _save_matrix("replay", seed, A)
    return A


# =====================================================================
# EWC at lambda = 1e6 (best operating point from 3-task sweep)
# =====================================================================
def run_ewc5(split, seed, epochs=30, ewc_lambda=1e6):
    rng = np.random.default_rng(seed)
    n_in = split.X_train.shape[1]; n_out = split.y_train.shape[1]
    p = _init_mlp(rng, n_in, 128, 80, n_out)
    opt = _init_adam(p)
    fisher_acc = {k: np.zeros_like(v) for k, v in p.items()}
    p_anchor = None
    A = np.full((N_TASKS5, N_TASKS5), np.nan)
    for stage, tid in enumerate(TASK_ORDER5):
        Xt = task_slice(split.X_train, split.task_train, tid)
        yt = task_slice(split.y_train, split.task_train, tid)
        rng2 = np.random.default_rng(seed + 100*stage)
        for ep in range(epochs):
            perm = rng2.permutation(Xt.shape[0])
            for s in range(0, len(perm), 1024):
                b = perm[s:s+1024]
                yhat, cache = _mlp_forward(p, Xt[b])
                grads = _mlp_backward(p, cache, yt[b], yhat)
                if p_anchor is not None:
                    for k in grads:
                        grads[k] = grads[k] + ewc_lambda * fisher_acc[k] * (p[k] - p_anchor[k])
                _adam_step(p, grads, opt, lr=1.5e-3)
        F_t = _compute_fisher(p, Xt, yt)
        for k in fisher_acc:
            fisher_acc[k] = fisher_acc[k] + F_t[k]
        p_anchor = {k: v.copy() for k, v in p.items()}
        y_pred = _mlp_predict(p, split.X_test)
        per = _r2_per_task(split, y_pred)
        for jb in TASK_ORDER5:
            A[stage, jb] = per[jb]
        _save_matrix("ewc", seed, A)
    return A


# =====================================================================
# Tree methods refit-from-scratch
# =====================================================================
def run_xgb5(split, seed):
    A = np.full((N_TASKS5, N_TASKS5), np.nan)
    final = None
    for stage, tid in enumerate(TASK_ORDER5):
        Xt = task_slice(split.X_train, split.task_train, tid)
        yt = task_slice(split.y_train, split.task_train, tid)
        m = XGBWrapper(n_estimators=500, max_depth=7,
                        learning_rate=0.06, seed=seed).fit(Xt, yt)
        y_pred = m.predict(split.X_test)
        per = _r2_per_task(split, y_pred)
        for jb in TASK_ORDER5:
            A[stage, jb] = per[jb]
        _save_matrix("xgb", seed, A)
        final = m
    return A


def run_rf5(split, seed):
    A = np.full((N_TASKS5, N_TASKS5), np.nan)
    for stage, tid in enumerate(TASK_ORDER5):
        Xt = task_slice(split.X_train, split.task_train, tid)
        yt = task_slice(split.y_train, split.task_train, tid)
        m = RFWrapper(n_estimators=200, seed=seed).fit(Xt, yt)
        y_pred = m.predict(split.X_test)
        per = _r2_per_task(split, y_pred)
        for jb in TASK_ORDER5:
            A[stage, jb] = per[jb]
        _save_matrix("rf", seed, A)
    return A


# =====================================================================
# CLI
# =====================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["cms_freeze","indep","replay","ewc","xgb","rf"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    df = preprocess(load_raw(ROOT / "data" / "colab_dataset_microstrip.csv"))
    split = make_5task_splits(df, test_frac=0.20, seed=0)

    t0 = time.time()
    if args.method == "cms_freeze":
        A = run_cms5(split, args.seed, args.epochs)
    elif args.method == "indep":
        A = run_indep5(split, args.seed, args.epochs)
    elif args.method == "replay":
        A = run_replay5(split, args.seed, args.epochs)
    elif args.method == "ewc":
        A = run_ewc5(split, args.seed, args.epochs)
    elif args.method == "xgb":
        A = run_xgb5(split, args.seed)
    elif args.method == "rf":
        A = run_rf5(split, args.seed)
    rt = time.time() - t0
    bwt = backward_transfer(A)
    fm = forgetting_measure(A)
    print(f"{args.method} seed={args.seed} {rt:.1f}s  "
          f"BWT={bwt:+.4f}  FM={fm:+.4f}  "
          f"final={A[-1].mean():+.4f}  diag={[round(A[i,i],3) for i in range(N_TASKS5)]}")
