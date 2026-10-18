"""
bench.py -- one continual-learning run: one method, one seed, one task sequence.

After every training stage i the model is evaluated on the test rows of every
task j, giving T x T matrices for six metrics:
    s11_r2, s11_mae [dB], s11_mae_matched [dB, rows with true S11 < -10 dB;
    NaN for tasks without such rows], gain_r2, gain_mae [dB],
    fr_r2 (on log10 f), fr_mape [%]
Entries for tasks not yet trained (j > i) are recorded as well; for the
independent and progressive methods, which have no model for an unseen task,
they are NaN. BWT and the other summaries only use j <= i.

Usage:
    python3 bench.py --method cms --seed 0 --tasks 5 --epochs 300
Writes results/runs/T{tasks}_e{epochs}[_extra]/{method}_seed{seed}.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from data import load, make_split, labels
import models as M

RUNS = Path(__file__).resolve().parent.parent / "results" / "runs"
BS = 1024
LR_MLP = 1.5e-3
LR_CMS_SLOW, LR_CMS_FAST = 1e-3, 3e-3


# ------------------------------------------------------------------ metrics
def metrics(split, yhat_std, mask):
    yt = split.y_phys(split.y_test[mask]); yp = split.y_phys(yhat_std)

    def r2(a, b):
        return float(1 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum())
    f_t, f_p = 10 ** yt[:, 2], 10 ** yp[:, 2]
    matched = yt[:, 0] < -10.0          # designs matched at the design frequency
    mm = float(np.abs(yt[matched, 0] - yp[matched, 0]).mean()) if matched.any() else float("nan")
    return dict(s11_r2=r2(yt[:, 0], yp[:, 0]), s11_mae=float(np.abs(yt[:, 0] - yp[:, 0]).mean()),
                s11_mae_matched=mm,
                gain_r2=r2(yt[:, 1], yp[:, 1]), gain_mae=float(np.abs(yt[:, 1] - yp[:, 1]).mean()),
                fr_r2=r2(yt[:, 2], yp[:, 2]), fr_mape=float(100 * (np.abs(f_p - f_t) / f_t).mean()))


KEYS = ["s11_r2", "s11_mae", "s11_mae_matched", "gain_r2", "gain_mae", "fr_r2", "fr_mape"]


class Recorder:
    def __init__(self, split):
        self.s = split; T = split.n_tasks
        self.A = {k: np.full((T, T), np.nan) for k in KEYS}
        self.pooled_matched = [float("nan")] * T     # row-weighted, all trained tasks
        s11 = split.y_phys(split.y_test)[:, 0]
        self.n_matched = [int((s11[split.t_test == j] < -10.0).sum()) for j in range(T)]

    def record(self, stage, predict):
        """predict(X, task) -> standardised predictions, or None if no model."""
        errs = []
        for j in range(self.s.n_tasks):
            m = self.s.t_test == j
            yhat = predict(self.s.X_test[m], j)
            if yhat is None:
                continue
            for k, v in metrics(self.s, yhat, m).items():
                self.A[k][stage, j] = v
            yt = self.s.y_phys(self.s.y_test[m])[:, 0]; yp = self.s.y_phys(yhat)[:, 0]
            errs.append(np.abs(yt - yp)[yt < -10.0])
        e = np.concatenate(errs) if errs else np.zeros(0)
        if len(e):
            self.pooled_matched[stage] = float(e.mean())


def batches(rng, n, bs=BS):
    idx = rng.permutation(n)
    for s in range(0, n, bs):
        yield idx[s:s + bs]


def task_data(split, t):
    m = split.t_train == t
    return split.X_train[m], split.y_train[m]


# ------------------------------------------------------------------ MLP family
def run_mlp_family(split, seed, epochs, order, kind, lam=0.0, replay_frac=0.0):
    """kind in {finetune, ewc, replay}; one network trained through all tasks."""
    rec = Recorder(split)
    p = M.mlp_init(np.random.default_rng([seed, 0]))
    opt = M.Adam(p, LR_MLP)
    keys = list(p)
    F_acc = {k: np.zeros_like(v) for k, v in p.items()}
    anchor = None
    bufX = np.zeros((0, 5)); bufy = np.zeros((0, 3))
    for stage, t in enumerate(order):
        X, y = task_data(split, t)
        rng = np.random.default_rng([seed, 100 + stage])
        n_rep = int(round(BS * replay_frac)) if (kind == "replay" and len(bufX)) else 0
        bs_cur = BS - n_rep
        for _ in range(epochs):
            for b in batches(rng, len(X), bs_cur):
                Xb, yb = X[b], y[b]
                if n_rep:
                    r = rng.choice(len(bufX), size=min(n_rep, len(bufX)), replace=False)
                    Xb = np.concatenate([Xb, bufX[r]]); yb = np.concatenate([yb, bufy[r]])
                yhat, c = M.mlp_forward(p, Xb)
                g = M.mlp_backward(p, c, yb, yhat)
                if kind == "ewc" and anchor is not None:
                    for k in g:
                        g[k] = g[k] + lam * F_acc[k] * (p[k] - anchor[k])
                M.add_l2(p, g)
                opt.step(p, g, keys)
        if kind == "ewc":
            Ft = M.mlp_fisher(p, X, y)
            for k in F_acc:
                F_acc[k] += Ft[k]
            anchor = {k: v.copy() for k, v in p.items()}
        if kind == "replay":
            keep = rng.choice(len(X), size=int(round(replay_frac * len(X))), replace=False)
            bufX = np.concatenate([bufX, X[keep]]); bufy = np.concatenate([bufy, y[keep]])
        rec.record(stage, lambda Xe, j: M.mlp_forward(p, Xe)[0])
    extra = dict(params=M.n_params(p), buffer_rows=int(len(bufX)))
    return rec, extra


def run_indep(split, seed, epochs, order):
    rec = Recorder(split); nets = {}
    for stage, t in enumerate(order):
        X, y = task_data(split, t)
        p = M.mlp_init(np.random.default_rng([seed, stage]))
        opt = M.Adam(p, LR_MLP); keys = list(p)
        rng = np.random.default_rng([seed, 100 + stage])
        for _ in range(epochs):
            for b in batches(rng, len(X)):
                yhat, c = M.mlp_forward(p, X[b])
                g = M.mlp_backward(p, c, y[b], yhat); M.add_l2(p, g)
                opt.step(p, g, keys)
        nets[t] = p
        rec.record(stage, lambda Xe, j: M.mlp_forward(nets[j], Xe)[0] if j in nets else None)
    return rec, dict(params=int(sum(M.n_params(p) for p in nets.values())))


def run_prognet(split, seed, epochs, order):
    rec = Recorder(split); cols = []; col_of = {}
    for stage, t in enumerate(order):
        X, y = task_data(split, t)
        col = M.prog_column_init(np.random.default_rng([seed, stage]), n_prev=len(cols))
        opt = M.Adam(col, LR_MLP); keys = list(col)
        prev = M.prog_hidden(cols, X)
        rng = np.random.default_rng([seed, 100 + stage])
        for _ in range(epochs):
            for b in batches(rng, len(X)):
                yhat, c = M.prog_forward(col, X[b], [h[b] for h in prev])
                g = M.prog_backward(col, c, y[b], yhat); M.add_l2(col, g)
                opt.step(col, g, keys)
        cols.append(col); col_of[t] = len(cols) - 1

        def pred(Xe, j):
            if j not in col_of:
                return None
            k = col_of[j]
            return M.prog_forward(cols[k], Xe, M.prog_hidden(cols[:k], Xe))[0]
        rec.record(stage, pred)
    return rec, dict(params=int(sum(M.n_params(c) for c in cols)))


def run_cms(split, seed, epochs, order):
    T = split.n_tasks
    rec = Recorder(split)
    p = M.cms_init(np.random.default_rng([seed, 0]), T)
    opt = M.Adam(p, LR_CMS_FAST)
    enc = ["W1", "b1", "W2", "b2"]
    for stage, t in enumerate(order):
        X, y = task_data(split, t)
        rng = np.random.default_rng([seed, 100 + stage])
        head = [f"E{t}", f"W3_{t}", f"b3_{t}", f"W4_{t}", f"b4_{t}"]
        for _ in range(epochs):
            for b in batches(rng, len(X)):
                yhat, c = M.cms_forward(p, X[b], t)
                g = M.cms_backward(p, c, y[b], yhat, t); M.add_l2(p, g)
                if stage == 0:
                    opt.step(p, g, enc, lr=LR_CMS_SLOW)
                opt.step(p, g, head, lr=LR_CMS_FAST)
        rec.record(stage, lambda Xe, j: M.cms_forward(p, Xe, j)[0])
    return rec, dict(params=M.cms_task_params(p, T))


# ------------------------------------------------------------------ trees
def _tree(kind, seed):
    if kind == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=1)
    import xgboost as xgb
    from sklearn.multioutput import MultiOutputRegressor
    return MultiOutputRegressor(xgb.XGBRegressor(
        n_estimators=500, max_depth=7, learning_rate=0.06, tree_method="hist",
        random_state=seed, n_jobs=1, verbosity=0))


def run_tree(split, seed, order, kind):
    rec = Recorder(split)
    for stage, t in enumerate(order):
        X, y = task_data(split, t)
        m = _tree(kind, seed).fit(X, y)
        rec.record(stage, lambda Xe, j: m.predict(Xe))
    return rec, dict(params=None)


# ------------------------------------------------------------------ pooled reference
def run_joint(split, seed, epochs, kind):
    """Non-sequential reference: one model trained on all tasks at once.
    Recorded in row 0 of the matrices."""
    rec = Recorder(split)
    X, y = split.X_train, split.y_train
    if kind == "joint_xgb":
        m = _tree("xgb", seed).fit(X, y)
        rec.record(0, lambda Xe, j: m.predict(Xe))
        return rec, dict(params=None)
    p = M.mlp_init(np.random.default_rng([seed, 0])); opt = M.Adam(p, LR_MLP); keys = list(p)
    rng = np.random.default_rng([seed, 100])
    for _ in range(epochs):
        for b in batches(rng, len(X)):
            yhat, c = M.mlp_forward(p, X[b])
            g = M.mlp_backward(p, c, y[b], yhat); M.add_l2(p, g); opt.step(p, g, keys)
    rec.record(0, lambda Xe, j: M.mlp_forward(p, Xe)[0])
    return rec, dict(params=M.n_params(p))


# ------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["finetune", "ewc", "replay", "indep", "prognet", "cms",
                             "rf", "xgb", "joint_mlp", "joint_xgb"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--tasks", type=int, default=5, choices=[5, 10, 20])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--replay_frac", type=float, default=0.10)
    ap.add_argument("--order", type=str, default="", help="comma-separated task order")
    ap.add_argument("--tag", type=str, default="")
    a = ap.parse_args()

    split = make_split(load(), a.tasks, a.seed)
    order = [int(x) for x in a.order.split(",")] if a.order else list(range(a.tasks))
    t0 = time.time()
    if a.method in ("finetune", "ewc", "replay"):
        rec, extra = run_mlp_family(split, a.seed, a.epochs, order, a.method,
                                    lam=a.lam, replay_frac=a.replay_frac)
    elif a.method == "indep":
        rec, extra = run_indep(split, a.seed, a.epochs, order)
    elif a.method == "prognet":
        rec, extra = run_prognet(split, a.seed, a.epochs, order)
    elif a.method == "cms":
        rec, extra = run_cms(split, a.seed, a.epochs, order)
    elif a.method in ("rf", "xgb"):
        rec, extra = run_tree(split, a.seed, order, a.method)
    else:
        rec, extra = run_joint(split, a.seed, a.epochs, a.method)
    secs = time.time() - t0

    sub = f"T{a.tasks}_e{a.epochs}" + (f"_{a.tag}" if a.tag else "")
    out = RUNS / sub; out.mkdir(parents=True, exist_ok=True)
    name = a.method + (f"_lam{a.lam:g}" if a.method == "ewc" else "") + \
        (f"_buf{a.replay_frac:g}" if a.method == "replay" else "")
    res = dict(method=a.method, seed=a.seed, tasks=a.tasks, epochs=a.epochs, lam=a.lam,
               replay_frac=a.replay_frac, order=order, labels=labels(a.tasks),
               seconds=secs, threads=os.environ.get("OMP_NUM_THREADS"), **extra,
               pooled_matched_mae=rec.pooled_matched, n_matched_test=rec.n_matched,
               matrices={k: v.tolist() for k, v in rec.A.items()})
    (out / f"{name}_seed{a.seed}.json").write_text(json.dumps(res))
    A = rec.A["s11_r2"]
    if a.method.startswith("joint"):
        print(f"{sub} {name} s{a.seed} {secs:.1f}s  per-task R2 {np.round(A[0], 3)}")
    else:
        T = len(order)
        diag = np.array([A[i, order[i]] for i in range(T)])
        bwt = np.mean([A[T - 1, order[i]] - A[i, order[i]] for i in range(T - 1)])
        print(f"{sub} {name} s{a.seed} {secs:.1f}s  BWT {bwt:+.4f}  final {A[-1].mean():+.4f}  diag {diag.mean():+.4f}")


if __name__ == "__main__":
    main()
