"""
ordering_ablation.py -- Task-ordering sensitivity for NestAnt-CMS.
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
import numpy as np

from data import load_raw, preprocess, task_slice
from data5 import make_5task_splits, TASK5_LABELS
from metrics import r2_score
from nestant_cms import (CMSConfig, init_cms, init_cms_opt,
                         cms_forward, cms_backward, cms_update, cms_predict)
from nestant import iterate_minibatches

RES = Path(__file__).resolve().parent.parent / "results"
RES.mkdir(exist_ok=True)
N_TASKS = 5


def run_one_order(order, seed, epochs, df):
    cfg = CMSConfig(n_features=5, n_targets=3, n_bands=N_TASKS,
                    hidden=128, embed=64, band_embed=16, fast_hidden=80,
                    slow_lr=1e-3, fast_lr=3e-3,
                    epochs=epochs, batch_size=1024,
                    weight_decay=1e-5, seed=seed)

    split = make_5task_splits(df, test_frac=0.20, seed=seed)
    p = init_cms(cfg)
    opt = init_cms_opt(p, N_TASKS)
    rng = np.random.default_rng(seed)

    s11 = split.target_names.index("s11_at_target")
    A = np.full((N_TASKS, N_TASKS), np.nan)

    for stage, tid in enumerate(order):
        Xt = task_slice(split.X_train, split.task_train, tid)
        yt = task_slice(split.y_train, split.task_train, tid)
        tt = task_slice(split.task_train, split.task_train, tid)

        slow_lr = cfg.slow_lr if stage == 0 else 0.0

        for _ in range(cfg.epochs):
            for b in iterate_minibatches(rng, Xt.shape[0], cfg.batch_size):
                yhat, cache = cms_forward(p, Xt[b], tt[b])
                grads, _ = cms_backward(p, cache, yt[b], yhat,
                                        cfg.n_bands, cfg.weight_decay)
                cms_update(p, grads, opt, slow_lr, cfg.fast_lr,
                           cfg.weight_decay, active_bands=[tid])

        y_pred = cms_predict(p, split.X_test, split.task_test)
        y_true = split.y_test * split.y_std + split.y_mean
        yp = y_pred * split.y_std + split.y_mean
        for jb in range(N_TASKS):
            m = split.task_test == jb
            A[stage, jb] = r2_score(y_true[m, s11:s11+1], yp[m, s11:s11+1])

    return A


def bwt_for_order(A, order):
    deltas = []
    for stage, tid in enumerate(order[:-1]):
        trained_at = A[stage, tid]
        at_end = A[-1, tid]
        deltas.append(at_end - trained_at)
    return float(np.mean(deltas))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_orders", type=int, default=6)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    df = preprocess(load_raw(str(here.parent / "data" / "colab_dataset_microstrip.csv")))

    base = list(range(N_TASKS))
    orders = [tuple(base)]
    rng = np.random.default_rng(12345)
    while len(orders) < args.n_orders:
        cand = tuple(rng.permutation(N_TASKS).tolist())
        if cand not in orders:
            orders.append(cand)

    rows = []
    t0 = time.time()
    for oi, order in enumerate(orders):
        for seed in args.seeds:
            A = run_one_order(list(order), seed, args.epochs, df)
            bwt = bwt_for_order(A, list(order))
            final = float(np.nanmean(A[-1]))
            diag = float(np.nanmean([A[s, t] for s, t in enumerate(order)]))
            label = "->".join(TASK5_LABELS[t] for t in order)
            rows.append(dict(order=list(order), order_label=label, seed=seed,
                             bwt=bwt, final_mean=final, diag_mean=diag,
                             first_task=TASK5_LABELS[order[0]]))
            print(f"[{time.time()-t0:6.1f}s] order {oi} seed {seed}  "
                  f"BWT={bwt:+.4f}  final={final:+.4f}  diag={diag:+.4f}  "
                  f"({label})")
            np.savetxt(RES / f"ordering_A_o{oi}_seed{seed}.csv", A,
                       delimiter=",", fmt="%.6f")

    (RES / "ordering_ablation.json").write_text(json.dumps(rows, indent=2))

    bwts = np.array([r["bwt"] for r in rows])
    finals = np.array([r["final_mean"] for r in rows])
    print("\n================ SUMMARY ================")
    print(f"orders x seeds        : {len(orders)} x {len(args.seeds)} = {len(rows)}")
    print(f"BWT   mean +/- std    : {bwts.mean():+.4f} +/- {bwts.std():.4f}")
    print(f"BWT   min / max       : {bwts.min():+.4f} / {bwts.max():+.4f}")
    print(f"final mean +/- std    : {finals.mean():+.4f} +/- {finals.std():.4f}")
    print(f"final min / max       : {finals.min():+.4f} / {finals.max():+.4f}")
    print("\nBy first task (which task the slow prior is learned from):")
    for t in range(N_TASKS):
        sel = [r for r in rows if r["order"][0] == t]
        if sel:
            f = np.array([r["final_mean"] for r in sel])
            print(f"  {TASK5_LABELS[t]:>12}: final R2 = "
                  f"{f.mean():+.4f} +/- {f.std():.4f}  (n={len(sel)})")
    print("=========================================")


if __name__ == "__main__":
    main()
