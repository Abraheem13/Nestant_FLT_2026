"""
aggregate.py -- summarise results/runs/*/*.json into results/summary.json.

Per run (S11 R^2 matrix A, task order o, T tasks):
    diag  = mean_i A[i, o_i]                       accuracy right after training
    final = mean_j A[T-1, j]                       accuracy after the last task
    BWT   = mean_{i<T-1} (A[T-1, o_i] - A[i, o_i])
    FM    = mean_{i<T-1} max_{s in [i, T-2]} A[s, o_i] - A[T-1, o_i]
The same final/diag means are reported for S11 MAE [dB], S11 MAE on matched
designs [dB] (tasks without matched test rows are skipped), gain MAE [dB]
and resonance-frequency error [%]; for MAE-type metrics the "bwt" entry is
the mean increase in error on earlier tasks (positive = forgetting). Statistics over seeds: mean and sample standard
deviation (ddof = 1). Paired t-tests pair runs that share a seed (and hence the
same split and initialisation streams).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from data import load, make_split, labels

RES = Path(__file__).resolve().parent.parent / "results"


def summarise_run(r):
    o = r["order"]; T = len(o)
    out = {}
    if r["method"].startswith("joint"):
        for k, A in r["matrices"].items():
            out[f"final_{k}"] = float(np.nanmean(A[0]))
            out[f"pertask_{k}"] = A[0]
        out["final_matched_pooled"] = r["pooled_matched_mae"][0]
        return out
    for k, A in r["matrices"].items():
        A = np.array(A)
        out[f"final_{k}"] = float(np.nanmean(A[T - 1]))
        out[f"diag_{k}"] = float(np.nanmean([A[i, o[i]] for i in range(T)]))
        out[f"bwt_{k}"] = float(np.nanmean([A[T - 1, o[i]] - A[i, o[i]] for i in range(T - 1)]))
        out[f"pertask_final_{k}"] = A[T - 1].tolist()
        out[f"pertask_diag_{k}"] = [A[i, o[i]] for i in range(T)]
    out["final_matched_pooled"] = r["pooled_matched_mae"][T - 1]
    A = np.array(r["matrices"]["s11_r2"])
    out["fm_s11_r2"] = float(np.mean([max(A[s, o[i]] for s in range(i, T - 1)) - A[T - 1, o[i]]
                                      for i in range(T - 1)]))
    # worst single off-diagonal drop, to show what drives a mean BWT
    drops = [(A[T - 1, o[i]] - A[i, o[i]], i) for i in range(T - 1)]
    out["worst_drop"] = min(drops)[0]; out["worst_drop_task"] = int(o[min(drops)[1]])
    out["median_drop"] = float(np.median([d for d, _ in drops]))
    return out


def ms(v):
    v = np.asarray(v, float)
    return dict(mean=float(v.mean()), std=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                n=int(len(v)), values=v.tolist())


def main():
    runs = defaultdict(list)
    for f in sorted((RES / "runs").glob("*/*.json")):
        r = json.loads(f.read_text())
        runs[(f.parent.name, f.stem.rsplit("_seed", 1)[0])].append(r)

    summary = {}
    for (cfg, name), rs in sorted(runs.items()):
        rs = sorted(rs, key=lambda r: r["seed"])
        per = [summarise_run(r) for r in rs]
        d = dict(seeds=[r["seed"] for r in rs], params=rs[0].get("params"),
                 seconds=ms([r["seconds"] for r in rs]), labels=rs[0]["labels"])
        for k in per[0]:
            if isinstance(per[0][k], list):
                d[k] = np.mean([p[k] for p in per], 0).tolist()
            else:
                d[k] = ms([p[k] for p in per])
        if not name.startswith("joint"):
            d["matrix_seed0_s11_r2"] = rs[0]["matrices"]["s11_r2"]
            d["matrix_mean_s11_r2"] = np.mean([r["matrices"]["s11_r2"] for r in rs], 0).tolist()
        summary.setdefault(cfg, {})[name] = d

    # EWC: lambda giving the lowest mean final S11 MAE in each configuration
    # (selected on the test partition, which can only favour EWC)
    for cfg, ms_ in summary.items():
        ewc = {k: v for k, v in ms_.items() if k.startswith("ewc_lam")}
        if ewc:
            best = min(ewc, key=lambda k: ewc[k]["final_s11_mae"]["mean"])
            ms_["ewc_best"] = dict(ewc[best], selected=best)
            ms_["ewc_sweep"] = {k: dict(final_r2=v["final_s11_r2"]["mean"], bwt_r2=v["bwt_s11_r2"]["mean"],
                                        diag_r2=v["diag_s11_r2"]["mean"], final_mae=v["final_s11_mae"]["mean"],
                                        diag_mae=v["diag_s11_mae"]["mean"], delta_mae=v["bwt_s11_mae"]["mean"])
                                for k, v in ewc.items()}

    # paired tests on final S11 R^2
    tests = {}
    for cfg in ("T5_e300", "T10_e300"):
        S = summary.get(cfg, {})
        for a, b in [("cms", "indep"), ("prognet", "cms"), ("prognet", "indep"), ("prognet", "xgb"),
                     ("joint_xgb", "prognet"), ("joint_mlp", "prognet")]:
            for metric in ("final_s11_r2", "diag_s11_r2", "final_s11_mae", "diag_s11_mae"):
                if a in S and b in S and metric in S[a] and metric in S[b]:
                    common = sorted(set(S[a]["seeds"]) & set(S[b]["seeds"]))
                    x = np.array([S[a][metric]["values"][S[a]["seeds"].index(sd)] for sd in common])
                    y = np.array([S[b][metric]["values"][S[b]["seeds"].index(sd)] for sd in common])
                    if len(common) < 2:
                        continue
                    t = stats.ttest_rel(x, y)
                    tests[f"{cfg}:{a}-{b}:{metric}"] = dict(diff=float((x - y).mean()), n=len(common),
                                                         t=float(t.statistic), p=float(t.pvalue))
    summary["tests"] = tests

    # replicate ceiling: R^2 of predicting each test row by the mean of all
    # test rows with the same geometry (an upper bound for any function of
    # the recorded inputs), and the replicate spread in dB
    df = load(); ceil = {}
    for T in (5, 10):
        vals, maes, sds = [], [], []
        for seed in range(5):
            s = make_split(df, T, seed)
            y = s.y_phys(s.y_test)[:, 0]
            row, rmae, rsd = [], [], []
            for t in range(T):
                m = s.t_test == t
                g = s.g_test[m]; yt = y[m]
                gm = np.array([yt[g == k].mean() for k in g])
                row.append(1 - ((yt - gm) ** 2).sum() / ((yt - yt.mean()) ** 2).sum())
                rmae.append(np.abs(yt - gm).mean()); rsd.append(yt.std())
            vals.append(row); maes.append(rmae); sds.append(rsd)
        ceil[f"T{T}"] = dict(per_task=np.mean(vals, 0).tolist(), mean=float(np.mean(vals)),
                             mae_per_task=np.mean(maes, 0).tolist(), mae_mean=float(np.mean(maes)),
                             s11_sd_per_task=np.mean(sds, 0).tolist(), labels=labels(T))
    gid = df.groupby(["target_freq", "width", "length", "ground_plane", "x_0"])
    spread = gid["s11_at_target"].std(ddof=1)
    summary["data"] = dict(rows=len(df), geometries=int(gid.ngroups),
                           reps=gid.size().value_counts().sort_index().to_dict(),
                           s11_rep_sd_median=float(spread.median()), s11_rep_sd_mean=float(spread.mean()),
                           s11_positive_rows=int((df.s11_at_target > 0).sum()),
                           matched_frac=float((df.s11_at_target < -10).mean()),
                           ceiling=ceil)
    summary["data"]["reps"] = {str(k): int(v) for k, v in summary["data"]["reps"].items()}
    rows = {}
    for T in (5, 10):
        counts = [[int((make_split(df, T, sd).t_train == t).sum()) for t in range(T)] for sd in range(5)]
        rows[f"T{T}"] = dict(min=int(np.min(counts)), max=int(np.max(counts)))
    summary["data"]["train_rows"] = rows
    allruns = [json.loads(f.read_text()) for f in (RES / "runs").glob("*/*.json")]
    summary["compute"] = dict(n_runs=len(allruns), cpu_hours=sum(r["seconds"] for r in allruns) / 3600)
    summary["gradient_check"] = json.loads((RES / "gradient_check.json").read_text())["overall"]
    (RES / "summary.json").write_text(json.dumps(summary, indent=1))

    for cfg in sorted(k for k in summary if k.startswith("T")):
        print(f"\n== {cfg}")
        for name, v in sorted(summary[cfg].items(), key=lambda kv: -kv[1]["final_s11_r2"]["mean"]
                              if "final_s11_r2" in kv[1] else 0):
            if name == "ewc_sweep":
                continue
            f = v["final_s11_r2"]; line = f"{name:<18} final {f['mean']:+.4f}±{f['std']:.4f}"
            if "bwt_s11_r2" in v:
                b = v["bwt_s11_r2"]; dg = v["diag_s11_r2"]
                line += f"  bwt {b['mean']:+.4f}±{b['std']:.4f}  diag {dg['mean']:+.4f}±{dg['std']:.4f}"
                line += f"  maeF {v['final_s11_mae']['mean']:.3f} maeD {v['diag_s11_mae']['mean']:.3f}"
            line += f"  n={f['n']}  params={v['params']}  {v['seconds']['mean']:.0f}s"
            print(line)
    print("\nEWC selections:", {c: summary[c]["ewc_best"]["selected"] for c in summary
                                if c.startswith("T") and "ewc_best" in summary[c]})
    for k, v in tests.items():
        print(k, v)
    print("ceiling", {k: round(v["mean"], 4) for k, v in ceil.items()})


if __name__ == "__main__":
    main()
