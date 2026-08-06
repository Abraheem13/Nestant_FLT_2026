import json, numpy as np, pandas as pd
from pathlib import Path
RES = Path(__file__).resolve().parent.parent / "results"

FILEKEY = {"progressive": "prognet"}

NAME = {'cms_freeze':'NestAnt-CMS (ours)', 'indep':'Independent per-task MLP',
        'replay':'Replay (10% buffer)', 'ewc':'EWC (lambda=1e6)',
        'xgb':'XGBoost (refit)', 'rf':'Random Forest (refit)',
        'progressive':'Progressive Networks'}

def bwt(A):
    T = A.shape[0]
    return float(np.mean([A[T-1,i] - A[i,i] for i in range(T-1)]))

def fm(A):
    T = A.shape[0]
    return float(np.mean([max(A[j,i] for j in range(i,T-1)) - A[T-1,i]
                          for i in range(T-1)]))

summary = {}
for key in NAME:
    mats = []
    for s in range(5):
        f = RES / f"forgetting5_{FILEKEY.get(key,key)}_seed{s}_v3.csv"
        if f.exists():
            mats.append(pd.read_csv(f, index_col=0).values)
    if not mats:
        print(f"MISSING: {key}");  continue
    M = np.stack(mats)
    summary[key] = dict(
        name=NAME[key], n_seeds=len(mats),
        bwt_mean=float(np.mean([bwt(m) for m in mats])),
        bwt_std=float(np.std([bwt(m) for m in mats])),
        fm_mean=float(np.mean([fm(m) for m in mats])),
        fm_std=float(np.std([fm(m) for m in mats])),
        final_mean=float(np.mean([m[-1].mean() for m in mats])),
        final_std=float(np.std([m[-1].mean() for m in mats])),
        diag_mean=float(np.mean([np.diag(m).mean() for m in mats])),
        diag_std=float(np.std([np.diag(m).mean() for m in mats])),
        matrix_mean=M.mean(0).tolist(), matrix_std=M.std(0).tolist())

(RES / "multiseed_summary_5task.json").write_text(json.dumps(summary, indent=2))

print(f"\n{'Method':<26} {'BWT':>18} {'Final R2':>18} {'Diag R2':>18}")
print("-"*84)
for k, v in sorted(summary.items(), key=lambda x: -x[1]['final_mean']):
    print(f"{v['name']:<26} {v['bwt_mean']:+.4f}+/-{v['bwt_std']:.4f}   "
          f"{v['final_mean']:+.4f}+/-{v['final_std']:.4f}   "
          f"{v['diag_mean']:+.4f}+/-{v['diag_std']:.4f}")
