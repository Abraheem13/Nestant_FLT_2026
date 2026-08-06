"""
convergence_study.py -- Do the zero-forgetting methods improve with more
compute, and does the gap between them change?

Sweeps epochs and fast-head width for NestAnt-CMS, Independent per-task,
and Progressive Networks under IDENTICAL budgets.
"""
import sys, os, json, time, argparse, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parent.parent / "results"

FILEKEY = {"progressive": "prognet"}

def run(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout

def read_matrix(key, seed):
    f = RES / f"forgetting5_{FILEKEY.get(key,key)}_seed{seed}_v3.csv"
    return pd.read_csv(f, index_col=0).values if f.exists() else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, nargs="+", default=[30, 100, 300])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    rows = []
    t0 = time.time()
    for ep in args.epochs:
        for seed in args.seeds:
            for key, cmd in [
                ("cms_freeze", f"python3 continual5_runner.py --method cms_freeze --seed {seed} --epochs {ep}"),
                ("indep",      f"python3 continual5_runner.py --method indep --seed {seed} --epochs {ep}"),
                ("progressive",f"python3 progressive_nets.py --seed {seed} --epochs {ep}"),
            ]:
                run(cmd)
                A = read_matrix(key, seed)
                if A is None:
                    print(f"  !! no matrix for {key} seed {seed}");  continue
                rows.append(dict(method=key, epochs=ep, seed=seed,
                                 final=float(A[-1].mean()),
                                 diag=float(np.diag(A).mean()),
                                 per_task=[float(A[i,i]) for i in range(5)]))
                print(f"[{time.time()-t0:6.1f}s] {key:<12} ep={ep:<4} seed={seed}  "
                      f"final={rows[-1]['final']:+.4f}  "
                      f"task3={rows[-1]['per_task'][2]:+.3f}")

    (RES / "convergence_study.json").write_text(json.dumps(rows, indent=2))

    print("\n================ SUMMARY ================")
    print(f"{'epochs':>7} {'method':<14} {'final R2':>18} {'2.4GHz task':>14}")
    print("-"*58)
    for ep in args.epochs:
        for key in ["cms_freeze", "indep", "progressive"]:
            sel = [r for r in rows if r["epochs"]==ep and r["method"]==key]
            if not sel: continue
            f = np.array([r["final"] for r in sel])
            t3 = np.array([r["per_task"][2] for r in sel])
            print(f"{ep:>7} {key:<14} {f.mean():+.4f}+/-{f.std():.4f}   "
                  f"{t3.mean():+.3f}+/-{t3.std():.3f}")
        print()
    print("Gap CMS - Progressive, per epoch budget:")
    for ep in args.epochs:
        c = np.array([r["final"] for r in rows if r["epochs"]==ep and r["method"]=="cms_freeze"])
        p = np.array([r["final"] for r in rows if r["epochs"]==ep and r["method"]=="progressive"])
        if len(c) and len(p):
            gap = c.mean()-p.mean()
            sig = gap/np.sqrt(c.std()**2 + p.std()**2 + 1e-12)
            print(f"  ep={ep:<4} gap={gap:+.4f}  ({sig:+.2f} sigma)")
    print("=========================================")

if __name__ == "__main__":
    main()
