"""make_jobs.py -- write results/jobs.txt, one bench.py argument line per run."""
from pathlib import Path

LAMBDAS = ["1", "10", "100", "1000", "10000", "100000", "1000000", "10000000", "100000000"]
jobs = []
for s in range(5):
    for T in (5, 10, 20):
        for m in ["finetune", "indep", "prognet", "cms", "rf", "xgb", "joint_mlp", "joint_xgb"]:
            jobs.append(f"--method {m} --seed {s} --tasks {T} --epochs 300")
        jobs.append(f"--method replay --replay_frac 0.1 --seed {s} --tasks {T} --epochs 300")
        for lam in LAMBDAS:
            jobs.append(f"--method ewc --lam {lam} --seed {s} --tasks {T} --epochs 300")
    for f in ["0.05", "0.2", "0.4"]:
        jobs.append(f"--method replay --replay_frac {f} --seed {s} --tasks 5 --epochs 300")
    for e in (30, 100):
        for m in ["finetune", "indep", "prognet", "cms"]:
            jobs.append(f"--method {m} --seed {s} --tasks 5 --epochs {e}")
        jobs.append(f"--method replay --replay_frac 0.1 --seed {s} --tasks 5 --epochs {e}")
        for lam in LAMBDAS:
            jobs.append(f"--method ewc --lam {lam} --seed {s} --tasks 5 --epochs {e}")
    for k in range(1, 5):                      # NestAnt-CMS with each task first
        order = [k] + [t for t in range(5) if t != k]
        jobs.append(f"--method cms --seed {s} --tasks 5 --epochs 300 "
                    f"--order {','.join(map(str, order))} --tag first{k}")
jobs.sort(key=lambda j: ("--epochs 300" not in j, "rf" in j))   # long runs first
out = Path(__file__).resolve().parent.parent / "results" / "jobs.txt"
out.write_text("\n".join(jobs) + "\n")
print(len(jobs), "jobs ->", out)
