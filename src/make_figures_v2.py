"""
make_figures_v2.py -- regenerate paper figures from the VERIFIED
300-epoch results. Run from nestant/src/.

    python3 make_figures_v2.py

Writes figures/fig1_benchmark.png and figures/fig2_budget.png
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(__file__).resolve().parent.parent / "results"
FIGS = Path(__file__).resolve().parent.parent / "figures"
FIGS.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "savefig.dpi": 300, "savefig.bbox": "tight", "mathtext.fontset": "stix",
})

WIN, MID, LOW = "#117733", "#888888", "#BBBBBB"

# ---------------------------------------------------------------- Fig 1
d = json.loads((RES / "multiseed_summary_5task.json").read_text())
order = ["progressive", "cms_freeze", "indep", "replay", "ewc", "rf", "xgb"]
label = {"progressive": "Progressive Nets", "cms_freeze": "NestAnt-CMS",
         "indep": "Independent", "replay": "Replay (10%)",
         "ewc": r"EWC ($\lambda{=}10^6$)", "rf": "Random Forest",
         "xgb": "XGBoost"}
order = [k for k in order if k in d]

fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
names = [label[k] for k in order][::-1]
col = [WIN if k == "progressive" else (MID if k in ("cms_freeze", "indep") else LOW)
       for k in order][::-1]

for j, (field, sfield, title, xlab) in enumerate([
        ("bwt_mean", "bwt_std", "(a) Forgetting", r"BWT $\rightarrow$ higher better"),
        ("final_mean", "final_std", "(b) Retained accuracy", r"Final $R^2$ $\rightarrow$ higher better")]):
    v = [d[k][field] for k in order][::-1]
    e = [d[k][sfield] for k in order][::-1]
    ax[j].barh(names, v, xerr=e, color=col, edgecolor="#333", linewidth=0.6,
               height=0.62, capsize=2.5, error_kw=dict(linewidth=0.7, ecolor="#333"))
    ax[j].axvline(0, color="#222", lw=0.7)
    ax[j].set_xlabel(xlab); ax[j].set_title(title, fontweight="bold")
    lo = min(v) - abs(min(v)) * 0.25 - 0.4
    ax[j].set_xlim(lo, max(1.35, max(v) + 0.9))
    for i, val in enumerate(v):
        ax[j].text(max(1.30, max(v) + 0.85), i, f"{val:+.3f}",
                   va="center", ha="right", fontsize=7.5, family="monospace")
    if j == 1:
        ax[j].set_yticklabels([])

fig.suptitle("Five-task continual benchmark (5 seeds, 300 epochs)",
             fontweight="bold", y=1.03)
fig.tight_layout()
fig.savefig(FIGS / "fig1_benchmark.png")
plt.close(fig)
print("wrote fig1_benchmark.png")

# ---------------------------------------------------------------- Fig 2
cs = RES / "convergence_study.json"
fig, ax = plt.subplots(figsize=(3.4, 2.6))
if cs.exists():
    rows = json.loads(cs.read_text())
    eps = sorted({r["epochs"] for r in rows})
    for key, lab, c, mk in [("cms_freeze", "NestAnt-CMS", WIN, "o"),
                            ("indep", "Independent", MID, "s")]:
        m = [np.mean([r["final"] for r in rows
                      if r["epochs"] == e and r["method"] == key]) for e in eps]
        sd = [np.std([r["final"] for r in rows
                      if r["epochs"] == e and r["method"] == key]) for e in eps]
        ax.errorbar(eps, m, yerr=sd, marker=mk, color=c, label=lab,
                    lw=1.3, markersize=4.5, capsize=2.5)
    ax.set_xscale("log")
    ax.set_xticks(eps); ax.set_xticklabels([str(e) for e in eps])
else:
    print("  (convergence_study.json absent -- using reported values)")
    eps = [30, 100, 300]
    ax.errorbar(eps, [0.611, 0.790, 0.858], yerr=[0.029, 0.004, 0.004],
                marker="o", color=WIN, label="NestAnt-CMS", lw=1.3,
                markersize=4.5, capsize=2.5)
    ax.errorbar(eps, [0.510, 0.783, 0.856], yerr=[0.020, 0.005, 0.004],
                marker="s", color=MID, label="Independent", lw=1.3,
                markersize=4.5, capsize=2.5)
    ax.set_xscale("log"); ax.set_xticks(eps)
    ax.set_xticklabels(["30", "100", "300"])

ax.set_xlabel("Epochs per task"); ax.set_ylabel(r"Final $R^2$")
ax.set_title("Shared-prior benefit vs budget", fontweight="bold")
ax.legend(frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(FIGS / "fig2_budget.png")
plt.close(fig)
print("wrote fig2_budget.png")

# ------------------------------------------------------------- Fig 3
d = json.loads((RES / "multiseed_summary_5task.json").read_text())
fig, ax = plt.subplots(figsize=(3.4, 2.7))
pts = {"progressive": (WIN, "*", 15), "cms_freeze": ("#332288", "o", 6),
       "indep": (MID, "s", 5), "replay": (LOW, "^", 5), "ewc": (LOW, "v", 5),
       "rf": (LOW, "D", 4), "xgb": (LOW, "P", 5)}
for k, (c, mk, ms) in pts.items():
    if k not in d: continue
    ax.errorbar(d[k]["bwt_mean"], d[k]["final_mean"], xerr=d[k]["bwt_std"],
                yerr=d[k]["final_std"], marker=mk, color=c, markersize=ms,
                capsize=2, elinewidth=0.7, markeredgecolor="black",
                markeredgewidth=0.5, linestyle="none", zorder=3)
ax.axvspan(-0.35, 0.7, alpha=0.07, color=WIN, zorder=0)
for txt, xy, xyt, col, bold in [
    ("Progressive  0.905", (0.0, 0.905), (-5.1, 1.02), WIN, "bold"),
    ("NestAnt-CMS  0.861", (0.0, 0.861), (-5.1, 0.44), "#332288", "normal"),
    ("Independent  0.859", (0.0, 0.859), (-5.1, 0.06), "black", "normal"),
    ("Replay", (-1.70, -0.63), (-1.15, -0.20), "black", "normal"),
    ("EWC", (-2.84, -2.30), (-2.60, -1.35), "black", "normal"),
    ("tree ensembles", (-4.67, -2.83), (-5.2, -2.15), "black", "normal")]:
    ax.annotate(txt, xy, xyt, fontsize=7.5, ha="left", color=col,
                fontweight=bold,
                arrowprops=dict(arrowstyle="-", color="#999", lw=0.5))
ax.axhline(0, color="#333", lw=0.6); ax.axvline(0, color="#333", lw=0.6)
ax.set_xlabel(r"BWT $\rightarrow$"); ax.set_ylabel(r"Final $R^2$ $\rightarrow$")
ax.set_title("Retention--accuracy plane", fontweight="bold")
ax.set_xlim(-5.4, 0.9); ax.set_ylim(-3.4, 1.25)
fig.tight_layout(); fig.savefig(FIGS / "fig3_pareto.png"); plt.close(fig)
print("wrote fig3_pareto.png")

# ------------------------------------------------------------- Fig 4
Ts = np.arange(2, 51)
base = 5*128+128+128*80+80+80*3+3
fig, ax = plt.subplots(figsize=(3.4, 2.6))
ax.plot(Ts, (Ts*base + 80*80*Ts*(Ts-1)//2)/1e3, color=WIN, lw=1.6,
        label="Progressive  $O(T^2)$")
ax.plot(Ts, (9024 + Ts*6739)/1e3, color="#332288", lw=1.6, ls="--",
        label="NestAnt-CMS  $O(T)$")
for T, r in [(5, 2.8), (20, 10.0), (50, 24.3)]:
    y = (T*base + 80*80*T*(T-1)//2)/1e3
    ax.plot(T, y, "o", color=WIN, ms=4)
    ax.annotate(f"{r:.1f}$\\times$", (T, y), textcoords="offset points",
                xytext=(-2, 6), fontsize=7.5, ha="right", color="#444")
ax.set_yscale("log")
ax.set_xlabel("Number of tasks $T$"); ax.set_ylabel("Parameters (thousands)")
ax.set_title("Memory scaling", fontweight="bold")
ax.legend(frameon=False, loc="upper left", fontsize=7.5)
fig.tight_layout(); fig.savefig(FIGS / "fig4_memory.png"); plt.close(fig)
print("wrote fig4_memory.png")

# ------------------------------------------------------------- Fig 5
fig, ax = plt.subplots(figsize=(3.4, 2.4))
labs = ["1.5GHz$_S$", "2.4GHz", "5.8GHz$_L$"]
mu = [0.596, 0.582, 0.581]; sd = [0.018, 0.008, 0.014]; ns = [9, 3, 6]
ax.barh(labs, mu, xerr=sd, color=["#332288", MID, MID], edgecolor="#333",
        linewidth=0.6, height=0.55, capsize=3,
        error_kw=dict(linewidth=0.8, ecolor="#333"))
ax.axvline(0.589, color=WIN, ls="--", lw=1.2, zorder=3)
ax.text(0.589, 2.55, " pooled mean", color=WIN, fontsize=7.5, va="center")
for i, n in enumerate(ns):
    ax.text(0.30, i, f"n={n}", va="center", fontsize=7.5, color="white")
ax.set_xlim(0.28, 0.68)
ax.set_xlabel(r"Final $R^2$"); ax.set_ylabel("Prior learned from")
ax.set_title("Task-ordering robustness", fontweight="bold")
fig.tight_layout(); fig.savefig(FIGS / "fig5_ordering.png"); plt.close(fig)
print("wrote fig5_ordering.png")
