"""
make_figures.py -- figures for the paper, drawn only from results/summary.json.

    fig_heatmaps.pdf : S11 R^2 after every stage, seed 0, T=5
    fig_scaling.pdf  : (a) final R^2 versus epochs per task, (b) parameters versus T
Fonts are embedded as TrueType (pdf.fonttype 42).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "figures"; FIGS.mkdir(exist_ok=True)
S = json.loads((ROOT / "results" / "summary.json").read_text())

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 7.5, "axes.titlesize": 8,
    "axes.labelsize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.linewidth": 0.6, "pdf.fonttype": 42,
    "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})

LAB = ["1.5$_S$", "1.5$_L$", "2.4", "5.8$_S$", "5.8$_L$"]
cmap = LinearSegmentedColormap.from_list("d", [(0.0, "#a50026"), (0.3, "#f46d43"),
                                               (0.5, "#ffffbf"), (0.75, "#66bd63"), (1.0, "#1a9850")])

# ------------------------------------------------------------------ heatmaps
panels = [("xgb", "XGBoost (refit)"), ("finetune", "Fine-tuned MLP"),
          ("cms", "NestAnt-CMS"), ("prognet", "Progressive Networks")]
fig, axes = plt.subplots(1, 4, figsize=(7.16, 1.95))
for ax, (k, title) in zip(axes, panels):
    M = np.array(S["T5_e300"][k]["matrix_seed0_s11_r2"], float)
    mask = np.triu(np.ones_like(M, bool), 1)
    C = np.where(mask, np.nan, np.clip(M, -1, 1))
    im = ax.imshow(C, vmin=-1, vmax=1, cmap=cmap)
    for i in range(5):
        for j in range(i + 1):
            v = M[i, j]
            txt = f"{v:.2f}" if abs(v) < 10 else f"{v:.0f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=5.6,
                    color="white" if (C[i, j] < -0.2 or C[i, j] > 0.6) else "black")
    ax.set_xticks(range(5)); ax.set_xticklabels(LAB, rotation=45, ha="right")
    ax.set_yticks(range(5)); ax.set_yticklabels([f"after {l}" for l in LAB] if ax is axes[0] else [])
    ax.set_title(title, fontsize=7.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
axes[0].set_ylabel("training stage")
cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.012, ticks=[-1, 0, 1])
cb.set_label(r"$R^2$ on $S_{11}$ (clipped to $[-1,1]$)", fontsize=6.5)
cb.outline.set_linewidth(0.4)
fig.savefig(FIGS / "fig_heatmaps.pdf")
plt.close(fig)

# ------------------------------------------------------------------ scaling
fig, (a, b) = plt.subplots(1, 2, figsize=(3.45, 1.55))
eps = [30, 100, 300]
style = {"prognet": ("Progressive", "#117733", "o"), "cms": ("NestAnt-CMS", "#332288", "s"),
         "indep": ("Independent", "#888888", "^")}
for k, (lab, c, mk) in style.items():
    m = [S[f"T5_e{e}"][k]["final_s11_mae"]["mean"] for e in eps]
    sd = [S[f"T5_e{e}"][k]["final_s11_mae"]["std"] for e in eps]
    a.errorbar(eps, m, yerr=sd, marker=mk, ms=3, lw=1, color=c, label=lab, capsize=1.5)
a.set_xscale("log"); a.set_xticks(eps); a.set_xticklabels([str(e) for e in eps])
a.minorticks_off()
a.set_xlabel("epochs per task"); a.set_ylabel(r"final $S_{11}$ MAE (dB)")
a.set_title("(a) training budget, $T{=}5$")
a.legend(frameon=False, loc="upper right", handlelength=1.2, fontsize=5.5)

T = np.arange(1, 51)
col = 5 * 128 + 128 + 128 * 80 + 80 + 80 * 3 + 3
prog = T * col + 80 * 80 * T * (T - 1) // 2
cms = (5 * 128 + 128 + 128 * 64 + 64) + T * (16 + 80 * 80 + 80 + 80 * 3 + 3)
b.plot(T, prog / 1e3, color="#117733", lw=1, label="Progressive")
b.plot(T, cms / 1e3, color="#332288", lw=1, ls="--", label="NestAnt-CMS")
b.plot(T, T * col / 1e3, color="#888888", lw=1, ls=":", label="Independent")
for t in (5, 10, 20):
    for k, c in (("prognet", "#117733"), ("cms", "#332288"), ("indep", "#888888")):
        b.plot(t, S[f"T{t}_e300"][k]["params"] / 1e3, "o", ms=2.5, color=c)
b.set_yscale("log"); b.set_xlabel("number of tasks $T$"); b.set_ylabel("parameters (thousands)")
b.set_title("(b) memory; dots = trained")
b.legend(frameon=False, loc="lower right", handlelength=1.6, fontsize=5.5)
fig.tight_layout(w_pad=0.6)
fig.savefig(FIGS / "fig_scaling.pdf")
plt.close(fig)
print("wrote", [p.name for p in FIGS.glob("fig_*.pdf")])

# ------------------------------------------------------------------ EWC and replay sweeps
C = S["T5_e300"]
fig, (a, b) = plt.subplots(1, 2, figsize=(3.45, 1.55))
lams = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]
key = {lam: f"ewc_lam{lam:g}" for lam in lams}
for metric, lab, c, mk in (("diag_s11_mae", "on task just learned", "#888888", "s"),
                           ("final_s11_mae", "final, all tasks", "#a50026", "o")):
    m = [C[key[l]][metric]["mean"] for l in lams]
    sd = [C[key[l]][metric]["std"] for l in lams]
    a.errorbar(lams, m, yerr=sd, marker=mk, ms=2.5, lw=0.9, color=c, label=lab, capsize=1.2)
a.axhline(C["indep"]["final_s11_mae"]["mean"], color="#332288", lw=0.7, ls="--")
a.text(1.3, C["indep"]["final_s11_mae"]["mean"] / 1.5, "independent MLPs", fontsize=5.5, color="#332288")
a.set_ylim(0.25, 120)
a.set_xscale("log"); a.set_yscale("log")
a.set_xticks([1, 1e2, 1e4, 1e6, 1e8])
a.set_xlabel(r"EWC penalty $\lambda$"); a.set_ylabel(r"$S_{11}$ MAE (dB)")
a.set_title(r"(a) EWC, $T{=}5$")
a.legend(frameon=False, loc="upper left", fontsize=5.5, handlelength=1.4)

rhos = [0.0, 0.05, 0.1, 0.2, 0.4]
rk = {0.0: "finetune", 0.05: "replay_buf0.05", 0.1: "replay_buf0.1", 0.2: "replay_buf0.2", 0.4: "replay_buf0.4"}
for metric, lab, c, mk in (("final_s11_mae", "final, all tasks", "#a50026", "o"),
                           ("bwt_s11_mae", r"$\Delta$ (forgetting)", "#f46d43", "^"),
                           ("diag_s11_mae", "on task just learned", "#888888", "s")):
    m = [C[rk[r]][metric]["mean"] for r in rhos]
    sd = [C[rk[r]][metric]["std"] for r in rhos]
    b.errorbar([100 * r for r in rhos], m, yerr=sd, marker=mk, ms=2.5, lw=0.9, color=c, label=lab, capsize=1.2)
b.axhline(C["indep"]["final_s11_mae"]["mean"], color="#332288", lw=0.7, ls="--")
b.set_xlabel(r"replay buffer $\rho$ (%)"); b.set_ylabel(r"$S_{11}$ MAE (dB)")
b.set_title(r"(b) replay, $T{=}5$ ($\rho{=}0$: fine-tuning)")
b.set_xticks([0, 5, 10, 20, 40])
b.legend(frameon=False, loc="upper right", fontsize=5.5, handlelength=1.4)
fig.tight_layout(w_pad=0.5)
fig.savefig(FIGS / "fig_sweeps.pdf")
plt.close(fig)

# ------------------------------------------------------------------ prediction scatter
PRED = ROOT / "results" / "preds" / "T5_e300"
X = np.load(PRED / "xgb_seed0.npz"); P = np.load(PRED / "prognet_seed0.npz")
task = 3                                            # 5.8 GHz small patches
sel = X["task"] == task
y = X["true_s11"][sel]
panels = [(X["pred_s11"][3, sel], "XGBoost, after 5.8$_S$"),
          (X["pred_s11"][4, sel], "XGBoost, after 5.8$_L$"),
          (P["pred_s11"][4, sel], "Progressive, after 5.8$_L$")]
lo = min(y.min(), min(p.min() for p, _ in panels)) - 0.5
fig, axes = plt.subplots(1, 3, figsize=(3.45, 1.32), sharey=True)
for ax, (p, title) in zip(axes, panels):
    ax.plot([lo, 0.5], [lo, 0.5], color="#999999", lw=0.6)
    ax.scatter(y, p, s=1.2, alpha=0.35, color="#332288", linewidths=0)
    ax.set_xlim(lo, 0.5); ax.set_ylim(lo, 0.5); ax.set_aspect("equal")
    ax.set_title(title, fontsize=6.3)
    ax.text(0.04, 0.96, f"MAE {np.abs(p - y).mean():.2f} dB", transform=ax.transAxes,
            fontsize=5.8, va="top")
    ax.set_xlabel("simulated (dB)", fontsize=6)
axes[0].set_ylabel("predicted (dB)", fontsize=6)
fig.tight_layout(w_pad=0.3)
fig.savefig(FIGS / "fig_scatter.pdf")
plt.close(fig)
print("wrote fig_sweeps.pdf, fig_scatter.pdf")

# ------------------------------------------------------------------ task order (NestAnt-CMS)
keys = ["T5_e300", "T5_e300_first1", "T5_e300_first2", "T5_e300_first3", "T5_e300_first4"]
m = [S[k]["cms"]["final_s11_mae"]["mean"] for k in keys]
sd = [S[k]["cms"]["final_s11_mae"]["std"] for k in keys]
fig, ax = plt.subplots(figsize=(3.45, 1.25))
ax.bar(range(5), m, yerr=sd, color="#332288", alpha=0.85, width=0.6, capsize=2,
       error_kw=dict(lw=0.7))
for ref, lab, c, ls in (("indep", "independent MLPs", "#888888", "--"),
                        ("prognet", "Progressive Nets", "#117733", ":")):
    v = S["T5_e300"][ref]["final_s11_mae"]["mean"]
    ax.axhline(v, color=c, lw=0.9, ls=ls, label=lab)
ax.set_xticks(range(5)); ax.set_xticklabels(LAB)
ax.set_xlabel("first task (encoder trained on it)"); ax.set_ylabel(r"final $S_{11}$ MAE (dB)")
ax.set_ylim(0, 0.9)
ax.legend(frameon=False, fontsize=5.5, loc="upper left", ncol=2, handlelength=1.6)
fig.tight_layout()
fig.savefig(FIGS / "fig_order.pdf")
plt.close(fig)
print("wrote fig_order.pdf")

