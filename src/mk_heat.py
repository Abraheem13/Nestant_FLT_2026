"""Three-panel per-stage accuracy heatmaps (Fig. 2 in the paper)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import pandas as pd

plt.rcParams.update({"font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8,
    "axes.titlesize": 9, "mathtext.fontset": "stix",
    "savefig.dpi": 300, "savefig.bbox": "tight"})

HERE = Path(__file__).resolve().parent
RES, FIGS = HERE.parent / "results", HERE.parent / "figures"
FIGS.mkdir(exist_ok=True)
L = ["1.5G$_S$", "1.5G$_L$", "2.4G", "5.8G$_S$", "5.8G$_L$"]

def load(key, seed=0):
    return pd.read_csv(RES / f"forgetting5_{key}_seed{seed}_v3.csv",
                       index_col=0).values

panels = [(load("prognet"), "Progressive Networks"),
          (load("cms_freeze"), "NestAnt-CMS"),
          (load("xgb"), "XGBoost (refit)")]

cmap = LinearSegmentedColormap.from_list("d", [(0.0, "#a50026"),
    (0.30, "#f46d43"), (0.5, "#ffffbf"), (0.75, "#66bd63"), (1.0, "#1a9850")])

fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.45))
for ax, (M, t) in zip(axes, panels):
    D = M.astype(float).copy()
    mask = np.triu(np.ones_like(D, bool), 1)
    D[mask] = np.nan
    C = np.clip(D, -1.0, 1.0)
    im = ax.imshow(C, vmin=-1, vmax=1, cmap=cmap)
    for i in range(5):
        for j in range(5):
            if mask[i, j]: continue
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}" if abs(v) < 10 else f"{v:.0f}",
                    ha="center", va="center", fontsize=6.4,
                    color="white" if C[i, j] < -0.15 or C[i, j] > 0.55 else "black")
    ax.set_xticks(range(5))
    ax.set_xticklabels(L, rotation=45, ha="right", fontsize=6.4)
    ax.set_yticks(range(5))
    ax.set_yticklabels([f"after {l}" for l in L] if ax is axes[0] else [],
                       fontsize=6.4)
    ax.set_title(t, fontweight="bold", fontsize=8.5)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(length=0)
axes[0].set_ylabel("training stage", fontsize=7.5)
cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.015, ticks=[-1, -0.5, 0, 0.5, 1])
cb.set_label("$R^2$ on $S_{11}$ (clipped)", fontsize=7.5)
cb.ax.tick_params(labelsize=6.5); cb.outline.set_linewidth(0.5)
fig.savefig(FIGS / "fig6_heatmaps.png")
print("wrote fig6_heatmaps.png")
