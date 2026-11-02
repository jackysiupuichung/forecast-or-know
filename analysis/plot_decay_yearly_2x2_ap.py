"""The paper's decay-figure design with per-query AP (or MRR) on the y-axis.

Panels A1, A2, A2+bio, A3, one curve per model, black pooled linear trend across models (milli-units per year),
trial history (ta_breadth) as a grey dashed reference, cutoff lines. Unit: per-query metric on the full time-aware 1:10 sets,
binned by decision year (2021-2026). Pool adjustment as in the paper: the metric residualised on log(list length)
within each arm (at 1:10 the list length is 11 x the number of positives, so this adjusts for the number of positives;
check_decay_adjust.py shows it matches a joint regression with log n_pos + log n_cand to within 0.2 milli/yr).
Trend line and slope: query-level OLS of the adjusted metric on year with model fixed effects (disease-clustered SEs),
so each query counts once and the small 2026 slice (33 queries) is not over-weighted; star = 95% CI excludes zero.

  python plot_decay_yearly_2x2_ap.py [ap|mrr]          2x2 review preview -> preview/decay_options/decay_yearly_2x2_<met>.png
  python plot_decay_yearly_2x2_ap.py [ap|mrr] --paper  1x4 main-text figure -> <paper>/figures_camera/decay_yearly_<met>.png"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D)); sys.path.insert(0, str(D.parent))
from camera_data import load  # noqa: E402
from src.evaluation.fig_style import MODEL_COLORS, clean_axes, paper_style, FS, LW  # noqa: E402

from src.paths import FIGURES as PAPER  # noqa: E402
PAPER.mkdir(parents=True, exist_ok=True)
MODELS = [("Llama 3.1-8B", "llama3.1-8b"), ("Llama 3.1-70B", "llama3.1-70b"), ("Qwen 3-8B", "qwen3-8b"),
          ("Qwen 3-32B", "qwen3-32b"), ("Gemma 2-9B", "gemma-2-9b"), ("Gemma 2-27B", "gemma-2-27b")]
ARMS = [("A1 (no retrieval)", "A1"), ("A2 (dated graph)", "A2"),
        ("A3 (newest graph)", "A3")]   # A2+bio panel dropped: A2+bio is appendix-only (2026-10-26)
CUTS = [(2023.5, "#555"), (2024.5, "#999")]       # Llama post-cutoff from 2024; Qwen/Gemma from 2025
YEARS = list(range(2021, 2027))
MET = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "ap"
NAME = {"ap": "AP", "mrr": "MRR"}[MET]
PAPER_MODE = "--paper" in sys.argv

q = load(); q["logpool"] = np.log(q.n_cand)


def adjust(P):
    r = smf.ols(f"{MET} ~ logpool", P).fit()
    return P[MET] - r.params["logpool"] * (P.logpool - P.logpool.mean())


def slope(P, fe):
    f = smf.ols(f"adj ~ year{fe}", P).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(P.diseaseId)[0]})
    lo, hi = f.conf_int().loc["year"]
    return f.params.year, "*" if lo > 0 or hi < 0 else "", f


R = q[q.series == "ta_breadth"].copy(); R["adj"] = adjust(R)
ref = R.groupby("year").adj.mean()
rs, rst, rfit = slope(R, "")
print(f"trial history: {rs * 1000:+.1f} milli/yr p={rfit.pvalues.year:.3f}")
panels = []
for alabel, arm in ARMS:
    P = q[(q.arm == arm) & q.series.isin([s for _, s in MODELS])].copy(); P["adj"] = adjust(P)
    sl, st, f = slope(P, " + C(series)")
    lo, hi = f.conf_int().loc["year"]
    print(f"{arm}: {sl * 1000:+.1f} milli/yr [{lo * 1000:+.1f}, {hi * 1000:+.1f}] p={f.pvalues.year:.3f}")
    panels.append((alabel, P, sl, st))

if PAPER_MODE:
    paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.1), sharey=True)
else:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.4), sharey=True)
handles = None
for ax, (alabel, P, sl, st) in zip(axes.ravel(), panels):
    for cy, cc in CUTS:
        ax.axvline(cy, color=cc, ls=(0, (5, 3)), lw=0.8 if PAPER_MODE else 1.0, alpha=0.7)
    for mlabel, slug in MODELS:
        m = P[P.series == slug].groupby("year").adj.mean()
        ax.plot(m.index, m.values, "-o", color=MODEL_COLORS[slug], lw=LW["data"] if PAPER_MODE else 1.6,
                ms=2.2 if PAPER_MODE else 4, label=mlabel)
    xr = np.array([YEARS[0], YEARS[-1]])
    ax.plot(xr, P.adj.mean() + sl * (xr - P.year.mean()), color="black", lw=LW["trend"] if PAPER_MODE else 2.2,
            label="pooled trend")
    ax.plot(ref.index, ref.values, "--", color="#8a8981", lw=LW["data"] if PAPER_MODE else 1.6, label="trial history")
    if PAPER_MODE:
        ax.set_title(alabel, fontsize=FS["title"], pad=3)
        # slopes are reported in the caption, not on the panels
        ax.set_xticks(YEARS); ax.set_xticklabels([f"'{y % 100}" for y in YEARS])
        ax.tick_params(labelsize=FS["tick"], length=2, pad=1.5)
    else:
        ax.set_title(f"{alabel}: {sl * 1000:+.1f} milli/yr{st}", fontsize=11, weight="bold")
        ax.set_xticks(YEARS)
    clean_axes(ax)
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
rlab = "trial history"
print(f"caption slopes (AP/yr): trial history {rs:+.4f}{rst}; " + ", ".join(f"{a} {sl:+.4f}{st}" for a, _, sl, st in panels))
labels = [rlab if l == "trial history" else l for l in labels]
if PAPER_MODE:
    for ax in axes:
        ax.set_xlabel("decision year", fontsize=FS["label"], labelpad=1)
    axes[0].set_ylabel(f"adjusted {NAME}", fontsize=FS["label"], labelpad=2)
    fig.legend(handles, labels, ncol=4, fontsize=FS["legend"], frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               handlelength=1.8, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.16, 1, 1), w_pad=0.6)
    out = PAPER / f"decay_yearly_{MET}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
else:
    for ax in axes[-1]:
        ax.set_xlabel("decision year")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"pool-adj. {NAME} (per query)")
    fig.legend(handles, labels, ncol=8, fontsize=8.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = D / "preview" / "decay_options" / f"decay_yearly_2x2_{MET}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
plt.close(fig)
print("wrote", out)
