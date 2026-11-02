"""Camera-ready main-text Fig. 2: gain from the dated graph (A2-A1) and from the newest graph (A3-A2) per model,
before and after each model's training cutoff, in AP (top) and AUROC (bottom). Per-query paired differences,
95% CIs from 2,000 query resamples. No A2+bio (appendix-only). Sized for the page (text width ~5.5 in).

    python analysis/plot_gaps_main.py   -> $OUTDIR/figures/camera_gaps_2bar.png (+ .csv)
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluation.fig_style import MODEL_COLORS, clean_axes, paper_style, FS, LW  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from camera_data import load  # noqa: E402
from src.paths import FIGURES as OUT  # noqa: E402
OUT.mkdir(parents=True, exist_ok=True)
K = ["diseaseId", "year"]
MODELS = [("Llama 3.1-8B", "llama3.1-8b"), ("Llama 3.1-70B", "llama3.1-70b"), ("Qwen 3-8B", "qwen3-8b"),
          ("Qwen 3-32B", "qwen3-32b"), ("Gemma 2-9B", "gemma-2-9b"), ("Gemma 2-27B", "gemma-2-27b")]
POSTY = {"llama3.1-8b": 2024, "llama3.1-70b": 2024}
# condition colours follow Fig. 1 (blue = decision-time KG evidence, red = post-decision KG evidence),
# kept apart from the per-model palette so no colour means both a model and a condition
# CUSP's decomposition: knowledge gap (pre-cutoff information) and forecasting gap (full minus pre-cutoff)
GAPS = [("A2", "A1", r"knowledge gap, $\Delta_\mathrm{know}$ (A2$-$A1)", "#2E6E8E"),
        ("A3", "A2", r"forecasting gap, $\Delta_\mathrm{fore}$ (A3$-$A2)", "#B5544B")]
q = load(ids=True)
q["post"] = [y >= POSTY.get(s, 2025) for s, y in zip(q.series, q.year)]
rng = np.random.default_rng(0)

def gap(slug, a, b, side, met):
    g = q[(q.series == slug) & (q.post == (side == "post"))]
    d = (g[g.arm == a].set_index(K)[met] - g[g.arm == b].set_index(K)[met]).dropna().to_numpy()
    lo, hi = np.percentile(d[rng.integers(0, len(d), (2000, len(d)))].mean(1), [2.5, 97.5])
    return d.mean(), lo, hi

paper_style()
fig, axes = plt.subplots(2, 2, figsize=(5.5, 3.5), sharey="row")
x, w = np.arange(len(MODELS)), 0.36   # bar pair per model
rows = []
for r, (met, mlab) in enumerate([("ap", "gain in AP"), ("auroc", "gain in AUROC")]):
    for c, side in enumerate(["pre", "post"]):
        ax = axes[r, c]
        for i, (a, b, lab, col) in enumerate(GAPS):
            v = np.array([gap(s, a, b, side, met) for _, s in MODELS])
            rows += [(met, side, n, lab, *vv) for (n, _), vv in zip(MODELS, v)]
            ax.bar(x + (i - 0.5) * w, v[:, 0], w, color=col, edgecolor="white", linewidth=1,
                   label=lab if (r, c) == (0, 0) else None)
            ax.errorbar(x + (i - 0.5) * w, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]],
                        fmt="none", ecolor="#444444", elinewidth=LW["err"], capsize=1.5)
        ax.axhline(0, color="#222222", lw=LW["axis"])
        ax.set_xticks(x)
        ax.set_xticklabels([n for n, _ in MODELS] if r == 1 else [], rotation=25, ha="right", fontsize=FS["tick"])
        if r == 0:
            ax.set_title(f"{side}-cutoff query years", fontsize=FS["title"], pad=3)
        if c == 0:
            ax.set_ylabel(mlab, fontsize=FS["label"], labelpad=2)
        ax.tick_params(labelsize=FS["tick"])
        clean_axes(ax)
        if r == 1:   # model names in their fixed colour, matching the per-model lines in Fig. 3
            for t, (_, s_) in zip(ax.get_xticklabels(), MODELS):
                t.set_color(MODEL_COLORS[s_])
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, ncol=2, fontsize=FS["legend"], frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=0.6, w_pad=0.8)
fig.savefig(OUT / "camera_gaps_2bar.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
import pandas as pd
pd.DataFrame(rows, columns=["metric", "side", "model", "gap", "mean", "lo", "hi"]).to_csv(
    OUT / "camera_gaps_2bar.csv", index=False, float_format="%.4f")
print("wrote", OUT / "camera_gaps_2bar.png")
