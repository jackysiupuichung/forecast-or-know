#!/usr/bin/env python3
"""
Binary-ECE reliability figure for the workshop paper, per model, all four access
conditions overlaid. Pure re-read of the disjoint per-year caches
(candidate_scores[].p_yes / .label) pooled over the 2021--2026 windows: no new
inference.

One panel per model (2x3), each with the four arms' reliability curves
(mean p_yes vs empirical positive rate, equal-width bins) and a per-arm ECE in
the legend. The diagonal is perfect calibration; marker area is proportional to
bin count. Reuses src.evaluation.binary_metrics.{ece,calibration_curve_bins}.

Output: OUTDIR/binary_ece_reliability.png (+ binary_ece.csv with per-(model,arm)
scalar ECE / Brier / n).
"""
import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_outputs import records  # noqa: E402
from src.paths import FIGURES  # noqa: E402
from src.evaluation.binary_metrics import ece, brier, calibration_curve_bins  # noqa: E402
from src.evaluation.fig_style import clean_axes, paper_style, FS, LW  # noqa: E402

WIN = range(2021, 2027)
# (arm label, results subdir, cache tag, colour)
# condition colours follow Fig. 1: purple = parametric (A1), blue = dated graph (A2), red = newest graph (A3)
ARMS = [("A1", "a1", "A1", "#7A6E96"),
        ("A2", "a2", "A2", "#2E6E8E"),
        ("A3", "a3", "A3", "#B5544B")]
MODELS = {"Llama 3.1-8B": "llama3.1-8b",
          "Llama 3.1-70B": "llama3.1-70b",
          "Qwen 3-8B": "qwen3-8b",
          "Qwen 3-32B": "qwen3-32b",
          "Gemma 2-9B": "gemma-2-9b",
          "Gemma 2-27B": "gemma-2-27b"}
N_BINS = int(os.environ.get("N_BINS", "10"))
MIN_BIN = int(os.environ.get("MIN_BIN", "15"))  # min candidates to plot a bin
OUTDIR = str(FIGURES)


def load_arm(slug, subdir, tag):
    """Pooled (y_true, p_yes) over all windows for one (model, arm)."""
    y, p = [], []
    for w in WIN:
        for d in records(slug, tag, w):
            for c in d.get("candidate_scores", []):
                if c.get("p_yes") is None:
                    continue
                y.append(int(c["label"]))
                p.append(float(c["p_yes"]))
    return np.array(y), np.array(p)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    rows = []
    paper_style()
    fig, axes = plt.subplots(2, 3, figsize=(5.5, 3.9), sharex=True, sharey=True)
    for ax, (mname, slug) in zip(axes.ravel(), MODELS.items()):
        ax.plot([0, 1], [0, 1], ls="--", lw=LW["ref"], color="grey", zorder=0)
        for label, subdir, tag, colour in ARMS:
            yt, pp = load_arm(slug, subdir, tag)
            if len(yt) == 0:
                continue
            e = ece(yt, pp, N_BINS)
            b = brier(yt, pp)
            rows.append([mname, label, len(yt), f"{e:.4f}", f"{b:.4f}"])
            # Plot only bins with enough support so a 1-candidate bin does not
            # distort the curve; ECE above is still computed on all candidates.
            bins = [d for d in calibration_curve_bins(yt, pp, N_BINS)
                    if d["count"] >= MIN_BIN]
            xs = [d["p_mean"] for d in bins]
            ys = [d["y_mean"] for d in bins]
            sz = [max(2, 60 * d["count"] / len(yt)) for d in bins]
            ax.plot(xs, ys, "-", color=colour, lw=LW["data"], zorder=2)
            ax.scatter(xs, ys, s=sz, color=colour, alpha=0.7, zorder=3,
                       label=f"{label} (ECE {e:.2f})")
        ax.set_title(mname, fontsize=FS["title"], pad=2)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.36); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        clean_axes(ax, grid_axis="both")
        ax.legend(frameon=False, fontsize=FS["legend"], loc="upper left", handletextpad=0.2, borderaxespad=0.2)
    for ax in axes[:, 0]:
        ax.set_ylabel("empirical positive rate")
    for ax in axes[-1, :]:
        ax.set_xlabel("mean predicted $p_\\mathrm{yes}$")
    fig.tight_layout()
    out = os.path.join(OUTDIR, "binary_ece_reliability.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("Wrote", out)
    with open(os.path.join(OUTDIR, "binary_ece.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "arm", "n", "ece", "brier"])
        w.writerows(rows)
    print("Wrote", os.path.join(OUTDIR, "binary_ece.csv"))
    # console summary
    for r in rows:
        print(f"  {r[0]:14s} {r[1]:7s} n={r[2]:>6} ECE={r[3]} Brier={r[4]}")


if __name__ == "__main__":
    main()
