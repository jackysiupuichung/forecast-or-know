#!/usr/bin/env python3
"""Post-cutoff disease-macro MRR by therapeutic area, four arms (A1, A2, A2+bio,
A3) in a 2x2. Each panel: one grouped bar per model per area (6 models), a black
diamond for the cross-model mean, and a dotted line connecting the means across
areas. Error bars are SEM of the per-disease-window MRR for that model x area.
Shared TA order (by post-cutoff support) and y-axis. Saved to the paper figures
dir."""
import glob, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_outputs import records  # noqa: E402
from src.paths import FIGURES, OT_DIR  # noqa: E402
from src.evaluation.fig_style import MODEL_COLORS, clean_axes, paper_style, FS, LW  # noqa: E402
FIGURES.mkdir(parents=True, exist_ok=True)
OUT = str(FIGURES / "per_ta_post_mrr.png")
DIS_PARQUET = str(OT_DIR / "diseases")
# model slug -> (display name, fractional cutoff, colour)
MODELS = {
 "llama3.1-8b":  ("Llama 3.1-8B",  2024.0, MODEL_COLORS["llama3.1-8b"]),
 "llama3.1-70b": ("Llama 3.1-70B", 2024.0, MODEL_COLORS["llama3.1-70b"]),
 "qwen3-8b":     ("Qwen 3-8B",     2024.5, MODEL_COLORS["qwen3-8b"]),
 "qwen3-32b":    ("Qwen 3-32B",    2024.5, MODEL_COLORS["qwen3-32b"]),
 "gemma-2-9b":   ("Gemma 2-9B",    2024.5, MODEL_COLORS["gemma-2-9b"]),
 "gemma-2-27b":  ("Gemma 2-27B",   2024.5, MODEL_COLORS["gemma-2-27b"]),
}
WIN = [2021, 2022, 2023, 2024, 2025, 2026]
CANONICAL = {
 "MONDO_0045024":"cancer","EFO_0000319":"cardiovascular","EFO_0001379":"endocrine",
 "EFO_0010282":"GI","OTAR_0000018":"genetic","EFO_0005803":"hematologic",
 "EFO_0000540":"immune","EFO_0010285":"integumentary","OTAR_0000006":"musculoskeletal",
 "EFO_0000618":"nervous","OTAR_0000017":"reproductive","OTAR_0000010":"respiratory",
 "EFO_0009690":"urinary"}
# (title, results-dir, filename-tag)
ARMS = [("A1 (no retrieval)", "a1", "A1"), ("A2 (dated graph)", "a2", "A2"),
        ("A3 (newest graph)", "a3", "A3")]   # A2+bio removed from the paper (2026-10-26)


def load_ta_map():
    df = pd.concat([pd.read_parquet(f, columns=["id", "therapeuticAreas"])
                    for f in glob.glob(f"{DIS_PARQUET}/*.parquet")], ignore_index=True)
    m = {}
    for _, r in df.iterrows():
        tas = r["therapeuticAreas"]
        if tas is None:
            continue
        for t in list(tas):
            if t in CANONICAL:
                m[r["id"]] = CANONICAL[t]; break
    return m


def collect(root, tag, ta_map):
    """{ta: {slug: [post-cutoff per-disease MRRs]}}."""
    acc = {}
    for slug, (_, cut, _) in MODELS.items():
        for y in WIN:
            if y < cut:                      # post-cutoff only
                continue
            for d in records(slug, tag, y):
                mrr = (d.get("metrics") or {}).get("mrr")
                ta = ta_map.get(d.get("disease_id"))
                if mrr is None or ta is None:
                    continue
                acc.setdefault(ta, {}).setdefault(slug, []).append(float(mrr))
    return acc


def shorten(ta):
    return (ta.replace(" disease", "").replace(" or thoracic", "")
            .replace(" or connective tissue", "").replace(", familial or congenital", "")
            .replace(" system", "").replace(" or other complication", "")
            .replace(" or benign tumor", "").replace(" or breast", ""))


def main():
    ta_map = load_ta_map()
    data = {tag: collect(root, tag, ta_map) for _, root, tag in ARMS}
    # TA order + inclusion: enough post-cutoff support in A2 (>=10 disease-windows)
    a2 = data["A2"]
    order = [ta for ta, per in sorted(
                a2.items(), key=lambda kv: -sum(len(v) for v in kv[1].values()))
             if sum(len(v) for v in per.values()) >= 10]
    labels = [shorten(t) for t in order]
    slugs = list(MODELS)
    nm = len(slugs)
    x = np.arange(len(order))
    bw = 0.8 / nm                            # bar width within each area group
    ymax = 0.0
    for tag in data:
        for ta in order:
            for slug in slugs:
                v = data[tag].get(ta, {}).get(slug, [])
                if v:
                    ymax = max(ymax, np.mean(v) + (np.std(v) / max(1, len(v) ** 0.5)))

    paper_style()
    fig, axes = plt.subplots(3, 1, figsize=(5.5, 5.4), sharey=True, sharex=True)
    for ax, (title, root, tag) in zip(axes.ravel(), ARMS):
        acc = data[tag]
        means_by_area = []
        for j, slug in enumerate(slugs):
            _, _, colour = MODELS[slug]
            offs = (j - (nm - 1) / 2) * bw
            m = [np.mean(acc.get(t, {}).get(slug, []) or [np.nan]) for t in order]
            se = [(np.std(acc.get(t, {}).get(slug, [])) /
                   max(1, len(acc.get(t, {}).get(slug, [])) ** 0.5))
                  if acc.get(t, {}).get(slug) else 0 for t in order]
            ax.bar(x + offs, m, bw, yerr=se, color=colour, ecolor="0.35",
                   error_kw={"lw": LW["err"]}, label=MODELS[slug][0])
        # cross-model mean diamond + dotted connector
        for t in order:
            per_model = [np.mean(acc[t][s]) for s in slugs if acc.get(t, {}).get(s)]
            means_by_area.append(np.mean(per_model) if per_model else np.nan)
        print(title, {shorten(t): round(v, 3) for t, v in zip(order, means_by_area)}, flush=True)
        ax.plot(x, means_by_area, ls=":", color="black", lw=LW["data"], zorder=5)
        ax.scatter(x, means_by_area, marker="D", s=10, color="black", zorder=6)
        ax.axhline(0.5, ls="--", lw=LW["ref"], color="0.6", zorder=0)
        ax.set_title(title, fontsize=FS["title"], pad=2)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=FS["tick"])
        ax.set_ylim(0, ymax * 1.12); ax.margins(x=0.01)
        clean_axes(ax)
    for ax in axes:
        ax.set_ylabel("post-cutoff MRR")
    axes[0].legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.12), fontsize=FS["legend"], ncol=3,
                   handlelength=1.2, columnspacing=1.0)
    fig.tight_layout()
    fig.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("Wrote", OUT, "| TAs:", len(order))


if __name__ == "__main__":
    main()
