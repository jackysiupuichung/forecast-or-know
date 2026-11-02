#!/usr/bin/env python3
"""Dataset-description plots for the time-aware 1:10 test sets (App. C). Positives are read from the
scored A1 outputs of Llama 3.1-8B (ground truth);
therapeutic area is resolved from the OpenTargets 26.03 disease parquet
(therapeuticAreas -> names, restricted to the 13 canonical TAs), giving 100%
coverage. First-trial dates from the raw clinical_precedence parquets.

Three plots into $OUTDIR/figures:
  dataset_ta_counts   unique diseases + unique targets per TA
  dataset_ta_reuse    target-reuse index per TA (pairs / unique targets)
  dataset_time_ta     positives per year, stacked by TA
"""
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
OUT = str(FIGURES); FIGURES.mkdir(parents=True, exist_ok=True)
from src.evaluation.fig_style import paper_style, clean_axes, FS, LW, NEUTRAL  # noqa: E402
DIS_PARQUET = str(OT_DIR / "diseases")
CANONICAL = {  # id -> short name (13 canonical TAs, from trial_entry_dataset_builder)
 "MONDO_0045024":"cancer","EFO_0000319":"cardiovascular","EFO_0001379":"endocrine",
 "EFO_0010282":"GI","OTAR_0000018":"genetic","EFO_0005803":"hematologic",
 "EFO_0000540":"immune","EFO_0010285":"integumentary","OTAR_0000006":"musculoskeletal",
 "EFO_0000618":"nervous","OTAR_0000017":"reproductive","OTAR_0000010":"respiratory",
 "EFO_0009690":"urinary"}

def ta_map():
    df = pd.concat([pd.read_parquet(f, columns=["id","therapeuticAreas"])
                    for f in glob.glob(f"{DIS_PARQUET}/*.parquet")], ignore_index=True)
    m = {}
    for _, r in df.iterrows():
        tas = r["therapeuticAreas"]
        if tas is None: continue
        for t in list(tas):
            if t in CANONICAL:            # first canonical TA hit = the disease's major TA
                m[r["id"]] = CANONICAL[t]; break
    return m

def positives():
    """(disease, target, year) scored positives from the A1 outputs."""
    rows = []
    for y in range(2021, 2027):
      for d in records("llama3.1-8b", "A1", y):
        for c in d["candidate_scores"]:
            if c["label"] == 1:
                rows.append((d["disease_id"], c["target_id"], y))
    return pd.DataFrame(rows, columns=["disease","target","year"])

CP_DIR = str(OT_DIR / "sourceId=clinical_precedence")
def first_trial_dates():
    """(disease,target) -> earliest resolvedTrialDate."""
    df = pd.concat([pd.read_parquet(f, columns=["targetId","diseaseId","resolvedTrialDate"])
                    for f in glob.glob(f"{CP_DIR}/*.parquet")], ignore_index=True)
    df["resolvedTrialDate"] = pd.to_datetime(df["resolvedTrialDate"], errors="coerce")
    df = df.dropna(subset=["resolvedTrialDate"])
    return df.groupby(["diseaseId","targetId"])["resolvedTrialDate"].min()

def main():
    tam = ta_map()
    pos = positives()
    pos["ta"] = pos["disease"].map(tam)
    cov = pos["ta"].notna().mean()
    print(f"positives: {len(pos)} | TA coverage: {cov:.0%}")
    pos = pos.dropna(subset=["ta"])
    ORDER = sorted(pos["ta"].unique(), key=lambda t: (-pos[pos.ta==t]["disease"].nunique(), t))  # ties: alphabetical

    # Three separate plots (combined into one figure in LaTeX):
    #   dataset_ta_counts, dataset_ta_reuse, dataset_time_ta
    g = pos.groupby("ta").agg(dis=("disease","nunique"), tgt=("target","nunique")).reindex(ORDER)
    reuse = pos.groupby("ta")[["target"]].apply(lambda d: len(d)/d["target"].nunique()).reindex(ORDER)
    y = np.arange(len(ORDER)); h = 0.4

    # (a) unique diseases + targets per area, values annotated
    paper_style()
    fig, ax = plt.subplots(figsize=(2.7, 3.0))
    ax.barh(y+h/2, g["dis"], h, label="unique diseases", color=NEUTRAL["dark"])
    ax.barh(y-h/2, g["tgt"], h, label="unique targets", color=NEUTRAL["light"])
    for yi, v in zip(y+h/2, g["dis"]): ax.text(v+3, yi, str(int(v)), va="center", fontsize=FS["annot"])
    for yi, v in zip(y-h/2, g["tgt"]): ax.text(v+3, yi, str(int(v)), va="center", fontsize=FS["annot"])
    ax.set_yticks(y); ax.set_yticklabels(ORDER); ax.invert_yaxis()
    ax.set_xlabel("unique count per area")
    ax.set_xlim(0, g[["dis","tgt"]].values.max()*1.18); ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2, handlelength=1.0, columnspacing=0.8)
    clean_axes(ax, grid_axis=None)
    fig.tight_layout(); fig.savefig(f"{OUT}/dataset_ta_counts.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close(fig)

    # (b) target reuse per area, values annotated
    fig, ax = plt.subplots(figsize=(2.7, 3.0))
    ax.barh(y, reuse.values, color=NEUTRAL["dark"])
    for yi, v in zip(y, reuse.values): ax.text(v+0.03, yi, f"{v:.1f}", va="center", fontsize=FS["annot"])
    ax.set_yticks(y); ax.set_yticklabels(ORDER); ax.invert_yaxis()
    ax.set_xlabel("positive pairs per target")
    ax.set_xlim(0, reuse.max()*1.18)
    clean_axes(ax, grid_axis=None)
    fig.tight_layout(); fig.savefig(f"{OUT}/dataset_ta_reuse.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close(fig)

    # (c) cumulative positives per TA by EXACT trial-entry date (2021-2026)
    dates = first_trial_dates()
    pos["date"] = pos.set_index(["disease","target"]).index.map(dates)
    d = pos.dropna(subset=["date"]).copy()
    d = d[(d["date"] >= "2021-01-01") & (d["date"] < "2027-01-01")]
    fig, ax = plt.subplots(figsize=(4.95, 2.6))
    cmap = plt.get_cmap("tab20")
    for i, ta in enumerate(ORDER):
        s = np.sort(d[d["ta"] == ta]["date"].values)
        if len(s) == 0: continue
        ax.step(s, np.arange(1, len(s) + 1), where="post", color=cmap(i), lw=LW["data"], label=ta)
    # bold cumulative TOTAL across all areas
    st = np.sort(d["date"].values)
    ax.step(st, np.arange(1, len(st) + 1), where="post", color="black", lw=LW["trend"],
            label=f"all areas ({len(st):,})")
    ax.set_yscale("log")
    ax.set_xlabel("trial-entry date"); ax.set_ylabel("cumulative positives (log)")
    ax.legend(frameon=False, fontsize=FS["legend"], title="therapeutic area", title_fontsize=FS["legend"],
              loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=1, handlelength=1.2, labelspacing=0.25)
    clean_axes(ax, grid_axis=None)
    ax.margins(x=0.01)
    fig.tight_layout(); fig.savefig(f"{OUT}/dataset_time_ta.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close(fig)

    print("Wrote 3 plots to", OUT)
    print("Wrote dataset_ta_counts, dataset_ta_reuse, dataset_time_ta")

if __name__ == "__main__":
    main()
