#!/usr/bin/env python3
"""
Build ratio-based (1:k) Temporal TargetBench val/test sets, TGB-2.0 framing.

Query = (disease d, year Y), cutoff C = Y-1, graph snapshot_C.
  positives  genes whose FIRST trial for d is in year Y
  pool       genes trialled by C for any disease sharing a canonical therapeutic
             area with d (type constraint), minus genes trialled for d by Y.
             Pair-level, time-aware: genes first trialled for d AFTER Y stay
             eligible negatives. Nothing after Y is used, and no target-level
             future filter is applied (unlike trial_entry_dataset_builder).
  negatives  k * n_pos drawn uniformly without replacement from the pool.
             A query whose pool is smaller than k * n_pos is dropped and logged,
             so every kept query is exactly 1:k.

Uniform sampling keeps AUROC an unbiased estimate of AUROC over the whole pool.
Rank metrics (MRR, P@k) depend on k and must be reported with it.

Eligible disease: canonical-TA, a graph disease node, >= 1 positive in Y.
Universe: protein-coding graph targets (src/evaluation/targetbench.py).

Output per year: <out-root>/<split>_yr<Y>/trial_entry_ratio<k>.csv holding ids,
label, first_trial_year, cutoff_year, the prompt text columns, and label-blind
stage-1 scores at C (for baselines). Also stats.json and dropped_queries.csv.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation.targetbench import (  # noqa: E402
    ROOT, Stage1, load_disease_meta, load_pairs, load_universe,
)
from src.evaluation.trial_entry_dataset_builder import attach_names, load_nodes  # noqa: E402

SCORE_COLS = ["breadth", "recent", "ta_breadth", "onto", "rrf"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=["val", "test"], required=True)
    p.add_argument("--years", type=int, nargs="+", default=None)
    p.add_argument("--ratio", type=int, default=10)
    p.add_argument("--out-root", default="data/evaluation/ratio10")
    p.add_argument("--graph", default="data/snapshots/snapshot_2015_datatype.pt")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    years = args.years or (list(range(2021, 2026)) if args.split == "test" else list(range(2017, 2021)))
    if args.split == "val":
        assert min(years) >= 2017, "val queries must be after the selection-train event range (<=2016)"
    k = args.ratio
    rng = np.random.default_rng(args.seed)

    genes, gidx, graph_diseases = load_universe(args.graph)
    d_tas, d_rel = load_disease_meta()
    pairs = load_pairs()
    pairs = pairs[pairs.targetId.isin(gidx)].copy()
    targets, diseases = load_nodes(ROOT)
    print(f"universe {len(genes):,} | split {args.split} | years {years} | 1:{k}")

    for Y in years:
        C = Y - 1
        st = Stage1(pairs, gidx, C, d_tas, d_rel)
        byY = pairs[pairs.year <= Y].groupby("diseaseId").targetId.apply(
            lambda s: s.map(gidx).to_numpy()).to_dict()
        posY = pairs[pairs.year == Y]
        posY = posY[posY.diseaseId.map(lambda d: bool(d_tas.get(d)) and d in graph_diseases)]
        pos_by = posY.groupby("diseaseId").targetId.apply(lambda s: s.map(gidx).to_numpy()).to_dict()
        rows, per_q, dropped = [], [], []
        for d in sorted(pos_by):
            pos = pos_by[d]
            sc = st.disease_scores(d)
            pool = sc["ta_breadth"] > 0
            pool[byY.get(d, np.array([], int))] = False     # pair-level: trialled for d by Y
            pool_idx = np.flatnonzero(pool)
            need = k * len(pos)
            if len(pool_idx) < need:
                dropped.append(dict(diseaseId=d, n_pos=len(pos), pool=len(pool_idx)))
                continue
            neg = rng.choice(pool_idx, size=need, replace=False)
            for idx, lab in [(pos, 1), (neg, 0)]:
                for x in idx:
                    rows.append((genes[x], d, lab, C, *[float(sc[c][x]) for c in SCORE_COLS]))
            per_q.append(dict(diseaseId=d, n_pos=len(pos), n_neg=need, pool=len(pool_idx)))
        df = pd.DataFrame(rows, columns=["targetId", "diseaseId", "label", "cutoff_year", *SCORE_COLS])
        df["first_trial_year"] = np.where(df.label == 1, Y, np.nan)
        assert not df.duplicated(["targetId", "diseaseId"]).any()
        full = attach_names(df, targets, diseases)
        assert len(full) == len(df)
        out = Path(args.out_root) / f"{args.split}_yr{Y}"
        out.mkdir(parents=True, exist_ok=True)
        full.to_csv(out / f"trial_entry_ratio{k}.csv", index=False)
        pd.DataFrame(dropped, columns=["diseaseId", "n_pos", "pool"]).to_csv(out / "dropped_queries.csv", index=False)
        q = pd.DataFrame(per_q)
        stats = dict(split=args.split, year=Y, cutoff_year=C, ratio=k, n_queries=len(q),
                     n_pos=int(q.n_pos.sum()), n_neg=int(q.n_neg.sum()), rows=len(df),
                     dropped_queries=len(dropped), dropped_pos=int(sum(x["n_pos"] for x in dropped)),
                     median_pool=int(q.pool.median()), seed=args.seed,
                     pool_def="same-canonical-TA targets trialled by C, minus pairs trialled for d by Y")
        (out / "stats.json").write_text(json.dumps(stats, indent=2))
        print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
