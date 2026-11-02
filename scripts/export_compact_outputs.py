#!/usr/bin/env python3
"""Compact copy of the inference outputs for the repository: every candidate's label, p_yes, Yes/No
log-probabilities and parse flag, in the original candidate order, plus each disease's runner MRR.
analysis/run_outputs.py reads it when the full run directories are absent.

    python scripts/export_compact_outputs.py   (reads results/<model>/<run>/predictions.jsonl)
        -> data/predictions/candidate_scores.parquet, data/predictions/run_metrics.parquet
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
from run_outputs import MODELS, PRED, RUN, YEARS, run_file  # noqa: E402

cand, met = [], []
for m in MODELS:
    for arm in RUN:
        for Y in YEARS:
            f = run_file(m, arm, Y)
            for line in open(f):
                r = json.loads(line)
                d = r["disease_id"]
                met.append((m, arm, Y, d, (r.get("metrics") or {}).get("mrr")))
                for c in r["candidate_scores"]:
                    cand.append((m, arm, Y, d, c["target_id"], int(c["label"]), c.get("p_yes"),
                                 c.get("logprob_yes"), c.get("logprob_no"), bool(c.get("parse_ok"))))
        print(m, arm, flush=True)
C = pd.DataFrame(cand, columns=["model", "arm", "year", "disease_id", "target_id", "label", "p_yes",
                                "logprob_yes", "logprob_no", "parse_ok"])
M = pd.DataFrame(met, columns=["model", "arm", "year", "disease_id", "mrr"])
for df in (C, M):
    for col in ["model", "arm", "disease_id"] + (["target_id"] if "target_id" in df else []):
        df[col] = df[col].astype("category")
    df["year"] = df["year"].astype("int16")
C["label"] = C["label"].astype("int8")
PRED.mkdir(parents=True, exist_ok=True)
C.to_parquet(PRED / "candidate_scores.parquet", index=False, compression="zstd")
M.to_parquet(PRED / "run_metrics.parquet", index=False, compression="zstd")
print(len(C), "candidates;", len(M), "runs x diseases")
