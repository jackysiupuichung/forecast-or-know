"""Recompute the per-query metrics from scores_long.parquet with subset_engine, save them as q_full.parquet
(used by r6_r8_checks.py), and check they match analysis/per_query.py.

    python analysis/validate_engine.py   -> $AGG_DIR/q_full.parquet"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from camera_data import load  # noqa: E402
from subset_engine import load_long, query_metrics  # noqa: E402
from src.paths import AGG_DIR  # noqa: E402

t = time.time(); long = load_long(); q = query_metrics(long)
print("engine time %.0fs, queries %d" % (time.time() - t, len(q)), flush=True)
q.to_parquet(AGG_DIR / "q_full.parquet", index=False)
ref = load(ids=True)
m = q.merge(ref, on=["series", "arm", "diseaseId", "year"], suffixes=("", "_ref"))
print("matched rows", len(m), "of", len(ref))
for c in ["auroc", "ap", "mrr"]:
    d = float(np.abs(m[c] - m[c + "_ref"]).max())
    print(c, "max |diff|", d)
    assert d < 1e-9, c
