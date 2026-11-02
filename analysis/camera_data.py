"""Shared loader: per-query metrics on the full time-aware 1:10 test sets (analysis/per_query.py).

Columns: series (model slug or ta_breadth), arm (A0-A3, or baseline), diseaseId, year, n_pos,
n_cand (list length) and the metrics auroc, mrr, hit1, ndcg3, p3, p5, ap, rprec.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import AGG_DIR  # noqa: E402


def load(ids=False):
    """All per-query rows; A0 (identifiers only) rows are included only when ids=True."""
    q = pd.read_csv(AGG_DIR / "per_query.csv")
    return q if ids else q[q.arm != "A0"].reset_index(drop=True)
