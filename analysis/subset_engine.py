"""Per-query metrics from scores_long.parquet (build_scores_long.py), optionally on subsets of positives.

A subset keeps every negative and only the positives selected by a mask, so the question per query
stays "rank these positives above the same negatives". Queries left with no positive are dropped.
Metrics match analysis/per_query.py on the full lists: AUROC (ties count one half), AP and MRR with ties
broken against positives, plus n_pos / n_cand for the query-size adjustment."""
from pathlib import Path

import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import AGG_DIR as D  # noqa: E402
K = ["diseaseId", "year"]
MODELS = ["llama3.1-8b", "llama3.1-70b", "qwen3-8b", "qwen3-32b", "gemma-2-9b", "gemma-2-27b"]
POSTY = {"llama3.1-8b": 2024, "llama3.1-70b": 2024}          # first post-cutoff query year; others 2025


def load_long():
    long = pd.read_parquet(D / "scores_long.parquet")
    for c in ["series", "arm", "diseaseId", "targetId"]:
        long[c] = long[c].astype(str)
    return long


def _query(lab, s):
    npos, n = lab.sum(), len(lab)
    r = rankdata(s)                                   # average ranks for ties
    auroc = (r[lab == 1].sum() - npos * (npos + 1) / 2) / (npos * (n - npos))
    order = np.lexsort((lab, -s)); ls = lab[order]; hits = np.flatnonzero(ls)
    return auroc, np.mean(np.arange(1, npos + 1) / (hits + 1)), 1.0 / (hits[0] + 1), npos, n


def query_metrics(long, keep_pos=None):
    """keep_pos: boolean Series aligned with `long` (True = keep this positive); negatives are always kept."""
    df = long if keep_pos is None else long[(long.label == 0) | keep_pos.to_numpy()]
    rows = []
    for (se, arm, Y, d), g in df.groupby(["series", "arm", "year", "diseaseId"], sort=False, observed=True):
        lab = g.label.to_numpy(); s = g.score.to_numpy(float)
        if 0 < lab.sum() < len(lab):
            a, ap, mrr, npos, n = _query(lab, s)
            rows.append((se, arm, Y, d, a, ap, mrr, npos, n))
    q = pd.DataFrame(rows, columns=["series", "arm", "year", "diseaseId", "auroc", "ap", "mrr", "n_pos", "n_cand"])
    q["post"] = [y >= POSTY.get(s, 2025) for s, y in zip(q.series, q.year)]
    return q


def boot_ci(v, n=2000, seed=0):
    v = np.asarray(v, float); rng = np.random.default_rng(seed)
    b = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return v.mean(), lo, hi


def paired(q, a, b, metric, series=None, side=None, base_series=None):
    """Per-query paired difference arm a minus arm b (or minus a heuristic when base_series is set)."""
    sel = q if series is None else q[q.series.isin(series)]
    if side is not None:
        sel = sel[sel.post == (side == "post")]
    x = sel[sel.arm == a].set_index(["series"] + K)[metric]
    if base_series is not None:
        y = q[(q.series == base_series)].set_index(K)[metric]
        x = x.reset_index().set_index(K)
        return (x[metric] - y.reindex(x.index)).dropna()
    y = sel[sel.arm == b].set_index(["series"] + K)[metric]
    return (x - y).dropna()


def fmt(t, d=3):
    m, lo, hi = t
    return f"{m:+.{d}f}" + ("*" if lo > 0 or hi < 0 else "")
