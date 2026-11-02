"""Binary-classification metrics for the Task 1 logprob-scoring pipeline.

ROC-AUC, PR-AUC, Brier score, and ECE — all computed both globally (over
the full pooled set of (target, disease) pairs) and per-disease (with a
bootstrap CI across diseases). The per-disease aggregation matches the
existing rubric pipeline so headline numbers are directly comparable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .bootstrap import bootstrap_mean_ci


def roc_auc(y_true: Sequence[int], p: Sequence[float]) -> Optional[float]:
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y_true, dtype=int)
    if y.sum() == 0 or y.sum() == len(y):
        return None
    return float(roc_auc_score(y, np.asarray(p, dtype=float)))


def pr_auc(y_true: Sequence[int], p: Sequence[float]) -> Optional[float]:
    from sklearn.metrics import average_precision_score
    y = np.asarray(y_true, dtype=int)
    if y.sum() == 0:
        return None
    return float(average_precision_score(y, np.asarray(p, dtype=float)))


def brier(y_true: Sequence[int], p: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=float)
    pp = np.asarray(p, dtype=float)
    return float(np.mean((pp - y) ** 2))


def ece(y_true: Sequence[int], p: Sequence[float], n_bins: int = 10) -> float:
    """Expected Calibration Error with equal-width bins on [0, 1]."""
    y = np.asarray(y_true, dtype=float)
    pp = np.asarray(p, dtype=float)
    if len(y) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    n = len(y)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (pp >= lo) & (pp < hi if i < n_bins - 1 else pp <= hi)
        if not mask.any():
            continue
        bin_p = float(pp[mask].mean())
        bin_y = float(y[mask].mean())
        total += (mask.sum() / n) * abs(bin_p - bin_y)
    return float(total)


def _per_disease_pairs(records: List[Dict[str, Any]]) -> List[Tuple[List[int], List[float]]]:
    """Extract (labels, p_yes) per disease for records with usable predictions."""
    out: List[Tuple[List[int], List[float]]] = []
    for r in records:
        ys: List[int] = []
        ps: List[float] = []
        for c in r.get("candidate_scores", []):
            if c.get("p_yes") is None:
                continue
            ys.append(int(c["label"]))
            ps.append(float(c["p_yes"]))
        if ys and ps:
            out.append((ys, ps))
    return out


def aggregate_binary_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute pooled and per-disease bootstrap-CI binary metrics.

    Pooled: one number across all (pair, p_yes, label) triples in the run.
    Per-disease: compute the metric for each disease separately, then bootstrap
                 the mean across diseases.
    """
    n_total = len(records)
    parseable = [r for r in records if r.get("n_parse_ok", 0) > 0]
    n_signal = len(parseable)

    pooled_y: List[int] = []
    pooled_p: List[float] = []
    for r in parseable:
        for c in r.get("candidate_scores", []):
            if c.get("p_yes") is None:
                continue
            pooled_y.append(int(c["label"]))
            pooled_p.append(float(c["p_yes"]))

    pooled = {
        "roc_auc": roc_auc(pooled_y, pooled_p) if pooled_y else None,
        "pr_auc": pr_auc(pooled_y, pooled_p) if pooled_y else None,
        "brier": brier(pooled_y, pooled_p) if pooled_y else None,
        "ece": ece(pooled_y, pooled_p) if pooled_y else None,
        "n_pairs": len(pooled_y),
        "n_positives": int(sum(pooled_y)),
    }

    per_disease = _per_disease_pairs(parseable)
    per_disease_metrics = {
        "roc_auc": [roc_auc(y, p) for y, p in per_disease],
        "pr_auc": [pr_auc(y, p) for y, p in per_disease],
        "brier": [brier(y, p) for y, p in per_disease],
        "ece": [ece(y, p) for y, p in per_disease],
    }
    bootstrapped = {
        name: bootstrap_mean_ci(values)
        for name, values in per_disease_metrics.items()
    }

    return {
        "n_diseases": n_total,
        "n_diseases_with_signal": n_signal,
        "n_diseases_no_signal": n_total - n_signal,
        "coverage": (n_signal / n_total) if n_total else 0.0,
        "pooled": pooled,
        "per_disease_macro": bootstrapped,
        # Calibration / answer-bias diagnostics (CUSP Fig 4a/4c). Pooled over all
        # (pair, p_yes, label) triples so the reliability curve and the answer
        # bias reflect the whole run, not a per-disease macro-average.
        "calibration_bins": calibration_curve_bins(pooled_y, pooled_p) if pooled_y else [],
        "answer_bias": answer_bias_by_truth(parseable),
    }


def calibration_curve_bins(
    y_true: Sequence[int], p: Sequence[float], n_bins: int = 10
) -> List[Dict[str, float]]:
    """Reliability-curve bins for a P(yes)-vs-empirical-frequency plot.

    Equal-width bins on [0, 1], matching `ece()`'s binning so the curve and the
    scalar ECE are consistent. Returns one dict per NON-EMPTY bin with:
        p_mean   — mean predicted P(yes) in the bin (x-coordinate)
        y_mean   — empirical positive rate in the bin (y-coordinate)
        count    — number of pairs in the bin
        frac     — count / N (bin weight; the ECE weight)
        bin_lo, bin_hi — bin edges

    This is the data behind CUSP Fig 4c (calibration curves). Perfect
    calibration is the diagonal p_mean == y_mean.
    """
    y = np.asarray(y_true, dtype=float)
    pp = np.asarray(p, dtype=float)
    n = len(y)
    if n == 0:
        return []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: List[Dict[str, float]] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (pp >= lo) & (pp < hi if i < n_bins - 1 else pp <= hi)
        c = int(mask.sum())
        if c == 0:
            continue
        bins.append({
            "bin_lo": float(lo),
            "bin_hi": float(hi),
            "p_mean": float(pp[mask].mean()),
            "y_mean": float(y[mask].mean()),
            "count": c,
            "frac": c / n,
        })
    return bins


def answer_bias_by_truth(
    records: List[Dict[str, Any]], threshold: float = 0.5, use_top_token: bool = False
) -> Dict[str, Any]:
    """Yes/No answer share split by ground-truth label — CUSP Fig 4a.

    Recovers the model's hard Yes/No answer and reports, separately for truth=Yes
    and truth=No pairs, the fraction the model answered "Yes". A well-behaved
    model answers "Yes" more often when the truth is Yes; a model that answers
    "Yes" (or "No") almost regardless of the truth exposes an answer-distribution
    bias that a shrinking pool / base-rate shift can otherwise disguise.

    Hard-answer source:
        use_top_token=False (default) — threshold P(yes) at `threshold`. Works
            whenever p_yes is present.
        use_top_token=True — use the greedy `top_token` the model actually emitted
            ("Yes"/"No"), matching scripts/plot_cusp_fig4_diagnostics.py. Falls
            back to the p_yes threshold for any candidate missing a top_token, so
            the two figure families report the same number on the same run.

    Also returns `overall_yes_rate` (the model's marginal propensity to say Yes)
    and `yes_rate_gap = yes_rate[truth=Yes] - yes_rate[truth=No]`, a single
    discriminability scalar (0 == answers ignore the truth).
    """
    yes_when_pos = 0
    n_pos = 0
    yes_when_neg = 0
    n_neg = 0
    for r in records:
        for c in r.get("candidate_scores", []):
            if c.get("p_yes") is None:
                continue
            tok = (c.get("top_token") or "").strip().lower() if use_top_token else ""
            if tok:
                said_yes = tok.startswith("yes")
            else:
                said_yes = float(c["p_yes"]) >= threshold
            if int(c["label"]) == 1:
                n_pos += 1
                yes_when_pos += int(said_yes)
            else:
                n_neg += 1
                yes_when_neg += int(said_yes)
    n_total = n_pos + n_neg
    yr_pos = (yes_when_pos / n_pos) if n_pos else None
    yr_neg = (yes_when_neg / n_neg) if n_neg else None
    return {
        "threshold": threshold,
        "n_pairs": n_total,
        "n_positives": n_pos,
        "n_negatives": n_neg,
        "yes_rate_truth_yes": yr_pos,
        "yes_rate_truth_no": yr_neg,
        "yes_rate_gap": (yr_pos - yr_neg) if (yr_pos is not None and yr_neg is not None) else None,
        "overall_yes_rate": ((yes_when_pos + yes_when_neg) / n_total) if n_total else None,
    }


def distributional_diagnostics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Quick sanity stats on the P(yes) distribution: spread, distinct values."""
    ps: List[float] = []
    for r in records:
        for c in r.get("candidate_scores", []):
            if c.get("p_yes") is not None:
                ps.append(float(c["p_yes"]))
    if not ps:
        return {"n": 0}
    arr = np.asarray(ps, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "stdev": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n_distinct_2dp": int(len({round(p, 2) for p in ps})),
        "n_distinct_4dp": int(len({round(p, 4) for p in ps})),
    }
