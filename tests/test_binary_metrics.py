"""Tests for the calibration / answer-bias diagnostics added for the CUSP-style
Task 1 binary-probe figures (calibration_curve_bins, answer_bias_by_truth) and
their wiring into aggregate_binary_metrics."""

from __future__ import annotations

import math

from src.evaluation.binary_metrics import (
    aggregate_binary_metrics,
    answer_bias_by_truth,
    calibration_curve_bins,
    ece,
)


def _rec(pairs):
    """Build a per-disease record from (label, p_yes) pairs."""
    cs = [{"target_id": f"t{i}", "label": lab, "p_yes": p}
          for i, (lab, p) in enumerate(pairs)]
    return {"disease_id": "d0", "n_parse_ok": len(cs), "candidate_scores": cs}


# ── calibration_curve_bins ──────────────────────────────────────────────────

def test_calibration_bins_perfect_calibration():
    # Predicted prob equals empirical frequency in each bin => ECE ~ 0.
    # Low bin: 20 pairs at p=0.05, exactly 1 positive (5%).
    # High bin: 20 pairs at p=0.95, exactly 19 positives (95%).
    y = [1] + [0] * 19 + [0] + [1] * 19
    p = [0.05] * 20 + [0.95] * 20
    bins = calibration_curve_bins(y, p, n_bins=10)
    assert len(bins) == 2
    low, high = bins[0], bins[1]
    assert math.isclose(low["p_mean"], 0.05) and math.isclose(low["y_mean"], 0.05)
    assert math.isclose(high["p_mean"], 0.95) and math.isclose(high["y_mean"], 0.95)
    # frac sums to 1 over non-empty bins
    assert math.isclose(sum(b["frac"] for b in bins), 1.0)
    # perfectly calibrated -> ECE ~ 0
    assert ece(y, p, n_bins=10) < 1e-9


def test_calibration_miscalibration_shows_in_bins():
    # Overconfident: p=0.95 but only 50% actually positive -> ECE ~ 0.45.
    y = [1, 0] * 10
    p = [0.95] * 20
    bins = calibration_curve_bins(y, p, n_bins=10)
    assert len(bins) == 1
    assert math.isclose(bins[0]["p_mean"], 0.95)
    assert math.isclose(bins[0]["y_mean"], 0.5)
    assert math.isclose(ece(y, p, n_bins=10), 0.45, abs_tol=1e-9)


def test_calibration_bins_skips_empty_and_covers_all():
    y = [0, 1, 0, 1]
    p = [0.1, 0.2, 0.9, 0.85]
    bins = calibration_curve_bins(y, p, n_bins=10)
    # only two occupied bins ([0.1,0.2) and [0.8,0.9]/[0.9,1.0])
    assert all(b["count"] > 0 for b in bins)
    assert sum(b["count"] for b in bins) == 4


def test_calibration_bins_empty_input():
    assert calibration_curve_bins([], []) == []


def test_calibration_top_edge_inclusive():
    # p == 1.0 must land in the last bin, not be dropped.
    bins = calibration_curve_bins([1, 1], [1.0, 1.0], n_bins=10)
    assert len(bins) == 1 and bins[0]["count"] == 2


# ── answer_bias_by_truth ────────────────────────────────────────────────────

def test_answer_bias_discriminating_model():
    # says Yes on all positives, No on all negatives -> gap = 1.0
    recs = [_rec([(1, 0.9), (1, 0.8), (0, 0.1), (0, 0.2)])]
    ab = answer_bias_by_truth(recs)
    assert ab["yes_rate_truth_yes"] == 1.0
    assert ab["yes_rate_truth_no"] == 0.0
    assert ab["yes_rate_gap"] == 1.0
    assert ab["n_positives"] == 2 and ab["n_negatives"] == 2


def test_answer_bias_constant_yes_model():
    # always says Yes -> gap = 0 (ignores truth), overall_yes_rate = 1
    recs = [_rec([(1, 0.99), (0, 0.99), (0, 0.7)])]
    ab = answer_bias_by_truth(recs)
    assert ab["yes_rate_truth_yes"] == 1.0
    assert ab["yes_rate_truth_no"] == 1.0
    assert ab["yes_rate_gap"] == 0.0
    assert math.isclose(ab["overall_yes_rate"], 1.0)


def test_answer_bias_threshold_boundary():
    # p == threshold counts as Yes (>=)
    recs = [_rec([(1, 0.5), (0, 0.49)])]
    ab = answer_bias_by_truth(recs, threshold=0.5)
    assert ab["yes_rate_truth_yes"] == 1.0
    assert ab["yes_rate_truth_no"] == 0.0


def test_answer_bias_ignores_unparsed():
    recs = [{"candidate_scores": [
        {"label": 1, "p_yes": None},
        {"label": 1, "p_yes": 0.9},
        {"label": 0, "p_yes": 0.1},
    ]}]
    ab = answer_bias_by_truth(recs)
    assert ab["n_pairs"] == 2


def test_answer_bias_top_token_overrides_threshold():
    # p_yes would say "No" (<0.5) but the emitted token is "Yes": with
    # use_top_token the token wins; without it the threshold wins.
    recs = [{"candidate_scores": [
        {"label": 1, "p_yes": 0.30, "top_token": "Yes"},
        {"label": 0, "p_yes": 0.80, "top_token": "No"},
    ]}]
    ab_tok = answer_bias_by_truth(recs, use_top_token=True)
    assert ab_tok["yes_rate_truth_yes"] == 1.0   # token "Yes"
    assert ab_tok["yes_rate_truth_no"] == 0.0    # token "No"
    ab_thr = answer_bias_by_truth(recs, use_top_token=False)
    assert ab_thr["yes_rate_truth_yes"] == 0.0   # p_yes 0.30 < 0.5
    assert ab_thr["yes_rate_truth_no"] == 1.0    # p_yes 0.80 >= 0.5


def test_answer_bias_top_token_falls_back_when_missing():
    # Missing top_token -> fall back to the p_yes threshold even in token mode.
    recs = [{"candidate_scores": [
        {"label": 1, "p_yes": 0.9},               # no top_token
        {"label": 0, "p_yes": 0.1, "top_token": ""},
    ]}]
    ab = answer_bias_by_truth(recs, use_top_token=True)
    assert ab["yes_rate_truth_yes"] == 1.0
    assert ab["yes_rate_truth_no"] == 0.0


def test_answer_bias_all_one_class():
    recs = [_rec([(1, 0.9), (1, 0.2)])]
    ab = answer_bias_by_truth(recs)
    assert ab["yes_rate_truth_no"] is None
    assert ab["yes_rate_gap"] is None  # cannot form the gap without negatives


# ── aggregate wiring ────────────────────────────────────────────────────────

def test_aggregate_emits_calibration_and_bias():
    recs = [_rec([(1, 0.9), (0, 0.1), (1, 0.6), (0, 0.3)])]
    agg = aggregate_binary_metrics(recs)
    assert "calibration_bins" in agg and isinstance(agg["calibration_bins"], list)
    assert agg["calibration_bins"], "expected at least one occupied bin"
    assert "answer_bias" in agg
    assert agg["answer_bias"]["n_pairs"] == 4
    # pooled metrics still present
    assert agg["pooled"]["roc_auc"] is not None
