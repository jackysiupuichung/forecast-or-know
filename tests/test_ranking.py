"""Tests for src/evaluation/ranking.py — symbol↔ID resolution + ranking metrics."""

from __future__ import annotations

import math

import pytest

from src.evaluation.ranking import (
    canonical_rankings,
    first_positive_rank,
    hit_at_k,
    mrr,
    ndcg_at_k,
    recall_at_k,
    resolve_to_id,
)


# Reuse the smoke-test fixture (one positive, five negatives, gene-symbol rankings).
GROUND_TRUTH = {
    "ENSG00000137869": 1,  # CYP19A1 — positive
    "ENSG00000134460": 0,
    "ENSG00000135902": 0,
    "ENSG00000005844": 0,
    "ENSG00000106991": 0,
    "ENSG00000102096": 0,
}
SYMBOL_TO_ID = {
    "CYP19A1": "ENSG00000137869",
    "ITGAL": "ENSG00000134460",
    "PIM2": "ENSG00000135902",
    "IL2RA": "ENSG00000005844",
    "ENG": "ENSG00000106991",
    "CHRND": "ENSG00000102096",
}


# ────────────────────────────────────────────────────────────────────────────
# resolve_to_id / canonical_rankings
# ────────────────────────────────────────────────────────────────────────────

def test_resolve_to_id_maps_known_symbol():
    assert resolve_to_id("CYP19A1", SYMBOL_TO_ID) == "ENSG00000137869"


def test_resolve_to_id_passthrough_for_unknown():
    # Already-canonical IDs and unrecognised entries must round-trip unchanged.
    assert resolve_to_id("ENSG00000137869", SYMBOL_TO_ID) == "ENSG00000137869"
    assert resolve_to_id("UNKNOWN_GENE", SYMBOL_TO_ID) == "UNKNOWN_GENE"


def test_canonical_rankings_resolves_mixed_input():
    mixed = ["CYP19A1", "ENSG00000134460", "UNKNOWN"]
    assert canonical_rankings(mixed, SYMBOL_TO_ID) == [
        "ENSG00000137869",
        "ENSG00000134460",
        "UNKNOWN",
    ]


# ────────────────────────────────────────────────────────────────────────────
# first_positive_rank
# ────────────────────────────────────────────────────────────────────────────

def test_first_positive_rank_symbol_at_position_1():
    rankings = ["CYP19A1", "ITGAL", "PIM2", "IL2RA", "ENG", "CHRND"]
    assert first_positive_rank(rankings, GROUND_TRUTH, SYMBOL_TO_ID) == 1


def test_first_positive_rank_symbol_at_position_4():
    rankings = ["ITGAL", "PIM2", "IL2RA", "CYP19A1", "ENG", "CHRND"]
    assert first_positive_rank(rankings, GROUND_TRUTH, SYMBOL_TO_ID) == 4


def test_first_positive_rank_returns_none_when_positive_absent():
    # All ranked entries are negatives.
    rankings = ["ITGAL", "PIM2", "IL2RA"]
    assert first_positive_rank(rankings, GROUND_TRUTH, SYMBOL_TO_ID) is None


def test_first_positive_rank_returns_none_for_empty_rankings():
    assert first_positive_rank([], GROUND_TRUTH, SYMBOL_TO_ID) is None


def test_first_positive_rank_with_canonical_ids_no_symbol_map_needed():
    rankings = ["ENSG00000137869", "ENSG00000134460"]
    # symbol_to_id omitted (defaults to None) — IDs match directly.
    assert first_positive_rank(rankings, GROUND_TRUTH) == 1


def test_first_positive_rank_no_symbol_map_skips_symbol_only_hits():
    # Without a mapping, a gene-symbol ranking can't resolve to an ID.
    assert first_positive_rank(["CYP19A1"], GROUND_TRUTH) is None


# ────────────────────────────────────────────────────────────────────────────
# recall_at_k
# ────────────────────────────────────────────────────────────────────────────

def test_recall_at_1_hits_with_symbol_at_top():
    rankings = ["CYP19A1", "ITGAL", "PIM2"]
    assert recall_at_k(rankings, GROUND_TRUTH, k=1, symbol_to_id=SYMBOL_TO_ID) == 1.0


def test_recall_at_k_below_first_positive_is_zero():
    rankings = ["ITGAL", "PIM2", "IL2RA", "CYP19A1"]
    assert recall_at_k(rankings, GROUND_TRUTH, k=3, symbol_to_id=SYMBOL_TO_ID) == 0.0


def test_recall_at_k_at_first_positive_is_one():
    rankings = ["ITGAL", "PIM2", "IL2RA", "CYP19A1"]
    assert recall_at_k(rankings, GROUND_TRUTH, k=4, symbol_to_id=SYMBOL_TO_ID) == 1.0


def test_recall_at_k_with_two_positives_partial_hit():
    gt = {"A": 1, "B": 1, "C": 0, "D": 0}
    rankings = ["A", "C", "D", "B"]
    # k=2 captures one of two positives → 0.5.
    assert recall_at_k(rankings, gt, k=2) == pytest.approx(0.5)


def test_recall_at_k_returns_zero_when_ground_truth_has_no_positives():
    gt = {"A": 0, "B": 0}
    assert recall_at_k(["A", "B"], gt, k=2) == 0.0


# ────────────────────────────────────────────────────────────────────────────
# ndcg_at_k
# ────────────────────────────────────────────────────────────────────────────

def test_ndcg_at_k_perfect_ranking_is_one():
    rankings = ["CYP19A1", "ITGAL", "PIM2", "IL2RA", "ENG", "CHRND"]
    assert ndcg_at_k(rankings, GROUND_TRUTH, k=6, symbol_to_id=SYMBOL_TO_ID) == pytest.approx(1.0)


def test_ndcg_at_k_with_positive_at_position_4():
    # DCG = 1/log2(5) (single hit at i=4); ideal = 1/log2(2) = 1.0
    rankings = ["ITGAL", "PIM2", "IL2RA", "CYP19A1"]
    expected = (1.0 / math.log2(5)) / 1.0
    assert ndcg_at_k(rankings, GROUND_TRUTH, k=4, symbol_to_id=SYMBOL_TO_ID) == pytest.approx(expected)


def test_ndcg_at_k_returns_zero_when_no_hit_in_topk():
    rankings = ["ITGAL", "PIM2", "IL2RA"]  # CYP19A1 absent
    assert ndcg_at_k(rankings, GROUND_TRUTH, k=3, symbol_to_id=SYMBOL_TO_ID) == 0.0


def test_ndcg_at_k_returns_zero_when_no_positives_in_ground_truth():
    assert ndcg_at_k(["A", "B"], {"A": 0, "B": 0}, k=2) == 0.0


def test_ndcg_at_k_two_positives_separated():
    # GT has two positives (A, B); ranking surfaces A at 1 and B at 4.
    gt = {"A": 1, "B": 1, "C": 0, "D": 0}
    rankings = ["A", "C", "D", "B"]
    dcg = 1.0 / math.log2(2) + 1.0 / math.log2(5)
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    assert ndcg_at_k(rankings, gt, k=4) == pytest.approx(dcg / idcg)


# ────────────────────────────────────────────────────────────────────────────
# hit_at_k
# ────────────────────────────────────────────────────────────────────────────

def test_hit_at_1_is_one_when_top_is_positive():
    rankings = ["CYP19A1", "ITGAL", "PIM2"]
    assert hit_at_k(rankings, GROUND_TRUTH, k=1, symbol_to_id=SYMBOL_TO_ID) == 1


def test_hit_at_1_is_zero_when_top_is_negative():
    rankings = ["ITGAL", "CYP19A1", "PIM2"]
    assert hit_at_k(rankings, GROUND_TRUTH, k=1, symbol_to_id=SYMBOL_TO_ID) == 0


def test_hit_at_k_only_counts_topk_window():
    rankings = ["ITGAL", "PIM2", "IL2RA", "CYP19A1"]
    # Positive at rank 4 — k=3 is a miss, k=4 is a hit.
    assert hit_at_k(rankings, GROUND_TRUTH, k=3, symbol_to_id=SYMBOL_TO_ID) == 0
    assert hit_at_k(rankings, GROUND_TRUTH, k=4, symbol_to_id=SYMBOL_TO_ID) == 1


def test_hit_at_k_uses_canonical_id_when_no_symbol_map():
    rankings = ["ENSG00000137869"]
    assert hit_at_k(rankings, GROUND_TRUTH, k=1) == 1


def test_hit_at_k_returns_zero_when_ground_truth_has_no_positives():
    assert hit_at_k(["A", "B"], {"A": 0, "B": 0}, k=2) == 0


def test_hit_at_k_with_two_positives_either_in_topk_counts():
    gt = {"A": 1, "B": 1, "C": 0, "D": 0}
    # B is at rank 2; A is at rank 4. k=2 catches B.
    rankings = ["C", "B", "D", "A"]
    assert hit_at_k(rankings, gt, k=2) == 1


# ────────────────────────────────────────────────────────────────────────────
# mrr (per-disease reciprocal rank)
# ────────────────────────────────────────────────────────────────────────────

def test_mrr_is_one_when_first_is_positive():
    rankings = ["CYP19A1", "ITGAL"]
    assert mrr(rankings, GROUND_TRUTH, symbol_to_id=SYMBOL_TO_ID) == pytest.approx(1.0)


def test_mrr_is_inverse_of_first_positive_rank():
    rankings = ["ITGAL", "PIM2", "IL2RA", "CYP19A1"]  # positive at rank 4
    assert mrr(rankings, GROUND_TRUTH, symbol_to_id=SYMBOL_TO_ID) == pytest.approx(0.25)


def test_mrr_returns_zero_when_no_positive_in_ranking():
    rankings = ["ITGAL", "PIM2", "IL2RA"]  # CYP19A1 absent
    assert mrr(rankings, GROUND_TRUTH, symbol_to_id=SYMBOL_TO_ID) == 0.0


def test_mrr_returns_zero_for_empty_rankings():
    assert mrr([], GROUND_TRUTH, symbol_to_id=SYMBOL_TO_ID) == 0.0


def test_mrr_with_canonical_id_no_symbol_map():
    rankings = ["ENSG00000134460", "ENSG00000137869"]  # positive at rank 2
    assert mrr(rankings, GROUND_TRUTH) == pytest.approx(0.5)
