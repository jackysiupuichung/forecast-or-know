#!/usr/bin/env python3
"""
Helpers for scoring LLM-produced target rankings against ground-truth labels.

LLMs frequently return gene symbols (e.g. "CYP19A1") instead of the canonical
Ensembl IDs (e.g. "ENSG00000137869") even when both are presented in the
prompt. Production scoring must resolve symbols back to IDs before lookup,
or every ranking will silently miss every positive.

Public API:
    resolve_to_id(label, symbol_to_id) -> str
        Map a single ranking entry (symbol or ID) to its canonical ID.
    canonical_rankings(rankings, symbol_to_id) -> List[str]
        Apply resolve_to_id to a full ranked list.
    first_positive_rank(rankings, ground_truth, symbol_to_id) -> Optional[int]
        1-indexed position of the first positive in the ranking, or None.
    hit_at_k(rankings, ground_truth, k, symbol_to_id) -> int
        1 if any positive appears in the top-k, else 0. Per-disease binary
        signal (vs. recall_at_k which is a fraction over all positives).
    mrr(rankings, ground_truth, symbol_to_id) -> float
        Reciprocal rank of the first positive (0.0 if none found).
        Per-disease scalar; macro-average across diseases gives MRR.
    recall_at_k(rankings, ground_truth, k, symbol_to_id) -> float
    ndcg_at_k(rankings, ground_truth, k, symbol_to_id) -> float
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence


def resolve_to_id(label: str, symbol_to_id: Dict[str, str]) -> str:
    """Resolve a ranking entry to a canonical ID. Pass-through if unknown."""
    return symbol_to_id.get(label, label)


def canonical_rankings(
    rankings: Sequence[str],
    symbol_to_id: Dict[str, str],
) -> List[str]:
    return [resolve_to_id(r, symbol_to_id) for r in rankings]


def first_positive_rank(
    rankings: Sequence[str],
    ground_truth: Dict[str, int],
    symbol_to_id: Optional[Dict[str, str]] = None,
) -> Optional[int]:
    """
    1-indexed position of the first positive (label==1) in the ranking.
    Returns None if no positive appears in the ranking.
    """
    sym = symbol_to_id or {}
    for i, ranked in enumerate(rankings, start=1):
        tid = resolve_to_id(ranked, sym)
        if ground_truth.get(tid) == 1:
            return i
    return None


def hit_at_k(
    rankings: Sequence[str],
    ground_truth: Dict[str, int],
    k: int,
    symbol_to_id: Optional[Dict[str, str]] = None,
) -> int:
    """
    Binary: 1 if any positive appears in the top-k of the ranking, else 0.

    Macro-averaging Hit@k across diseases gives the standard "fraction of
    diseases for which the model surfaces ≥1 positive in the top-k" metric.

    Returns 0 if there are no positives in ground_truth.
    """
    if not any(v == 1 for v in ground_truth.values()):
        return 0
    sym = symbol_to_id or {}
    for ranked in rankings[:k]:
        tid = resolve_to_id(ranked, sym)
        if ground_truth.get(tid) == 1:
            return 1
    return 0


def mrr(
    rankings: Sequence[str],
    ground_truth: Dict[str, int],
    symbol_to_id: Optional[Dict[str, str]] = None,
) -> float:
    """
    Reciprocal rank of the first positive (0.0 if none found in the ranking).

    Per-disease scalar in [0, 1]. Macro-average across diseases gives MRR.
    """
    rank = first_positive_rank(rankings, ground_truth, symbol_to_id)
    return 1.0 / rank if rank is not None else 0.0


def recall_at_k(
    rankings: Sequence[str],
    ground_truth: Dict[str, int],
    k: int,
    symbol_to_id: Optional[Dict[str, str]] = None,
) -> float:
    """
    Fraction of all ground-truth positives appearing in the top-k of the ranking.
    Returns 0.0 if there are no positives in ground_truth.
    """
    n_positives = sum(1 for v in ground_truth.values() if v == 1)
    if n_positives == 0:
        return 0.0
    sym = symbol_to_id or {}
    top_ids = {resolve_to_id(r, sym) for r in rankings[:k]}
    hits = sum(1 for tid, lbl in ground_truth.items() if lbl == 1 and tid in top_ids)
    return hits / n_positives


def precision_at_k(
    rankings: Sequence[str],
    ground_truth: Dict[str, int],
    k: int,
    symbol_to_id: Optional[Dict[str, str]] = None,
) -> float:
    """
    Fraction of the top-k ranked candidates that are ground-truth positives —
    the decision-relevant "of the top-k I would validate, how many are real"
    quantity. Denominator is min(k, list length), so short candidate lists are
    not penalised for having fewer than k items. Returns 0.0 if no positives.
    """
    if not any(v == 1 for v in ground_truth.values()):
        return 0.0
    sym = symbol_to_id or {}
    top = rankings[:k]
    if not top:
        return 0.0
    hits = sum(1 for r in top if ground_truth.get(resolve_to_id(r, sym)) == 1)
    return hits / len(top)


def ndcg_at_k(
    rankings: Sequence[str],
    ground_truth: Dict[str, int],
    k: int,
    symbol_to_id: Optional[Dict[str, str]] = None,
) -> float:
    """
    Normalised discounted cumulative gain at k, with binary relevance.

        DCG@k = sum_{i=1..k} rel_i / log2(i + 1)
        nDCG@k = DCG@k / ideal_DCG@k

    Returns 0.0 if no positives exist in ground_truth.
    """
    n_positives = sum(1 for v in ground_truth.values() if v == 1)
    if n_positives == 0:
        return 0.0
    sym = symbol_to_id or {}

    dcg = 0.0
    for i, ranked in enumerate(rankings[:k], start=1):
        tid = resolve_to_id(ranked, sym)
        if ground_truth.get(tid) == 1:
            dcg += 1.0 / math.log2(i + 1)

    ideal_hits = min(n_positives, k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0
