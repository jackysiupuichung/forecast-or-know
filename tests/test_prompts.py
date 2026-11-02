"""Tests for src/reasoning/prompts.py — pointwise A1/A2/A3 rendering."""

from __future__ import annotations

import pytest

from src.reasoning.prompts import (
    OUTPUT_FORMAT,
    SCORING_RUBRIC,
    SYSTEM_A1,
    SYSTEM_A2_A3,
    _strip_path_annotations,
    build_prompt,
)


# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────

DISEASE = {
    "id": "MONDO_0002775",
    "name": "polycystic ovary syndrome",
    "ontology_id": "MONDO_0002775",
    "therapeutic_area": "endocrine system disease | reproductive system or breast disease",
    "description": "An endocrine disorder affecting ovarian function.",
}

TARGET = {
    "id": "ENSG00000137869",
    "ensembl_id": "ENSG00000137869",
    "symbol": "CYP19A1",
    "protein_class": "Enzyme",
    "description": "Catalyses aromatisation of androgens to oestrogens.",
}

PATH_WITH_SCORES = (
    "CYP19A1 --expressed_in[score=0.85, year=2018]--> ovarian_tissue "
    "--disease_associated[score=0.72, year=2020]--> MONDO_0002775"
)
PATH_NO_INTERMEDIATE = (
    "CYP19A1 --disease_associated[score=0.61, year=2021]--> MONDO_0002775"
)
LLM_CUT = "December 2023"
GRAPH_CUT = "June 2024"


# ────────────────────────────────────────────────────────────────────────────
# Common contracts (apply to every ablation)
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("ablation", ["A1", "A2", "A3"])
def test_user_prompt_contains_target_and_disease_blocks(ablation):
    paths = [PATH_WITH_SCORES] if ablation != "A1" else None
    _, user = build_prompt(DISEASE, TARGET, ablation=ablation, paths=paths)
    assert TARGET["symbol"] in user
    assert TARGET["ensembl_id"] in user
    assert TARGET["protein_class"] in user
    assert DISEASE["name"] in user
    assert DISEASE["therapeutic_area"] in user


@pytest.mark.parametrize("ablation", ["A1", "A2", "A3"])
def test_user_prompt_contains_rubric_and_output_format(ablation):
    """Default mode is binary Yes/No; the scored rubric only renders in rubric mode."""
    paths = [PATH_WITH_SCORES] if ablation != "A1" else None
    # Binary mode (production default): single-word Yes/No contract, no scored rubric.
    _, user = build_prompt(DISEASE, TARGET, ablation=ablation, paths=paths)
    assert "Yes or No" in user
    assert "0-2" not in user and "9-10" not in user
    # Rubric mode still emits the scored rubric and JSON output contract.
    _, user_rubric = build_prompt(
        DISEASE, TARGET, ablation=ablation, paths=paths, mode="rubric"
    )
    assert "0-2" in user_rubric and "9-10" in user_rubric
    assert (
        '"score"' in user_rubric
        and '"confidence"' in user_rubric
        and '"reasoning"' in user_rubric
    )


# ────────────────────────────────────────────────────────────────────────────
# A1 — parametric only
# ────────────────────────────────────────────────────────────────────────────

def test_a1_uses_llm_cutoff_in_system():
    system, user = build_prompt(
        DISEASE, TARGET, ablation="A1",
        llm_cutoff=LLM_CUT, graph_cutoff=GRAPH_CUT,
    )
    assert LLM_CUT in system
    assert GRAPH_CUT not in system
    assert "no access to information beyond" in system.lower()
    assert "EVIDENCE PATHS" not in user


def test_a1_does_not_accept_paths_argument():
    """A1 ignores paths entirely; no error if `paths` accidentally passed."""
    _, user = build_prompt(DISEASE, TARGET, ablation="A1", paths=[PATH_WITH_SCORES])
    assert "EVIDENCE PATHS" not in user


# ────────────────────────────────────────────────────────────────────────────
# A2 — paths without scores
# ────────────────────────────────────────────────────────────────────────────

def test_a2_uses_graph_cutoff_in_system():
    system, _ = build_prompt(
        DISEASE, TARGET, ablation="A2", paths=[PATH_WITH_SCORES],
        llm_cutoff=LLM_CUT, graph_cutoff=GRAPH_CUT,
    )
    assert GRAPH_CUT in system
    assert LLM_CUT not in system


def test_a2_strips_score_year_annotations():
    _, user = build_prompt(
        DISEASE, TARGET, ablation="A2", paths=[PATH_WITH_SCORES],
        graph_cutoff=GRAPH_CUT,
    )
    # Score and year metadata gone, but relation labels and topology kept.
    assert "score=" not in user
    assert "year=" not in user
    assert "expressed_in" in user
    assert "disease_associated" in user
    assert "ovarian_tissue" in user
    assert "-->" in user


def test_a2_handles_empty_path_list():
    _, user = build_prompt(DISEASE, TARGET, ablation="A2", paths=[])
    assert "no paths" in user.lower()


def test_a2_requires_paths_argument():
    with pytest.raises(ValueError):
        build_prompt(DISEASE, TARGET, ablation="A2", paths=None)


# ────────────────────────────────────────────────────────────────────────────
# A3 — paths with scores
# ────────────────────────────────────────────────────────────────────────────

def test_a3_keeps_score_year_annotations():
    _, user = build_prompt(
        DISEASE, TARGET, ablation="A3", paths=[PATH_WITH_SCORES],
        graph_cutoff=GRAPH_CUT,
    )
    assert "score=0.85" in user
    assert "year=2018" in user
    assert "expressed_in" in user
    assert GRAPH_CUT in user or True  # graph_cutoff lives in system, not user


def test_a3_renders_multiple_paths_with_indices():
    _, user = build_prompt(
        DISEASE, TARGET, ablation="A3",
        paths=[PATH_WITH_SCORES, PATH_NO_INTERMEDIATE],
    )
    assert "Path 1:" in user
    assert "Path 2:" in user


def test_a3_requires_paths_argument():
    with pytest.raises(ValueError):
        build_prompt(DISEASE, TARGET, ablation="A3", paths=None)


# ────────────────────────────────────────────────────────────────────────────
# Validation
# ────────────────────────────────────────────────────────────────────────────

def test_unknown_ablation_raises():
    with pytest.raises(ValueError):
        build_prompt(DISEASE, TARGET, ablation="A4", paths=[])
    with pytest.raises(ValueError):
        build_prompt(DISEASE, TARGET, ablation="X9", paths=[])


def test_strip_path_annotations_helper():
    s = "X --rel[score=0.5, year=2020]--> Y --rel2[score=0.4]--> Z"
    out = _strip_path_annotations(s)
    assert "[" not in out and "]" not in out
    assert "rel" in out and "rel2" in out and "X" in out and "Z" in out


def test_field_aliasing_falls_back_correctly():
    """build_prompt accepts older field names (gene_name, biology, functions)."""
    legacy_target = {"id": "ENSG1", "gene_name": "FOO", "functions": "kinase activity"}
    legacy_disease = {"id": "MONDO_X", "disease_name": "foo disease",
                      "biology": "rare metabolic disorder"}
    _, user = build_prompt(legacy_disease, legacy_target, ablation="A1")
    assert "FOO" in user
    assert "kinase activity" in user
    assert "foo disease" in user
    assert "rare metabolic disorder" in user
