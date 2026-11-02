#!/usr/bin/env python3
"""
Prompt templates for pointwise per-pair scoring.

Task: Prospective target prioritisation. Each (target, disease) pair is
scored independently on a 0-10 viability scale. Per-disease rankings are
recovered by sorting candidates by score post-hoc. No listwise prompts.

Three ablation levels:
    A1 — Parametric only (no graph evidence)
    A2 — Paths without confidence scores (annotations stripped)
    A3 — Paths with confidence scores and metadata

Temporal constraints:
    A1     — LLM's own training cutoff date
    A2/A3  — Graph cutoff date

Output: JSON `{"score": int 0..10, "confidence": float 0..1, "reasoning": str}`

Public API:
    build_prompt(disease, target, ablation, paths=None,
                 llm_cutoff="December 2023", graph_cutoff="June 2024")
        → (system_prompt, user_prompt)
"""

from __future__ import annotations

import re
from typing import Dict, List, Literal, Optional, Tuple


# ────────────────────────────────────────────────────────────────────────────────
# Anchored scoring rubric (consistent across ablations)
# ────────────────────────────────────────────────────────────────────────────────

SCORING_RUBRIC = """\
Score the therapeutic viability on a 0-10 scale:

  0-2: No plausible mechanistic connection to disease biology.
       Target and disease operate in unrelated biological domains.
  3-4: Indirect connection through generic biological processes
       (e.g., apoptosis, inflammation, cell cycle) that connect to many
       diseases without disease-specific relevance.
  5-6: Moderate evidence with some mechanistic rationale, but key links
       are weak (low confidence), indirect (3+ hops), or based on a
       single line of evidence.
  7-8: Strong mechanistic rationale supported by multiple independent
       evidence types (e.g., genetic association AND pathway involvement).
       Direct or short-hop connections with high confidence scores.
  9-10: Compelling multi-modal evidence across genetics, pathways,
        protein interactions, and druggability. Multiple high-confidence
        direct connections. Target is in a pharmacologically tractable
        protein class."""


# ────────────────────────────────────────────────────────────────────────────────
# System prompts
# ────────────────────────────────────────────────────────────────────────────────

SYSTEM_A1 = (
    "You are evaluating whether a drug target is likely to be therapeutically "
    "viable for a specific disease. You have no access to information beyond "
    "{cutoff_date}."
)

# Strong Simulated Ignorance variant (per Li et al. 2026): explicitly instruct
# the model to behave as if its knowledge cutoff is `cutoff_date`, even when its
# actual training data extends beyond that date. Used when --si-mode=strong.
SYSTEM_A1_SI_STRONG = (
    "You are evaluating whether a drug target is likely to be therapeutically "
    "viable for a specific disease. For this task, behave as if your knowledge "
    "cutoff is {cutoff_date}. Do not use any information that became available "
    "after {cutoff_date}, even if it was in your training corpus. Reason only "
    "from what was publicly known on or before {cutoff_date}."
)

SYSTEM_A2_A3 = (
    "You are evaluating whether a drug target is likely to be therapeutically "
    "viable for a specific disease. All evidence presented below was available "
    "as of {cutoff_date}. You have no access to information beyond this date."
)


# Trial-entry framing (binary mode). The binary question asks whether the target
# ENTERS A CLINICAL TRIAL as a therapeutic hypothesis, so the system prompt is
# framed the same way rather than as generic "therapeutic viability". Rubric mode
# keeps the viability framing above (its rubric scores mechanistic viability).
SYSTEM_A1_TRIAL = (
    "You are forecasting whether a drug target will enter a clinical trial as a "
    "therapeutic hypothesis for a specific disease. You have no access to "
    "information beyond {cutoff_date}."
)
SYSTEM_A1_TRIAL_SI_STRONG = (
    "You are forecasting whether a drug target will enter a clinical trial as a "
    "therapeutic hypothesis for a specific disease. For this task, behave as if "
    "your knowledge cutoff is {cutoff_date}. Do not use any information that "
    "became available after {cutoff_date}, even if it was in your training "
    "corpus. Reason only from what was publicly known on or before {cutoff_date}."
)
SYSTEM_A1_TRIAL_RECALL = (
    "You are assessing whether a drug target has already entered a clinical "
    "trial as a therapeutic hypothesis for a specific disease. You have no "
    "access to information beyond {cutoff_date}."
)
SYSTEM_A2_A3_TRIAL = (
    "You are forecasting whether a drug target will enter a clinical trial as a "
    "therapeutic hypothesis for a specific disease. All evidence presented below "
    "was available as of {cutoff_date}. You have no access to information beyond "
    "this date."
)


# ────────────────────────────────────────────────────────────────────────────────
# Output contract
# ────────────────────────────────────────────────────────────────────────────────

OUTPUT_FORMAT = """\
Respond as a single JSON object with exactly these fields:
  "score": integer 0 to 10
  "confidence": number in [0.0, 1.0]
  "reasoning": string explaining your assessment
Do not include any text outside the JSON object."""


BINARY_INSTRUCTION = (
    "Answer with a single word: Yes or No. "
    "Output only that word, with no punctuation or explanation."
)


# ────────────────────────────────────────────────────────────────────────────────
# Block formatters
# ────────────────────────────────────────────────────────────────────────────────

def _format_target_block(target: Dict[str, str]) -> str:
    """Format the TARGET section of the user prompt."""
    symbol = target.get("symbol") or target.get("gene_name") or target["id"]
    ensembl = target.get("ensembl_id") or target["id"]
    protein_class = target.get("protein_class") or "Unknown"
    description = (
        target.get("description")
        or target.get("functions")
        or target.get("target_description")
        or "(no description available)"
    )
    return (
        f"TARGET: {symbol} ({ensembl})\n"
        f"  Protein class: {protein_class}\n"
        f"  Known functions: {description}"
    )


def _format_disease_block(disease: Dict[str, str]) -> str:
    """Format the DISEASE section of the user prompt."""
    name = disease.get("name") or disease.get("disease_name") or disease["id"]
    ontology_id = (
        disease.get("ontology_id")
        or disease.get("mondo_id")
        or disease.get("efo_id")
        or disease["id"]
    )
    therapeutic_area = disease.get("therapeutic_area") or "Unknown"
    biology = (
        disease.get("biology")
        or disease.get("description")
        or disease.get("disease_description")
        or "(no description available)"
    )
    return (
        f"DISEASE: {name} ({ontology_id})\n"
        f"  Therapeutic area: {therapeutic_area}\n"
        f"  Known biology: {biology}"
    )


# Match a `[ ... ]` annotation block on an edge (compact serialiser format).
# Used by A2 to strip score/year metadata while leaving the relation label
# and the path structure intact.
_EDGE_META_BRACKET = re.compile(r"\[[^\[\]]*\]")


def _strip_path_annotations(path: str) -> str:
    """A2 helper: remove `[score=..., year=...]` annotations from a path string,
    leaving relation labels and path topology intact."""
    return _EDGE_META_BRACKET.sub("", path)


def _format_paths_block(paths: List[str], strip_scores: bool) -> str:
    """Format a list of pre-serialised path strings (compact format from
    PathSerialiser) into the EVIDENCE PATHS block."""
    if not paths:
        return "(no paths connecting target to disease found in the knowledge graph)"
    lines: List[str] = []
    for i, p in enumerate(paths, start=1):
        if strip_scores:
            p = _strip_path_annotations(p)
        lines.append(f"Path {i}: {p}")
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────────────────
# Main prompt builder
# ────────────────────────────────────────────────────────────────────────────────

ALLOWED_ABLATIONS = ("A1", "A2", "A3")


def build_prompt(
    disease: Dict[str, str],
    target: Dict[str, str],
    ablation: str = "A3",
    paths: Optional[List[str]] = None,
    llm_cutoff: str = "December 2023",
    graph_cutoff: str = "June 2024",
    trial_window_year: int = 2023,
    label_year: Optional[int] = None,
    si_mode: Literal["off", "strong"] = "off",
    mode: Literal["rubric", "binary"] = "binary",
    recall_mode: bool = False,
    spec=None,
) -> Tuple[str, str]:
    """
    Build a pointwise scoring prompt for a single (target, disease) pair.

    Args:
        disease:      dict — required keys: id; optional: name, ontology_id,
                      therapeutic_area, description (or biology).
        target:       dict — required keys: id; optional: symbol (gene_name),
                      ensembl_id, protein_class, description (or functions /
                      target_description).
        ablation:     'A1' | 'A2' | 'A3'.
        paths:        Pre-serialised compact path strings (from PathSerialiser).
                      Required for A2 / A3. Pass `[]` for "no paths found".
        llm_cutoff:   Date in the A1 system prompt.
        graph_cutoff: Date in the A2 / A3 system prompt.
        trial_window_year: Year in the legacy open-ended binary question
                      "...enter a clinical trial ... after YEAR?" (= cutoff C).
        label_year:   If set (binary, non-recall), ask the one-year TGB-style
                      question matching the label "first trial in Y":
                      "...enter its first clinical trial ... in Y?". None keeps
                      the legacy wording byte-identical for old runs.

    Returns:
        (system_prompt, user_prompt)
    """
    # Resolve the arm. An explicit GraphContextSpec (variant harness) governs
    # graph-on/off and PRE-renders the paths upstream; otherwise fall back to the
    # frozen ablation names with the legacy internal strip (byte-identical).
    if spec is not None:
        has_graph = bool(spec.graph_context)
        strip_scores = False               # paths already rendered by the re-serialiser
    else:
        if ablation not in ALLOWED_ABLATIONS:
            raise ValueError(
                f"Unknown ablation: {ablation!r}. Must be one of {ALLOWED_ABLATIONS}."
            )
        has_graph = (ablation != "A1")
        strip_scores = (ablation == "A2")
    if has_graph and paths is None:
        raise ValueError("graph-context arm requires `paths` (pass `[]` if none were found).")
    if recall_mode and has_graph:
        raise NotImplementedError(
            "recall_mode is only supported for the parametric (A1) arm.")

    # ── System ──
    # Binary mode uses the trial-entry framing so the system prompt matches the
    # binary question ("...enter a clinical trial..."). Rubric mode keeps the
    # therapeutic-viability framing (it scores mechanistic viability 0-10).
    trial = (mode == "binary")
    if not has_graph:
        if trial and recall_mode:
            template = SYSTEM_A1_TRIAL_RECALL
        elif trial and si_mode == "strong":
            template = SYSTEM_A1_TRIAL_SI_STRONG
        elif trial:
            template = SYSTEM_A1_TRIAL
        elif si_mode == "strong":
            template = SYSTEM_A1_SI_STRONG
        else:
            template = SYSTEM_A1
        system = template.format(cutoff_date=llm_cutoff)
    else:
        template = SYSTEM_A2_A3_TRIAL if trial else SYSTEM_A2_A3
        system = template.format(cutoff_date=graph_cutoff)

    # ── User ──
    sections: List[str] = []
    sections.append(_format_target_block(target))
    sections.append("")
    sections.append(_format_disease_block(disease))

    if has_graph:
        symbol = target.get("symbol") or target.get("gene_name") or target["id"]
        name = disease.get("name") or disease.get("disease_name") or disease["id"]
        sections.append("")
        sections.append(f"EVIDENCE PATHS connecting {symbol} to {name}:")
        sections.append(_format_paths_block(paths or [], strip_scores=strip_scores))

    symbol = target.get("symbol") or target.get("gene_name") or target["id"]
    name = disease.get("name") or disease.get("disease_name") or disease["id"]
    sections.append("")
    if mode == "binary":
        if not has_graph:
            if recall_mode:
                sections.append(
                    f"Based on your knowledge available as of {llm_cutoff}, has "
                    f"{symbol} entered a clinical trial as a therapeutic target for "
                    f"{name} on or before {trial_window_year}?"
                )
            elif label_year is not None:
                sections.append(
                    f"Based on your knowledge available as of {llm_cutoff}, will "
                    f"{symbol} enter its first clinical trial as a therapeutic "
                    f"target for {name} in {label_year}?"
                )
            else:
                sections.append(
                    f"Based on your knowledge available as of {llm_cutoff}, will "
                    f"{symbol} enter a clinical trial as a therapeutic target for "
                    f"{name} after {trial_window_year}?"
                )
        elif label_year is not None:
            sections.append(
                f"Based on the evidence above (available as of {graph_cutoff}), "
                f"will {symbol} enter its first clinical trial as a therapeutic "
                f"target for {name} in {label_year}?"
            )
        else:
            sections.append(
                f"Based on the evidence above (available as of {graph_cutoff}), "
                f"will {symbol} enter a clinical trial as a therapeutic target "
                f"for {name} after {trial_window_year}?"
            )
        sections.append("")
        sections.append(BINARY_INSTRUCTION)
    else:
        if not has_graph:
            sections.append(
                f"Based on your knowledge, assess the therapeutic viability of "
                f"targeting {symbol} for {name}."
            )
        else:
            sections.append(
                f"Based on the evidence above, assess the therapeutic viability of "
                f"targeting {symbol} for {name}."
            )

        sections.append("")
        sections.append(SCORING_RUBRIC)
        sections.append("")
        sections.append(OUTPUT_FORMAT)

    return system, "\n".join(sections)


# ── Row -> prompt dicts (shared by the eval runner and training serialisers) ──

def _cell(row, col, default):
    """row[col] as str, or `default` if the column is absent or the value is
    missing (NaN/None/empty). Missing values used to render as the literal
    string "nan"."""
    try:
        v = row[col]
    except (KeyError, IndexError):
        return default
    if v is None or (isinstance(v, float) and v != v) or str(v) == "":
        return default
    return str(v)


def target_from_row(row) -> Dict[str, str]:
    tid = str(row["targetId"])
    return {
        "id": tid,
        "ensembl_id": tid,
        "symbol": _cell(row, "gene_name", tid),
        "protein_class": _cell(row, "protein_class", "Unknown"),
        "description": _cell(row, "target_description", ""),
    }


def disease_from_row(row) -> Dict[str, str]:
    did = str(row["diseaseId"])
    return {
        "id": did,
        "ontology_id": did,
        "name": _cell(row, "disease_name", did),
        "therapeutic_area": _cell(row, "therapeutic_area", "Unknown"),
        "description": _cell(row, "disease_description", ""),
    }


# ── Citation stripping (description-leak control) ──
# Target descriptions come from the 2026 UniProt release and carry evidence tags
# citing papers up to 2026, e.g. "{ECO:0000269|PubMed:37315560}" or
# "(PubMed:2950091, PubMed:37315560)". Stripping them removes a post-decision
# recency/attention signal at ~zero cost to accuracy (description ablation,
# 2026-10-02).
import re as _re

_CITE_TAG = _re.compile(r"\s*\{[^{}]*(?:ECO:|PubMed:|UniProtKB:)[^{}]*\}")
_CITE_PAREN = _re.compile(r"\s*\((?:\s*PubMed:\d+\s*,?)+\)")


def strip_citations(text):
    if not isinstance(text, str):
        return text
    out = _CITE_PAREN.sub("", _CITE_TAG.sub("", text))
    return _re.sub(r"\s+([.,;])", r"\1", _re.sub(r"[ \t]{2,}", " ", out)).strip()
