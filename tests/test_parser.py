"""Tests for src/reasoning/parser.py — JSON extraction + Task1/Task2 parsers."""

from __future__ import annotations

import textwrap

import pytest

from src.reasoning.parser import (
    PointwiseParsed,
    Task1Parsed,
    Task2Parsed,
    parse_pointwise_response,
    parse_task1_response,
    parse_task2_response,
)


# ────────────────────────────────────────────────────────────────────────────
# Task 2 — happy paths
# ────────────────────────────────────────────────────────────────────────────

def test_task2_clean_json_success():
    text = '{"prediction": "SUCCESS", "confidence": 0.82, "reasoning": "strong genetic evidence"}'
    p = parse_task2_response(text)
    assert p.ok
    assert p.prediction == 1
    assert p.confidence == pytest.approx(0.82)
    assert p.reasoning == "strong genetic evidence"
    assert p.error is None


def test_task2_clean_json_failure():
    text = '{"prediction": "FAILURE", "confidence": 0.41, "reasoning": "weak"}'
    p = parse_task2_response(text)
    assert p.ok
    assert p.prediction == 0
    assert p.confidence == pytest.approx(0.41)


def test_task2_json_in_markdown_fence():
    text = textwrap.dedent("""
        Here is my response:

        ```json
        {"prediction": "SUCCESS", "confidence": 0.7, "reasoning": "reasonable"}
        ```

        Hope that helps.
    """).strip()
    p = parse_task2_response(text)
    assert p.ok
    assert p.prediction == 1
    assert p.confidence == pytest.approx(0.7)


def test_task2_json_in_bare_fence():
    text = "```\n{\"prediction\": \"FAILURE\", \"confidence\": 0.3}\n```"
    p = parse_task2_response(text)
    assert p.ok
    assert p.prediction == 0


def test_task2_json_with_surrounding_prose():
    # Greedy {...} match — falls through Strategy 1 + 2, hits Strategy 3.
    text = (
        "After consideration: {\"prediction\": \"SUCCESS\", \"confidence\": 0.9, "
        '"reasoning": "ok"} — that is my answer.'
    )
    p = parse_task2_response(text)
    assert p.ok
    assert p.prediction == 1


def test_task2_numeric_prediction():
    text = '{"prediction": 1, "confidence": 0.6}'
    p = parse_task2_response(text)
    assert p.ok
    assert p.prediction == 1


def test_task2_lowercase_strings_normalised():
    text = '{"prediction": "success", "confidence": 0.5}'
    p = parse_task2_response(text)
    assert p.ok
    assert p.prediction == 1


# ────────────────────────────────────────────────────────────────────────────
# Task 2 — fallback paths
# ────────────────────────────────────────────────────────────────────────────

def test_task2_regex_fallback_when_json_missing_confidence():
    # JSON parses but `confidence` is missing -> the JSON branch returns no result
    # and the regex fallback kicks in. Regex finds prediction; confidence is None.
    text = "I think the answer is: prediction: SUCCESS, but I'm unsure of the confidence."
    p = parse_task2_response(text)
    assert p.ok
    assert p.prediction == 1
    assert p.confidence is None
    assert "regex" in (p.error or "").lower()


def test_task2_regex_fallback_with_bare_keywords():
    text = "Based on the evidence, this trial is likely a FAILURE. Confidence around 0.35."
    p = parse_task2_response(text)
    assert p.ok
    assert p.prediction == 0
    assert p.confidence == pytest.approx(0.35)


def test_task2_returns_failure_on_empty_string():
    p = parse_task2_response("")
    assert not p.ok
    assert "empty" in (p.error or "").lower()


def test_task2_returns_failure_on_pure_refusal():
    text = "I cannot provide medical predictions in this context."
    p = parse_task2_response(text)
    assert not p.ok


def test_task2_rejects_out_of_range_confidence():
    # Confidence 1.5 is invalid; without a valid pair the JSON branch falls
    # through. Regex still finds "SUCCESS"; confidence comes back as None.
    text = '{"prediction": "SUCCESS", "confidence": 1.5}'
    p = parse_task2_response(text)
    assert p.ok
    assert p.prediction == 1
    assert p.confidence is None


def test_task2_handles_garbage():
    p = parse_task2_response("!@#$%^&*()")
    assert not p.ok


# ────────────────────────────────────────────────────────────────────────────
# Task 1 — happy paths
# ────────────────────────────────────────────────────────────────────────────

def test_task1_clean_json():
    text = textwrap.dedent("""
        {
          "rankings": ["GENE_A", "GENE_B", "GENE_C"],
          "confidences": {"GENE_A": 0.9, "GENE_B": 0.7, "GENE_C": 0.5},
          "reasoning": {"GENE_A": "best", "GENE_B": "ok", "GENE_C": "weak"}
        }
    """).strip()
    p = parse_task1_response(text)
    assert p.ok
    assert p.rankings == ["GENE_A", "GENE_B", "GENE_C"]
    assert p.confidences["GENE_A"] == pytest.approx(0.9)
    assert p.reasoning["GENE_C"] == "weak"


def test_task1_in_markdown_fence():
    text = textwrap.dedent("""
        ```json
        {"rankings": ["X", "Y"], "confidences": {"X": 0.8, "Y": 0.4}, "reasoning": {}}
        ```
    """).strip()
    p = parse_task1_response(text)
    assert p.ok
    assert p.rankings == ["X", "Y"]
    assert p.reasoning == {}


def test_task1_rankings_only_no_confidences():
    # Confidences/reasoning are best-effort; missing them is allowed.
    text = '{"rankings": ["A", "B"]}'
    p = parse_task1_response(text)
    assert p.ok
    assert p.rankings == ["A", "B"]
    assert p.confidences == {}
    assert p.reasoning == {}


def test_task1_coerces_non_string_ids_to_strings():
    text = '{"rankings": [1, 2, 3], "confidences": {"1": 0.9}}'
    p = parse_task1_response(text)
    assert p.ok
    assert p.rankings == ["1", "2", "3"]


def test_task1_drops_invalid_confidence_entries():
    text = textwrap.dedent("""
        {"rankings": ["A", "B", "C"],
         "confidences": {"A": 0.5, "B": 1.7, "C": "not a number"}}
    """).strip()
    p = parse_task1_response(text)
    assert p.ok
    # Only A is in [0, 1]; B is out of range, C is non-numeric.
    assert p.confidences == {"A": pytest.approx(0.5)}


# ────────────────────────────────────────────────────────────────────────────
# Task 1 — failure paths
# ────────────────────────────────────────────────────────────────────────────

def test_task1_empty_string_fails():
    p = parse_task1_response("")
    assert not p.ok
    assert "empty" in (p.error or "").lower()


def test_task1_no_json_fails():
    p = parse_task1_response("Sorry, I cannot rank these targets.")
    assert not p.ok
    assert p.error == "no JSON object found"


def test_task1_missing_rankings_field_fails():
    p = parse_task1_response('{"confidences": {"A": 0.5}}')
    assert not p.ok
    assert "rankings" in (p.error or "")


def test_task1_empty_rankings_list_fails():
    p = parse_task1_response('{"rankings": []}')
    assert not p.ok


def test_task1_rankings_wrong_type_fails():
    p = parse_task1_response('{"rankings": "not a list"}')
    assert not p.ok


# ────────────────────────────────────────────────────────────────────────────
# Robustness — Qwen 1.5B-style real output (from smoke test)
# ────────────────────────────────────────────────────────────────────────────

def test_task1_real_qwen_output_with_truncated_reasoning():
    # Captured shape from results/smoke_test/smoke_test_llm.csv — fenced JSON,
    # final reasoning string truncated mid-sentence by max_tokens.
    text = textwrap.dedent('''
        ```json
        {
          "rankings": ["CYP19A1", "ITGAL", "PIM2", "IL2RA", "ENG", "CHRND"],
          "confidences": {
            "CYP19A1": 0.9,
            "ITGAL": 0.85,
            "PIM2": 0.8,
            "IL2RA": 0.75,
            "ENG": 0.7,
            "CHRND": 0.65
          },
          "reasoning": {
            "CYP19A1": "Highly relevant to ovarian function.",
            "ITGAL": "Cell adhesion role."
          }
        }
        ```
    ''').strip()
    p = parse_task1_response(text)
    assert p.ok
    assert p.rankings[0] == "CYP19A1"
    assert len(p.rankings) == 6
    # Only the two reasoning strings the model actually wrote.
    assert set(p.reasoning.keys()) == {"CYP19A1", "ITGAL"}


# ────────────────────────────────────────────────────────────────────────────
# JSONC hardening — JS-style comments and trailing commas
# ────────────────────────────────────────────────────────────────────────────

def test_task1_jsonc_line_comment_after_value():
    # Real Qwen 1.5B failure mode (MONDO_0008056 frequency_1to5 smoke run):
    # the model abbreviated with a `// Additional entries follow…` comment
    # and strict json.loads() rejected the output.
    text = textwrap.dedent('''
        ```json
        {
          "rankings": ["A", "B", "C"],
          "confidences": {"A": 0.9, "B": 0.7, "C": 0.5},
          "reasoning": {
            "A": "high confidence",
            "B": "moderate"
            // Additional entries follow similar justifications.
          }
        }
        ```
    ''').strip()
    p = parse_task1_response(text)
    assert p.ok, f"Parser should tolerate JS line comments; got error: {p.error}"
    assert p.rankings == ["A", "B", "C"]


def test_task1_jsonc_block_comment():
    text = textwrap.dedent('''
        {
          "rankings": ["X", "Y"],
          "confidences": {"X": 0.8, "Y": 0.4},
          /* This is a block comment.
             Multiple lines. */
          "reasoning": {"X": "ok"}
        }
    ''').strip()
    p = parse_task1_response(text)
    assert p.ok
    assert p.rankings == ["X", "Y"]


def test_task1_jsonc_trailing_comma_in_object():
    text = '{"rankings": ["A", "B"], "confidences": {"A": 0.5,}}'
    p = parse_task1_response(text)
    assert p.ok
    assert p.rankings == ["A", "B"]
    assert p.confidences == {"A": pytest.approx(0.5)}


def test_task1_jsonc_trailing_comma_in_array():
    text = '{"rankings": ["A", "B", "C",], "confidences": {}}'
    p = parse_task1_response(text)
    assert p.ok
    assert p.rankings == ["A", "B", "C"]


def test_task2_jsonc_line_comment_does_not_break_extraction():
    text = textwrap.dedent('''
        {
          "prediction": "SUCCESS",  // Best supported by genetics
          "confidence": 0.8,
          "reasoning": "GWAS hit + concordant rare-variant burden"
        }
    ''').strip()
    p = parse_task2_response(text)
    assert p.ok
    assert p.prediction == 1
    assert p.confidence == pytest.approx(0.8)


def test_strict_json_still_parses_fast_path():
    # Sanity: valid JSON should not be touched by the JSONC stripper.
    # (Behavioural assertion — strict path is tried first in _try_load.)
    text = '{"rankings": ["A"], "confidences": {"A": 0.5}}'
    p = parse_task1_response(text)
    assert p.ok
    assert p.rankings == ["A"]


# ────────────────────────────────────────────────────────────────────────────
# Pointwise parser — per-(target, disease) viability score
# ────────────────────────────────────────────────────────────────────────────

def test_pointwise_clean_json_success():
    text = '{"score": 7, "confidence": 0.82, "reasoning": "strong genetic evidence"}'
    p = parse_pointwise_response(text)
    assert p.ok
    assert p.score == 7
    assert p.confidence == pytest.approx(0.82)
    assert p.reasoning == "strong genetic evidence"
    assert p.error is None


def test_pointwise_score_zero_is_valid():
    p = parse_pointwise_response('{"score": 0, "confidence": 0.5, "reasoning": "no link"}')
    assert p.ok and p.score == 0


def test_pointwise_score_ten_is_valid():
    p = parse_pointwise_response('{"score": 10, "confidence": 0.9, "reasoning": "x"}')
    assert p.ok and p.score == 10


def test_pointwise_float_score_rounds():
    p = parse_pointwise_response('{"score": 7.4, "confidence": 0.5, "reasoning": "x"}')
    assert p.ok and p.score == 7
    p = parse_pointwise_response('{"score": 7.6, "confidence": 0.5, "reasoning": "x"}')
    assert p.ok and p.score == 8


def test_pointwise_score_out_of_range_fails():
    for bad in (-1, 11, 100):
        p = parse_pointwise_response(f'{{"score": {bad}, "confidence": 0.5, "reasoning": "x"}}')
        assert not p.ok, f"score={bad} should be rejected"


def test_pointwise_in_markdown_fence():
    text = textwrap.dedent("""
        Here is my assessment:
        ```json
        {"score": 5, "confidence": 0.6, "reasoning": "moderate"}
        ```
    """)
    p = parse_pointwise_response(text)
    assert p.ok and p.score == 5


def test_pointwise_with_surrounding_prose():
    text = 'My answer is: {"score": 3, "confidence": 0.4, "reasoning": "weak"} thanks.'
    p = parse_pointwise_response(text)
    assert p.ok and p.score == 3


def test_pointwise_regex_fallback_when_json_invalid():
    # Bare key=value style (not JSON) — regex fallback should still extract.
    text = "score: 6, confidence: 0.55"
    p = parse_pointwise_response(text)
    assert p.ok
    assert p.score == 6
    assert p.confidence == pytest.approx(0.55)
    assert p.error  # noted regex fallback


def test_pointwise_missing_confidence_still_ok_via_fallback():
    text = "score: 4"
    p = parse_pointwise_response(text)
    assert p.ok and p.score == 4 and p.confidence is None


def test_pointwise_empty_string_fails():
    p = parse_pointwise_response("")
    assert not p.ok


def test_pointwise_pure_refusal_fails():
    p = parse_pointwise_response("I cannot help with that.")
    assert not p.ok


def test_pointwise_invalid_confidence_dropped():
    # Score is fine, confidence out of [0,1] → score returned, confidence None.
    p = parse_pointwise_response('{"score": 7, "confidence": 1.5, "reasoning": "x"}')
    assert p.ok and p.score == 7 and p.confidence is None
