#!/usr/bin/env python3
"""
Parse LLM free-text responses into structured predictions.

LLMs are instructed to return JSON, but they sometimes wrap it in markdown code
fences, add prose before/after, or truncate. This module extracts the JSON object,
validates required fields, and falls back to regex-based extraction for the key
scalars if JSON parsing fails.

Public API:
    parse_pointwise_response(text) → PointwiseParsed
        Per-(target, disease) viability score (0..10), confidence, reasoning.
        Used by the pointwise prompts in `src.reasoning.prompts.build_prompt`.

    parse_task1_response(text) → Task1Parsed
    parse_task2_response(text) → Task2Parsed
        Legacy listwise/binary parsers retained for backwards-compatibility
        with run_task2_experiment.py. New code should use
        parse_pointwise_response.

All return dataclasses with an `ok` flag — if parsing fails, `ok=False` and
the caller can log + retry or exclude the sample.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ────────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ────────────────────────────────────────────────────────────────────────────────

@dataclass
class PointwiseParsed:
    """Result of parsing a pointwise (per-pair) viability scoring response."""
    ok: bool
    score: Optional[int] = None              # integer 0..10
    confidence: Optional[float] = None       # 0..1
    reasoning: Optional[str] = None
    raw_text: str = ""
    error: Optional[str] = None


@dataclass
class PointwiseBinaryParsed:
    """Result of parsing a binary Yes/No response with logprob extraction.

    `p_yes` is computed from the `top_logprobs` at the answer position —
    the first generated token whose *sampled* token is a Yes/No form:
    softmax over (logsumexp of yes-like tokens, logsumexp of no-like tokens).

    For models that answer tersely the answer position is token 0, so the
    readout is identical to the legacy first-token probe. For models that
    wrap the answer in JSON (e.g. Llama-3.1-70B emitting {"answer": "Yes"}),
    the scan finds the Yes/No where the model actually commits instead of
    reading noise off the `{"` token.
    """
    ok: bool
    p_yes: Optional[float] = None
    logprob_yes: Optional[float] = None
    logprob_no: Optional[float] = None
    top_token: Optional[str] = None      # first sampled token (format diagnostic)
    answer_token: Optional[str] = None   # sampled token at the answer position
    answer_position: Optional[int] = None
    n_yes_tokens: int = 0
    n_no_tokens: int = 0
    error: Optional[str] = None


# Token classifier — case/whitespace-insensitive match against the first
# generated token. Covers common tokenisations llama.cpp emits for "Yes"/"No"
# (with or without a leading space, with or without trailing punctuation).
_YES_FORMS = {"yes", "y", "true", "1"}
_NO_FORMS = {"no", "n", "false", "0"}


def _classify_yes_no_token(tok: str) -> Optional[str]:
    """Return 'yes', 'no', or None for a single top-logprob token string."""
    s = tok.strip().strip(".,!?:;\"'`").lower()
    if s in _YES_FORMS:
        return "yes"
    if s in _NO_FORMS:
        return "no"
    return None


def _is_sampled_answer_token(tok: str) -> bool:
    """Stricter predicate used to *locate* the answer position in a sampled
    token stream. Only literal yes/no words qualify — the digit and boolean
    forms accepted by `_classify_yes_no_token` for logprob pooling would
    misfire on JSON scaffolding (a `0` in `"confidence": 0.9`, a `true` in
    an unrelated boolean field)."""
    return tok.strip().strip(".,!?:;\"'`").lower() in {"yes", "no"}


def parse_pointwise_binary(
    logprobs: Optional[Dict[str, Any]],
    floor_offset: float = 5.0,
) -> PointwiseBinaryParsed:
    """Compute P("Yes") from llama.cpp `choices[0].logprobs`.

    The probe position is the first generated token whose *sampled* token is a
    Yes/No form. Terse models answer at position 0, so their readout is
    byte-identical to the legacy single-token probe; JSON-wrapping models
    (e.g. Llama-3.1-70B emitting {"answer": "Yes"}) are read at the token
    where the answer actually appears. If no sampled Yes/No is found within
    the generated window, falls back to position 0 (legacy behaviour) and
    tags the record via `error` so downstream analysis can filter.

    Args:
        logprobs:     The `logprobs` dict from a chat-completion response,
                      shape `{"content": [{"token", "logprob", "top_logprobs":
                      [{"token", "logprob", "bytes"}, ...]}, ...]}`.
        floor_offset: When only one of {yes, no} buckets has any tokens in the
                      top-K, the missing logprob is set to
                      `min(observed_top_logprobs) - floor_offset` so that
                      P_yes is well-defined but pinned away from 0/1.

    Returns:
        PointwiseBinaryParsed with `ok=True` if at least one yes/no token is
        present in `top_logprobs` at the probe position, otherwise `ok=False`.
    """
    import math

    if not logprobs or "content" not in logprobs or not logprobs["content"]:
        return PointwiseBinaryParsed(ok=False, error="no logprobs.content in response")

    content = logprobs["content"]
    top_token = content[0].get("token")

    answer_pos: Optional[int] = None
    for i, entry in enumerate(content):
        if _is_sampled_answer_token(entry.get("token") or ""):
            answer_pos = i
            break

    fallback_err = None
    if answer_pos is None:
        fallback_err = (
            f"no yes/no token sampled in {len(content)} generated tokens; "
            f"fell back to first-token probe"
        )
        answer_pos = 0

    first = content[answer_pos]
    answer_token = first.get("token")
    top_lps = first.get("top_logprobs") or []
    if not top_lps:
        return PointwiseBinaryParsed(
            ok=False, top_token=top_token,
            answer_token=answer_token, answer_position=answer_pos,
            error="no top_logprobs entries at probe position",
        )

    yes_lps: List[float] = []
    no_lps: List[float] = []
    for entry in top_lps:
        tok = entry.get("token", "")
        lp = entry.get("logprob")
        if lp is None:
            continue
        cls = _classify_yes_no_token(tok)
        if cls == "yes":
            yes_lps.append(float(lp))
        elif cls == "no":
            no_lps.append(float(lp))

    n_yes, n_no = len(yes_lps), len(no_lps)
    if n_yes == 0 and n_no == 0:
        return PointwiseBinaryParsed(
            ok=False, top_token=top_token,
            answer_token=answer_token, answer_position=answer_pos,
            n_yes_tokens=0, n_no_tokens=0,
            error="no yes/no token in top_logprobs at probe position",
        )

    def _logsumexp(xs: List[float]) -> float:
        m = max(xs)
        return m + math.log(sum(math.exp(x - m) for x in xs))

    all_lps = [float(e["logprob"]) for e in top_lps if e.get("logprob") is not None]
    floor = (min(all_lps) - floor_offset) if all_lps else -50.0

    lse_yes = _logsumexp(yes_lps) if yes_lps else floor
    lse_no = _logsumexp(no_lps) if no_lps else floor

    # softmax over (lse_yes, lse_no)
    m = max(lse_yes, lse_no)
    e_yes = math.exp(lse_yes - m)
    e_no = math.exp(lse_no - m)
    p_yes = e_yes / (e_yes + e_no)

    errs = [fallback_err] if fallback_err else []
    if n_yes == 0 or n_no == 0:
        errs.append(
            f"single-bucket-coverage (yes={n_yes}, no={n_no}); floored missing side"
        )

    return PointwiseBinaryParsed(
        ok=True,
        p_yes=float(p_yes),
        logprob_yes=lse_yes,
        logprob_no=lse_no,
        top_token=top_token,
        answer_token=answer_token,
        answer_position=answer_pos,
        n_yes_tokens=n_yes,
        n_no_tokens=n_no,
        error="; ".join(errs) if errs else None,
    )


@dataclass
class Task2Parsed:
    ok: bool
    prediction: Optional[int] = None     # 1=SUCCESS, 0=FAILURE
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    raw_text: str = ""
    error: Optional[str] = None


@dataclass
class Task1Parsed:
    ok: bool
    rankings: List[str] = field(default_factory=list)
    confidences: Dict[str, float] = field(default_factory=dict)
    reasoning: Dict[str, str] = field(default_factory=dict)
    raw_text: str = ""
    error: Optional[str] = None


# ────────────────────────────────────────────────────────────────────────────────
# JSON extraction
# ────────────────────────────────────────────────────────────────────────────────

_CODE_FENCE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL | re.IGNORECASE)
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

# Models occasionally emit JSONC (JSON-with-Comments) instead of strict JSON:
#   - "// elide rest" line comments after a value
#   - /* ... */ block comments
#   - trailing commas before closing ] or }
# Strict json.loads() rejects all three. We strip them as a fallback only —
# valid JSON parses on the fast path with no preprocessing.
_LINE_COMMENT_RE = re.compile(r"//[^\n]*")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def _strip_jsonc(text: str) -> str:
    """Remove JS-style comments and trailing commas to coerce JSONC → JSON.

    Naive — does not respect string-literal boundaries. The risk is removing
    a literal `//` inside a string; we accept that, since real model outputs
    here are gene symbols, ENSG IDs, and one-sentence rationales — none of
    which contain `//` or `/*` legitimately.
    """
    text = _BLOCK_COMMENT_RE.sub("", text)
    text = _LINE_COMMENT_RE.sub("", text)
    text = _TRAILING_COMMA_RE.sub(r"\1", text)
    return text


def _try_load(text: str) -> Optional[Dict[str, Any]]:
    """Attempt strict JSON, then JSONC-style fallback. None on failure."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        return json.loads(_strip_jsonc(text))
    except (json.JSONDecodeError, ValueError):
        return None


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Try a few strategies to pull a JSON object out of a text blob:
      1. Parse the text directly (strict, then JSONC-stripped)
      2. Extract from ```json ... ``` fence
      3. Greedy match for {...}
    Returns None if nothing parses.
    """
    # Strategy 1: direct
    obj = _try_load(text)
    if obj is not None:
        return obj

    # Strategy 2: code fence
    m = _CODE_FENCE.search(text)
    if m:
        obj = _try_load(m.group(1))
        if obj is not None:
            return obj

    # Strategy 3: greedy { ... }
    m = _JSON_OBJECT.search(text)
    if m:
        obj = _try_load(m.group(0))
        if obj is not None:
            return obj

    return None


# ────────────────────────────────────────────────────────────────────────────────
# Pointwise parser — per-(target, disease) viability score
# ────────────────────────────────────────────────────────────────────────────────

# Keyed regex fallbacks: tolerate JSON-ish outputs that fail strict parsing.
_POINTWISE_SCORE_RE = re.compile(
    r'"?score"?\s*[:=]\s*"?(-?\d+(?:\.\d+)?)"?', re.IGNORECASE
)
_POINTWISE_CONF_RE = re.compile(
    r'"?confidence"?\s*[:=]\s*"?(\d+(?:\.\d+)?)"?', re.IGNORECASE
)


def _normalise_score(val: Any) -> Optional[int]:
    """Score must be an integer in [0, 10]. Floats round to nearest int and
    are accepted if they fall in range; out-of-range or non-numeric → None."""
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    i = int(round(f))
    return i if 0 <= i <= 10 else None


def parse_pointwise_response(text: str) -> PointwiseParsed:
    """
    Parse a pointwise response. Expected schema:
        {"score": int 0..10, "confidence": float 0..1, "reasoning": str}
    """
    if not text or not text.strip():
        return PointwiseParsed(ok=False, raw_text=text, error="empty response")

    obj = _extract_json(text)
    if obj is not None:
        score = _normalise_score(obj.get("score"))
        conf = _normalise_confidence(obj.get("confidence"))
        reasoning = obj.get("reasoning")
        if score is not None:
            return PointwiseParsed(
                ok=True,
                score=score,
                confidence=conf,
                reasoning=reasoning if isinstance(reasoning, str) else None,
                raw_text=text,
            )

    # Fallback: regex sweep. Confidence is best-effort; missing it is OK.
    score_m = _POINTWISE_SCORE_RE.search(text)
    conf_m = _POINTWISE_CONF_RE.search(text)
    score = _normalise_score(score_m.group(1)) if score_m else None
    conf = _normalise_confidence(conf_m.group(1)) if conf_m else None

    if score is not None:
        return PointwiseParsed(
            ok=True,
            score=score,
            confidence=conf,
            reasoning=None,
            raw_text=text,
            error="json parse failed; extracted via regex fallback",
        )

    return PointwiseParsed(
        ok=False,
        raw_text=text,
        error="could not extract score from response",
    )


# ────────────────────────────────────────────────────────────────────────────────
# Task 2 parser — binary outcome prediction (legacy)
# ────────────────────────────────────────────────────────────────────────────────

_TASK2_PRED_RE = re.compile(
    r'"?prediction"?\s*[:=]\s*"?(SUCCESS|FAILURE|1|0)"?', re.IGNORECASE
)
_TASK2_PRED_BARE_RE = re.compile(r'\b(SUCCESS|FAILURE)\b', re.IGNORECASE)
_TASK2_CONF_RE = re.compile(
    r'"?confidence"?\s*[:=]\s*"?([01](?:\.\d+)?)"?'
)
_TASK2_CONF_BARE_RE = re.compile(r'\b([01]\.\d+)\b')


def _normalise_prediction(val: Any) -> Optional[int]:
    if isinstance(val, bool):
        return int(val)
    if isinstance(val, (int, float)):
        v = int(val)
        return v if v in (0, 1) else None
    if isinstance(val, str):
        s = val.strip().upper()
        if s in {"SUCCESS", "1", "TRUE", "YES", "SUCCEED", "PASS"}:
            return 1
        if s in {"FAILURE", "0", "FALSE", "NO", "FAIL"}:
            return 0
    return None


def _normalise_confidence(val: Any) -> Optional[float]:
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return f if 0.0 <= f <= 1.0 else None


def parse_task2_response(text: str) -> Task2Parsed:
    """
    Parse a Task 2 LLM response. Expected JSON schema:
        {"prediction": "SUCCESS"|"FAILURE", "confidence": float, "reasoning": str}
    """
    if not text or not text.strip():
        return Task2Parsed(ok=False, raw_text=text, error="empty response")

    obj = _extract_json(text)
    if obj is not None:
        pred = _normalise_prediction(obj.get("prediction"))
        conf = _normalise_confidence(obj.get("confidence"))
        reasoning = obj.get("reasoning")
        if pred is not None and conf is not None:
            return Task2Parsed(
                ok=True,
                prediction=pred,
                confidence=conf,
                reasoning=reasoning if isinstance(reasoning, str) else None,
                raw_text=text,
            )

    # Fallback: regex scan. Try keyed form first ("prediction: SUCCESS"), then bare ("SUCCESS").
    pred_m = _TASK2_PRED_RE.search(text) or _TASK2_PRED_BARE_RE.search(text)
    conf_m = _TASK2_CONF_RE.search(text) or _TASK2_CONF_BARE_RE.search(text)
    pred = _normalise_prediction(pred_m.group(1)) if pred_m else None
    conf = _normalise_confidence(conf_m.group(1)) if conf_m else None

    if pred is not None:
        return Task2Parsed(
            ok=True,
            prediction=pred,
            confidence=conf,  # may be None
            reasoning=None,
            raw_text=text,
            error="json parse failed; extracted via regex fallback",
        )

    return Task2Parsed(
        ok=False,
        raw_text=text,
        error="could not extract prediction from response",
    )


# ────────────────────────────────────────────────────────────────────────────────
# Task 1 parser — ranking
# ────────────────────────────────────────────────────────────────────────────────

def parse_task1_response(text: str) -> Task1Parsed:
    """
    Parse a Task 1 LLM response. Expected JSON schema:
        {
          "rankings":    [target_id, ...],
          "confidences": {target_id: float, ...},
          "reasoning":   {target_id: str, ...}
        }
    """
    if not text or not text.strip():
        return Task1Parsed(ok=False, raw_text=text, error="empty response")

    obj = _extract_json(text)
    if obj is None:
        return Task1Parsed(ok=False, raw_text=text, error="no JSON object found")

    rankings = obj.get("rankings")
    if not isinstance(rankings, list) or not rankings:
        return Task1Parsed(
            ok=False,
            raw_text=text,
            error="missing or empty 'rankings' field",
        )
    rankings = [str(r) for r in rankings]

    confidences_raw = obj.get("confidences", {})
    confidences: Dict[str, float] = {}
    if isinstance(confidences_raw, dict):
        for k, v in confidences_raw.items():
            c = _normalise_confidence(v)
            if c is not None:
                confidences[str(k)] = c

    reasoning_raw = obj.get("reasoning", {})
    reasoning: Dict[str, str] = {}
    if isinstance(reasoning_raw, dict):
        for k, v in reasoning_raw.items():
            if isinstance(v, str):
                reasoning[str(k)] = v

    return Task1Parsed(
        ok=True,
        rankings=rankings,
        confidences=confidences,
        reasoning=reasoning,
        raw_text=text,
    )
