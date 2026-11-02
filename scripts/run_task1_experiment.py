#!/usr/bin/env python3
"""
Run a Task 1 (target prioritisation) LLM reasoning experiment — pointwise.

Each (target, disease) pair in the eval CSV is scored independently on a
0-10 viability scale. Per-disease rankings are recovered post-hoc by
sorting the disease's candidates by score (ties broken by confidence).

Per disease:
  1. Collect candidate targets (positives + sampled negatives).
  2. (A2/A3) Extract paths for each (disease, target) pair from the graph.
  3. For every candidate, build a pointwise scoring prompt and call the LLM.
  4. Sort candidates by (score desc, confidence desc) → ranking.
  5. Score the ranking with Hit@K / MRR / NDCG@K / Recall@K against labels.

Writes:
    {output_dir}/predictions.jsonl   — one JSON line per disease (record below)
    {output_dir}/metrics.json        — aggregate metrics + bootstrap CIs
    {output_dir}/run_config.json     — full CLI config

Per-disease record:
    disease_id, ablation, model, n_candidates, n_positives,
    candidate_scores: [{target_id, gene_name, label, score, confidence,
                       reasoning, parse_ok, error, raw_text}],
    ranking: [target_id, ...]   (sorted desc by score, conf)
    metrics: {hit_at_k..., recall_at_k..., ndcg_at_k..., mrr,
             first_positive_rank}

Resumability: per-disease cache file in {output_dir}/cache/.

Usage: see scripts/run_inference.sh, which gives the exact arguments of every run in the paper.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, replace as _dc_replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.ranking import (
    precision_at_k,
    first_positive_rank,
    hit_at_k,
    mrr,
    ndcg_at_k,
    recall_at_k,
)
from src.reasoning.llm_client import (
    DETERMINISTIC,
    GenerationConfig,
    LLMClient,
)
from src.reasoning.parser import (
    PointwiseBinaryParsed,
    PointwiseParsed,
    parse_pointwise_binary,
    parse_pointwise_response,
)
from src.reasoning.prompts import build_prompt, disease_from_row, target_from_row
from src.evaluation.binary_metrics import (
    aggregate_binary_metrics,
    distributional_diagnostics,
)
from src.retrieval.paths import PathExtractor, KPathExtractor
from src.retrieval.serialise import PathSerialiser, build_name_map_from_parquets
from src.reasoning.graph_context import resolve_spec, render_pairs_for_spec


# Schema produced by src/evaluation/trial_entry_dataset_builder.py.
COL_TARGET_ID = "targetId"
COL_DISEASE_ID = "diseaseId"
COL_GENE = "gene_name"
COL_PROTEIN_CLASS = "protein_class"
COL_TARGET_DESC = "target_description"
COL_DISEASE_NAME = "disease_name"
COL_DISEASE_TA = "therapeutic_area"
COL_DISEASE_DESC = "disease_description"
COL_LABEL = "label"

K_VALUES = (1, 2, 3, 5, 10)

# Per-call max_tokens budget. Pointwise is one call per pair → constant N.
# Empirically: ~80 tokens of JSON scaffolding + room for a 4-5 sentence
# rationale. 512 is a healthy default; --max-tokens overrides.
_DEFAULT_MAX_TOKENS = 512


# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────

def cache_key(disease_id: str, model: str, ablation: str) -> str:
    safe = disease_id.replace("/", "_").replace(":", "_")
    return f"{safe}__{model.replace('/', '_')}__{ablation}.json"


def _load_usable_cache(cpath: Path) -> Optional[dict]:
    """Return a cached per-disease record, or None to force re-inference.

    A cache hit is only honoured if at least one candidate actually parsed.
    Entries where every candidate has parse_ok=False carry no signal, so
    replaying them silently rebuilds metrics.json from a failed run and
    reports coverage=0 as though it were a result. That happened twice with
    qwen3-8b: the model emitted `<think>` for every probe (thinking not
    disabled), and the resulting all-failed cache masked the later client-side
    fix because inference was skipped entirely. Treat such entries as misses.
    """
    if not cpath.exists():
        return None
    try:
        record = json.loads(cpath.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    scores = record.get("candidate_scores")
    if scores and not any(c.get("parse_ok") for c in scores):
        return None
    return record


def _server_alive(server_url: str, timeout: float = 5.0) -> bool:
    """Probe llama.cpp server `/health` (or `/v1/models` as fallback)."""
    base = server_url.rstrip("/")
    for path in ("/health", "/v1/models"):
        try:
            with urllib.request.urlopen(base + path, timeout=timeout) as resp:
                if 200 <= resp.status < 500:
                    return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
            continue
    return False


def wait_for_server(
    server_url: str,
    max_wait_sec: float = 300.0,
    probe_every_sec: float = 5.0,
) -> bool:
    """Block until the server responds or timeout expires. Returns True if alive."""
    deadline = time.time() + max_wait_sec
    waited = 0.0
    while time.time() < deadline:
        if _server_alive(server_url):
            if waited > 0:
                print(f"  [health] server back online after {waited:.0f}s", flush=True)
            return True
        time.sleep(probe_every_sec)
        waited += probe_every_sec
        if waited % 30 == 0:
            print(f"  [health] still waiting on {server_url} ({waited:.0f}s)", flush=True)
    return False


def load_graph(graph_path: str):
    print(f"Loading graph from {graph_path}...", flush=True)
    return torch.load(graph_path, weights_only=False)


def load_mappings(mappings_path: Optional[str], graph) -> Dict:
    if mappings_path and os.path.exists(mappings_path):
        print(f"Loading mappings from {mappings_path}...", flush=True)
        return torch.load(mappings_path, weights_only=False)
    print("No mappings file; building from graph.node_id lists", flush=True)
    return {
        "node_mapping": {
            nt: {nid: i for i, nid in enumerate(graph[nt].node_id)}
            for nt in graph.node_types
        }
    }


def _row_to_target(row) -> Dict[str, str]:
    # NaN-safe; shared with the training serialisers so prompts match exactly.
    return target_from_row(row)


def _row_to_disease(row) -> Dict[str, str]:
    return disease_from_row(row)


def _rank_candidates(candidate_scores: List[Dict[str, Any]]) -> List[str]:
    """Sort by score desc, confidence desc, with ties broken stably by gene
    symbol (deterministic). Pairs that failed to parse are pushed to the end
    with score = -1 so they don't interfere with the legitimate ranking."""
    def key(c):
        score = c["score"] if c["score"] is not None else -1
        conf = c["confidence"] if c["confidence"] is not None else -1.0
        return (-score, -conf, c.get("gene_name", ""))
    return [c["target_id"] for c in sorted(candidate_scores, key=key)]


def _rank_candidates_binary(candidate_scores: List[Dict[str, Any]]) -> List[str]:
    """Sort by p_yes desc; failed parses pushed to the end. Stable tiebreak
    on gene symbol for reproducibility."""
    def key(c):
        p = c["p_yes"] if c.get("p_yes") is not None else -1.0
        return (-p, c.get("gene_name", ""))
    return [c["target_id"] for c in sorted(candidate_scores, key=key)]


# ────────────────────────────────────────────────────────────────────────────────
# Per-disease execution
# ────────────────────────────────────────────────────────────────────────────────

def run_one_disease(
    disease_id: str,
    rows: pd.DataFrame,
    ablation: str,
    llm_cutoff: str,
    graph_cutoff: str,
    trial_window_year: int,
    si_mode: str,
    recall_mode: bool,
    serialiser: Optional[PathSerialiser],
    extractor,
    llm: LLMClient,
    max_paths: int,
    max_paths_extract: int,
    max_hops: int,
    config: GenerationConfig,
    max_tokens_override: Optional[int] = None,
    paths_cache: Optional[Dict[Tuple[str, str], List[str]]] = None,
    scoring_mode: str = "binary",
    binary_max_tokens: int = 12,
    spec=None,
    label_year: Optional[int] = None,
) -> Dict[str, Any]:
    """Score every candidate target for one disease, then rank."""
    # Graph-context on/off: the variant spec governs it; else fall back to ablation.
    has_graph = bool(spec.graph_context) if spec is not None else (ablation != "A1")
    first = rows.iloc[0]
    disease = _row_to_disease(first)
    candidates = [_row_to_target(r) for _, r in rows.iterrows()]
    ground_truth = {str(r[COL_TARGET_ID]): int(r[COL_LABEL]) for _, r in rows.iterrows()}
    symbol_to_id = {target_from_row(r)["symbol"]: str(r[COL_TARGET_ID]) for _, r in rows.iterrows()}

    # Path extraction (skipped for A1).
    paths_per_target: Dict[str, List[str]] = {}
    n_paths_retrieved = 0
    if has_graph:
        for t in candidates:
            cached = None
            if paths_cache is not None:
                cached = paths_cache.get((disease["id"], t["id"]))
            if cached is not None:
                serialised = cached[:max_paths]
                n_paths_retrieved += len(cached)
                paths_per_target[t["id"]] = serialised
                continue
            if extractor is None or serialiser is None:
                raise RuntimeError(
                    f"Ablation {ablation} requires --graph (and ideally --nodes-dir) "
                    f"or a --paths-cache that covers all pairs"
                )
            try:
                p_list = extractor.find_paths(
                    src_id=disease["id"],
                    dst_id=t["id"],
                    src_type="disease",
                    dst_type="target",
                    max_hops=max_hops,
                    max_paths=max_paths_extract,
                )
            except Exception as e:
                print(f"  [path-extract] {disease['id']}→{t['id']}: {type(e).__name__}: {e}", flush=True)
                p_list = []
            n_paths_retrieved += len(p_list)
            paths_per_target[t["id"]] = [
                serialiser.serialise_path(p, mode="compact") for p in p_list[:max_paths]
            ]

    # Per-pair pointwise scoring.
    if scoring_mode == "binary":
        # Yes/No with top_logprobs for P("Yes"), read at the answer position.
        # A small multi-token window (not max_tokens=1) lets format-locked
        # models (Llama-3.1-70B opens with `{"` regardless of instruction)
        # reach their Yes/No inside the JSON they insist on emitting; terse
        # models still answer at token 0, so greedy decoding leaves their
        # probe readout unchanged.
        budget = binary_max_tokens
        call_config = _dc_replace(
            config,
            max_tokens=budget,
            request_logprobs=True,
            top_logprobs=10,
        )
    else:
        budget = int(max_tokens_override) if max_tokens_override is not None else _DEFAULT_MAX_TOKENS
        call_config = _dc_replace(config, max_tokens=budget)

    candidate_scores: List[Dict[str, Any]] = []
    t_disease = time.time()
    for tgt in candidates:
        paths_arg: Optional[List[str]] = None
        if has_graph:
            paths_arg = paths_per_target.get(tgt["id"], [])
        system, user = build_prompt(
            disease=disease,
            target=tgt,
            ablation=ablation,
            paths=paths_arg,
            llm_cutoff=llm_cutoff,
            graph_cutoff=graph_cutoff,
            trial_window_year=trial_window_year,
            label_year=label_year,
            si_mode=si_mode,
            mode=scoring_mode,
            recall_mode=recall_mode,
            spec=spec,
        )
        t0 = time.time()
        if scoring_mode == "binary":
            try:
                response = llm.complete(system=system, user=user, config=call_config)
                parsed_bin: PointwiseBinaryParsed = parse_pointwise_binary(response.logprobs)
                elapsed = time.time() - t0
                usage = response.usage
                raw = response.content
                err = parsed_bin.error
                parse_ok = parsed_bin.ok
                p_yes = parsed_bin.p_yes
                lp_yes = parsed_bin.logprob_yes
                lp_no = parsed_bin.logprob_no
                top_token = parsed_bin.top_token
                answer_token = parsed_bin.answer_token
                answer_position = parsed_bin.answer_position
                n_yes_tokens = parsed_bin.n_yes_tokens
                n_no_tokens = parsed_bin.n_no_tokens
            except Exception as e:
                elapsed = time.time() - t0
                usage = {}
                raw = ""
                err = f"{type(e).__name__}: {e}"
                parse_ok = False
                p_yes = lp_yes = lp_no = None
                top_token = None
                answer_token = None
                answer_position = None
                n_yes_tokens = n_no_tokens = 0

            candidate_scores.append({
                "target_id": tgt["id"],
                "gene_name": tgt["symbol"],
                "label": ground_truth[tgt["id"]],
                "n_paths": len(paths_per_target.get(tgt["id"], [])) if ablation != "A1" else 0,
                "p_yes": p_yes,
                "logprob_yes": lp_yes,
                "logprob_no": lp_no,
                "top_token": top_token,
                "answer_token": answer_token,
                "answer_position": answer_position,
                "n_yes_tokens": n_yes_tokens,
                "n_no_tokens": n_no_tokens,
                "score": None,
                "confidence": None,
                "reasoning": None,
                "parse_ok": parse_ok,
                "error": err,
                "raw_response": raw,
                "elapsed_sec": round(elapsed, 3),
                "usage": usage,
            })
        else:
            try:
                response = llm.complete(system=system, user=user, config=call_config)
                parsed: PointwiseParsed = parse_pointwise_response(response.content)
                elapsed = time.time() - t0
                usage = response.usage
                raw = response.content
                err = parsed.error
            except Exception as e:
                parsed = PointwiseParsed(
                    ok=False, raw_text="", error=f"{type(e).__name__}: {e}"
                )
                elapsed = time.time() - t0
                usage = {}
                raw = ""
                err = parsed.error

            candidate_scores.append({
                "target_id": tgt["id"],
                "gene_name": tgt["symbol"],
                "label": ground_truth[tgt["id"]],
                "n_paths": len(paths_per_target.get(tgt["id"], [])) if ablation != "A1" else 0,
                "score": parsed.score,
                "confidence": parsed.confidence,
                "reasoning": parsed.reasoning,
                "parse_ok": parsed.ok,
                "error": err,
                "raw_response": raw,
                "elapsed_sec": round(elapsed, 3),
                "usage": usage,
            })

    elapsed_disease = time.time() - t_disease

    # Recover ranking from pointwise scores.
    if scoring_mode == "binary":
        ranking = _rank_candidates_binary(candidate_scores)
    else:
        ranking = _rank_candidates(candidate_scores)

    # Score against ground truth.
    metrics_per_disease: Dict[str, Optional[float]] = {}
    n_parse_ok = sum(1 for c in candidate_scores if c["parse_ok"])
    if n_parse_ok > 0:
        for k in K_VALUES:
            metrics_per_disease[f"hit_at_{k}"] = hit_at_k(
                ranking, ground_truth, k=k, symbol_to_id=symbol_to_id
            )
            metrics_per_disease[f"recall_at_{k}"] = recall_at_k(
                ranking, ground_truth, k=k, symbol_to_id=symbol_to_id
            )
            metrics_per_disease[f"precision_at_{k}"] = precision_at_k(
                ranking, ground_truth, k=k, symbol_to_id=symbol_to_id
            )
            metrics_per_disease[f"ndcg_at_{k}"] = ndcg_at_k(
                ranking, ground_truth, k=k, symbol_to_id=symbol_to_id
            )
        metrics_per_disease["mrr"] = mrr(ranking, ground_truth, symbol_to_id)
        metrics_per_disease["first_positive_rank"] = first_positive_rank(
            ranking, ground_truth, symbol_to_id
        )
    else:
        for k in K_VALUES:
            metrics_per_disease[f"hit_at_{k}"] = None
            metrics_per_disease[f"recall_at_{k}"] = None
            metrics_per_disease[f"precision_at_{k}"] = None
            metrics_per_disease[f"ndcg_at_{k}"] = None
        metrics_per_disease["mrr"] = None
        metrics_per_disease["first_positive_rank"] = None

    return {
        "disease_id": disease["id"],
        "ablation": ablation,
        "model": llm.model,
        "provider": llm.provider,
        "n_candidates": len(candidates),
        "n_positives": sum(ground_truth.values()),
        "n_paths_retrieved": n_paths_retrieved,
        "n_parse_ok": n_parse_ok,
        "max_tokens_used": budget,
        "ground_truth_labels": ground_truth,
        "symbol_to_id": symbol_to_id,
        "candidate_scores": candidate_scores,
        "ranking": ranking,
        "elapsed_sec": round(elapsed_disease, 3),
        "metrics": metrics_per_disease,
    }


# ────────────────────────────────────────────────────────────────────────────────
# Aggregation
# ────────────────────────────────────────────────────────────────────────────────

from src.evaluation.bootstrap import bootstrap_mean_ci as _bootstrap_mean_ci  # noqa: E402


def aggregate_metrics(records: List[Dict]) -> Dict:
    """Per-disease metrics → bootstrap-CI macro-averages."""
    # A disease counts toward metrics if at least one candidate parsed OK.
    parseable = [r for r in records if r.get("n_parse_ok", 0) > 0]
    n_total = len(records)
    n_signal = len(parseable)
    out: Dict[str, Any] = {
        "n_diseases": n_total,
        "n_diseases_with_signal": n_signal,
        "n_diseases_no_signal": n_total - n_signal,
        "coverage": (n_signal / n_total) if n_total else 0.0,
    }
    for k in K_VALUES:
        out[f"hit_at_{k}"] = _bootstrap_mean_ci(
            [r["metrics"][f"hit_at_{k}"] for r in parseable]
        )
        out[f"recall_at_{k}"] = _bootstrap_mean_ci(
            [r["metrics"][f"recall_at_{k}"] for r in parseable]
        )
        out[f"ndcg_at_{k}"] = _bootstrap_mean_ci(
            [r["metrics"][f"ndcg_at_{k}"] for r in parseable]
        )
    out["mrr"] = _bootstrap_mean_ci(
        [r["metrics"]["mrr"] for r in parseable if r["metrics"]["mrr"] is not None]
    )
    out["first_positive_rank"] = _bootstrap_mean_ci(
        [r["metrics"]["first_positive_rank"] for r in parseable
         if r["metrics"]["first_positive_rank"] is not None]
    )
    return out


# ────────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Pointwise Task 1 LLM experiment")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--graph", default=None,
                    help="Path to HeteroData snapshot — required for A2/A3")
    ap.add_argument("--mappings", default=None)
    ap.add_argument("--nodes-dir", default=None,
                    help="Dir for path-name lookup (evidenceDated/ or nodes/)")
    ap.add_argument("--provider", default="auto",
                    choices=["auto", "vllm", "transformers", "llama_cpp_server"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None,
                    help="Optional LoRA adapter dir (PEFT) for the finetuned arm; "
                         "transformers provider only. Base = --model, + this adapter.")
    ap.add_argument("--quantization", default=None,
                    choices=[None, "awq", "gptq", "bitsandbytes"])
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--server-url", default=None,
                    help="Required for --provider llama_cpp_server "
                         "(e.g. http://127.0.0.1:8080)")
    ap.add_argument("--server-wait-sec", type=float, default=300.0,
                    help="Max seconds to wait for llama.cpp server to recover "
                         "after a mid-run crash before aborting (default 300)")
    ap.add_argument("--min-coverage", type=float, default=0.95,
                    help="Fail the metrics summary if "
                         "n_diseases_with_signal/n_diseases falls below this "
                         "fraction (default 0.95). Set to 0 to disable.")
    ap.add_argument("--ablation", required=True, choices=["A1", "A2", "A3"])
    ap.add_argument("--variant", default=None,
                    help="Graph-context variant (src/reasoning/graph_context.py "
                         "NAMED_SPECS key, e.g. A2_year, gloss, verbose, shuffle, "
                         "corrupt, A3_time_gated). Overrides --ablation's rendering; "
                         "re-serialises the cached hops under the variant's RenderSpec. "
                         "Absent = frozen --ablation behaviour (byte-identical).")
    ap.add_argument("--llm-cutoff", default="December 2023",
                    help='LLM training cutoff (used in A1 system prompt)')
    ap.add_argument("--graph-cutoff", default="June 2024",
                    help='Graph cutoff date (used in A2/A3 system prompt)')
    ap.add_argument("--trial-window-year", type=int, default=2023,
                    help='Year used in the binary question "...enter a trial '
                         'after YEAR?". Must match the eval set\'s cutoff_year '
                         'so positive labels and the question align.')
    ap.add_argument("--label-year", type=int, default=None,
                    help='If set, ask the one-year question "...enter its first '
                         'clinical trial ... in LABEL_YEAR?" (TGB-style query; '
                         'label = first trial in that year). Use with '
                         '--trial-window-year = LABEL_YEAR-1. Unset = legacy wording.')
    ap.add_argument("--si-mode", choices=["off", "strong"], default="off",
                    help='Simulated Ignorance mode (per Li et al. 2026). "off": '
                         'system prompt simply states the cutoff_date. "strong": '
                         'explicit roleplay instruction telling the model to '
                         'behave as if its cutoff is the given date.')
    ap.add_argument("--recall-mode", action="store_true", default=False,
                    help='Switch the A1 binary question from prediction ("...will '
                         'X enter a trial after Y?") to recall ("...has X '
                         'entered a trial on or before Y?"). Only valid with '
                         '--ablation A1.')
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-hops", type=int, default=3)
    ap.add_argument("--max-paths", type=int, default=8,
                    help="Paths shown to the LLM per (target, disease) pair")
    ap.add_argument("--max-paths-extract", type=int, default=30)
    ap.add_argument("--retrieval-method",
                    choices=["bfs", "kpaths", "no_filter", "neighbors_only"],
                    default="bfs")
    ap.add_argument("--paths-cache", default=None,
                    help="Optional precomputed paths JSON (from "
                         "scripts/precompute_kpaths.py). When provided, "
                         "(disease, target) pairs found in the cache reuse "
                         "the precomputed compact strings; missing pairs "
                         "fall back to live extraction.")
    ap.add_argument("--max-tokens", type=int, default=_DEFAULT_MAX_TOKENS)
    ap.add_argument("--scoring-mode", choices=["rubric", "binary"], default="binary",
                    help="binary: Yes/No with P(Yes) from logprobs at the answer "
                         "position (default). rubric: legacy 0-10 integer JSON "
                         "output.")
    ap.add_argument("--binary-max-tokens", type=int, default=12,
                    help="Generation window for binary scoring. The probe reads "
                         "logprobs at the first sampled Yes/No token, so terse "
                         "models are unaffected (greedy answer at token 0); a "
                         "window >1 lets JSON-wrapping models (Llama-3.1-70B) "
                         "reach the Yes/No inside their JSON. Set 1 to force "
                         "the legacy single-token probe.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit to first N diseases (debug)")
    ap.add_argument("--skip-inference", action="store_true",
                    help="Recompute metrics from existing predictions.jsonl")
    args = ap.parse_args()

    if (
        args.ablation != "A1"
        and args.graph is None
        and args.paths_cache is None
        and not args.skip_inference
    ):
        ap.error(f"--graph or --paths-cache is required for ablation {args.ablation}")

    if args.recall_mode and args.ablation != "A1":
        ap.error("--recall-mode is only supported with --ablation A1")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = out / "cache"
    cache_dir.mkdir(exist_ok=True)
    with open(out / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    predictions_path = out / "predictions.jsonl"

    if not args.skip_inference:
        df = pd.read_csv(args.dataset)
        diseases = list(dict.fromkeys(df[COL_DISEASE_ID].tolist()))
        if args.limit:
            diseases = diseases[: args.limit]
        print(f"Dataset: {len(df):,} rows / {len(diseases):,} unique diseases",
              flush=True)

        spec_obj = resolve_spec(args.variant) if args.variant else None
        has_graph_run = (bool(spec_obj.graph_context) if spec_obj is not None
                         else args.ablation != "A1")
        extractor = None
        serialiser: Optional[PathSerialiser] = None
        paths_cache_map: Optional[Dict[Tuple[str, str], List[str]]] = None
        if has_graph_run:
            if args.paths_cache:
                print(f"Loading precomputed paths from {args.paths_cache}...", flush=True)
                with open(args.paths_cache) as f:
                    cache_blob = json.load(f)
                pairs = cache_blob.get("pairs", {})
                if spec_obj is not None:
                    # Variant: re-serialise the cached HOPS under the spec's RenderSpec
                    # (year/novelty/gloss/verbose) + apply shuffle/corrupt/time-gate.
                    # name_map from the cache's own nodes_dir so display names match.
                    sys.path.insert(0, str(Path(__file__).resolve().parent))
                    from precompute_kpaths import build_name_map_evidence_dated
                    ndir = cache_blob.get("config", {}).get("nodes_dir")
                    print(f"  [variant {spec_obj.name}] building name_map from {ndir} ...",
                          flush=True)
                    vmap = build_name_map_evidence_dated(ndir) if ndir else {}
                    paths_cache_map = render_pairs_for_spec(
                        pairs, spec_obj, vmap, cutoff_year=args.trial_window_year)
                    print(f"  [variant {spec_obj.name}] rendered {len(paths_cache_map):,} pairs "
                          f"(render={spec_obj.render}, control={spec_obj.control}, "
                          f"time_gate={spec_obj.time_gate}, max_paths={spec_obj.max_paths_shown})",
                          flush=True)
                else:
                    paths_cache_map = {}
                    for k, v in pairs.items():
                        if "||" not in k:
                            continue
                        did_k, tid_k = k.split("||", 1)
                        paths_cache_map[(did_k, tid_k)] = list(v.get("compact", []))
                    print(f"  loaded {len(paths_cache_map):,} (disease, target) pairs",
                          flush=True)
            if args.graph is not None:
                graph = load_graph(args.graph)
                mappings = load_mappings(args.mappings, graph)
                if args.retrieval_method == "bfs":
                    extractor = PathExtractor(graph, mappings=mappings)
                else:
                    extractor = KPathExtractor(graph, mappings=mappings,
                                               mode=args.retrieval_method)
                name_map = {}
                if args.nodes_dir:
                    print(f"Building name map from {args.nodes_dir}...", flush=True)
                    name_map = build_name_map_from_parquets(args.nodes_dir)
                serialiser = PathSerialiser(name_map=name_map)

        config = GenerationConfig(
            max_tokens=args.max_tokens,
            temperature=DETERMINISTIC.temperature,
            top_p=DETERMINISTIC.top_p,
            top_k=DETERMINISTIC.top_k,
            seed=DETERMINISTIC.seed,
        )
        llm = LLMClient(
            provider=args.provider,
            model=args.model,
            quantization=args.quantization,
            gpu_memory_utilization=args.gpu_memory_utilization,
            server_url=args.server_url,
            default_config=config,
            adapter=args.adapter,
        )

        # Establish baseline server health before starting (llama.cpp only).
        if args.provider == "llama_cpp_server":
            if not args.server_url:
                ap.error("--server-url is required for --provider llama_cpp_server")
            if not wait_for_server(args.server_url, max_wait_sec=60.0):
                raise RuntimeError(
                    f"llama.cpp server at {args.server_url} unreachable at startup"
                )
            print(f"  [health] server reachable at {args.server_url}", flush=True)

        n_done = n_skipped = n_failed = 0
        with open(predictions_path, "w") as out_f:
            for idx, did in enumerate(diseases):
                rows = df[df[COL_DISEASE_ID] == did]
                cpath = cache_dir / cache_key(str(did), args.model, args.ablation)
                cached = _load_usable_cache(cpath)
                if cached is not None:
                    record = cached
                    n_skipped += 1
                else:
                    # Health-check the server before each uncached disease so a
                    # mid-run crash blocks rather than silently producing
                    # `Connection refused` for every remaining call.
                    if args.provider == "llama_cpp_server":
                        if not _server_alive(args.server_url):
                            print(f"[{idx}] server down at {args.server_url} — "
                                  f"waiting up to {args.server_wait_sec}s",
                                  flush=True)
                            if not wait_for_server(
                                args.server_url,
                                max_wait_sec=args.server_wait_sec,
                            ):
                                raise RuntimeError(
                                    f"llama.cpp server at {args.server_url} did "
                                    f"not recover within {args.server_wait_sec}s "
                                    f"— aborting at disease {idx}/{len(diseases)} "
                                    f"({did}) to avoid silently corrupting metrics"
                                )
                    try:
                        record = run_one_disease(
                            disease_id=str(did),
                            rows=rows,
                            ablation=args.ablation,
                            llm_cutoff=args.llm_cutoff,
                            graph_cutoff=args.graph_cutoff,
                            trial_window_year=args.trial_window_year,
                            label_year=args.label_year,
                            si_mode=args.si_mode,
                            recall_mode=args.recall_mode,
                            serialiser=serialiser,
                            extractor=extractor,
                            llm=llm,
                            max_paths=args.max_paths,
                            max_paths_extract=args.max_paths_extract,
                            max_hops=args.max_hops,
                            config=config,
                            max_tokens_override=args.max_tokens,
                            paths_cache=paths_cache_map,
                            scoring_mode=args.scoring_mode,
                            binary_max_tokens=args.binary_max_tokens,
                            spec=spec_obj,
                        )
                        cpath.write_text(json.dumps(record))
                        n_done += 1
                    except Exception as e:
                        print(f"[{idx}] FAILED {did}: {type(e).__name__}: {e}",
                              flush=True)
                        record = {
                            "disease_id": str(did),
                            "ablation": args.ablation,
                            "model": args.model,
                            "error": f"{type(e).__name__}: {e}",
                            "n_parse_ok": 0,
                            "metrics": {**{f"hit_at_{k}": None for k in K_VALUES},
                                        **{f"recall_at_{k}": None for k in K_VALUES},
                                        **{f"ndcg_at_{k}": None for k in K_VALUES},
                                        "mrr": None, "first_positive_rank": None},
                        }
                        n_failed += 1

                out_f.write(json.dumps(record) + "\n")
                if (idx + 1) % 10 == 0:
                    print(f"  [{idx + 1}/{len(diseases)}] done={n_done} "
                          f"skipped={n_skipped} failed={n_failed}", flush=True)

        print(f"\nInference complete: done={n_done} skipped={n_skipped} "
              f"failed={n_failed}", flush=True)

    # ── Metrics ──
    print("\nComputing metrics...", flush=True)
    records = [json.loads(line) for line in open(predictions_path)]
    metrics_out = aggregate_metrics(records)
    if args.scoring_mode == "binary":
        metrics_out["binary"] = aggregate_binary_metrics(records)
        metrics_out["binary"]["distribution"] = distributional_diagnostics(records)
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print(f"TASK 1 METRICS — ablation {args.ablation} (pointwise)")
    print("=" * 60)
    n_total = metrics_out["n_diseases"]
    n_signal = metrics_out["n_diseases_with_signal"]
    coverage = n_signal / n_total if n_total else 0.0
    print(f"n diseases = {n_total}  "
          f"(with_signal={n_signal}, "
          f"no_signal={metrics_out['n_diseases_no_signal']}, "
          f"coverage={coverage:.1%})")
    if coverage < args.min_coverage:
        print("!" * 60)
        print(f"!! WARNING: coverage {coverage:.1%} < min_coverage "
              f"{args.min_coverage:.1%}")
        print(f"!! Reported metrics aggregate only {n_signal}/{n_total} diseases.")
        print(f"!! Likely cause: llama.cpp server crashed mid-run "
              f"(check predictions.jsonl errors).")
        print("!" * 60)
    def _ci(d, fmt=".3f"):
        if d.get("lo") is None or d.get("hi") is None:
            return f"  CI [n/a, n/a] (n={d.get('n', 0)})"
        return f"  CI [{format(d['lo'], fmt)}, {format(d['hi'], fmt)}]"

    for k in K_VALUES:
        h = metrics_out[f"hit_at_{k}"]
        r = metrics_out[f"recall_at_{k}"]
        n = metrics_out[f"ndcg_at_{k}"]
        if h["mean"] is not None:
            print(f"  Hit@{k:<2}    = {h['mean']:.3f}{_ci(h)}")
        if r["mean"] is not None:
            print(f"  Recall@{k:<2} = {r['mean']:.3f}{_ci(r)}")
        if n["mean"] is not None:
            print(f"  NDCG@{k:<2}   = {n['mean']:.3f}{_ci(n)}")
    m_mrr = metrics_out["mrr"]
    if m_mrr["mean"] is not None:
        print(f"  MRR        = {m_mrr['mean']:.3f}{_ci(m_mrr)}")
    fpr = metrics_out["first_positive_rank"]
    if fpr["mean"] is not None:
        print(f"  Mean first-positive rank = {fpr['mean']:.2f}{_ci(fpr, '.2f')}")

    if args.scoring_mode == "binary" and "binary" in metrics_out:
        b = metrics_out["binary"]
        pooled = b["pooled"]
        macro = b["per_disease_macro"]
        dist = b["distribution"]
        print("\n" + "=" * 60)
        print("BINARY HEADLINE (P(Yes) from logprobs)")
        print("=" * 60)
        if pooled["roc_auc"] is not None:
            print(f"  Pooled ROC-AUC = {pooled['roc_auc']:.3f}  PR-AUC = {pooled['pr_auc']:.3f}")
            print(f"  Pooled Brier   = {pooled['brier']:.4f}     ECE    = {pooled['ece']:.4f}")
            print(f"  n_pairs={pooled['n_pairs']}  n_positives={pooled['n_positives']}")
        for name in ("roc_auc", "pr_auc", "brier", "ece"):
            m = macro.get(name)
            if m and m.get("mean") is not None:
                print(f"  Per-disease {name:<8} = {m['mean']:.3f}{_ci(m)}  n={m['n']}")
        if dist.get("n", 0) > 0:
            print(f"  P(Yes) dist:  mean={dist['mean']:.3f}  stdev={dist['stdev']:.3f}  "
                  f"distinct@2dp={dist['n_distinct_2dp']}  range=[{dist['min']:.3f}, {dist['max']:.3f}]")


if __name__ == "__main__":
    main()
