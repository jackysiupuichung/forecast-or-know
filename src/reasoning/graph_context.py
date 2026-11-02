#!/usr/bin/env python3
"""Graph-context variant harness — how graph information is fed to the LLM.

Implements the design in notes/graph_retrieval/variant_harness_design.md:
a single `GraphContextSpec` abstraction so feeding-variants are *data*, not code
branches, and the frozen A1/A2/A3 reference arms are reproduced byte-for-byte.

Two dataclasses:
  RenderSpec       — Layers 2-3: how cached hops are rendered (score/year/novelty/
                     evidence visibility, relation schema-vs-gloss, compact/verbose).
  GraphContextSpec — the full arm: graph-context on/off (A1), which cache
                     (topology/snapshot), the RenderSpec, path count, time-gate,
                     and control (shuffle/corrupt).

The two axes A2→A3 conflate are kept separate here: `cache_basename` carries the
snapshot/topology axis; `render` carries the metadata axis (see design §1, the
"A2/A3 overload" risk).

Determinism: a spec is a frozen dataclass fixed per run. Stochastic controls
(shuffle/corrupt) derive randomness from `(pair_key, spec.seed)` ONLY — never RNG
call order or model output — so the rendered EVIDENCE block is a pure function of
(cache, spec) and is byte-identical across all models.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Optional


@dataclass(frozen=True)
class RenderSpec:
    """How a cached hop list is rendered into path strings (Layers 2-3)."""
    show_score: bool = True          # score={weight:.3f}
    show_year: bool = False          # year={edge_time}   (latent leakage cue)
    show_novelty: bool = False       # novelty={edge_novelty:.3f}
    show_evidence: bool = False      # literature snippet + PMID (Tier-2; needs C1 cache)
    relation_style: str = "schema"   # "schema" (clinical_trial_positive) | "gloss" (prose)
    serialise_mode: str = "compact"  # "compact" (arrow lines) | "verbose" (prose)
    max_evidence_chars: int = 200    # deterministic snippet cap (C1)

    def __post_init__(self):
        if self.relation_style not in ("schema", "gloss"):
            raise ValueError(f"relation_style must be schema|gloss, got {self.relation_style!r}")
        if self.serialise_mode not in ("compact", "verbose"):
            raise ValueError(f"serialise_mode must be compact|verbose, got {self.serialise_mode!r}")


@dataclass(frozen=True)
class GraphContextSpec:
    """One arm of the graph->LLM ladder."""
    name: str                                   # registry key
    graph_context: bool                         # False == A1 (no EVIDENCE block, parametric framing)
    cache_basename: Optional[str] = None        # Tier-2 selector: which cache in <evaldir>/paths/
    render: RenderSpec = field(default_factory=RenderSpec)
    max_paths_shown: int = 10                   # Layer 1.4 — paths fed to the LLM (K-Paths default 10)
    time_gate: Optional[str] = None             # None | "cutoff" (drop paths with any hop edge_time > cutoff)
    control: Optional[str] = None               # None | "shuffle" | "corrupt"
    seed: int = 42

    # The ablation family this arm belongs to (A1 vs A2/A3) — governs the prompt
    # framing in build_prompt. graph_context already encodes it, but keep an
    # explicit accessor so the runner/prompt code reads cleanly.
    @property
    def prompt_ablation(self) -> str:
        """Which frozen prompt family this arm renders under. graph_context=False
        -> A1 framing; True -> A2/A3 framing (identical framing; only the rendered
        EVIDENCE block differs, which is what the spec controls)."""
        return "A1" if not self.graph_context else "A3"

    def __post_init__(self):
        if self.time_gate not in (None, "cutoff"):
            raise ValueError(f"time_gate must be None|'cutoff', got {self.time_gate!r}")
        if self.control not in (None, "shuffle", "corrupt"):
            raise ValueError(f"control must be None|'shuffle'|'corrupt', got {self.control!r}")
        if self.graph_context and self.cache_basename is None:
            raise ValueError(f"spec {self.name!r}: graph_context=True requires a cache_basename")


# ── Reference arms (byte-identical to the frozen A1/A2/A3 prompts) ──────────────
# A2 = re-serialise dated cache with show_score=False -> the bracket is empty, i.e.
#   --advancement-->, exactly what _strip_path_annotations produces today.
# A3 = re-serialise newest (snap2026) cache with show_score=True -> score={:.3f},
#   the current default.  (Golden-master test in tests/ enforces this.)
#
# NOTE on cache_basename: these are the winnat reference names. For the disjoint
# runs the caches are advancement-only variants — pass --paths-cache to override,
# or use the *_noadv specs below.
_A2 = GraphContextSpec("A2", graph_context=True,
                       cache_basename="kpaths_h3_p20.json",
                       render=RenderSpec(show_score=False))
_A3 = GraphContextSpec("A3", graph_context=True,
                       cache_basename="kpaths_h3_p20_snap2026.json",
                       render=RenderSpec(show_score=True))
_A1 = GraphContextSpec("A1", graph_context=False, cache_basename=None)

NAMED_SPECS: Dict[str, GraphContextSpec] = {
    # ── reference (frozen) ──
    "A1": _A1,
    "A2": _A2,
    "A3": _A3,

    # ── disjoint advancement-only cache variants (what run_disjoint_paths.sh built) ──
    "A2_noadv": replace(_A2, name="A2_noadv", cache_basename="kpaths_h3_p20_noadv.json"),
    "A3_noadv": replace(_A3, name="A3_noadv", cache_basename="kpaths_h3_p20_snap2026_noadv.json"),

    # ── Tier-1 metadata-richness rungs (off the DATED cache; no post-cutoff leak) ──
    # Built on A2's dated cache but with score shown, then +year, then +novelty:
    "A2_score": replace(_A2, name="A2_score", render=RenderSpec(show_score=True)),
    "A2_year": replace(_A2, name="A2_year",
                       render=RenderSpec(show_score=True, show_year=True)),
    "A2_year_novelty": replace(_A2, name="A2_year_novelty",
                               render=RenderSpec(show_score=True, show_year=True, show_novelty=True)),

    # ── rendering-style variants ──
    "gloss": replace(_A2, name="gloss",
                     render=RenderSpec(show_score=True, relation_style="gloss")),
    "verbose": replace(_A2, name="verbose",
                       render=RenderSpec(show_score=True, serialise_mode="verbose")),

    # ── leakage-correctness: time-gate the NEWEST cache back to the cutoff ──
    # (only meaningful on a cache that CONTAINS post-cutoff edges, i.e. snap2026)
    "A3_time_gated": replace(_A3, name="A3_time_gated", time_gate="cutoff"),

    # ── Tier-2 topology: 2-hop ego neighbourhood (B2). Needs a neigh_r2 cache
    #    built via `precompute_kpaths.py --retrieval-mode neighbors2`. Same render
    #    as A3 so only the topology differs (clean B2-vs-path contrast). ──
    "B2": GraphContextSpec("B2", graph_context=True,
                           cache_basename="neigh_r2_p20.json",
                           render=RenderSpec(show_score=True)),

    # ── Tier-2 topology: PCST connected subgraph (B1 / G-Retriever). Needs a pcst
    #    cache built via `precompute_kpaths.py --retrieval-mode pcst`. Same render
    #    as A3, so only the topology (connected tree vs disjoint paths) differs. ──
    "B1": GraphContextSpec("B1", graph_context=True,
                           cache_basename="pcst_p20.json",
                           render=RenderSpec(show_score=True)),

    # ── Substrate variants for the ∆fore leak decomposition (clinical_outcome
    #    family excluded = advancement + all clinical_trial_*, i.e. the label edge
    #    removed). Same K-Paths topology + score render as A2_score/A3; ONLY the
    #    graph substrate differs. A2_bioonly = dated bio-only; A3_bioonly = fresh
    #    (2026) bio-only. A3−A3_bioonly isolates the direct trial-edge leak;
    #    A3_bioonly−A2 asks whether fresh NON-label biology forecasts. ──
    # A3_bio_only (TIME-GATED): newest-2026 K-Paths with only POST-cutoff clinical/
    # label edges removed (pre-cutoff clinical history KEPT, = A2's). Built date-
    # conditionally (precompute --exclude-after-year=cutoff), p10 = default shown,
    # so each pair gets a full set of leak-free paths (not a render-time deficit).
    # Isolates the leak: A3 − A3_bioonly_tg = post-cutoff trial-edge leak;
    # A3_bioonly_tg − A2 = fresh non-label biology. Framed "December 2026" like A3.
    "A3_bioonly_tg": GraphContextSpec("A3_bioonly_tg", graph_context=True,
                                      cache_basename="kpaths_h3_p10_snap2026_bioonly_tg.json",
                                      render=RenderSpec(show_score=True)),

    # ── control / negative arms (does the model USE the graph?) ──
    # shuffle: A2-style rendering but a DIFFERENT pair's paths (topology test).
    "shuffle": replace(_A2, name="shuffle", control="shuffle"),
    # corrupt MUST show the metadata it randomises, else the test is a no-op —
    # so it renders score+year (the corrupted values), unlike A2.
    "corrupt": replace(_A2, name="corrupt", control="corrupt",
                       render=RenderSpec(show_score=True, show_year=True)),
}


_UNDATED = -9223372036854775808


def _time_gate(hops_lists, cutoff_year):
    """Drop a whole path if ANY hop has edge_time > cutoff (undated/ontology hops
    are always admissible). Leakage-critical: a path is admissible evidence only
    if every dated edge predates the window cutoff."""
    if cutoff_year is None:
        return hops_lists
    out = []
    for hl in hops_lists:
        if any(h.get("edge_time") not in (None, _UNDATED) and h["edge_time"] > cutoff_year
               for h in hl):
            continue
        out.append(hl)
    return out


def _corrupt(hops_lists, pair_key, seed):
    """Randomise edge_weight/novelty/year on a COPY; topology + entities untouched.
    Per-pair-seeded so it's identical across models."""
    import random
    rng = random.Random(hash(pair_key) ^ seed)
    out = []
    for hl in hops_lists:
        nl = []
        for h in hl:
            h2 = dict(h)
            h2["edge_weight"] = rng.random()
            h2["edge_novelty"] = rng.random()
            if h2.get("edge_time") not in (None, _UNDATED):
                h2["edge_time"] = rng.randint(2000, 2025)
            nl.append(h2)
        out.append(nl)
    return out


def render_pairs_for_spec(pairs, spec: GraphContextSpec, name_map, cutoff_year=None):
    """Render every cached pair under `spec`, applying shuffle/corrupt/time-gate.

    `pairs` is the cache's `pairs` map: {"<disease>||<target>": {compact, hops, ...}}.
    Returns {(disease_id, target_id): [rendered path strings, capped to max_paths_shown]}.
    Pure function of (pairs, spec, name_map, cutoff_year) — byte-identical across models.
    """
    import random
    from src.retrieval.serialise import render_paths_from_hops
    keys = sorted(pairs.keys())
    donor = {}
    if spec.control == "shuffle":
        perm = list(keys)
        random.Random(spec.seed).shuffle(perm)     # deterministic donor permutation
        donor = dict(zip(keys, perm))
    out = {}
    for k in keys:
        if "||" not in k:
            continue
        did, tid = k.split("||", 1)
        src = donor.get(k, k)                       # shuffle: render a DIFFERENT pair's hops
        hl = pairs.get(src, {}).get("hops") or []
        if spec.time_gate == "cutoff":
            hl = _time_gate(hl, cutoff_year)
        if spec.control == "corrupt":
            hl = _corrupt(hl, k, spec.seed)
        out[(did, tid)] = render_paths_from_hops(hl, name_map, spec.render)[: spec.max_paths_shown]
    return out


def resolve_spec(variant: Optional[str], ablation: Optional[str] = None) -> GraphContextSpec:
    """Resolve a run to a GraphContextSpec.

    Precedence: explicit --variant NAME > --ablation (backward-compat). Exactly
    one must be resolvable. Unknown names raise with the registry keys listed.
    """
    key = variant or ablation
    if key is None:
        raise ValueError("resolve_spec: provide a variant or an ablation")
    if key not in NAMED_SPECS:
        raise ValueError(
            f"unknown graph-context variant {key!r}. Known: {sorted(NAMED_SPECS)}")
    return NAMED_SPECS[key]
