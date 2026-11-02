#!/usr/bin/env python3
"""
Precompute K-path retrievals for every (disease, target) pair in the
evaluation set.

Output is written once and reused across all 6 LLMs × {A2, A3} ablations,
guaranteeing identical path inputs across model comparisons (path
extraction is deterministic given a fixed graph).

Reads:  data/evaluation/trial_entry_cutoff_2023/*.csv  (all 9 ratio CSVs)
Loads:  data/snapshots/snapshot_<C>_datatype.pt
Names:  $OT_DIR/{diseases,targets,go,reactome,molecule}/*.parquet
Writes: data/evaluation/trial_entry_cutoff_2023/paths/kpaths_h3_p30.json

Output JSON shape:
    {
      "config": {mode, max_hops, max_paths, graph, n_pairs, ...},
      "pairs": {
        "<disease_id>||<target_id>": {
          "n_paths": int,
          "compact": [str, ...],     # serialised, ready to feed build_prompt
          "hops":    [[hop_dict, ...], ...]   # raw, for re-serialisation
        },
        ...
      }
    }
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.paths import KPathExtractor, clinical_outcome_relations
from src.retrieval.serialise import PathSerialiser


COL_TARGET_ID = "targetId"
COL_DISEASE_ID = "diseaseId"


# evidenceDated/ uses multi-part parquets and column names that differ from the
# `evidences/nodes/` layout `build_name_map_from_parquets` was written for, so
# we build the name_map directly here.
_NAME_MAP_SPEC = [
    ("disease",  "diseases/*.parquet",  "name"),
    ("target",   "targets/*.parquet",   "approvedSymbol"),
    ("go",       "go/*.parquet",        "label"),
    ("reactome", "reactome/*.parquet",  "label"),
    ("molecule", "molecule/*.parquet",  "name"),
]


def build_name_map_evidence_dated(root: str) -> Dict[str, Dict[str, str]]:
    import glob
    name_map: Dict[str, Dict[str, str]] = {}
    for node_type, pattern, name_col in _NAME_MAP_SPEC:
        files = sorted(glob.glob(f"{root}/{pattern}"))
        if not files:
            continue
        frames = []
        for f in files:
            try:
                frames.append(pd.read_parquet(f, columns=["id", name_col]))
            except (KeyError, ValueError):
                pass
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        df = df.dropna(subset=["id", name_col]).drop_duplicates(subset=["id"])
        name_map[node_type] = dict(zip(df["id"].astype(str), df[name_col].astype(str)))
    return name_map


def collect_pairs(csv_dir: Path) -> List[Tuple[str, str]]:
    """Dedup (disease_id, target_id) across every CSV in csv_dir."""
    seen: Set[Tuple[str, str]] = set()
    for csv in sorted(csv_dir.glob("trial_entry_*.csv")):
        df = pd.read_csv(csv, usecols=[COL_DISEASE_ID, COL_TARGET_ID], dtype=str)
        for d, t in zip(df[COL_DISEASE_ID], df[COL_TARGET_ID]):
            seen.add((d, t))
    return sorted(seen)


def main() -> None:
    ap = argparse.ArgumentParser(description="Precompute K-paths for eval pairs")
    ap.add_argument("--graph", default="data/snapshots/snapshot_2023_datatype.pt")
    ap.add_argument("--mappings", default=None)
    ap.add_argument(
        "--nodes-dir",
        default=os.environ.get("OT_DIR", "data/opentargets/26.03/evidenceDated"),
    )
    ap.add_argument(
        "--csv-dir", default="data/evaluation/trial_entry_cutoff_2023"
    )
    ap.add_argument(
        "--output",
        default="data/evaluation/trial_entry_cutoff_2023/paths/kpaths_h3_p30.json",
    )
    ap.add_argument("--max-hops", type=int, default=3)
    ap.add_argument("--max-paths", type=int, default=30)
    ap.add_argument("--retrieval-mode",
                    choices=["kpaths", "no_filter", "neighbors_only", "neighbors2", "pcst"],
                    default="kpaths",
                    help="Topology retrieved per pair. 'kpaths' (default) = Yen "
                         "shortest simple paths (K-Paths). 'neighbors2' = 2-hop ego "
                         "neighbourhood of each endpoint (B2/LightRAG-style). Recorded "
                         "in the cache config.mode. Name the --output distinctly "
                         "(e.g. neigh_r2_p20.json) so it doesn't overwrite the path cache.")
    ap.add_argument(
        "--exclude-relations", default=None,
        help="Relations to drop from the retrieval graph so no extracted path "
             "routes through them (leakage-safe substrate). One of: the named "
             "family 'clinical_outcome' (all clinical_trial* + "
             "clinically_associated + advancement, resolved against the graph); "
             "a comma-separated list of exact relation names; or 'none'. "
             "Recorded in the cache config as exclude_relations[_family].")
    ap.add_argument("--exclude-relations-after", default=None,
                    help="DATE-CONDITIONAL exclusion set (separate from the always-"
                         "unconditional --exclude-relations): these relations are "
                         "dropped only for edges dated strictly after "
                         "--exclude-after-year (the leak); edges dated <= it and "
                         "undated are KEPT. For the time-gated A3_bio_only arm pass "
                         "the clinical family here (keep pre-cutoff clinical history, "
                         "strip only post-cutoff) and put 'advancement' in "
                         "--exclude-relations so it is removed unconditionally. Same "
                         "grammar as --exclude-relations (family name or CSV list).")
    ap.add_argument("--exclude-after-year", type=int, default=None,
                    help="Cutoff year for --exclude-relations-after: those relations "
                         "are dropped only for edges dated strictly after this year. "
                         "No effect unless --exclude-relations-after is also set.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit to first N pairs (debug)")
    ap.add_argument("--shard", default=None,
                    help="Process only shard i of N, format 'i/N' (0-indexed). "
                         "Pairs are deterministically partitioned by stable sort "
                         "order, so N array tasks cover the pair set with no "
                         "overlap. Each shard writes a partial cache to "
                         "<output>.shard<i>of<N>; merge with --merge-shards. "
                         "Enables embarrassingly-parallel array jobs (the graph "
                         "is read-only, loaded once per task).")
    ap.add_argument("--merge-shards", type=int, default=None, metavar="N",
                    help="Merge mode: read <output>.shard<i>of<N> for i in [0,N) "
                         "and write the unified cache to <output>. No graph load. "
                         "Run after all shard array tasks complete.")
    ap.add_argument("--progress-every", type=int, default=200)
    ap.add_argument("--checkpoint-every", type=int, default=2000,
                    help="Atomically write the merged cache to --output every "
                         "N processed pairs so a Slurm timeout / SIGTERM "
                         "preserves progress. Set to 0 to disable.")
    ap.add_argument("--resume-from", default=None,
                    help="Path to an existing cache JSON. Pairs already "
                         "present in that cache are skipped; only new pairs "
                         "are extracted, then the merged result is written "
                         "to --output. Use this to extend the cache after "
                         "adding a new ratio (e.g. 1:5 → 1:10) without "
                         "recomputing the existing pairs.")
    args = ap.parse_args()

    # Resolved set of excluded relations (populated after the graph is loaded).
    # Kept None on the merge / re-emit paths, which inherit provenance instead.
    exclude = None

    csv_dir = Path(args.csv_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Merge mode: stitch shard partials into the unified cache, no graph load ──
    if args.merge_shards is not None:
        N = args.merge_shards
        merged: Dict[str, Dict] = {}
        n_found = 0
        # Inherit graph/nodes_dir from the SHARD configs — the shards were built
        # at the correct per-window snapshot, whereas this merge process does NOT
        # load a graph and so `args.graph` is just the CLI default (snapshot_2023).
        # Stamping args.graph here mislabels the merged config and breaks the
        # leakage check. Take the snapshot provenance from shard 0's config.
        shard_graph = None
        shard_nodes_dir = None
        shard_exclude = None
        shard_exclude_family = None
        for i in range(N):
            shard_path = out_path.with_name(f"{out_path.name}.shard{i}of{N}")
            if not shard_path.exists():
                raise SystemExit(
                    f"merge: missing shard {shard_path} — did array task {i} finish?"
                )
            blob = json.loads(shard_path.read_text())
            if i == 0:
                shard_cfg = blob.get("config", {}) or {}
                shard_graph = shard_cfg.get("graph")
                shard_nodes_dir = shard_cfg.get("nodes_dir")
                shard_exclude = shard_cfg.get("exclude_relations")
                shard_exclude_family = shard_cfg.get("exclude_relations_family")
            merged.update(blob.get("pairs", {}) or {})
            n_found += 1
        n_with_paths = sum(1 for v in merged.values() if v.get("n_paths", 0) > 0)
        n_zero = sum(1 for v in merged.values() if v.get("n_paths", 0) == 0)
        payload = {
            "config": {
                "mode": "kpaths",
                "max_hops": args.max_hops,
                "max_paths": args.max_paths,
                # provenance from the shards (correct snapshot), not the CLI default
                "graph": shard_graph if shard_graph is not None else args.graph,
                "nodes_dir": shard_nodes_dir if shard_nodes_dir is not None else args.nodes_dir,
                "csv_dir": str(csv_dir),
                "n_pairs": len(merged),
                "n_with_paths": n_with_paths,
                "n_zero": n_zero,
                "merged_from_shards": N,
                # provenance from the shards (built with the graph loaded)
                "exclude_relations_family": shard_exclude_family,
                "exclude_relations": shard_exclude,
            },
            "pairs": merged,
        }
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        with open(tmp_path, "w") as f:
            json.dump(payload, f)
        tmp_path.replace(out_path)
        print(f"Merged {n_found} shards → {out_path} "
              f"({len(merged):,} pairs, with_paths={n_with_paths:,})", flush=True)
        return

    print(f"Collecting pairs from {csv_dir}...", flush=True)
    all_pairs = collect_pairs(csv_dir)
    print(f"  {len(all_pairs):,} unique (disease, target) pairs across CSVs",
          flush=True)

    # ── Shard mode: deterministic 1/N partition of the (stably-sorted) pair set ──
    shard_i = shard_n = None
    if args.shard is not None:
        try:
            shard_i, shard_n = (int(x) for x in args.shard.split("/"))
        except ValueError:
            raise SystemExit(f"--shard must be 'i/N', got {args.shard!r}")
        if not (0 <= shard_i < shard_n):
            raise SystemExit(f"--shard i must be in [0, N); got {args.shard}")
        # Stride partition: pair k goes to shard (k % N). Balances load even when
        # cost correlates with sort position (e.g. disease-grouped runs).
        all_pairs = [p for k, p in enumerate(all_pairs) if k % shard_n == shard_i]
        # Redirect output to a per-shard partial; merge step assembles the whole.
        out_path = out_path.with_name(f"{out_path.name}.shard{shard_i}of{shard_n}")
        print(f"  shard {shard_i}/{shard_n}: {len(all_pairs):,} pairs → {out_path.name}",
              flush=True)

    # Resume mode: pre-load the existing cache and skip pairs that are
    # already present. New pairs (= the delta between the latest CSVs and
    # the cache) are processed; everything is then re-emitted to the new
    # output as a single merged cache.
    existing_pairs: Dict[str, Dict] = {}
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            raise SystemExit(f"--resume-from path does not exist: {resume_path}")
        print(f"Loading existing cache from {resume_path}...", flush=True)
        existing_blob = json.loads(resume_path.read_text())
        existing_pairs = existing_blob.get("pairs", {}) or {}
        print(f"  loaded {len(existing_pairs):,} cached (disease, target) pairs",
              flush=True)
        skip_keys = set(existing_pairs.keys())
        pairs = [(d, t) for (d, t) in all_pairs if f"{d}||{t}" not in skip_keys]
        print(f"  delta to extract: {len(pairs):,} new pairs", flush=True)
    else:
        pairs = all_pairs

    if args.limit:
        pairs = pairs[: args.limit]

    if not pairs and not existing_pairs:
        raise SystemExit("nothing to do: no pairs to extract and no cache to merge")

    if not pairs:
        print("No new pairs to extract — re-emitting existing cache to --output",
              flush=True)
        out_pairs: Dict[str, Dict] = dict(existing_pairs)
        n_with_paths = sum(1 for v in out_pairs.values() if v.get("n_paths", 0) > 0)
        n_zero = sum(1 for v in out_pairs.values() if v.get("n_paths", 0) == 0)
        n_err = 0
        elapsed = 0.0
        payload = {
            "config": {
                "mode": "kpaths",
                "max_hops": args.max_hops,
                "max_paths": args.max_paths,
                "graph": args.graph,
                "nodes_dir": args.nodes_dir,
                "csv_dir": str(csv_dir),
                "n_pairs": len(out_pairs),
                "n_with_paths": n_with_paths,
                "n_zero": n_zero,
                "n_err": n_err,
                "elapsed_sec": round(elapsed, 1),
                "resumed_from": args.resume_from,
                "exclude_relations_family": args.exclude_relations,
                "exclude_relations": sorted(exclude) if exclude else None,
            },
            "pairs": out_pairs,
        }
        print(f"Writing {out_path}...", flush=True)
        with open(out_path, "w") as f:
            json.dump(payload, f)
        print(f"  size: {out_path.stat().st_size / 1e6:.1f} MB", flush=True)
        return

    print(f"Loading graph from {args.graph}...", flush=True)
    graph = torch.load(args.graph, weights_only=False)

    mappings = None
    if args.mappings:
        print(f"Loading mappings from {args.mappings}...", flush=True)
        mappings = torch.load(args.mappings, weights_only=False)

    if args.exclude_relations and args.exclude_relations.lower() not in ("none", ""):
        if args.exclude_relations == "clinical_outcome":
            exclude = clinical_outcome_relations(graph)
        else:
            exclude = frozenset(
                x.strip() for x in args.exclude_relations.split(",") if x.strip()
            )
        print(f"Excluding {len(exclude)} relations from retrieval graph "
              f"(family={args.exclude_relations!r}): {sorted(exclude)}", flush=True)

    # Date-conditional exclusion set (dropped only for edges dated > cutoff).
    # Resolved the same way as --exclude-relations (family name or CSV list).
    exclude_after = None
    if args.exclude_relations_after and args.exclude_relations_after.lower() not in ("none", ""):
        if args.exclude_relations_after == "clinical_outcome":
            exclude_after = clinical_outcome_relations(graph)
        else:
            exclude_after = frozenset(
                x.strip() for x in args.exclude_relations_after.split(",") if x.strip()
            )
        print(f"Date-gating {len(exclude_after)} relations at >{args.exclude_after_year} "
              f"(family={args.exclude_relations_after!r}): {sorted(exclude_after)}", flush=True)

    print(f"Building KPathExtractor (mode={args.retrieval_mode})...", flush=True)
    extractor = KPathExtractor(
        graph, mappings=mappings, mode=args.retrieval_mode, exclude_relations=exclude,
        exclude_relations_after=exclude_after, exclude_after_year=args.exclude_after_year
    )

    print(f"Building name map from {args.nodes_dir}...", flush=True)
    name_map = build_name_map_evidence_dated(args.nodes_dir)
    serialiser = PathSerialiser(name_map=name_map)
    print(f"  name_map covers: { {k: len(v) for k, v in name_map.items()} }",
          flush=True)

    # Start with the existing cache (empty if not resuming) so the final
    # JSON is a single merged map of cached + newly-extracted pairs.
    out_pairs: Dict[str, Dict] = dict(existing_pairs)
    t0 = time.time()
    n_with_paths = sum(1 for v in out_pairs.values() if v.get("n_paths", 0) > 0)
    n_zero = sum(1 for v in out_pairs.values() if v.get("n_paths", 0) == 0)
    n_err = 0

    def _write_cache(elapsed_sec: float, partial: bool = False) -> None:
        """Atomically write the merged cache to --output. Atomic via tmp+rename
        so a SIGKILL during the write cannot corrupt the previous file."""
        payload = {
            "config": {
                "mode": "kpaths",
                "max_hops": args.max_hops,
                "max_paths": args.max_paths,
                "graph": args.graph,
                "nodes_dir": args.nodes_dir,
                "csv_dir": str(csv_dir),
                "n_pairs": len(out_pairs),
                "n_pairs_extracted_this_run": len(pairs),
                "n_with_paths": n_with_paths,
                "n_zero": n_zero,
                "n_err": n_err,
                "elapsed_sec": round(elapsed_sec, 1),
                "resumed_from": args.resume_from,
                "partial": partial,
                "exclude_relations_family": args.exclude_relations,
                "exclude_relations": sorted(exclude) if exclude else None,
            },
            "pairs": out_pairs,
        }
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        with open(tmp_path, "w") as f:
            json.dump(payload, f)
        tmp_path.replace(out_path)

    for i, (did, tid) in enumerate(pairs):
        try:
            paths = extractor.find_paths(
                src_id=did,
                dst_id=tid,
                src_type="disease",
                dst_type="target",
                max_hops=args.max_hops,
                max_paths=args.max_paths,
            )
            compact = [serialiser.serialise_path(p, mode="compact") for p in paths]
            hops = [[asdict(h) for h in p.hops] for p in paths]
            out_pairs[f"{did}||{tid}"] = {
                "n_paths": len(paths),
                "compact": compact,
                "hops": hops,
            }
            if paths:
                n_with_paths += 1
            else:
                n_zero += 1
        except Exception as e:
            n_err += 1
            out_pairs[f"{did}||{tid}"] = {
                "n_paths": 0,
                "compact": [],
                "hops": [],
                "error": f"{type(e).__name__}: {e}",
            }

        if (i + 1) % args.progress_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(pairs) - (i + 1)) / rate if rate > 0 else 0
            print(
                f"  [{i+1:,}/{len(pairs):,}] "
                f"with_paths={n_with_paths:,} zero={n_zero:,} err={n_err} "
                f"rate={rate:.1f}/s eta={eta/60:.1f}m",
                flush=True,
            )

        if args.checkpoint_every > 0 and (i + 1) % args.checkpoint_every == 0:
            elapsed = time.time() - t0
            _write_cache(elapsed, partial=True)
            print(f"  [checkpoint] wrote {out_path} "
                  f"({len(out_pairs):,} pairs, partial=True)", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f}m. "
          f"with_paths={n_with_paths:,} zero={n_zero:,} err={n_err}", flush=True)
    print(f"Writing {out_path}...", flush=True)
    _write_cache(elapsed, partial=False)
    print(f"  size: {out_path.stat().st_size / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
