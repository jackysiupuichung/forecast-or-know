#!/usr/bin/env python3
"""
Build a time-sliced snapshot of the temporal heterograph (THBKG) for path extraction.

Take the full graph at `--graph` (the canonical 26.03-style HeteroData with
`edge_time` per edge), drop edges with `edge_time > cutoff_year`, then
collapse multi-year duplicates of the same `(src, rel, dst)` triple to a
single edge keeping the row with the *latest* pre-cutoff year. `edge_attr`
follows the kept row.

Ontology / structural edges (`is_subtype_of`, `is_subpathway_of`, etc.) carry
a sentinel `edge_time` (INT64_MIN) and are passed through untouched: no time
filter, no dedup.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch_geometric.data import HeteroData


# Sentinel used for non-temporal (ontology / structural) edges in the source
# graph — these carry no real year and must skip the time filter.
SENTINEL_TIME = torch.iinfo(torch.int64).min


def _has_real_time(edge_time: torch.Tensor) -> bool:
    """An edge type has real years iff at least one entry is not the sentinel."""
    return bool((edge_time != SENTINEL_TIME).any().item())


def _dedup_latest(
    edge_index: torch.Tensor,
    edge_time: torch.Tensor,
    edge_attr: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Per (src, dst) pair, keep the row with the largest `edge_time`. Ties go to
    the first occurrence (stable). Returns the deduped tensors.
    """
    # Empty after filtering (an edge type with no edges <= cutoff — happens for
    # early-year snapshots where e.g. clinical_trial_positive has no pre-2005
    # edges). Nothing to dedup; return empties of the right shape/dtype.
    if edge_index.numel() == 0:
        return (
            edge_index,
            edge_time,
            edge_attr if edge_attr is not None else None,
        )

    # Sort descending by edge_time (stable) so the first occurrence of each
    # unique (src, dst) is the latest year; tied years keep the first-seen row.
    order = torch.argsort(edge_time, descending=True, stable=True)
    src_s = edge_index[0][order]
    dst_s = edge_index[1][order]
    time_s = edge_time[order]
    attr_s = edge_attr[order] if edge_attr is not None else None

    # Vectorised first-occurrence-per-pair: pack (src, dst) into a single int64
    # key, then use torch.unique(return_inverse) on it. The minimum index per
    # group is the "first occurrence under the sort", i.e. the latest-year row.
    dst_max = int(dst_s.max().item()) + 1 if dst_s.numel() else 1
    pair_key = src_s.to(torch.int64) * dst_max + dst_s.to(torch.int64)
    _, inverse = torch.unique(pair_key, return_inverse=True, sorted=False)
    n_groups = int(inverse.max().item()) + 1
    # scatter_reduce 'amin' over arange(N) grouped by inverse → min row index per group.
    idx = torch.arange(pair_key.numel(), dtype=torch.long)
    keep = torch.full((n_groups,), pair_key.numel(), dtype=torch.long)
    keep.scatter_reduce_(0, inverse, idx, reduce="amin", include_self=True)

    new_ei = torch.stack([src_s[keep], dst_s[keep]], dim=0)
    new_time = time_s[keep]
    new_attr = attr_s[keep] if attr_s is not None else None
    return new_ei, new_time, new_attr


def build_snapshot(graph_path: str, cutoff_year: int, output_path: str) -> None:
    print(f"Loading graph from {graph_path}...", flush=True)
    g: HeteroData = torch.load(graph_path, weights_only=False, map_location="cpu")

    out = HeteroData()
    # Copy node stores verbatim.
    for nt in g.node_types:
        for key, val in g[nt].items():
            out[nt][key] = val
        print(f"  node {nt}: {g[nt].num_nodes:,}")

    print(f"\nFiltering edges to year ≤ {cutoff_year} and deduplicating per (src, dst)...")
    print(f"{'edge type':<60} {'in':>10} {'≤cutoff':>10} {'deduped':>10}")
    for et in g.edge_types:
        store = g[et]
        ei = store.edge_index
        et_t = store.edge_time if hasattr(store, "edge_time") else None
        attr = store.edge_attr if hasattr(store, "edge_attr") else None
        n_in = ei.shape[1]

        if et_t is None or not _has_real_time(et_t):
            # Structural / ontology — pass through unchanged.
            out[et].edge_index = ei
            if et_t is not None:
                out[et].edge_time = et_t
            if attr is not None:
                out[et].edge_attr = attr
            print(f"{str(et):<60} {n_in:>10,} {'pass':>10} {'pass':>10}")
            continue

        mask = et_t <= cutoff_year
        ei_f = ei[:, mask]
        et_f = et_t[mask]
        attr_f = attr[mask] if attr is not None else None
        n_filt = ei_f.shape[1]

        ei_d, et_d, attr_d = _dedup_latest(ei_f, et_f, attr_f)
        n_ded = ei_d.shape[1]

        out[et].edge_index = ei_d
        out[et].edge_time = et_d
        if attr_d is not None:
            out[et].edge_attr = attr_d
        print(f"{str(et):<60} {n_in:>10,} {n_filt:>10,} {n_ded:>10,}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output_path)
    print(f"\nSaved snapshot → {output_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", required=True,
                   help="Full hetero_graph_with_features_*.pt with edge_time")
    p.add_argument("--cutoff-year", type=int, required=True)
    p.add_argument("--output", required=True,
                   help="Where to write the snapshot .pt file")
    args = p.parse_args()
    build_snapshot(args.graph, args.cutoff_year, args.output)


if __name__ == "__main__":
    main()
