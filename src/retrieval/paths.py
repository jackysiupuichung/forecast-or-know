#!/usr/bin/env python3
"""
Multi-hop path extraction on the OpenTargets HeteroData graph.

Given a (disease, target) pair, enumerate all paths of length ≤ max_hops that
connect them via annotated edges. Each hop is stored with:
    - src/dst node types and string IDs
    - relation name
    - edge_weight (confidence score)
    - edge_novelty
    - edge_time (year)

Public API:
    PathExtractor(graph, mappings)
        .find_paths(disease_id, target_id, max_hops=3, max_paths=50) → List[Path]

A Path is a list of Hop dataclasses.
"""

from __future__ import annotations

import torch
import networkx as nx
from dataclasses import dataclass, field
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Iterable
from torch_geometric.data import HeteroData


def clinical_outcome_relations(graph) -> "frozenset[str]":
    """Relations that encode the Task-1 trial-entry label or a near-label
    clinical-progress proxy, resolved against the graph's actual edge types:
    any ``clinical_trial*`` relation (incl. odd variants such as
    ``clinical_trial_Unknown/Operational``), ``clinically_associated``, and
    ``advancement``.

    Mirrors the label predicate in
    ``src/evaluation/trial_entry_dataset_builder.py`` (``r.startswith(
    "clinical_trial") or r == "clinically_associated"``) plus the
    ``advancement`` proxy. Pass the result as ``exclude_relations`` to build a
    leakage-safe retrieval graph — see
    ``notes/graph_retrieval/variant_harness_design.md``.
    """
    out = set()
    for (_src, rel, _dst) in graph.edge_types:
        if rel.startswith("clinical_trial") or rel in ("clinically_associated", "advancement"):
            out.add(rel)
    return frozenset(out)


@dataclass
class Hop:
    src_type: str
    src_id: str
    relation: str
    dst_type: str
    dst_id: str
    edge_weight: float
    edge_novelty: float
    edge_time: Optional[int]
    # Off-graph literature grounding (Tier-2 / C1). Defaulted so Hop(**h) works on
    # both pre-C1 caches (keys absent) and C1-era caches (keys present).
    evidence_text: Optional[str] = None
    pmid: Optional[str] = None


@dataclass
class Path:
    hops: List[Hop] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.hops)

    @property
    def nodes(self) -> List[Tuple[str, str]]:
        """Returns list of (node_type, node_id) along the path."""
        if not self.hops:
            return []
        out = [(self.hops[0].src_type, self.hops[0].src_id)]
        for h in self.hops:
            out.append((h.dst_type, h.dst_id))
        return out


# Adjacency entry for a single outgoing edge from a node
@dataclass
class _OutEdge:
    relation: str
    dst_type: str
    dst_idx: int
    edge_weight: float
    edge_novelty: float
    edge_time: Optional[int]


class PathExtractor:
    """
    Build a sparse adjacency view of a HeteroData graph, then enumerate
    bounded-hop paths between (src, dst) node pairs via BFS.

    The adjacency is keyed by (node_type, node_idx) → List[_OutEdge] so BFS can
    explore outgoing edges in O(deg) per node.

    Args:
        graph:    HeteroData (the full graph, not necessarily masked)
        mappings: dict with 'node_mapping' {node_type: {node_id_str: idx}}
                  — if None, read node_id lists from graph[nt].node_id
    """

    def __init__(
        self,
        graph: HeteroData,
        mappings: Optional[dict] = None,
        exclude_relations: Optional[Iterable[str]] = None,
        exclude_relations_after: Optional[Iterable[str]] = None,
        exclude_after_year: Optional[int] = None,
    ) -> None:
        self.graph = graph
        # Relation names dropped from the retrieval adjacency entirely, so no
        # extracted path can route through them. Used to strip the Task-1 label
        # family (clinical_trial_* / clinically_associated / advancement) from
        # the graph the LLM sees. Compared against the forward relation name of
        # each edge type — the ~reverse mirror is built from the same edge type,
        # so excluding it removes both directions.
        self.exclude_relations = frozenset(exclude_relations or ())
        # DATE-CONDITIONAL exclusion. `exclude_relations_after` names relations
        # dropped ONLY for edges dated strictly after `exclude_after_year` (the
        # leak); edges dated <= that year (and undated) are KEPT. This retains a
        # target's legitimate pre-cutoff clinical history while stripping only the
        # post-cutoff clinical/label edges (A3_bio_only, time-gated).
        #
        # IMPORTANT: this is a SEPARATE set from `exclude_relations`, which is
        # ALWAYS unconditional. `advancement` (the trial-progression near-label
        # edge) must go in `exclude_relations` so it is removed regardless of date
        # — matching the noadv A2/A3 caches — while the `clinical_trial_*` history
        # goes in `exclude_relations_after`. A relation in BOTH is dropped whole
        # (unconditional wins). No effect unless `exclude_after_year` is also set.
        self.exclude_relations_after = frozenset(exclude_relations_after or ())
        self.exclude_after_year = exclude_after_year

        # Node ID mapping: idx → id and id → idx per node type
        if mappings is not None and "node_mapping" in mappings:
            self.node_mapping = mappings["node_mapping"]
            self.idx_to_id = {
                nt: [None] * len(m) for nt, m in self.node_mapping.items()
            }
            for nt, m in self.node_mapping.items():
                for node_id, idx in m.items():
                    self.idx_to_id[nt][idx] = node_id
        else:
            # Fall back to graph[nt].node_id lists
            self.idx_to_id = {}
            self.node_mapping = {}
            for nt in graph.node_types:
                ids = list(graph[nt].node_id)
                self.idx_to_id[nt] = ids
                self.node_mapping[nt] = {nid: i for i, nid in enumerate(ids)}

        # Build outgoing adjacency: (src_type, src_idx) → List[_OutEdge]
        self.adj: Dict[Tuple[str, int], List[_OutEdge]] = defaultdict(list)
        self._build_adjacency()

    def _build_adjacency(self) -> None:
        # Bidirectional traversal: each forward edge (a, rel, b) is mirrored as a
        # reverse edge (b, ~rel, a). The ~prefix makes direction visible in LLM
        # prompts (e.g. `target ← molecule` shows up as `--~modulated_by-->`),
        # which keeps the serialised path honest about which way the original
        # evidence flows. Self-loop edge types (same src/dst type, e.g.
        # disease is_subtype_of disease) still get reversed so the BFS can walk
        # both parent→child and child→parent.
        forward = 0
        reverse = 0
        excluded_edges = 0
        excluded_types: List[str] = []
        UND = -9223372036854775808  # undated sentinel — never gated (not provably post-cutoff)
        gate = self.exclude_after_year is not None
        for edge_type in self.graph.edge_types:
            src_type, relation, dst_type = edge_type
            store = self.graph[edge_type]
            ei = store.edge_index  # [2, E]
            E = ei.size(1)
            if E == 0:
                continue
            # Unconditional exclude: drop the whole edge type regardless of date
            # (e.g. advancement). Takes precedence over the date-conditional set.
            if relation in self.exclude_relations:
                excluded_edges += E
                excluded_types.append(relation)
                continue

            has_attr = hasattr(store, "edge_attr") and store.edge_attr is not None
            has_time = hasattr(store, "edge_time") and store.edge_time is not None

            ei_src = ei[0].tolist()
            ei_dst = ei[1].tolist()
            if has_attr:
                ea = store.edge_attr
                weights = ea[:, 0].tolist()
                novelties = ea[:, 1].tolist() if ea.size(1) >= 2 else [0.0] * E
            else:
                weights = [0.0] * E
                novelties = [0.0] * E
            times = store.edge_time.tolist() if has_time else [None] * E

            # Date-conditional exclude: for a relation in the date-gated set with a
            # cutoff set, keep edges dated <= cutoff (and undated); drop only
            # post-cutoff edges.
            gate_this = gate and (relation in self.exclude_relations_after)
            inv_relation = f"~{relation}"
            n_add = 0
            for i in range(E):
                if gate_this:
                    t = times[i]
                    if t is not None and t != UND and t > self.exclude_after_year:
                        excluded_edges += 1
                        continue
                self.adj[(src_type, ei_src[i])].append(
                    _OutEdge(
                        relation=relation,
                        dst_type=dst_type,
                        dst_idx=ei_dst[i],
                        edge_weight=weights[i],
                        edge_novelty=novelties[i],
                        edge_time=times[i],
                    )
                )
                self.adj[(dst_type, ei_dst[i])].append(
                    _OutEdge(
                        relation=inv_relation,
                        dst_type=src_type,
                        dst_idx=ei_src[i],
                        edge_weight=weights[i],
                        edge_novelty=novelties[i],
                        edge_time=times[i],
                    )
                )
                n_add += 1
            forward += n_add
            reverse += n_add
            if gate_this and n_add < E:
                excluded_types.append(f"{relation}(>{self.exclude_after_year})")
        print(f"PathExtractor: indexed {forward:,} forward + {reverse:,} reverse = "
              f"{forward + reverse:,} directed edges (bidirectional)")
        if self.exclude_relations or self.exclude_relations_after:
            print(f"PathExtractor: EXCLUDED {excluded_edges:,} edges across "
                  f"{len(set(excluded_types))} relation types "
                  f"(not indexed, not traversable): {sorted(set(excluded_types))}")

    def find_paths(
        self,
        src_id: str,
        dst_id: str,
        src_type: str = "disease",
        dst_type: str = "target",
        max_hops: int = 3,
        max_paths: int = 50,
        allow_revisit: bool = False,
    ) -> List[Path]:
        """
        Enumerate paths from (src_type, src_id) to (dst_type, dst_id) of length ≤ max_hops.

        Uses BFS up to max_hops. Paths that hit the target node of the correct type
        are collected; the search continues to explore paths of greater length too,
        up to max_hops and max_paths.

        Args:
            src_id, dst_id:     string node IDs
            src_type, dst_type: node types
            max_hops:           maximum path length (number of edges)
            max_paths:          early-stop once this many paths found
            allow_revisit:      if False, a path cannot revisit the same (type, idx) twice

        Returns:
            List of Path objects (possibly empty). Shorter paths appear first.
        """
        if src_id not in self.node_mapping.get(src_type, {}):
            return []
        if dst_id not in self.node_mapping.get(dst_type, {}):
            return []

        src_idx = self.node_mapping[src_type][src_id]
        dst_idx = self.node_mapping[dst_type][dst_id]

        paths: List[Path] = []

        # BFS queue: entries are (current_type, current_idx, hops_so_far, visited_set)
        # To avoid deep copies, keep hops as a list and pop after expansion.
        # Iterative BFS with a deque of (path_so_far, visited) tuples.
        init_visited = frozenset() if allow_revisit else frozenset({(src_type, src_idx)})
        queue: deque[Tuple[List[Hop], frozenset]] = deque()
        queue.append(([], init_visited))

        while queue and len(paths) < max_paths:
            path_hops, visited = queue.popleft()

            # Determine current node
            if not path_hops:
                cur_type, cur_idx = src_type, src_idx
            else:
                last = path_hops[-1]
                cur_type, cur_idx = last.dst_type, self.node_mapping[last.dst_type][last.dst_id]

            # Check termination
            if cur_type == dst_type and cur_idx == dst_idx and path_hops:
                paths.append(Path(hops=list(path_hops)))
                continue  # Don't extend beyond destination

            if len(path_hops) >= max_hops:
                continue

            # Expand outgoing edges
            for oe in self.adj.get((cur_type, cur_idx), []):
                next_key = (oe.dst_type, oe.dst_idx)
                if not allow_revisit and next_key in visited:
                    continue

                dst_id_str = self.idx_to_id[oe.dst_type][oe.dst_idx]
                new_hop = Hop(
                    src_type=cur_type,
                    src_id=self.idx_to_id[cur_type][cur_idx],
                    relation=oe.relation,
                    dst_type=oe.dst_type,
                    dst_id=dst_id_str,
                    edge_weight=oe.edge_weight,
                    edge_novelty=oe.edge_novelty,
                    edge_time=oe.edge_time,
                )
                new_visited = visited if allow_revisit else visited | {next_key}
                queue.append((path_hops + [new_hop], new_visited))

        return paths


class KPathExtractor(PathExtractor):
    """
    K-Paths retrieval (rsinghlab/K-Paths, KDD 2025) on the same HeteroData
    adjacency that PathExtractor builds.

    Three modes:
      - "kpaths":         shortest_simple_paths, deduplicated by relation sequence
      - "no_filter":      shortest_simple_paths, no dedup
      - "neighbors_only": one-hop neighbors of src and dst (no path search)

    Shares __init__ with PathExtractor and additionally builds an nx.DiGraph
    view of self.adj for nx.shortest_simple_paths.
    """

    VALID_MODES = ("kpaths", "no_filter", "neighbors_only", "neighbors2", "pcst")

    def __init__(
        self,
        graph: HeteroData,
        mappings: Optional[dict] = None,
        mode: str = "kpaths",
        exclude_relations: Optional[Iterable[str]] = None,
        exclude_relations_after: Optional[Iterable[str]] = None,
        exclude_after_year: Optional[int] = None,
    ) -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got {mode!r}")
        super().__init__(graph, mappings=mappings, exclude_relations=exclude_relations,
                         exclude_relations_after=exclude_relations_after,
                         exclude_after_year=exclude_after_year)
        self.mode = mode
        self._nx = self._build_nx()

    def _build_nx(self) -> nx.DiGraph:
        # nx.DiGraph collapses parallel edges; we keep the full _OutEdge list
        # per (u, v) under the "edges" attr so path materialisation can pick
        # the highest-confidence relation for that pair.
        g = nx.DiGraph()
        for (src_type, src_idx), edges in self.adj.items():
            u = (src_type, src_idx)
            for oe in edges:
                v = (oe.dst_type, oe.dst_idx)
                if g.has_edge(u, v):
                    g[u][v]["edges"].append(oe)
                else:
                    g.add_edge(u, v, edges=[oe])
        return g

    def _best_edge(self, u: Tuple[str, int], v: Tuple[str, int]) -> "_OutEdge":
        # Highest edge_weight among parallel relations between u and v.
        return max(self._nx[u][v]["edges"], key=lambda oe: oe.edge_weight)

    def _hop_from_oe(self, src_type: str, src_idx: int, oe: "_OutEdge") -> Hop:
        return Hop(
            src_type=src_type,
            src_id=self.idx_to_id[src_type][src_idx],
            relation=oe.relation,
            dst_type=oe.dst_type,
            dst_id=self.idx_to_id[oe.dst_type][oe.dst_idx],
            edge_weight=oe.edge_weight,
            edge_novelty=oe.edge_novelty,
            edge_time=oe.edge_time,
        )

    def _materialise(self, node_seq: List[Tuple[str, int]]) -> List[Hop]:
        hops: List[Hop] = []
        for i in range(len(node_seq) - 1):
            u, v = node_seq[i], node_seq[i + 1]
            oe = self._best_edge(u, v)
            hops.append(self._hop_from_oe(u[0], u[1], oe))
        return hops

    def _one_node_paths(
        self, node: Tuple[str, int], cap: int, dedup: bool
    ) -> List[Path]:
        out: List[Path] = []
        seen: set[str] = set()
        for oe in self.adj.get(node, []):
            if dedup and oe.relation in seen:
                continue
            seen.add(oe.relation)
            out.append(Path(hops=[self._hop_from_oe(node[0], node[1], oe)]))
            if len(out) >= cap:
                break
        return out

    def _ego_paths(
        self, node: Tuple[str, int], radius: int = 2, cap: int = 10, dedup: bool = True
    ) -> List[Path]:
        """k-hop ego neighbourhood of a single node (B2 / LightRAG-style): every
        walk of length 1..radius out of `node`, emitted as a Path so the cache
        schema and serialiser are reused unchanged. BFS, no immediate backtrack to
        the start; dedup on relation-sequence when `dedup`."""
        out: List[Path] = []
        seen_relseq: set = set()
        queue: deque[Tuple[List[Hop], Tuple[str, int]]] = deque()
        queue.append(([], node))
        while queue and len(out) < cap:
            hops, cur = queue.popleft()
            if hops:
                relseq = tuple(h.relation for h in hops)
                if not (dedup and relseq in seen_relseq):
                    seen_relseq.add(relseq)
                    out.append(Path(hops=list(hops)))
                    if len(out) >= cap:
                        break
            if len(hops) >= radius:
                continue
            for oe in self.adj.get(cur, []):
                nxt = (oe.dst_type, oe.dst_idx)
                if nxt == node:                       # no immediate backtrack to start
                    continue
                queue.append((hops + [self._hop_from_oe(cur[0], cur[1], oe)], nxt))
        return out

    def _pcst_subgraph(
        self,
        u: Tuple[str, int],
        v: Tuple[str, int],
        max_paths: int,
        max_hops: int,
        cand_budget: Optional[int] = None,
    ) -> List[Path]:
        """B1 / G-Retriever-style CONNECTED SUBGRAPH connecting the pair.

        The candidate is the K shortest simple u-v paths (already connected and
        query-relevant); we emit their **deduplicated edge set** in shortest-first
        order — so shared intermediate nodes are represented once and cross-links
        between paths are preserved. That deduplicated-connected view is exactly
        what B1 tests against the *disjoint*-path view of A2/A3. Edges are emitted
        as length-1 Paths so the cache schema and serialiser are reused unchanged.

        Note: exact Prize-Collecting Steiner (pcst_fast) would additionally *prune*
        low-value branches; the PyPI `pcst-fast==1.0.10` build returns incorrect
        indices (its own canonical example fails), so we use the shortest-path
        union, which yields the same connected-subgraph representation.
        """
        cand_budget = cand_budget or max(max_paths * 2, 20)
        try:
            seqs: List[List[Tuple[str, int]]] = []
            for node_seq in nx.shortest_simple_paths(self._nx, u, v):
                if len(node_seq) - 1 > max_hops:
                    break
                seqs.append(node_seq)
                if len(seqs) >= cand_budget:
                    break
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

        out: List[Path] = []
        seen_edge: set = set()
        for seq in seqs:                                  # shortest paths first
            for a, b in zip(seq[:-1], seq[1:]):
                key = frozenset((a, b))
                if key in seen_edge:                      # dedup shared edges
                    continue
                seen_edge.add(key)
                oe = self._best_edge(a, b)
                if oe is None:
                    continue
                out.append(Path(hops=[self._hop_from_oe(a[0], a[1], oe)]))
                if len(out) >= max_paths:
                    return out
        return out

    def find_paths(
        self,
        src_id: str,
        dst_id: str,
        src_type: str = "disease",
        dst_type: str = "target",
        max_hops: int = 3,
        max_paths: int = 10,
        allow_revisit: bool = False,  # accepted for API parity; ignored (simple paths)
        drop_pair_relations: Optional[Iterable[str]] = None,
    ) -> List[Path]:
        """`drop_pair_relations`: relation names (forward form; the reverse
        "~<rel>" is matched too) removed **only between the two queried
        endpoints** for this call, e.g. the clinical_trial* edges that encode the
        pair's own label (A3-minus-own arm). Retrieval is then re-run on the
        modified graph, so a remaining non-clinical direct edge can surface in
        place of the removed one. Both the path-search graph and the adjacency
        list (used by the no-path fallback) are filtered, and restored after."""
        if src_id not in self.node_mapping.get(src_type, {}):
            return []
        if dst_id not in self.node_mapping.get(dst_type, {}):
            return []

        u = (src_type, self.node_mapping[src_type][src_id])
        v = (dst_type, self.node_mapping[dst_type][dst_id])
        if drop_pair_relations:
            drop = {r.lstrip("~") for r in drop_pair_relations}
            saved = self._drop_pair_edges(u, v, drop)
            try:
                return self._find_paths_uv(u, v, max_hops, max_paths)
            finally:
                self._restore_pair_edges(saved)
        return self._find_paths_uv(u, v, max_hops, max_paths)

    def _drop_pair_edges(self, u, v, drop):
        """Remove relations in `drop` between u and v (both directions) from
        self._nx and self.adj. Returns what is needed to restore them."""
        saved = []
        for a, b in ((u, v), (v, u)):
            if self._nx.has_edge(a, b):
                edges = self._nx[a][b]["edges"]
                saved.append(("nx", a, b, edges))
                keep = [oe for oe in edges if oe.relation.lstrip("~") not in drop]
                if keep:
                    self._nx[a][b]["edges"] = keep
                else:
                    self._nx.remove_edge(a, b)
            if a in self.adj:
                lst = self.adj[a]
                saved.append(("adj", a, b, lst))
                self.adj[a] = [oe for oe in lst
                               if not ((oe.dst_type, oe.dst_idx) == b and oe.relation.lstrip("~") in drop)]
        return saved

    def _restore_pair_edges(self, saved):
        for kind, a, b, obj in saved:
            if kind == "nx":
                if self._nx.has_edge(a, b):
                    self._nx[a][b]["edges"] = obj
                else:
                    self._nx.add_edge(a, b, edges=obj)
            else:
                self.adj[a] = obj

    def _find_paths_uv(self, u, v, max_hops, max_paths) -> List[Path]:
        u_in = u in self._nx
        v_in = v in self._nx
        dedup = self.mode == "kpaths"

        if self.mode == "neighbors_only":
            half = max(1, max_paths // 2)
            return (
                self._one_node_paths(u, half, dedup=False)
                + self._one_node_paths(v, max_paths - half, dedup=False)
            )

        if self.mode == "neighbors2":
            half = max(1, max_paths // 2)
            return (
                self._ego_paths(u, radius=max_hops, cap=half, dedup=dedup)
                + self._ego_paths(v, radius=max_hops, cap=max_paths - half, dedup=dedup)
            )

        if self.mode == "pcst":
            if not (u in self._nx and v in self._nx):
                return []
            return self._pcst_subgraph(u, v, max_paths, max_hops)

        if not u_in and not v_in:
            return []
        if u_in and not v_in:
            return self._one_node_paths(u, max_paths, dedup=dedup)
        if v_in and not u_in:
            return self._one_node_paths(v, max_paths, dedup=dedup)

        out: List[Path] = []
        seen_relseq: set[Tuple[str, ...]] = set()
        try:
            for node_seq in nx.shortest_simple_paths(self._nx, u, v):
                if len(node_seq) - 1 > max_hops:
                    break
                hops = self._materialise(node_seq)
                if dedup:
                    relseq = tuple(h.relation for h in hops)
                    if relseq in seen_relseq:
                        continue
                    seen_relseq.add(relseq)
                out.append(Path(hops=hops))
                if len(out) >= max_paths:
                    break
        except nx.NetworkXNoPath:
            half = max(1, max_paths // 2)
            return (
                self._one_node_paths(u, half, dedup=dedup)
                + self._one_node_paths(v, max_paths - half, dedup=dedup)
            )
        return out
