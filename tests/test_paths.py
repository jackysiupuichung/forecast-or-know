"""Tests for src/retrieval/paths.py — bidirectional BFS path enumeration."""

from __future__ import annotations

import torch
from torch_geometric.data import HeteroData

from src.retrieval.paths import PathExtractor, KPathExtractor


def _make_graph() -> tuple[HeteroData, dict]:
    """
    Tiny synthetic graph:

        target T0 --animal_model--> disease D0
        target T1 --genetic_association--> disease D0
        molecule M0 --modulated_by--> target T0   (note: edge stored as molecule→target)
        molecule M0 --clinical_trial--> disease D1

    With bidirectional traversal we expect:
      - T0 → D0 directly via animal_model (1 hop, forward)
      - D0 → T0 via ~animal_model (1 hop, reverse)
      - D0 → D1 via T0 ← M0 → D1 is impossible (T0 has no edge to D1; M0 does)
      - T0 ↔ D1 via T0 ← M0 → D1 in 2 hops (reverse modulated_by then forward clinical_trial)
    """
    g = HeteroData()
    g["target"].num_nodes = 2
    g["target"].node_id = ["T0", "T1"]
    g["target"].x = torch.zeros(2, 4)
    g["disease"].num_nodes = 2
    g["disease"].node_id = ["D0", "D1"]
    g["disease"].x = torch.zeros(2, 4)
    g["molecule"].num_nodes = 1
    g["molecule"].node_id = ["M0"]
    g["molecule"].x = torch.zeros(1, 4)

    # T0 --animal_model--> D0
    # T1 --genetic_association--> D0
    g["target", "animal_model", "disease"].edge_index = torch.tensor([[0], [0]])
    g["target", "animal_model", "disease"].edge_attr = torch.tensor([[0.5, 0.1]])
    g["target", "animal_model", "disease"].edge_time = torch.tensor([2010])

    g["target", "genetic_association", "disease"].edge_index = torch.tensor([[1], [0]])
    g["target", "genetic_association", "disease"].edge_attr = torch.tensor([[0.8, 0.2]])
    g["target", "genetic_association", "disease"].edge_time = torch.tensor([2015])

    # M0 --modulated_by--> T0   (so target T0 is reachable from M0 only by reverse traversal)
    g["molecule", "modulated_by", "target"].edge_index = torch.tensor([[0], [0]])
    g["molecule", "modulated_by", "target"].edge_attr = torch.tensor([[0.7, 0.0]])
    g["molecule", "modulated_by", "target"].edge_time = torch.tensor([2018])

    # M0 --clinical_trial--> D1
    g["molecule", "clinical_trial", "disease"].edge_index = torch.tensor([[0], [1]])
    g["molecule", "clinical_trial", "disease"].edge_attr = torch.tensor([[0.9, 0.3]])
    g["molecule", "clinical_trial", "disease"].edge_time = torch.tensor([2020])

    mappings = {
        "node_mapping": {
            "target":   {"T0": 0, "T1": 1},
            "disease":  {"D0": 0, "D1": 1},
            "molecule": {"M0": 0},
        }
    }
    return g, mappings


def test_forward_target_to_disease_one_hop():
    g, m = _make_graph()
    ex = PathExtractor(g, mappings=m)
    paths = ex.find_paths("T0", "D0", src_type="target", dst_type="disease",
                          max_hops=1, max_paths=10)
    assert len(paths) == 1
    h = paths[0].hops[0]
    assert h.relation == "animal_model"
    assert h.src_id == "T0" and h.dst_id == "D0"


def test_reverse_disease_to_target_one_hop():
    """The forward edge is target→disease; with bidirectional adjacency, BFS
    from disease should still reach the target via the inverse relation."""
    g, m = _make_graph()
    ex = PathExtractor(g, mappings=m)
    paths = ex.find_paths("D0", "T0", src_type="disease", dst_type="target",
                          max_hops=1, max_paths=10)
    assert len(paths) == 1
    h = paths[0].hops[0]
    assert h.relation == "~animal_model"
    assert h.src_id == "D0" and h.dst_id == "T0"


def test_reverse_disease_to_target_picks_up_both_relations():
    """D0 should reach both T0 (~animal_model) and T1 (~genetic_association)."""
    g, m = _make_graph()
    ex = PathExtractor(g, mappings=m)
    p_t0 = ex.find_paths("D0", "T0", src_type="disease", dst_type="target", max_hops=1)
    p_t1 = ex.find_paths("D0", "T1", src_type="disease", dst_type="target", max_hops=1)
    assert len(p_t0) == 1 and p_t0[0].hops[0].relation == "~animal_model"
    assert len(p_t1) == 1 and p_t1[0].hops[0].relation == "~genetic_association"


def test_two_hop_target_to_disease_via_molecule():
    """T0 → D1 requires T0 ← M0 → D1: reverse-modulated_by then forward
    clinical_trial. Only possible if reverse adjacency exists."""
    g, m = _make_graph()
    ex = PathExtractor(g, mappings=m)
    paths = ex.find_paths("T0", "D1", src_type="target", dst_type="disease",
                          max_hops=2, max_paths=10)
    assert len(paths) == 1
    p = paths[0]
    assert len(p.hops) == 2
    # First hop: T0 → M0 via reversed modulated_by
    assert p.hops[0].relation == "~modulated_by"
    assert p.hops[0].dst_type == "molecule" and p.hops[0].dst_id == "M0"
    # Second hop: M0 → D1 via forward clinical_trial
    assert p.hops[1].relation == "clinical_trial"
    assert p.hops[1].dst_id == "D1"


def test_no_path_returns_empty():
    g, m = _make_graph()
    ex = PathExtractor(g, mappings=m)
    # T1 has no path to D1 within 2 hops (T1 → D0 is forward; D0 has no
    # outgoing edge to D1 in this graph)
    paths = ex.find_paths("T1", "D1", src_type="target", dst_type="disease",
                          max_hops=2, max_paths=10)
    assert paths == []


def test_unknown_node_returns_empty():
    g, m = _make_graph()
    ex = PathExtractor(g, mappings=m)
    assert ex.find_paths("UNKNOWN", "D0", src_type="target", dst_type="disease") == []
    assert ex.find_paths("T0", "UNKNOWN", src_type="target", dst_type="disease") == []


# -------------------- KPathExtractor --------------------

def _make_diversity_graph() -> tuple[HeteroData, dict]:
    """
    Graph with two parallel relation-distinct routes T0 → D0:
        T0 --animal_model--> D0           (1-hop, relseq = (animal_model,))
        T0 --rel_a--> M0 --rel_b--> D0   (2-hop, relseq = (rel_a, rel_b))
        T0 --rel_a--> M1 --rel_b--> D0   (2-hop, relseq = (rel_a, rel_b)) ← duplicate
    """
    g = HeteroData()
    g["target"].num_nodes = 1
    g["target"].node_id = ["T0"]
    g["target"].x = torch.zeros(1, 4)
    g["disease"].num_nodes = 1
    g["disease"].node_id = ["D0"]
    g["disease"].x = torch.zeros(1, 4)
    g["molecule"].num_nodes = 2
    g["molecule"].node_id = ["M0", "M1"]
    g["molecule"].x = torch.zeros(2, 4)

    g["target", "animal_model", "disease"].edge_index = torch.tensor([[0], [0]])
    g["target", "animal_model", "disease"].edge_attr = torch.tensor([[0.5, 0.1]])
    g["target", "animal_model", "disease"].edge_time = torch.tensor([2010])

    # T0 → M0, T0 → M1 (rel_a)
    g["target", "rel_a", "molecule"].edge_index = torch.tensor([[0, 0], [0, 1]])
    g["target", "rel_a", "molecule"].edge_attr = torch.tensor([[0.6, 0.0], [0.7, 0.0]])
    g["target", "rel_a", "molecule"].edge_time = torch.tensor([2018, 2019])

    # M0 → D0, M1 → D0 (rel_b)
    g["molecule", "rel_b", "disease"].edge_index = torch.tensor([[0, 1], [0, 0]])
    g["molecule", "rel_b", "disease"].edge_attr = torch.tensor([[0.8, 0.0], [0.9, 0.0]])
    g["molecule", "rel_b", "disease"].edge_time = torch.tensor([2020, 2021])

    mappings = {"node_mapping": {
        "target": {"T0": 0},
        "disease": {"D0": 0},
        "molecule": {"M0": 0, "M1": 1},
    }}
    return g, mappings


def test_kpaths_diversity_filter_drops_duplicate_relseq():
    g, m = _make_diversity_graph()
    ex = KPathExtractor(g, mappings=m, mode="kpaths")
    paths = ex.find_paths("T0", "D0", src_type="target", dst_type="disease",
                          max_hops=2, max_paths=10)
    relseqs = {tuple(h.relation for h in p.hops) for p in paths}
    # Diversity filter collapses the two (rel_a, rel_b) routes into one.
    assert ("animal_model",) in relseqs
    assert ("rel_a", "rel_b") in relseqs
    assert len(paths) == 2


def test_no_filter_keeps_duplicate_relseq():
    g, m = _make_diversity_graph()
    ex = KPathExtractor(g, mappings=m, mode="no_filter")
    paths = ex.find_paths("T0", "D0", src_type="target", dst_type="disease",
                          max_hops=2, max_paths=10)
    # All three simple paths kept: 1-hop direct + two 2-hop variants.
    assert len(paths) == 3


def test_neighbors_only_returns_endpoints_neighbors():
    g, m = _make_diversity_graph()
    ex = KPathExtractor(g, mappings=m, mode="neighbors_only")
    paths = ex.find_paths("T0", "D0", src_type="target", dst_type="disease",
                          max_hops=3, max_paths=8)
    # Each returned path is a single hop.
    assert all(len(p.hops) == 1 for p in paths)
    src_starts = {p.hops[0].src_id for p in paths}
    # Half of paths start at T0, the rest start at D0.
    assert "T0" in src_starts and "D0" in src_starts


def test_kpaths_disconnected_falls_back_to_one_node():
    """When src and dst sit in disjoint components, K-Paths should fall back
    to one-node neighbors via the NetworkXNoPath branch."""
    g = HeteroData()
    g["target"].num_nodes = 1
    g["target"].node_id = ["T0"]
    g["target"].x = torch.zeros(1, 4)
    g["disease"].num_nodes = 1
    g["disease"].node_id = ["D0"]
    g["disease"].x = torch.zeros(1, 4)
    g["molecule"].num_nodes = 2
    g["molecule"].node_id = ["M0", "M1"]
    g["molecule"].x = torch.zeros(2, 4)

    # Component A: T0 -- modulated_by -- M0  (no edges to disease side)
    g["molecule", "modulated_by", "target"].edge_index = torch.tensor([[0], [0]])
    g["molecule", "modulated_by", "target"].edge_attr = torch.tensor([[0.5, 0.0]])
    g["molecule", "modulated_by", "target"].edge_time = torch.tensor([2018])

    # Component B: M1 -- clinical_trial --> D0  (no edges to target side)
    g["molecule", "clinical_trial", "disease"].edge_index = torch.tensor([[1], [0]])
    g["molecule", "clinical_trial", "disease"].edge_attr = torch.tensor([[0.9, 0.0]])
    g["molecule", "clinical_trial", "disease"].edge_time = torch.tensor([2020])

    m = {"node_mapping": {
        "target": {"T0": 0}, "disease": {"D0": 0}, "molecule": {"M0": 0, "M1": 1},
    }}
    ex = KPathExtractor(g, mappings=m, mode="kpaths")
    paths = ex.find_paths("T0", "D0", src_type="target", dst_type="disease",
                          max_hops=3, max_paths=10)
    # No path between components → fallback to one-node neighbors of both endpoints.
    assert len(paths) >= 1
    assert all(len(p.hops) == 1 for p in paths)


# -------------------- drop_pair_relations (A3-minus-own arm) --------------------

def _make_own_edge_graph() -> tuple[HeteroData, dict]:
    """T0 -clinical_trial_ongoing(0.9)-> D0 and T0 -literature(0.3)-> D0 (parallel),
    plus M0 -modulated_by-> T0 and M0 -clinical_trial_ongoing-> D0 (a 2-hop route
    through a NEIGHBOURING trial edge, which must survive the per-pair drop)."""
    g = HeteroData()
    for nt, ids in [("target", ["T0"]), ("disease", ["D0"]), ("molecule", ["M0"])]:
        g[nt].num_nodes = len(ids)
        g[nt].node_id = ids
        g[nt].x = torch.zeros(len(ids), 4)
    for rel, s, w in [("clinical_trial_ongoing", 0.9, 2022), ("literature", 0.3, 2015)]:
        g["target", rel, "disease"].edge_index = torch.tensor([[0], [0]])
        g["target", rel, "disease"].edge_attr = torch.tensor([[s, 0.0]])
        g["target", rel, "disease"].edge_time = torch.tensor([w])
    g["molecule", "modulated_by", "target"].edge_index = torch.tensor([[0], [0]])
    g["molecule", "modulated_by", "target"].edge_attr = torch.tensor([[0.7, 0.0]])
    g["molecule", "modulated_by", "target"].edge_time = torch.tensor([2018])
    g["molecule", "clinical_trial_ongoing", "disease"].edge_index = torch.tensor([[0], [0]])
    g["molecule", "clinical_trial_ongoing", "disease"].edge_attr = torch.tensor([[0.8, 0.0]])
    g["molecule", "clinical_trial_ongoing", "disease"].edge_time = torch.tensor([2021])
    m = {"node_mapping": {"target": {"T0": 0}, "disease": {"D0": 0}, "molecule": {"M0": 0}}}
    return g, m


def _direct(paths):
    return [p for p in paths if len(p.hops) == 1]


def test_drop_pair_relations_replaces_direct_edge_and_keeps_neighbours():
    g, m = _make_own_edge_graph()
    ex = KPathExtractor(g, mappings=m, mode="kpaths")
    base = ex.find_paths("D0", "T0", max_hops=3, max_paths=10)
    assert _direct(base)[0].hops[0].relation.lstrip("~") == "clinical_trial_ongoing"
    own = ex.find_paths("D0", "T0", max_hops=3, max_paths=10,
                        drop_pair_relations={"clinical_trial_ongoing"})
    # direct path now carries the remaining literature edge, not the label edge
    assert [p.hops[0].relation.lstrip("~") for p in _direct(own)] == ["literature"]
    # the 2-hop route through the NEIGHBOURING trial edge (M0-D0) is kept
    assert any(len(p.hops) == 2 for p in own)
    # graph restored: a plain call sees the label edge again
    again = ex.find_paths("D0", "T0", max_hops=3, max_paths=10)
    assert [[h.relation for h in p.hops] for p in again] == [[h.relation for h in p.hops] for p in base]


def test_drop_pair_relations_fallback_never_reintroduces_dropped_edge():
    g = HeteroData()
    for nt, ids in [("target", ["T0"]), ("disease", ["D0"])]:
        g[nt].num_nodes = 1
        g[nt].node_id = ids
        g[nt].x = torch.zeros(1, 4)
    g["target", "clinical_trial_ongoing", "disease"].edge_index = torch.tensor([[0], [0]])
    g["target", "clinical_trial_ongoing", "disease"].edge_attr = torch.tensor([[0.9, 0.0]])
    g["target", "clinical_trial_ongoing", "disease"].edge_time = torch.tensor([2022])
    m = {"node_mapping": {"target": {"T0": 0}, "disease": {"D0": 0}}}
    ex = KPathExtractor(g, mappings=m, mode="kpaths")
    own = ex.find_paths("D0", "T0", max_hops=3, max_paths=10,
                        drop_pair_relations={"clinical_trial_ongoing"})
    assert all("clinical_trial" not in h.relation for p in own for h in p.hops)
