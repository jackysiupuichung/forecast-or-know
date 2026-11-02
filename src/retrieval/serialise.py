#!/usr/bin/env python3
"""
Path serialisation for LLM consumption.

Converts structured Path objects (from paths.py) into natural-language strings
annotated with edge metadata: relation type, confidence score, novelty, year.

Two output modes:
    "compact" — one line per hop, arrow-separated:
        TP53 (target) --clinical_trial_positive[score=0.84, year=2019]--> breast carcinoma (disease)
    "verbose" — one paragraph per path, prose style:
        "TP53 (a target) is linked via a 'clinical_trial_positive' relation
        (confidence 0.84, reported in 2019) to breast carcinoma (a disease)."

Public API:
    PathSerialiser(name_map)
        .serialise_path(path, mode='compact') → str
        .serialise_paths(paths, mode='compact', include_index=True) → str
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional
from src.retrieval.paths import Path, Hop


class PathSerialiser:
    """
    Render Path objects as strings suitable for LLM prompts.

    Args:
        name_map: Optional Dict[node_type → Dict[node_id → display_name]].
                  If a node's display name is not in the map, falls back to the ID.
                  E.g. {'target': {'ENSG00000141510': 'TP53'}, 'disease': {'EFO_0000275': 'breast carcinoma'}}
    """

    def __init__(self, name_map: Optional[Dict[str, Dict[str, str]]] = None,
                 render=None) -> None:
        self.name_map = name_map or {}
        # `render` is any object exposing the RenderSpec flags (show_score,
        # show_year, show_novelty, show_evidence, relation_style, max_evidence_chars).
        # Duck-typed so this module never imports graph_context (keeps the import
        # DAG acyclic). render=None == the frozen default (score shown, nothing
        # else), byte-identical to the pre-harness serialiser.
        self.render = render

    def _display(self, node_type: str, node_id: str) -> str:
        name = self.name_map.get(node_type, {}).get(node_id)
        if name and name != node_id:
            return f"{name} ({node_type})"
        return f"{node_id} ({node_type})"

    # Sentinel used by the graph builder for edges without a real year
    _UNDATED_SENTINEL = -9223372036854775808

    # Ontology / hierarchy relations carry no real confidence score and no
    # evidence date — they are definitional taxonomic links. Render bare.
    _ONTOLOGY_RELATIONS = frozenset({
        "is_subtype_of", "~is_subtype_of",
        "is_subpathway_of", "~is_subpathway_of",
    })

    # Schema-relation -> natural-language gloss (Layer 2.5). Ontology relations
    # keep their bare label. Only applied when render.relation_style == "gloss".
    _RELATION_GLOSS = {
        "clinical_trial_positive": "was positive in a trial for",
        "clinical_trial_ongoing": "is in an ongoing trial for",
        "clinical_trial_adverse_effects": "had adverse effects in a trial for",
        "clinical_trial_unmet_efficacy": "showed unmet efficacy in a trial for",
        "clinical_trial_Unknown/Operational": "is in a trial (status unknown) for",
        "clinically_associated": "is clinically associated with",
        "advancement": "advanced in development for",
        "associated_with": "is associated with",
        "interacts_with": "interacts with",
        "participates_in": "participates in",
        "annotated_with": "is annotated with",
        "expressed_in": "is expressed in",
    }

    def _rel_label(self, relation: str) -> str:
        bare = relation.lstrip("~")
        if getattr(self.render, "relation_style", "schema") == "gloss":
            return self._RELATION_GLOSS.get(bare, bare)
        return bare

    def _format_edge_meta(self, hop: Hop) -> str:
        if hop.relation in self._ONTOLOGY_RELATIONS:
            return ""
        r = self.render
        parts: List[str] = []
        # Defaults via getattr => render=None reproduces the frozen `score={:.3f}`.
        if getattr(r, "show_score", True):
            parts.append(f"score={hop.edge_weight:.3f}")
        if getattr(r, "show_year", False) and hop.edge_time not in (None, self._UNDATED_SENTINEL):
            parts.append(f"year={hop.edge_time}")
        if getattr(r, "show_novelty", False):
            parts.append(f"novelty={hop.edge_novelty:.3f}")
        if getattr(r, "show_evidence", False) and getattr(hop, "evidence_text", None):
            snip = hop.evidence_text[: getattr(r, "max_evidence_chars", 200)]
            pmid = getattr(hop, "pmid", None)
            parts.append(f'"{snip}"' + (f" PMID:{pmid}" if pmid else ""))
        return ", ".join(parts)

    def serialise_path(self, path: Path, mode: str = "compact") -> str:
        if not path.hops:
            return "(empty path)"

        if mode == "compact":
            # src --relation[meta]--> dst --relation[meta]--> ...
            parts = [self._display(path.hops[0].src_type, path.hops[0].src_id)]
            for h in path.hops:
                rel = self._rel_label(h.relation)
                meta = self._format_edge_meta(h)
                arrow = f"--{rel}[{meta}]-->" if meta else f"--{rel}-->"
                parts.append(arrow)
                parts.append(self._display(h.dst_type, h.dst_id))
            return " ".join(parts)

        if mode == "verbose":
            # Prose: "X is linked to Y via relation (meta). Y is linked to Z..."
            sentences = []
            for h in path.hops:
                src = self._display(h.src_type, h.src_id)
                dst = self._display(h.dst_type, h.dst_id)
                rel = self._rel_label(h.relation)
                meta = self._format_edge_meta(h)
                tail = f" ({meta})" if meta else ""
                sentences.append(
                    f"{src} is linked to {dst} via '{rel}'{tail}."
                )
            return " ".join(sentences)

        raise ValueError(f"Unknown mode: {mode!r}. Use 'compact' or 'verbose'.")

    def serialise_paths(
        self,
        paths: Iterable[Path],
        mode: str = "compact",
        include_index: bool = True,
    ) -> str:
        """Serialise a list of paths as a newline-separated string."""
        lines: List[str] = []
        for i, p in enumerate(paths, start=1):
            line = self.serialise_path(p, mode=mode)
            if include_index:
                line = f"[{i}] {line}"
            lines.append(line)
        return "\n".join(lines)


def render_paths_from_hops(hops_lists, name_map, render=None):
    """Re-serialise cached raw hop-dicts under a RenderSpec (Tier-1 entry point).

    `hops_lists` is a list of paths, each a list of hop dicts (the `hops` field of
    scripts/precompute_kpaths.py's cache). Reconstructs Hop objects and renders via
    the same PathSerialiser code that produced the cache — so render=None yields
    byte-identical strings to the cached `compact` list (enforced by the
    golden-master test).
    """
    ser = PathSerialiser(name_map, render=render)
    mode = getattr(render, "serialise_mode", "compact")
    out: List[str] = []
    for hl in hops_lists:
        out.append(ser.serialise_path(Path(hops=[Hop(**h) for h in hl]), mode=mode))
    return out


def build_name_map_from_parquets(
    nodes_dir: str,
    node_files: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, str]]:
    """
    Build a name_map from node parquet files produced by the preprocessing pipeline.

    Args:
        nodes_dir:   Directory containing node parquets (e.g. .../evidences/nodes/)
        node_files:  Optional override of {node_type: filename}. Defaults to:
                         target   → targets.parquet   (id, approvedSymbol)
                         disease  → diseases.parquet  (id, diseaseFromSource)
                         molecule → molecule.parquet  (id, canonicalSmiles)  [no name]
                         go       → go.parquet        (id, name)
                         reactome → reactome.parquet  (id, label)

    Returns:
        Dict[node_type → Dict[node_id → display_name]]
    """
    import os
    import pandas as pd

    default_files = {
        "target":   ("targets.parquet",   "approvedSymbol"),
        "disease":  ("diseases.parquet",  "diseaseFromSource"),
        "go":       ("go.parquet",        "name"),
        "reactome": ("reactome.parquet",  "label"),
    }

    name_map: Dict[str, Dict[str, str]] = {}
    for node_type, (fname, name_col) in default_files.items():
        path = os.path.join(nodes_dir, fname)
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_parquet(path, columns=["id", name_col])
        except (KeyError, ValueError):
            # Name column missing — skip
            continue
        name_map[node_type] = dict(zip(df["id"].astype(str), df[name_col].astype(str)))
    return name_map
