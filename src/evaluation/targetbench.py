"""
Shared pieces for building the candidate sets (scripts/build_ratio_eval.py): the candidate universe,
dated first-trial pairs, and the label-blind trial-count heuristics stored with each candidate.

Every score for cutoff C is computed from pairs with first-trial year <= C only.
"""
import glob

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.stats import rankdata

from src.evaluation.trial_entry_dataset_builder import CANONICAL_TA_IDS
from src.paths import OT_DIR

ROOT = str(OT_DIR)   # full Open Targets 26.03 evidence-dated layer (set OT_DIR)
POP_EDGES = [0, 1, 2, 4, 8, 16, 32, 64, 128]   # trial-breadth bins: 0, 1, 2-3, 4-7, ..., 128+


def pop_bin(b):
    return np.searchsorted(POP_EDGES, b, side="right") - 1


def load_universe(graph_path: str, root: str = ROOT):
    """Protein-coding genes (OT targets.biotype) that are target nodes in the graph.
    Returns (genes array, gene->index dict, set of graph disease ids)."""
    g = torch.load(graph_path, weights_only=False)
    graph_targets, graph_diseases = set(g["target"].node_id), set(g["disease"].node_id)
    del g
    tg = pd.read_parquet(f"{root}/targets", columns=["id", "biotype"])
    genes = np.array(sorted(set(tg[tg.biotype == "protein_coding"].id) & graph_targets))
    return genes, {x: i for i, x in enumerate(genes)}, graph_diseases


def load_disease_meta(root: str = ROOT):
    """(canonical TAs per disease, ontology ancestors|descendants per disease)."""
    ds = pd.read_parquet(f"{root}/diseases", columns=["id", "therapeuticAreas", "ancestors", "descendants"])
    d_tas = {r.id: set(r.therapeuticAreas) & set(CANONICAL_TA_IDS) if r.therapeuticAreas is not None else set()
             for r in ds.itertuples()}
    d_rel = {r.id: (set(r.ancestors) if r.ancestors is not None else set())
             | (set(r.descendants) if r.descendants is not None else set()) for r in ds.itertuples()}
    return d_tas, d_rel


def load_pairs(root: str = ROOT) -> pd.DataFrame:
    """First-trial year per (target, disease) from clinical_precedence."""
    files = sorted(glob.glob(f"{root}/sourceId=clinical_precedence/*.parquet"))
    cp = pd.concat([pd.read_parquet(f, columns=["targetId", "diseaseId", "resolvedTrialDate"])
                    for f in files], ignore_index=True)
    cp["resolvedTrialDate"] = pd.to_datetime(cp["resolvedTrialDate"], errors="coerce")
    cp = cp.dropna(subset=["resolvedTrialDate", "targetId", "diseaseId"])
    cp["year"] = cp.resolvedTrialDate.dt.year.astype(int)
    return cp.groupby(["targetId", "diseaseId"], as_index=False)["year"].min()


class Stage1:
    """Label-blind heuristics at cutoff C over the gene universe.

    breadth    #diseases the target was trialled for by C
    recent     #new trial pairs of the target in C-2..C
    ta_breadth #diseases sharing a canonical TA with d that the target was trialled for by C
    onto       #ontology relatives (ancestors/descendants) of d the target was trialled for by C
    rrf        reciprocal-rank fusion (k=60) of the four
    """

    def __init__(self, pairs: pd.DataFrame, gidx: dict, C: int, d_tas: dict, d_rel: dict):
        self.C = C
        self.G = len(gidx)
        p = pairs[pairs.targetId.isin(gidx)]
        self.all_d = sorted(p.diseaseId.unique())
        self.didx = {d: i for i, d in enumerate(self.all_d)}
        self.d_tas, self.d_rel = d_tas, d_rel
        pre = p[p.year <= C]
        gi, di = pre.targetId.map(gidx).to_numpy(), pre.diseaseId.map(self.didx).to_numpy()
        self.M = sp.csr_matrix((np.ones(len(pre)), (gi, di)), shape=(self.G, len(self.all_d)))
        self.breadth = np.asarray(self.M.sum(1)).ravel()
        rec = pre[pre.year >= C - 2]
        self.recent = np.bincount(rec.targetId.map(gidx), minlength=self.G).astype(float)
        self.bins = pop_bin(self.breadth)
        self._rk_breadth = rankdata(-self.breadth)
        self._rk_recent = rankdata(-self.recent)
        self.known_by_d = pre.groupby("diseaseId").targetId.apply(
            lambda s: s.map(gidx).to_numpy()).to_dict()

    def disease_scores(self, d: str) -> dict:
        tas, rel = self.d_tas.get(d, set()), self.d_rel.get(d, set())
        ta_cols = [self.didx[x] for x in self.all_d if self.d_tas.get(x, set()) & tas] if tas else []
        rel_cols = [self.didx[x] for x in rel if x in self.didx]
        ta = np.asarray(self.M[:, ta_cols].sum(1)).ravel() if ta_cols else np.zeros(self.G)
        onto = np.asarray(self.M[:, rel_cols].sum(1)).ravel() if rel_cols else np.zeros(self.G)
        rrf = (1 / (60 + self._rk_breadth) + 1 / (60 + self._rk_recent)
               + 1 / (60 + rankdata(-ta)) + 1 / (60 + rankdata(-onto)))
        return {"breadth": self.breadth, "recent": self.recent, "ta_breadth": ta, "onto": onto, "rrf": rrf}

    def known(self, d: str) -> np.ndarray:
        """Indices of genes already trialled for d by C."""
        return self.known_by_d.get(d, np.array([], int))
