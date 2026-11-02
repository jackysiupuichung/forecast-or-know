"""Genetic-association evidence of each candidate pair at the decision time C (Table 4 baselines).

genetic_direct  Open Targets genetic-association evidence score of (t, d) in snapshot_C (rows dated <= C)
genetic_prop    max of that score over d and its ontology descendants (Open Targets-style propagation)

Needs the graph snapshots (SNAP_DIR) and the full Open Targets diseases table (OT_DIR). The output
ships with the repository, so Table 4 rebuilds without them.

    python analysis/genetic_scores.py   -> $AGG_DIR/genetic_scores.csv.gz
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import AGG_DIR, OT_DIR, snapshot  # noqa: E402

ds = pd.read_parquet(OT_DIR / "diseases", columns=["id", "descendants"])
desc = {r.id: set(r.descendants) if r.descendants is not None else set() for r in ds.itertuples()}
frames = []
for Y in range(2021, 2027):
    C = Y - 1
    df = pd.read_csv(f"data/evaluation/ratio10/test_yr{Y}/trial_entry_ratio10.csv",
                     usecols=["targetId", "diseaseId"])
    g = torch.load(snapshot(C), weights_only=False)
    tid, did = np.asarray(g["target"].node_id), np.asarray(g["disease"].node_id)
    s = g[("target", "genetic_association", "disease")]
    assert (s.edge_time.numpy() <= C).all(), f"genetic edge after C in snapshot_{C}"
    e = pd.DataFrame({"t": tid[s.edge_index[0].numpy()], "d": did[s.edge_index[1].numpy()],
                      "score": s.edge_attr[:, 0].numpy()}).groupby(["t", "d"]).score.max()
    by_t = {}
    for (t, d), v in e.items():
        by_t.setdefault(t, {})[d] = v

    def direct(t, d):
        return by_t.get(t, {}).get(d, 0.0)

    def prop(t, d):
        sc = by_t.get(t)
        if not sc:
            return 0.0
        return max([sc.get(d, 0.0)] + [sc[x] for x in desc.get(d, ()) if x in sc])

    df["genetic_direct"] = [direct(t, d) for t, d in zip(df.targetId, df.diseaseId)]
    df["genetic_prop"] = [prop(t, d) for t, d in zip(df.targetId, df.diseaseId)]
    df["year"] = Y
    frames.append(df); del g
    print(Y, "genetic edges", len(e), flush=True)
AGG_DIR.mkdir(parents=True, exist_ok=True)
pd.concat(frames, ignore_index=True)[["year", "diseaseId", "targetId", "genetic_direct", "genetic_prop"]] \
    .to_csv(AGG_DIR / "genetic_scores.csv.gz", index=False)
print("wrote", AGG_DIR / "genetic_scores.csv.gz")
