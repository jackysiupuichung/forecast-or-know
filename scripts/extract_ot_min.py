#!/usr/bin/env python3
"""Small extract of the Open Targets 26.03 evidence-dated layer shipped as data/opentargets_min, enough
for the figures: the therapeutic areas of the evaluated diseases and the first-trial date of every
positive pair (its clinical-precedence records; the figures take the earliest resolvedTrialDate). Open Targets data are CC0.

    OT_DIR=/path/to/26.03/evidenceDated python scripts/extract_ot_min.py
"""
import os
from pathlib import Path

import pandas as pd

OT = Path(os.environ["OT_DIR"])
OUT = Path("data/opentargets_min")
cand = pd.concat([pd.read_csv(f"data/evaluation/ratio10/test_yr{y}/trial_entry_ratio10.csv",
                              usecols=["diseaseId", "targetId", "label"]) for y in range(2021, 2027)])
dis = pd.read_parquet(OT / "diseases", columns=["id", "therapeuticAreas"])
dis = dis[dis.id.isin(set(cand.diseaseId))].reset_index(drop=True)
pos = cand[cand.label == 1][["diseaseId", "targetId"]].drop_duplicates()
cp = pd.read_parquet(OT / "sourceId=clinical_precedence", columns=["targetId", "diseaseId", "resolvedTrialDate"])
cp = cp.merge(pos, on=["diseaseId", "targetId"])          # all records of the positive pairs, as stored
(OUT / "diseases").mkdir(parents=True, exist_ok=True)
(OUT / "sourceId=clinical_precedence").mkdir(parents=True, exist_ok=True)
dis.to_parquet(OUT / "diseases" / "part-0.parquet", index=False)
cp[["targetId", "diseaseId", "resolvedTrialDate"]].to_parquet(
    OUT / "sourceId=clinical_precedence" / "part-0.parquet", index=False)
print(len(dis), "diseases;", len(cp), "clinical-precedence records for", len(cp.groupby(["diseaseId", "targetId"])), "positive pairs")
