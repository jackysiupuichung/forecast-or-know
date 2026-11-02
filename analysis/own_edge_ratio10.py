"""Own-edge check: share of positives and negatives whose shown paths (the first 8 of the K-path cache) contain
a direct clinical-trial edge between the queried target and disease, per decision year and arm.
Needs the K-path caches (scripts/build_kpaths.sh).

    python analysis/own_edge_ratio10.py   -> $AGG_DIR/own_edge_disclosure.csv"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import AGG_DIR  # noqa: E402

CLIN = re.compile(r"clinical_trial|clinically_associated")
CACHES = {"A2": "kpaths_h3_p20_noadv.json", "A3": "kpaths_h3_p20_snap2026_noadv.json"}
rows = []
for y in range(2021, 2027):
    w = f"data/evaluation/ratio10/test_yr{y}"
    lab = pd.read_csv(f"{w}/trial_entry_ratio10.csv", usecols=["targetId", "diseaseId", "label"])
    for arm, name in CACHES.items():
        P = json.load(open(f"{w}/paths/{name}"))["pairs"]
        hit = np.array([any(p.count("-->") == 1 and CLIN.search(p) for p in P.get(f"{d}||{t}", {"compact": []})["compact"][:8])
                        for t, d in zip(lab.targetId, lab.diseaseId)])
        del P
        rows.append({"year": y, "arm": arm, "pos_with_own_trial_edge": hit[lab.label == 1].mean(),
                     "neg_with_own_trial_edge": hit[lab.label == 0].mean(),
                     "n_pos": int((lab.label == 1).sum()), "n_neg": int((lab.label == 0).sum())})
df = pd.DataFrame(rows)
AGG_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(AGG_DIR / "own_edge_disclosure.csv", index=False)
print(df.round(4).to_string(index=False))
