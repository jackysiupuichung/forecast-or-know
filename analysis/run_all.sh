#!/usr/bin/env bash
# Every table and figure of the paper from the shipped data (no GPU).
# Writes outputs/tables/*.tex, outputs/figures/*.png and prints the in-text numbers.
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/unpack_data.sh
python analysis/per_query.py            # per-query metrics          -> data/aggregates/per_query.csv
python analysis/build_scores_long.py    # per-candidate scores       -> data/aggregates/scores_long.parquet
python analysis/validate_engine.py      # query metrics, checked     -> data/aggregates/q_full.parquet
python analysis/make_tex_tables.py      # Tables 1, 3 and 5; caption counts and chance levels
python analysis/add_single_baselines.py # Table 4
python analysis/plot_gaps_main.py       # Figure 2
python analysis/plot_decay_yearly_2x2_ap.py ap --paper   # Figure 3 (slopes in the caption)
python analysis/plot_dataset_desc_current.py             # Appendix C figure
for a in a1 a2 a3; do ARM=$a python analysis/plot_expanding_arm_large.py; done   # Appendix F figure
python analysis/plot_per_ta_2x2.py      # Appendix G figure
python analysis/plot_binary_ece.py      # Appendix H figure
python analysis/r6_r8_checks.py         # Appendix E: floor across the cutoff and its sensitivity checks
