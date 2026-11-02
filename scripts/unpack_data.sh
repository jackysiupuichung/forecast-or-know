#!/usr/bin/env bash
# Unpack the shipped candidate sets (gzip) next to themselves.
set -euo pipefail
cd "$(dirname "$0")/.."
for f in data/evaluation/ratio10/test_yr*/trial_entry_ratio10.csv.gz; do gunzip -kf "$f"; done
echo "unpacked $(ls data/evaluation/ratio10/test_yr*/trial_entry_ratio10.csv | wc -l) candidate sets"
