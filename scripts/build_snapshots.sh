#!/usr/bin/env bash
# Yearly snapshots of the temporal knowledge graph (THBKG): snapshot_C keeps every edge dated <= C.
# snapshot_2015 gives the candidate universe; snapshot_2020..2025 are the dated graphs G[C] (A2,
# decision year Y = C + 1) and snapshot_2026 is the newest graph (A3).
#
#   SOURCE_GRAPH=/path/to/hetero_graph_with_features_datatype.pt bash scripts/build_snapshots.sh
set -euo pipefail
cd "$(dirname "$0")/.."
SOURCE_GRAPH="${SOURCE_GRAPH:?Set SOURCE_GRAPH to the full THBKG graph (.pt)}"
SNAP_DIR="${SNAP_DIR:-data/snapshots}"
mkdir -p "${SNAP_DIR}"
for C in ${CUTOFFS:-2015 2020 2021 2022 2023 2024 2025 2026}; do
    python -m src.graph.build_snapshot --graph "${SOURCE_GRAPH}" --cutoff-year "${C}" \
        --output "${SNAP_DIR}/snapshot_${C}_datatype.pt"
done
