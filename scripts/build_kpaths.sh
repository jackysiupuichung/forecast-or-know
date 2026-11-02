#!/usr/bin/env bash
# K-path caches for every candidate pair of each decision year Y (top 20 paths, <= 3 hops; the prompt
# shows the first 8). The 'advancement' relation is excluded from the retrieval graph.
#   A2 (dated graph):  snapshot_<Y-1>  -> <set>/paths/kpaths_h3_p20_noadv.json
#   A3 (newest graph): snapshot_2026   -> <set>/paths/kpaths_h3_p20_snap2026_noadv.json
#
#   OT_DIR=/path/to/26.03/evidenceDated ARM=A2|A3 [YEARS="2021 ... 2026"] [SHARDS=8] bash scripts/build_kpaths.sh
#
# Cost: about 1.5 pairs/s per process on snapshot_2026 and up to 32 GB of RAM per process; one cache
# for all six years takes 13-26 CPU-hours. SHARDS runs that many processes in parallel and merges them.
set -euo pipefail
cd "$(dirname "$0")/.."
OT_DIR="${OT_DIR:?Set OT_DIR to the Open Targets 26.03 evidence-dated layer}"
SNAP_DIR="${SNAP_DIR:-data/snapshots}"
ARM="${ARM:?Set ARM=A2 or ARM=A3}"
SHARDS="${SHARDS:-1}"
for Y in ${YEARS:-2021 2022 2023 2024 2025 2026}; do
    SET="data/evaluation/ratio10/test_yr${Y}"
    if [[ "${ARM}" == "A2" ]]; then
        GRAPH="${SNAP_DIR}/snapshot_$((Y - 1))_datatype.pt"; OUT="${SET}/paths/kpaths_h3_p20_noadv.json"
    else
        GRAPH="${SNAP_DIR}/snapshot_2026_datatype.pt"; OUT="${SET}/paths/kpaths_h3_p20_snap2026_noadv.json"
    fi
    mkdir -p "${SET}/paths"
    COMMON=(--nodes-dir "${OT_DIR}" --csv-dir "${SET}" --max-hops 3 --max-paths 20 --output "${OUT}")
    if [[ "${SHARDS}" == "1" ]]; then
        python scripts/precompute_kpaths.py --graph "${GRAPH}" "${COMMON[@]}" --exclude-relations advancement
    else
        seq 0 $((SHARDS - 1)) | xargs -P "${SHARDS}" -I{} python scripts/precompute_kpaths.py --graph "${GRAPH}" \
            "${COMMON[@]}" --exclude-relations advancement --shard "{}/${SHARDS}"
        python scripts/precompute_kpaths.py --graph "${GRAPH}" "${COMMON[@]}" --merge-shards "${SHARDS}"
        rm -f "${OUT}".shard*of"${SHARDS}"
    fi
done
