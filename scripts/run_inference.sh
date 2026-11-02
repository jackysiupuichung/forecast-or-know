#!/usr/bin/env bash
# Zero-shot scoring of one model on the time-aware 1:10 test sets, one decision year Y at a time,
# with the exact arguments used for the paper (Hugging Face transformers, greedy decoding, Yes/No
# log-probabilities read at the answer token).
#
#   MODEL=meta-llama/Llama-3.1-8B-Instruct SLUG=llama3.1-8b ARM=A0|A1|A2|A3 \
#       [LOAD_4BIT=1] [YEARS="2021 ... 2026"] bash scripts/run_inference.sh
#
# Models (pin these revisions for an exact rerun; LOAD_4BIT=1 = NF4 for the 27B-70B models):
#   meta-llama/Llama-3.1-8B-Instruct  0e9e39f2   meta-llama/Llama-3.1-70B-Instruct 1605565b (4-bit)
#   Qwen/Qwen3-8B                     b968826d   Qwen/Qwen3-32B                    9216db57 (4-bit)
#   google/gemma-2-9b-it              11c9b309   google/gemma-2-27b-it             aaf20e6b (4-bit)
# The Llama and Gemma repositories are gated: export HF_TOKEN.
#
# Arms: A1 descriptions, no retrieval; A0 the A1 prompt on the identifiers-only sets
# (scripts/make_ids_sets.py); A2 paths from the dated graph snapshot_<Y-1>; A3 paths from the newest
# graph snapshot_2026 (evidence date stated as December 2026). A0 and A1 state "December 2023" as the
# knowledge date for every model and year. A2 strips the [score=...] annotations from the paths; A3
# shows them (as in the runs reported in the paper).
#
# Output: results/<SLUG>/<set>_test_yr<Y>_<arm>/{predictions.jsonl,metrics.json,run_config.json}
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${MODEL:?Set MODEL=<hf id>}"; SLUG="${SLUG:?Set SLUG=<short name>}"; ARM="${ARM:?Set ARM}"
QUANT=(); [[ "${LOAD_4BIT:-0}" == "1" ]] && QUANT=(--quantization bitsandbytes)
for Y in ${YEARS:-2021 2022 2023 2024 2025 2026}; do
    C=$((Y - 1))
    if [[ "${ARM}" == "A0" ]]; then SETROOT=data/evaluation/ratio10_ids; ABL=A1; else SETROOT=data/evaluation/ratio10; ABL="${ARM}"; fi
    SET="${SETROOT}/test_yr${Y}"
    OUT="results/${SLUG}/$(basename "${SETROOT}")_test_yr${Y}_${ABL}"
    ARGS=(--dataset "${SET}/trial_entry_ratio10.csv" --provider transformers --model "${MODEL}"
          --ablation "${ABL}" --output-dir "${OUT}" --max-tokens 3 --scoring-mode binary --max-paths 8
          --trial-window-year "${C}" --label-year "${Y}" "${QUANT[@]}")
    case "${ARM}" in
        A2) ARGS+=(--graph-cutoff "December ${C}" --paths-cache "${SET}/paths/kpaths_h3_p20_noadv.json") ;;
        A3) ARGS+=(--graph-cutoff "December 2026" --paths-cache "${SET}/paths/kpaths_h3_p20_snap2026_noadv.json") ;;
        *)  ARGS+=(--graph-cutoff "December ${C}") ;;
    esac
    mkdir -p "${OUT}"
    echo ">> ${SLUG} ${ARM} yr${Y}"
    python scripts/run_task1_experiment.py "${ARGS[@]}"
done
