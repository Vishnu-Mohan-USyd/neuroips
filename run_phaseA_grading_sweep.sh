#!/usr/bin/env bash
# Task #11 orchestration: run Phase A training + V2 OSI sweep for each
# graded-iso L4→L2/3 sampling variant.
#
# Usage:
#   ./run_phaseA_grading_sweep.sh [variant1 variant2 ...]
#
#   With no args: trains all five variants {random, am, sharp, strict, gentle}.
#
# Each variant writes to /tmp/phaseA_grading_<variant>/ which contains:
#   trained_weights.bin              -- post-training L4→L2/3 weights
#   trained_weights.json             -- sibling metadata (existing pipeline)
#   v1_v2_phaseA_v2_osi.json         -- V2 OSI per L2/3 cell + summary stats
#   v1_v2_phaseA_*.json              -- other validation artefacts (V1/V3/V4/V5)
#   run.log                          -- captured stdout/stderr of the train run
#
# After all variants finish, runs plot_phaseA_grading.py to write
#   /tmp/phaseA_grading_summary.json
#   /tmp/phaseA_grading_summary.png

set -e -u -o pipefail

VARIANTS_DEFAULT=(random am sharp strict gentle)
if (( $# > 0 )); then
    VARIANTS=("$@")
else
    VARIANTS=("${VARIANTS_DEFAULT[@]}")
fi

# Phase A canonical training params (matches the original Phase A run).
N_TRIALS=1000
TRAIN_STIM_MS=500
SEED=42

PROJECT=/home/vishnu/coding_proj/refine_v1/neuroips
BIN=$PROJECT/build/v1_test
[[ -x $BIN ]] || { echo "ERROR: missing binary $BIN" >&2; exit 1; }

echo "task #11 sweep: variants=${VARIANTS[*]}"
echo "  N_TRIALS=$N_TRIALS  TRAIN_STIM_MS=$TRAIN_STIM_MS  SEED=$SEED"
date '+%Y-%m-%d %H:%M:%S start'
SWEEP_T0=$(date +%s)

for V in "${VARIANTS[@]}"; do
    OUT=/tmp/phaseA_grading_${V}
    rm -rf "$OUT"
    mkdir -p "$OUT"
    BIN_PATH=$OUT/trained_weights.bin
    LOG=$OUT/run.log
    echo
    echo "=================================================================="
    echo "variant: $V    out_dir: $OUT"
    echo "=================================================================="
    T0=$(date +%s)
    "$BIN" --train-stdp \
        --l4-l23-grading "$V" \
        --n-train-trials "$N_TRIALS" \
        --train-stim-ms "$TRAIN_STIM_MS" \
        --seed "$SEED" \
        --save-trained-weights "$BIN_PATH" \
        --out_dir "$OUT" \
        2>&1 | tee "$LOG"
    T1=$(date +%s)
    echo "variant=$V  wall_s=$((T1 - T0))"
done

SWEEP_T1=$(date +%s)
echo
echo "=================================================================="
echo "sweep wall total: $((SWEEP_T1 - SWEEP_T0)) s"
echo "=================================================================="

# Aggregate.
echo
echo "running aggregation..."
cd "$PROJECT"
python3 plot_phaseA_grading.py "${VARIANTS[@]}" || {
    echo "aggregation script failed (non-fatal: artefacts on disk)" >&2
}

date '+%Y-%m-%d %H:%M:%S done'
