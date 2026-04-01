#!/bin/bash
# Launch V2 input ablation (9 runs) + oracle predictor tests (3 runs)
# All run in parallel on GPU with nohup

set -e
cd "$(dirname "$0")/.."

# Disable torch.compile to avoid 30-60 min warmup per process
export TORCHDYNAMO_DISABLE=1

RESULTS_DIR="results/diagnostics"
mkdir -p "$RESULTS_DIR"

STEPS=15000
SEED=42
DEVICE="cuda:0"

# Mechanisms to test (skip adaptation_only for ablation — it has no V2 feedback)
MECHANISMS="dampening sharpening center_surround"
INPUT_MODES="l23 l4 l4_l23"

echo "=== V2 Input Ablation: 9 runs (Stage 1 + Stage 2) ==="
for mech in $MECHANISMS; do
    for mode in $INPUT_MODES; do
        LOG="$RESULTS_DIR/ablation_${mech}_${mode}.log"
        echo "  Launching: $mech / $mode -> $LOG"
        nohup python -m scripts.train \
            --mechanism "$mech" \
            --v2-input "$mode" \
            --stage2-steps "$STEPS" \
            --device "$DEVICE" \
            --seed "$SEED" \
            --output "$RESULTS_DIR/ablation_${mech}_${mode}" \
            > "$LOG" 2>&1 &
    done
done

echo ""
echo "=== Oracle Predictor Tests: 3 runs ==="
ORACLE_STEPS=5000
for mech in $MECHANISMS; do
    LOG="$RESULTS_DIR/oracle_${mech}.log"
    echo "  Launching: oracle $mech -> $LOG"
    nohup python -m scripts.oracle_test \
        --mechanism "$mech" \
        --steps "$ORACLE_STEPS" \
        --device "$DEVICE" \
        --seed "$SEED" \
        --output "$RESULTS_DIR/oracle" \
        > "$LOG" 2>&1 &
done

echo ""
echo "All 12 processes launched. Monitor with:"
echo "  tail -f $RESULTS_DIR/*.log"
echo "  nvidia-smi"
echo ""
echo "PIDs:"
jobs -l
