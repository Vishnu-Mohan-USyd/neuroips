#!/usr/bin/env bash
# run_sweep.sh — Run training + analysis for a single config point
#
# Usage:
#   ./scripts/run_sweep.sh <config_path> <output_dir> [seed] [device]
#
# Example:
#   ./scripts/run_sweep.sh config/exp_dampening_d1.yaml results/sweep/d1_sensory0 42 cuda
#
# What it does:
#   1. Runs full training (stage1 + stage2)
#   2. Runs analyze_representation.py on the checkpoint
#   3. Runs debug_expected_vs_unexpected.py on the checkpoint
#   4. Saves a JSON summary of key metrics
#
# All output is logged to <output_dir>/sweep.log

set -euo pipefail

CONFIG="${1:?Usage: $0 <config> <output_dir> [seed] [device]}"
OUTPUT_DIR="${2:?Usage: $0 <config> <output_dir> [seed] [device]}"
SEED="${3:-42}"
DEVICE="${4:-cuda}"

LABEL="$(basename "$CONFIG" .yaml)"
CHECKPOINT_DIR="${OUTPUT_DIR}/center_surround_seed${SEED}"
CHECKPOINT="${CHECKPOINT_DIR}/checkpoint.pt"
LOGFILE="${OUTPUT_DIR}/sweep.log"
SUMMARY="${OUTPUT_DIR}/summary.json"

mkdir -p "$OUTPUT_DIR"

echo "================================================================" | tee "$LOGFILE"
echo "SWEEP: $LABEL (seed=$SEED, device=$DEVICE)"                      | tee -a "$LOGFILE"
echo "Config:     $CONFIG"                                              | tee -a "$LOGFILE"
echo "Output:     $OUTPUT_DIR"                                          | tee -a "$LOGFILE"
echo "Checkpoint: $CHECKPOINT"                                          | tee -a "$LOGFILE"
echo "Started:    $(date -Iseconds)"                                    | tee -a "$LOGFILE"
echo "================================================================" | tee -a "$LOGFILE"

# ---- Step 1: Training ----
echo "" | tee -a "$LOGFILE"
echo "[1/3] Training..." | tee -a "$LOGFILE"
TRAIN_START=$(date +%s)

PYTHONPATH=. python3 -m scripts.train \
    --config "$CONFIG" \
    --seed "$SEED" \
    --output "$OUTPUT_DIR" \
    --device "$DEVICE" \
    2>&1 | tee -a "$LOGFILE"

TRAIN_END=$(date +%s)
TRAIN_SECS=$((TRAIN_END - TRAIN_START))
echo "[1/3] Training complete in ${TRAIN_SECS}s" | tee -a "$LOGFILE"

# Verify checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT" | tee -a "$LOGFILE"
    exit 1
fi

# ---- Step 2: Representation analysis ----
echo "" | tee -a "$LOGFILE"
echo "[2/3] Running representation analysis..." | tee -a "$LOGFILE"
REP_START=$(date +%s)

PYTHONPATH=. python3 scripts/analyze_representation.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --label "$LABEL" \
    --device "$DEVICE" \
    2>&1 | tee -a "${OUTPUT_DIR}/analysis_representation.log"

REP_END=$(date +%s)
REP_SECS=$((REP_END - REP_START))
echo "[2/3] Representation analysis complete in ${REP_SECS}s" | tee -a "$LOGFILE"

# ---- Step 3: Expected-vs-unexpected analysis ----
echo "" | tee -a "$LOGFILE"
echo "[3/3] Running expected-vs-unexpected analysis..." | tee -a "$LOGFILE"
EVU_START=$(date +%s)

PYTHONPATH=. python3 scripts/debug_expected_vs_unexpected.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --label "$LABEL" \
    --rng-seed "$SEED" \
    2>&1 | tee -a "${OUTPUT_DIR}/analysis_expected_vs_unexpected.log"

EVU_END=$(date +%s)
EVU_SECS=$((EVU_END - EVU_START))
echo "[3/3] Expected-vs-unexpected analysis complete in ${EVU_SECS}s" | tee -a "$LOGFILE"

# ---- Step 4: Extract key metrics into JSON summary ----
echo "" | tee -a "$LOGFILE"
echo "Extracting summary metrics..." | tee -a "$LOGFILE"

PYTHONPATH=. python3 -c "
import re, json, sys

rep_log = open('${OUTPUT_DIR}/analysis_representation.log').read()
evu_log = open('${OUTPUT_DIR}/analysis_expected_vs_unexpected.log').read()

def extract(pattern, text, default='null'):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else default

summary = {
    'config': '${CONFIG}',
    'label': '${LABEL}',
    'seed': ${SEED},
    'train_seconds': ${TRAIN_SECS},
    'M7_d3': extract(r'M7 acc δ=3° delta\s+\|\s+([\-\+\d\.]+)', rep_log),
    'M7_d5': extract(r'M7 acc δ=5° delta\s+\|\s+([\-\+\d\.]+)', rep_log),
    'M7_d10': extract(r'M7 acc δ=10° delta\s+\|\s+([\-\+\d\.]+)', rep_log),
    'M7_d15': extract(r'M7 acc δ=15° delta\s+\|\s+([\-\+\d\.]+)', rep_log),
    'global_amp': extract(r'M10 global mean ratio.*\|\s+([\d\.]+)', rep_log),
    'fwhm_on': extract(r'PopBump FWHM ON.*\|\s+([\d\.]+)', rep_log),
    'fwhm_off': extract(r'PopBump FWHM OFF.*\|\s+([\d\.]+)', rep_log),
    'fwhm_delta': extract(r'PopBump FWHM delta.*\|\s+([\-\+\d\.]+)', rep_log),
    'peak_on': extract(r'stim=90,ora=90:.*ON=([\d\.]+)', rep_log),
    'peak_off': extract(r'stim=90,ora=90:.*OFF=([\d\.]+)', rep_log),
    'fb_on_gap': extract(r'FB-ON gap.*:\s+([\-\+\d\.]+)', evu_log),
    'fb_off_gap': extract(r'FB-OFF gap.*:\s+([\-\+\d\.]+)', evu_log),
    'fb_contribution': extract(r'Feedback contribution:\s+([\-\+\d\.]+)', evu_log),
    'fb_direction': 'WIDENS' if 'WIDENS' in evu_log else ('NARROWS' if 'NARROWS' in evu_log else 'UNKNOWN'),
}

with open('${SUMMARY}', 'w') as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
" 2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "================================================================" | tee -a "$LOGFILE"
echo "SWEEP COMPLETE: $LABEL"                                           | tee -a "$LOGFILE"
echo "Summary:    $SUMMARY"                                             | tee -a "$LOGFILE"
echo "Finished:   $(date -Iseconds)"                                    | tee -a "$LOGFILE"
echo "================================================================" | tee -a "$LOGFILE"
