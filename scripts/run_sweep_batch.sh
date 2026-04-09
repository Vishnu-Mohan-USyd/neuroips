#!/usr/bin/env bash
# run_sweep_batch.sh — Run sweep over multiple configs sequentially
#
# Usage:
#   ./scripts/run_sweep_batch.sh <config_list_file> [seed] [device]
#
# Config list file format (one per line):
#   <config_path> <output_dir>
#
# Example config_list.txt:
#   config/exp_dampening_d1.yaml results/sweep/d1_sensory0
#   config/exp_dampening_d2.yaml results/sweep/d2_sensory03
#   config/exp_dampening_d3.yaml results/sweep/d3_sensory10
#
# After all runs complete, merges all summary.json files into sweep_results.json

set -euo pipefail

CONFIG_LIST="${1:?Usage: $0 <config_list_file> [seed] [device]}"
SEED="${2:-42}"
DEVICE="${3:-cuda}"

if [ ! -f "$CONFIG_LIST" ]; then
    echo "ERROR: Config list file not found: $CONFIG_LIST"
    exit 1
fi

SWEEP_DIR="$(dirname "$CONFIG_LIST")"
SWEEP_LOG="${SWEEP_DIR}/sweep_batch.log"
SWEEP_RESULTS="${SWEEP_DIR}/sweep_results.json"

echo "================================================================" | tee "$SWEEP_LOG"
echo "BATCH SWEEP — $(wc -l < "$CONFIG_LIST") configs"                 | tee -a "$SWEEP_LOG"
echo "Seed: $SEED, Device: $DEVICE"                                    | tee -a "$SWEEP_LOG"
echo "Started: $(date -Iseconds)"                                      | tee -a "$SWEEP_LOG"
echo "================================================================" | tee -a "$SWEEP_LOG"

TOTAL=0
PASSED=0
FAILED=0

while IFS=' ' read -r CONFIG OUTPUT_DIR; do
    # Skip empty lines and comments
    [[ -z "$CONFIG" || "$CONFIG" == \#* ]] && continue

    TOTAL=$((TOTAL + 1))
    LABEL="$(basename "$CONFIG" .yaml)"

    echo "" | tee -a "$SWEEP_LOG"
    echo "--- [$TOTAL] $LABEL ---" | tee -a "$SWEEP_LOG"

    if bash scripts/run_sweep.sh "$CONFIG" "$OUTPUT_DIR" "$SEED" "$DEVICE" 2>&1; then
        PASSED=$((PASSED + 1))
        echo "--- [$TOTAL] $LABEL: PASSED ---" | tee -a "$SWEEP_LOG"
    else
        FAILED=$((FAILED + 1))
        echo "--- [$TOTAL] $LABEL: FAILED ---" | tee -a "$SWEEP_LOG"
    fi
done < "$CONFIG_LIST"

# Merge all summary.json files
echo "" | tee -a "$SWEEP_LOG"
echo "Merging summaries..." | tee -a "$SWEEP_LOG"

python3 -c "
import json, glob, sys

results = []
while True:
    line = sys.stdin.readline()
    if not line:
        break
    parts = line.strip().split()
    if not parts or parts[0].startswith('#'):
        continue
    output_dir = parts[1]
    summary_path = f'{output_dir}/summary.json'
    try:
        with open(summary_path) as f:
            results.append(json.load(f))
    except FileNotFoundError:
        results.append({'output_dir': output_dir, 'error': 'summary not found'})

with open('${SWEEP_RESULTS}', 'w') as f:
    json.dump(results, f, indent=2)

print(f'Merged {len(results)} results to ${SWEEP_RESULTS}')
" < "$CONFIG_LIST" 2>&1 | tee -a "$SWEEP_LOG"

echo "" | tee -a "$SWEEP_LOG"
echo "================================================================" | tee -a "$SWEEP_LOG"
echo "BATCH COMPLETE: $PASSED passed, $FAILED failed, $TOTAL total"    | tee -a "$SWEEP_LOG"
echo "Results: $SWEEP_RESULTS"                                         | tee -a "$SWEEP_LOG"
echo "Finished: $(date -Iseconds)"                                     | tee -a "$SWEEP_LOG"
echo "================================================================" | tee -a "$SWEEP_LOG"
