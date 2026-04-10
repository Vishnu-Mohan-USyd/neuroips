#!/usr/bin/env bash
# launch_sweep_groups.sh — Launch sweep groups sequentially, runs within each group in parallel
#
# Usage:
#   ./scripts/launch_sweep_groups.sh [seed] [device]
#
# Runs on the CURRENT machine. For remote execution:
#   ssh vishnu@100.123.25.88 'cd ~/neuroips/freshstart && nohup ./scripts/launch_sweep_groups.sh 42 cuda > logs/sweep_master.log 2>&1 &'
#
# Group execution order:
#   A (6 runs, parallel) → wait → B (5 runs, parallel) → wait →
#   C (2 runs, parallel) → wait → D (3 runs, parallel) → wait →
#   E (2 runs, parallel) → wait → merge results

set -euo pipefail

SEED="${1:-42}"
DEVICE="${2:-cuda}"

LOG_DIR="logs/sweep"
mkdir -p "$LOG_DIR" results/sweep

echo "================================================================"
echo "SWEEP LAUNCHER — 19 configs across 5 groups"
echo "Seed: $SEED, Device: $DEVICE"
echo "Started: $(date -Iseconds)"
echo "================================================================"

launch_group() {
    local group_name="$1"
    shift
    local configs=("$@")
    local pids=()

    echo ""
    echo "=== GROUP $group_name: ${#configs[@]} runs ==="
    echo "Started: $(date -Iseconds)"

    for entry in "${configs[@]}"; do
        local name="${entry%% *}"
        local config="config/sweep/sweep_${name}.yaml"
        local output="results/sweep/${name}"

        echo "  Launching $name: $config → $output"
        nohup bash scripts/run_sweep.sh "$config" "$output" "$SEED" "$DEVICE" \
            > "${LOG_DIR}/${name}.log" 2>&1 &
        pids+=($!)
    done

    echo "  PIDs: ${pids[*]}"
    echo "  Waiting for group $group_name to complete..."

    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            failed=$((failed + 1))
        fi
    done

    echo "=== GROUP $group_name COMPLETE (${#configs[@]} runs, $failed failed) ==="
    echo "Finished: $(date -Iseconds)"
    return $failed
}

TOTAL_FAILED=0

# Group A: Phase boundary (7 runs)
launch_group "A" "a1" "a2" "a3" "a4" "a5" "a6" "a7" || TOTAL_FAILED=$((TOTAL_FAILED + $?))

# Group B: Energy interaction (5 runs)
launch_group "B" "b1" "b2" "b3" "b4" "b5" || TOTAL_FAILED=$((TOTAL_FAILED + $?))

# Group C: L2/3 weight (2 runs)
launch_group "C" "c1" "c2" || TOTAL_FAILED=$((TOTAL_FAILED + $?))

# Group D: Mismatch interaction (3 runs)
launch_group "D" "d1" "d2" "d3" || TOTAL_FAILED=$((TOTAL_FAILED + $?))

# Group E: Deconfound (2 runs)
launch_group "E" "e1" "e2" || TOTAL_FAILED=$((TOTAL_FAILED + $?))

# Merge all results
echo ""
echo "=== MERGING RESULTS ==="
python3 -c "
import json, os

results = []
for name in ['a1','a2','a3','a4','a5','a6','a7','b1','b2','b3','b4','b5','c1','c2','d1','d2','d3','e1','e2']:
    path = f'results/sweep/{name}/summary.json'
    if os.path.exists(path):
        with open(path) as f:
            results.append(json.load(f))
    else:
        results.append({'label': f'sweep_{name}', 'error': 'summary not found'})

with open('results/sweep/all_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Print summary table
print(f'{'Name':>6} {'λ_sens':>7} {'M7_d10':>8} {'Amp':>6} {'FWHM_Δ':>8} {'FB_dir':>8}')
print('-' * 50)
for r in results:
    if 'error' in r:
        print(f'{r[\"label\"]:>6} ERROR: {r[\"error\"]}')
    else:
        label = r.get('label','?').replace('sweep_','')
        m7 = r.get('M7_d10', '?')
        amp = r.get('global_amp', '?')
        fwhm = r.get('fwhm_delta', '?')
        fb = r.get('fb_direction', '?')
        m7s = f'{m7:+.3f}' if isinstance(m7, float) else str(m7)
        amps = f'{amp:.2f}' if isinstance(amp, float) else str(amp)
        fwhms = f'{fwhm:+.1f}' if isinstance(fwhm, float) else str(fwhm)
        print(f'{label:>6} {m7s:>8} {amps:>6} {fwhms:>8} {fb:>8}')
"

echo ""
echo "================================================================"
echo "SWEEP COMPLETE — 19 runs, $TOTAL_FAILED failed"
echo "Results: results/sweep/all_results.json"
echo "Finished: $(date -Iseconds)"
echo "================================================================"
