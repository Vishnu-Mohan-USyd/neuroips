#!/usr/bin/env bash
# run_rate_multiseed.sh — Multi-seed rate model validation for Phase 4 SNN targets
#
# Purpose:
#   Run 3 critical sweep configs (a1 / a2a / e1) with 3 NEW seeds each
#   to measure single-seed noise on M7 / M10 / FWHM metrics. This is a
#   HARD PRECONDITION for Phase 4 SNN gate finalization per the
#   Validator+Lead joint ruling 2026-04-10. See:
#     /home/vishnu/.claude/plans/snn_rate_model_targets.md §E.2
#
# Scope:
#   3 configs × 3 seeds = 9 total runs
#     - a1  (dampening):  lambda_sensory=0.0,  lambda_e=2.0, l23w=1.0
#     - a2a (transition): lambda_sensory=0.12, lambda_e=2.0, l23w=1.0
#     - e1  (sharpening): lambda_sensory=0.3,  lambda_e=2.0, l23w=3.0
#
# Seed selection:
#   Seeds {1, 7, 13} — chosen to be co-prime with the original sweep seed (42)
#   and small enough to avoid pickling issues with numpy.random.SeedSequence.
#   None collide with any existing run under results/sweep/.
#
# Usage:
#   ./scripts/run_rate_multiseed.sh [device]
#
# Examples:
#   ./scripts/run_rate_multiseed.sh              # device=cuda (default)
#   ./scripts/run_rate_multiseed.sh cuda:0       # pin to GPU 0
#   nohup ./scripts/run_rate_multiseed.sh cuda > logs/multiseed.log 2>&1 &
#
# Output layout:
#   results/multiseed/a1_seed1/    center_surround_seed1/checkpoint.pt, summary.json
#   results/multiseed/a1_seed7/    center_surround_seed7/checkpoint.pt, summary.json
#   results/multiseed/a1_seed13/   ...
#   results/multiseed/a2a_seed1/   ...
#   results/multiseed/a2a_seed7/   ...
#   results/multiseed/a2a_seed13/  ...
#   results/multiseed/e1_seed1/    ...
#   results/multiseed/e1_seed7/    ...
#   results/multiseed/e1_seed13/   ...
#   results/multiseed/all_seeds.json   — merged summary across all 9 runs
#   results/multiseed/per_config_stats.json   — mean±std per config per metric
#
# Expected runtime (single GPU, sequential):
#   ~1 hour per run (2000 stage1 steps + 5000 stage2 steps, batch=32)
#   9 runs × 1 hour = ~9 hours total
#
# Parallelism notes:
#   Runs are sequential by design — the original sweep uses 19-way parallel
#   launches via launch_sweep_groups.sh, but that assumes no VRAM pressure
#   on a single GPU. With 9 runs and ~5 GB VRAM per run, on a single 24 GB
#   GPU you could parallelise 3-way safely; on a 48 GB GPU, 6-way. Modify
#   the `for` loop to use `nohup ... &` with `wait` barriers if you want
#   to shorten wall-clock time — see launch_sweep_groups.sh for the pattern.
#
# Post-hoc analysis:
#   After completion, the script emits `per_config_stats.json` with:
#     { "a1":  {"M7_d10": {"mean": ..., "std": ...}, "global_amp": {...}, ...},
#       "a2a": {...},
#       "e1":  {...} }
#   These replace the single-seed point estimates in
#   /home/vishnu/.claude/plans/snn_rate_model_targets.md §D.2, enabling the
#   Validator to compute proper z-scores against Phase 4 SNN outputs.
#
# Status: PREP ONLY — do NOT execute until the Team Lead escalates after
# Phase 3 SNN training completes. Per Task #32.

set -euo pipefail

# Activate conda if available (needed on remote machines)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate base 2>/dev/null || true
fi

DEVICE="${1:-cuda}"
SEEDS=(1 7 13)
CONFIGS=("a1" "a2a" "e1")

OUTPUT_ROOT="results/multiseed"
LOG_DIR="logs/multiseed"

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"

echo "================================================================"
echo "MULTI-SEED RATE MODEL VALIDATION"
echo "Configs:     ${CONFIGS[*]}"
echo "Seeds:       ${SEEDS[*]}"
echo "Device:      $DEVICE"
echo "Total runs:  $((${#CONFIGS[@]} * ${#SEEDS[@]}))"
echo "Output root: $OUTPUT_ROOT"
echo "Started:     $(date -Iseconds)"
echo "================================================================"

TOTAL_FAILED=0
RUN_COUNT=0
TOTAL_RUNS=$((${#CONFIGS[@]} * ${#SEEDS[@]}))

for config_name in "${CONFIGS[@]}"; do
    config_path="config/sweep/sweep_${config_name}.yaml"

    if [ ! -f "$config_path" ]; then
        echo "ERROR: config not found: $config_path" >&2
        exit 2
    fi

    for seed in "${SEEDS[@]}"; do
        RUN_COUNT=$((RUN_COUNT + 1))
        output_dir="${OUTPUT_ROOT}/${config_name}_seed${seed}"
        run_log="${LOG_DIR}/${config_name}_seed${seed}.log"

        echo ""
        echo "---- [$RUN_COUNT/$TOTAL_RUNS] ${config_name} seed=${seed} ----"
        echo "Config:  $config_path"
        echo "Output:  $output_dir"
        echo "Log:     $run_log"
        echo "Started: $(date -Iseconds)"

        if bash scripts/run_sweep.sh "$config_path" "$output_dir" "$seed" "$DEVICE" \
                > "$run_log" 2>&1; then
            echo "  OK (see $run_log)"
        else
            rc=$?
            echo "  FAILED rc=$rc (see $run_log)"
            TOTAL_FAILED=$((TOTAL_FAILED + 1))
        fi
    done
done

# ---- Merge per-run summaries and compute per-config stats ----
echo ""
echo "=== MERGING RESULTS ==="

PYTHONPATH=. python3 - <<PYEOF
import json
import os
from statistics import mean, stdev

output_root = "${OUTPUT_ROOT}"
# Kept in lock-step with the bash arrays above — update both sites together.
configs = ["a1", "a2a", "e1"]
seeds = [1, 7, 13]

all_runs = []
for cfg in configs:
    for seed in seeds:
        summary_path = os.path.join(output_root, f"{cfg}_seed{seed}", "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                row = json.load(f)
            row["_config"] = cfg
            row["_seed"] = seed
            all_runs.append(row)
        else:
            all_runs.append({
                "_config": cfg,
                "_seed": seed,
                "error": f"summary not found at {summary_path}",
            })

with open(os.path.join(output_root, "all_seeds.json"), "w") as f:
    json.dump(all_runs, f, indent=2)

# Per-config stats on key metrics
KEY_METRICS = [
    "M7_d3", "M7_d5", "M7_d10", "M7_d15",
    "global_amp", "fwhm_delta",
    "fb_contribution",
]

def stats(values):
    vals = [v for v in values if isinstance(v, (int, float))]
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    if len(vals) == 1:
        return {"mean": vals[0], "std": 0.0, "n": 1}
    return {
        "mean": mean(vals),
        "std":  stdev(vals),
        "n":    len(vals),
    }

per_cfg = {}
for cfg in configs:
    rows = [r for r in all_runs if r.get("_config") == cfg and "error" not in r]
    per_cfg[cfg] = {m: stats([r.get(m) for r in rows]) for m in KEY_METRICS}
    per_cfg[cfg]["_seeds_completed"] = sorted(r["_seed"] for r in rows)

with open(os.path.join(output_root, "per_config_stats.json"), "w") as f:
    json.dump(per_cfg, f, indent=2)

# Pretty-print table
print()
print(f"{'config':>6} {'metric':>15} {'mean':>12} {'std':>10} {'n':>3}")
print("-" * 52)
for cfg in configs:
    for m in KEY_METRICS:
        s = per_cfg[cfg][m]
        mean_str = f"{s['mean']:+.4f}" if s["mean"] is not None else "  None  "
        std_str  = f"{s['std']:.4f}"   if s["std"]  is not None else "  None  "
        print(f"{cfg:>6} {m:>15} {mean_str:>12} {std_str:>10} {s['n']:>3}")
    print()
PYEOF

echo ""
echo "================================================================"
echo "MULTI-SEED RUN COMPLETE"
echo "Total runs:  $TOTAL_RUNS"
echo "Failed:      $TOTAL_FAILED"
echo "Per-seed:    $OUTPUT_ROOT/all_seeds.json"
echo "Per-config:  $OUTPUT_ROOT/per_config_stats.json"
echo "Finished:    $(date -Iseconds)"
echo "================================================================"

exit $TOTAL_FAILED
