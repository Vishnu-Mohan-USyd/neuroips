#!/bin/bash
# Batch 1: Oracle template manipulation (true/wrong/random/uniform) × 3 seeds
# 12 runs total: 4 configs × 3 seeds (42, 123, 456)
# Each run: Stage 1 (2000 steps) + Stage 2 (5000 steps)
# steps_on=12, steps_isi=4
# Runs 4 at a time on GPU in 3 waves

set -e

LOG_DIR="logs/batch1"
OUT_BASE="results/batch1"
mkdir -p "$LOG_DIR"

echo "=== Batch 1: 12 runs (3 waves × 4 parallel on GPU) ==="
echo "Start time: $(date)"

run_one() {
    local cfg="$1" seed="$2" name="$3"
    local out_dir="${OUT_BASE}/${name}"
    local log_file="${LOG_DIR}/${name}.log"
    echo "  Starting: $name (config=$cfg seed=$seed)"
    python3 -m scripts.train \
        --config "$cfg" \
        --seed "$seed" \
        --output "$out_dir" \
        --device cuda \
        > "$log_file" 2>&1
    echo "  Finished: $name (exit=$?)"
}

# --- Wave 1: true×42, true×123, wrong×42, wrong×123 ---
echo ""
echo "=== Wave 1 (4 runs) — started $(date) ==="
run_one config/apical_template_true.yaml  42  true_s42   &
run_one config/apical_template_true.yaml  123 true_s123  &
run_one config/apical_template_wrong.yaml 42  wrong_s42  &
run_one config/apical_template_wrong.yaml 123 wrong_s123 &
wait
echo "=== Wave 1 complete — $(date) ==="
echo "RAM after Wave 1:"
free -m | head -3

# --- Wave 2: true×456, wrong×456, random×42, random×123 ---
echo ""
echo "=== Wave 2 (4 runs) — started $(date) ==="
run_one config/apical_template_true.yaml   456 true_s456   &
run_one config/apical_template_wrong.yaml  456 wrong_s456  &
run_one config/apical_template_random.yaml 42  random_s42  &
run_one config/apical_template_random.yaml 123 random_s123 &
wait
echo "=== Wave 2 complete — $(date) ==="
echo "RAM after Wave 2:"
free -m | head -3

# --- Wave 3: random×456, uniform×42, uniform×123, uniform×456 ---
echo ""
echo "=== Wave 3 (4 runs) — started $(date) ==="
run_one config/apical_template_random.yaml  456 random_s456  &
run_one config/apical_template_uniform.yaml 42  uniform_s42  &
run_one config/apical_template_uniform.yaml 123 uniform_s123 &
run_one config/apical_template_uniform.yaml 456 uniform_s456 &
wait
echo "=== Wave 3 complete — $(date) ==="

echo ""
echo "=== Batch 1 complete ==="
echo "End time: $(date)"
echo "Checkpoints in: $OUT_BASE"
echo "Logs in: $LOG_DIR"

# Verify all 12 checkpoints
echo ""
echo "=== Checkpoint verification ==="
for d in true_s42 true_s123 true_s456 wrong_s42 wrong_s123 wrong_s456 random_s42 random_s123 random_s456 uniform_s42 uniform_s123 uniform_s456; do
    ckpt=$(find "${OUT_BASE}/${d}" -name "checkpoint.pt" 2>/dev/null | head -1)
    if [ -n "$ckpt" ]; then
        echo "  OK: $d — $ckpt"
    else
        echo "  MISSING: $d"
    fi
done
