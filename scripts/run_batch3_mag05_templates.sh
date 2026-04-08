#!/bin/bash
# Batch 3: Template manipulation at mag=0.5 (coincidence gate active)
# 4 configs × 3 seeds = 12 runs total
# Each run: Stage 1 (2000 steps) + Stage 2 (5000 steps)
# Stagger launches by 90s to avoid torch.compile RAM spikes
# Max 4 concurrent on GPU

set -e

LOG_DIR="logs/batch3"
OUT_BASE="results/batch3"
mkdir -p "$LOG_DIR"

run_one() {
    local cfg="$1" seed="$2" name="$3"
    local out_dir="${OUT_BASE}/${name}"
    local log_file="${LOG_DIR}/${name}.log"
    echo "  [$(date +%H:%M:%S)] Starting: $name"
    python3 -m scripts.train \
        --config "$cfg" \
        --seed "$seed" \
        --output "$out_dir" \
        --device cuda \
        > "$log_file" 2>&1
    echo "  [$(date +%H:%M:%S)] Finished: $name (exit=$?)"
}

ram_check() {
    echo "  RAM: $(free -m | awk '/Mem:/{print $3}') MB used"
}

echo "=== Batch 3: Template manipulation at mag=0.5 (12 runs) ==="
echo "Start time: $(date)"

# --- Wave A: true_s42, true_s123, wrong_s42, wrong_s123 ---
echo ""
echo "=== Wave A (4 runs, staggered) — $(date) ==="
run_one config/apical_template_true.yaml 42 true_s42 &
PID1=$!
sleep 90
ram_check
run_one config/apical_template_true.yaml 123 true_s123 &
PID2=$!
sleep 90
ram_check
run_one config/apical_template_wrong.yaml 42 wrong_s42 &
PID3=$!
sleep 90
ram_check
run_one config/apical_template_wrong.yaml 123 wrong_s123 &
PID4=$!
wait $PID1 $PID2 $PID3 $PID4
echo "=== Wave A done — $(date) ==="
ram_check

# --- Wave B: true_s456, wrong_s456, random_s42, random_s123 ---
echo ""
echo "=== Wave B (4 runs, staggered) — $(date) ==="
run_one config/apical_template_true.yaml 456 true_s456 &
PID1=$!
sleep 90
ram_check
run_one config/apical_template_wrong.yaml 456 wrong_s456 &
PID2=$!
sleep 90
ram_check
run_one config/apical_template_random.yaml 42 random_s42 &
PID3=$!
sleep 90
ram_check
run_one config/apical_template_random.yaml 123 random_s123 &
PID4=$!
wait $PID1 $PID2 $PID3 $PID4
echo "=== Wave B done — $(date) ==="
ram_check

# --- Wave C: random_s456, uniform_s42, uniform_s123, uniform_s456 ---
echo ""
echo "=== Wave C (4 runs, staggered) — $(date) ==="
run_one config/apical_template_random.yaml 456 random_s456 &
PID1=$!
sleep 90
ram_check
run_one config/apical_template_uniform.yaml 42 uniform_s42 &
PID2=$!
sleep 90
ram_check
run_one config/apical_template_uniform.yaml 123 uniform_s123 &
PID3=$!
sleep 90
ram_check
run_one config/apical_template_uniform.yaml 456 uniform_s456 &
PID4=$!
wait $PID1 $PID2 $PID3 $PID4
echo "=== Wave C done — $(date) ==="
ram_check

echo ""
echo "=== All 12 runs complete ==="
echo "End time: $(date)"

# Verify all 12 checkpoints
echo ""
echo "=== Checkpoint verification ==="
PASS=0
FAIL=0
for d in true_s42 true_s123 true_s456 wrong_s42 wrong_s123 wrong_s456 random_s42 random_s123 random_s456 uniform_s42 uniform_s123 uniform_s456; do
    ckpt=$(find "${OUT_BASE}/${d}" -name "checkpoint.pt" 2>/dev/null | head -1)
    if [ -n "$ckpt" ]; then
        echo "  OK: $d"
        PASS=$((PASS + 1))
    else
        echo "  MISSING: $d"
        FAIL=$((FAIL + 1))
    fi
done
echo ""
echo "Results: $PASS OK, $FAIL missing out of 12"
