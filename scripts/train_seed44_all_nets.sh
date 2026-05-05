#!/bin/bash
# Task #21 — Seed-robustness Part A: train all 5 nets at seed 44.
#
# Mirror of Task #4 (seed 43) protocol exactly. Only `--seed` changes
# (43 → 44). All other training params (configs, n_steps, lambdas, lrs)
# come from the per-net sweep YAMLs and are untouched.
#
# Per-net sweep config (must match Task #4):
#   r1r2  → config/sweep/sweep_rescue_1_2.yaml
#   a1    → config/sweep/sweep_a1.yaml
#   b1    → config/sweep/sweep_b1.yaml
#   c1    → config/sweep/sweep_c1.yaml
#   e1    → config/sweep/sweep_e1.yaml
#
# Resource ceiling per dispatch:
#   ≤10 GB GPU and ≤10 GB RAM per process. Serial 1-parallel.
#   (Task #4 peaked at ~3 GB GPU + ~7 GB RAM per process — well under.)
#
# Outputs (per net):
#   checkpoints/net_seed44_<net>.pt           — final ckpt (Stage-1 + Stage-2)
#   results/training_seed44_<net>.json        — wall-clock + memory + final loss
#   results/training_seed44_<net>.metrics.jsonl — per-100-step Stage-2 metrics
#   logs/seed44_<net>_train.log               — full Python stdout/stderr
#   logs/seed44_<net>_mem.log                 — per-30s RAM/GPU samples
#
# Total wall-clock estimate: ~58 min/net × 5 nets ≈ 5 hr (matches dispatch ETA).

set -euo pipefail

cd /mnt/c/Users/User/codingproj/freshstart

SEED=44

declare -A NET_CFG
NET_CFG[r1r2]="config/sweep/sweep_rescue_1_2.yaml"
NET_CFG[a1]="config/sweep/sweep_a1.yaml"
NET_CFG[b1]="config/sweep/sweep_b1.yaml"
NET_CFG[c1]="config/sweep/sweep_c1.yaml"
NET_CFG[e1]="config/sweep/sweep_e1.yaml"

mkdir -p logs checkpoints results

NETS=("r1r2" "a1" "b1" "c1" "e1")

echo "=== $(date +%T) Seed ${SEED} training launcher: ${#NETS[@]} nets serial ==="

for net in "${NETS[@]}"; do
  cfg="${NET_CFG[$net]}"
  out_tmp="/tmp/seed${SEED}_${net}_tmp"
  rm -rf "$out_tmp"
  mkdir -p "$out_tmp"

  log_train="logs/seed${SEED}_${net}_train.log"
  log_mem="logs/seed${SEED}_${net}_mem.log"
  : > "$log_train"
  : > "$log_mem"

  echo ""
  echo "--- $(date +%T) [$net] starting (cfg=$cfg seed=${SEED}) ---"

  # Background memory sampler (RAM = python proc RSS; GPU via nvidia-smi).
  # Mirrors the seed43_<net>_mem.log format exactly: "HH:MM:SS GPU=NMiB RAM=NMiB".
  (
    while true; do
      if [[ -f /tmp/seed${SEED}_${net}_train.pid ]]; then
        pid=$(cat /tmp/seed${SEED}_${net}_train.pid 2>/dev/null || echo "")
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
          ts=$(date +%T)
          gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
          ram=$(awk '/VmRSS/ {print int($2/1024)}' /proc/"$pid"/status 2>/dev/null || echo 0)
          echo "$ts GPU=${gpu}MiB RAM=${ram}MiB" >> "$log_mem"
        else
          break
        fi
      fi
      sleep 30
    done
  ) &
  mem_pid=$!

  # Launch training. Background it so we can capture PID for memory sampling.
  python3 -m scripts.train --config "$cfg" --seed "$SEED" --output "$out_tmp" \
    >> "$log_train" 2>&1 &
  train_pid=$!
  echo "$train_pid" > "/tmp/seed${SEED}_${net}_train.pid"

  # Wait for training to finish (foreground via wait — PID is captured).
  if ! wait "$train_pid"; then
    echo "[ERROR] $(date +%T) [$net] training FAILED (exit $?)" >&2
    kill "$mem_pid" 2>/dev/null || true
    rm -f "/tmp/seed${SEED}_${net}_train.pid"
    exit 1
  fi

  # Stop memory sampler.
  rm -f "/tmp/seed${SEED}_${net}_train.pid"
  kill "$mem_pid" 2>/dev/null || true
  wait "$mem_pid" 2>/dev/null || true

  # Finalize: copy ckpt → checkpoints/net_seed44_<net>.pt and write metrics JSON.
  python3 scripts/finalize_seed_training.py \
    --seed "$SEED" --net "$net" \
    --tmp-dir "$out_tmp" \
    --train-log "$log_train" \
    --mem-log "$log_mem" \
    --config-path "$cfg" \
    >> "$log_train" 2>&1

  echo "--- $(date +%T) [$net] complete ---"
  ls -la "checkpoints/net_seed${SEED}_${net}.pt" "results/training_seed${SEED}_${net}.json"
done

echo ""
echo "=== $(date +%T) All ${#NETS[@]} nets training complete ==="
echo ""
echo "Final ckpt + metrics inventory:"
ls -la "checkpoints/net_seed${SEED}_"*.pt "results/training_seed${SEED}_"*.json
