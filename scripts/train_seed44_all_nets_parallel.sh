#!/bin/bash
# Task #21 — Seed-robustness Part A (REMOTE PARALLEL): train all 5 nets at seed 44
# concurrently on the remote A6000 (48 GB).
#
# Variant of scripts/train_seed44_all_nets.sh: same per-net protocol,
# but launches all 5 nets simultaneously via background python+setsid
# inside tmux, instead of sequentially.
#
# Per-process resource ceiling per dispatch: ≤10 GB GPU, ≤10 GB RAM.
# With 5 nets × ~3 GB GPU = 15 GB, ~5 GB RAM each = 25 GB RAM, the
# remote A6000 (48 GB GPU) and 157 GB system RAM are well within bounds.
#
# Usage (intended to be run inside a detached tmux session on remote):
#   tmux new-session -d -s seed44_train_remote 'exec bash scripts/train_seed44_all_nets_parallel.sh'
#
# Outputs: same as the local serial launcher (checkpoints/net_seed44_<net>.pt
# + results/training_seed44_<net>.{json,metrics.jsonl}).

set -uo pipefail   # NOT -e: one net's failure must not kill the others.

cd "$(dirname "$0")/.."   # cwd = repo root

SEED=44

declare -A NET_CFG
NET_CFG[r1r2]="config/sweep/sweep_rescue_1_2.yaml"
NET_CFG[a1]="config/sweep/sweep_a1.yaml"
NET_CFG[b1]="config/sweep/sweep_b1.yaml"
NET_CFG[c1]="config/sweep/sweep_c1.yaml"
NET_CFG[e1]="config/sweep/sweep_e1.yaml"

# Pick the first available python with torch+CUDA. Caller can override.
PYTHON="${PYTHON:-/home/vishnu/miniconda3/bin/python}"
if ! "$PYTHON" -c "import torch" 2>/dev/null; then
  if command -v python3 >/dev/null 2>&1 && python3 -c "import torch" 2>/dev/null; then
    PYTHON=python3
  else
    echo "[FATAL] no python with torch found (tried $PYTHON and python3)" >&2
    exit 2
  fi
fi

mkdir -p logs checkpoints results

NETS=(r1r2 a1 b1 c1 e1)

echo "=== $(date +%T) Seed ${SEED} parallel training launcher: ${#NETS[@]} nets concurrent ==="
echo "Python: $PYTHON"
"$PYTHON" -c "import torch; print(f'CUDA: {torch.cuda.is_available()}; device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"
echo ""

declare -A TRAIN_PIDS

# --- Launch phase: spawn all 5 nets concurrently ---
for net in "${NETS[@]}"; do
  cfg="${NET_CFG[$net]}"
  out_tmp="/tmp/seed${SEED}_${net}_tmp"
  rm -rf "$out_tmp"
  mkdir -p "$out_tmp"

  log_train="logs/seed${SEED}_${net}_train.log"
  log_mem="logs/seed${SEED}_${net}_mem.log"
  : > "$log_train"
  : > "$log_mem"

  echo "--- $(date +%T) [$net] launching (cfg=$cfg seed=${SEED}) ---"

  # setsid + fd0/1/2 redirected to /dev/null/log files makes the python
  # immune to the parent shell's SIGHUP/SIGPIPE. (The tmux session itself
  # also survives SSH disconnects, but defence-in-depth.)
  setsid "$PYTHON" -m scripts.train \
      --config "$cfg" --seed "$SEED" --output "$out_tmp" \
      > "$log_train" 2>&1 < /dev/null &
  TRAIN_PIDS[$net]=$!
  echo "  $net: PID ${TRAIN_PIDS[$net]}  log=$log_train"
done

# --- Memory samplers (one per net, lightweight) ---
for net in "${NETS[@]}"; do
  pid="${TRAIN_PIDS[$net]}"
  log_mem="logs/seed${SEED}_${net}_mem.log"
  (
    while kill -0 "$pid" 2>/dev/null; do
      ts=$(date +%T)
      gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
      ram=$(awk '/VmRSS/ {print int($2/1024)}' /proc/"$pid"/status 2>/dev/null || echo 0)
      echo "$ts GPU=${gpu}MiB RAM=${ram}MiB" >> "$log_mem"
      sleep 30
    done
  ) </dev/null >/dev/null 2>&1 &
done

echo ""
echo "=== $(date +%T) All 5 nets launched — waiting for completion ==="
echo ""

# --- Wait phase: wait for each train pid ---
ANY_FAILED=0
for net in "${NETS[@]}"; do
  pid="${TRAIN_PIDS[$net]}"
  if wait "$pid"; then
    echo "[$(date +%T)] [$net] complete (PID $pid exit 0)"
  else
    rc=$?
    echo "[$(date +%T)] [$net] FAILED (PID $pid exit $rc)" >&2
    ANY_FAILED=1
  fi
done

# Give samplers a tick to flush
sleep 2

# --- Finalize phase: per-net ckpt + JSON ---
echo ""
echo "=== $(date +%T) Finalizing — moving ckpts + writing JSONs ==="
for net in "${NETS[@]}"; do
  out_tmp="/tmp/seed${SEED}_${net}_tmp"
  log_train="logs/seed${SEED}_${net}_train.log"
  log_mem="logs/seed${SEED}_${net}_mem.log"
  cfg="${NET_CFG[$net]}"
  echo "--- finalizing $net ---"
  if "$PYTHON" scripts/finalize_seed_training.py \
        --seed "$SEED" --net "$net" \
        --tmp-dir "$out_tmp" \
        --train-log "$log_train" \
        --mem-log "$log_mem" \
        --config-path "$cfg"; then
    :
  else
    echo "[WARN] finalize failed for $net" >&2
    ANY_FAILED=1
  fi
done

echo ""
echo "=== $(date +%T) All ${#NETS[@]} nets training complete ==="
ls -la "checkpoints/net_seed${SEED}_"*.pt 2>/dev/null
ls -la "results/training_seed${SEED}_"*.json 2>/dev/null

if [[ "$ANY_FAILED" -eq 1 ]]; then
  echo "[WARN] one or more nets had non-zero exit; see logs"
  exit 1
fi
exit 0
