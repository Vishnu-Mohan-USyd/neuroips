#!/bin/bash
# Task #21 — Seed-robustness Part B: paradigm matrix per seed.
#
# Runs the full 17-row paradigm matrix on the 5 (net × seed) ckpts at
# a given seed, using each ckpt's own joint-trained Dec A and the shared
# decoder_c.pt.
#
# Row layout (matches results/cross_decoder_comprehensive_20k_final.json):
#   1-4   R1+R2 HMM C1-C4 (paradigm_readout, all 4 conditions)
#   5-8   a1/b1/c1/e1 HMM C1 only (paradigm_readout --conditions C1_focused_native)
#   9-14  R1+R2 native cross_decoder_eval (NEW, M3R, HMS, HMS-T, P3P, VCD-test3)
#   15-17 R1+R2 modified cross_decoder_eval (M3R, HMS-T, VCD with --override-task-cue)
#
# Per-seed wall-clock estimate: ~8-10 min.
#
# Outputs (per seed, written to /tmp/paradigm_matrix_seed<SEED>/):
#   r1r2_paradigm.json     — rows 1-4
#   xdec_native.json       — rows 9-14
#   xdec_modified.json     — rows 15-17
#   legacy/<net>_C1.json   — rows 5-8 (one per legacy net)
#
# Usage:
#   bash scripts/run_seed_paradigm_matrix.sh <SEED>
#
# <SEED> determines per-net ckpt resolution:
#   seed=42  → R1+R2: results/simple_dual/emergent_seed42/checkpoint.pt
#              legacy: /tmp/remote_ckpts/<net>/checkpoint.pt
#   seed=43,44,...  → checkpoints/net_seed<SEED>_<net>.pt for all 5 nets

set -euo pipefail

cd /mnt/c/Users/User/codingproj/freshstart

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <SEED>" >&2
  exit 1
fi
SEED=$1

OUTDIR=/tmp/paradigm_matrix_seed${SEED}
mkdir -p "$OUTDIR/legacy"

LOGD=logs/paradigm_matrix_seed${SEED}
mkdir -p "$LOGD"

declare -A NET_CKPT
declare -A NET_CFG

if [[ "$SEED" == "42" ]]; then
  # Seed 42 baseline ckpts (heterogeneous origin: R1+R2 from simple_dual,
  # legacy from /tmp/remote_ckpts).
  NET_CKPT[r1r2]="results/simple_dual/emergent_seed42/checkpoint.pt"
  NET_CKPT[a1]="/tmp/remote_ckpts/a1/checkpoint.pt"
  NET_CKPT[b1]="/tmp/remote_ckpts/b1/checkpoint.pt"
  NET_CKPT[c1]="/tmp/remote_ckpts/c1/checkpoint.pt"
  NET_CKPT[e1]="/tmp/remote_ckpts/e1/checkpoint.pt"
else
  # Seed 43, 44, ...: fresh ckpts trained at this seed (Task #4 / Task #21 protocol).
  NET_CKPT[r1r2]="checkpoints/net_seed${SEED}_r1r2.pt"
  NET_CKPT[a1]="checkpoints/net_seed${SEED}_a1.pt"
  NET_CKPT[b1]="checkpoints/net_seed${SEED}_b1.pt"
  NET_CKPT[c1]="checkpoints/net_seed${SEED}_c1.pt"
  NET_CKPT[e1]="checkpoints/net_seed${SEED}_e1.pt"
fi

NET_CFG[r1r2]="config/sweep/sweep_rescue_1_2.yaml"
NET_CFG[a1]="config/sweep/sweep_a1.yaml"
NET_CFG[b1]="config/sweep/sweep_b1.yaml"
NET_CFG[c1]="config/sweep/sweep_c1.yaml"
NET_CFG[e1]="config/sweep/sweep_e1.yaml"

# Verify all ckpts exist before launching.
echo "=== $(date +%T) Seed ${SEED} paradigm matrix runner ==="
for net in r1r2 a1 b1 c1 e1; do
  if [[ ! -f "${NET_CKPT[$net]}" ]]; then
    echo "[ERROR] missing ckpt for $net: ${NET_CKPT[$net]}" >&2
    exit 1
  fi
  echo "  $net: ${NET_CKPT[$net]}"
done

# --- Step 1/3: R1+R2 paradigm_readout (rows 1-4) ---
echo ""
echo "=== $(date +%T) Step 1/3: R1+R2 paradigm_readout (rows 1-4) ==="
python3 scripts/r1r2_paradigm_readout.py \
  --checkpoint "${NET_CKPT[r1r2]}" \
  --config "${NET_CFG[r1r2]}" \
  --output-json "$OUTDIR/r1r2_paradigm.json" \
  --output-fig "/tmp/paradigm_matrix_seed${SEED}_fig_r1r2.png" \
  2>&1 | tee "$LOGD/r1r2_paradigm.log" | tail -20

# --- Step 2a/3: R1+R2 cross_decoder_eval native (rows 9-14) ---
echo ""
echo "=== $(date +%T) Step 2a/3: R1+R2 cross_decoder_eval native (rows 9-14) ==="
python3 scripts/cross_decoder_eval.py \
  --checkpoint "${NET_CKPT[r1r2]}" \
  --config "${NET_CFG[r1r2]}" \
  --output-json "$OUTDIR/xdec_native.json" \
  2>&1 | tee "$LOGD/xdec_native.log" | tail -20

# --- Step 2b/3: R1+R2 cross_decoder_eval modified (rows 15-17) ---
echo ""
echo "=== $(date +%T) Step 2b/3: R1+R2 cross_decoder_eval modified (rows 15-17) ==="
python3 scripts/cross_decoder_eval.py \
  --checkpoint "${NET_CKPT[r1r2]}" \
  --config "${NET_CFG[r1r2]}" \
  --override-task-cue \
  --strategies M3R HMS-T VCD \
  --output-json "$OUTDIR/xdec_modified.json" \
  2>&1 | tee "$LOGD/xdec_modified.log" | tail -20

# --- Step 3/3: 4 legacy paradigm_readout C1 only (rows 5-8) ---
echo ""
echo "=== $(date +%T) Step 3/3: 4 legacy paradigm_readout C1 (rows 5-8) ==="
for net in a1 b1 c1 e1; do
  echo "--- $net ---"
  python3 scripts/r1r2_paradigm_readout.py \
    --checkpoint "${NET_CKPT[$net]}" \
    --config "${NET_CFG[$net]}" \
    --conditions C1_focused_native \
    --output-json "$OUTDIR/legacy/${net}_C1.json" \
    --output-fig "/tmp/paradigm_matrix_seed${SEED}_fig_${net}.png" \
    2>&1 | tee "$LOGD/${net}_C1.log" | tail -10
done

echo ""
echo "=== $(date +%T) Seed ${SEED} paradigm matrix complete ==="
echo "Outputs:"
ls -la "$OUTDIR/"*.json "$OUTDIR/legacy/"*.json
