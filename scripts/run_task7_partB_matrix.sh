#!/bin/bash
# Task #7 Part B — re-run 17-row matrix with 20k-step Dec A' across all 5 nets.
#
# Patches each net's ckpt with that net's 20k Dec A' (in loss_heads + decoder_state),
# re-runs the 3 R1+R2 pipelines + 4 legacy paradigm_readout runs, then aggregates.
# Mirrors the 2026-04-24 Dec E rerun pattern (per-ckpt patch trick).
#
# Outputs:
#   /tmp/task7_decAprime20k/r1r2_paradigm.json     — rows 1-4 (R1+R2 paradigm_readout)
#   /tmp/task7_decAprime20k/xdec_native.json       — rows 9-14 (R1+R2 cross_decoder_eval native)
#   /tmp/task7_decAprime20k/xdec_modified.json     — rows 15-17 (R1+R2 cross_decoder_eval modified)
#   /tmp/task7_decAprime20k/legacy/{a1,b1,c1,e1}_C1.json  — rows 5-8 (per-legacy paradigm_readout C1)
#   /tmp/task7_decAprime20k/patched/{net}_ckpt.pt  — 5 patched ckpts (1 per net)
#
# Final merger then produces:
#   results/cross_decoder_comprehensive_20k_final.{json,md}
#
# Hard rules per Task #7 brief: no doc edits; no training-param changes.

set -euo pipefail

cd /mnt/c/Users/User/codingproj/freshstart

OUTDIR=/tmp/task7_decAprime20k
mkdir -p "$OUTDIR/patched" "$OUTDIR/legacy"
LOGD=logs/task7_decA_prime_20k_legacy

# --- Per-net config table ---
declare -A NET_CKPT
declare -A NET_CFG
declare -A NET_DECA20K
NET_CKPT[r1r2]="results/simple_dual/emergent_seed42/checkpoint.pt"
NET_CFG[r1r2]="config/sweep/sweep_rescue_1_2.yaml"
NET_DECA20K[r1r2]="checkpoints/decoder_a_prime_20k_r1r2.pt"
for net in a1 b1 c1 e1; do
  NET_CKPT[$net]="/tmp/remote_ckpts/$net/checkpoint.pt"
  NET_CFG[$net]="config/sweep/sweep_$net.yaml"
  NET_DECA20K[$net]="checkpoints/decoder_a_prime_20k_$net.pt"
done

# --- 1. Patch each net's ckpt with its own 20k Dec A' ---
echo "=== $(date +%T) Step 1/4: patching 5 ckpts with 20k Dec A' ==="
for net in r1r2 a1 b1 c1 e1; do
  python3 scripts/_make_decAprime_ckpt.py \
    --src-ckpt "${NET_CKPT[$net]}" \
    --dec-a-prime "${NET_DECA20K[$net]}" \
    --out "$OUTDIR/patched/${net}_ckpt.pt" \
    2>&1 | tail -3
done

# --- 2. R1+R2 paradigm_readout (rows 1-4) ---
echo "=== $(date +%T) Step 2a/4: R1+R2 paradigm_readout (rows 1-4) ==="
python3 scripts/r1r2_paradigm_readout.py \
  --checkpoint "$OUTDIR/patched/r1r2_ckpt.pt" \
  --config "${NET_CFG[r1r2]}" \
  --output-json "$OUTDIR/r1r2_paradigm.json" \
  --output-fig /tmp/task7_paradigm_r1r2.png \
  2>&1 | tee "$LOGD/r1r2_paradigm_partB.log" | tail -15

# --- 2b. R1+R2 cross_decoder_eval native (rows 9-14) ---
echo "=== $(date +%T) Step 2b/4: R1+R2 cross_decoder_eval native (rows 9-14) ==="
python3 scripts/cross_decoder_eval.py \
  --checkpoint "$OUTDIR/patched/r1r2_ckpt.pt" \
  --config "${NET_CFG[r1r2]}" \
  --output-json "$OUTDIR/xdec_native.json" \
  2>&1 | tee "$LOGD/r1r2_xdec_native_partB.log" | tail -15

# --- 2c. R1+R2 cross_decoder_eval modified (rows 15-17) ---
echo "=== $(date +%T) Step 2c/4: R1+R2 cross_decoder_eval modified (rows 15-17) ==="
python3 scripts/cross_decoder_eval.py \
  --checkpoint "$OUTDIR/patched/r1r2_ckpt.pt" \
  --config "${NET_CFG[r1r2]}" \
  --override-task-cue \
  --strategies M3R HMS-T VCD \
  --output-json "$OUTDIR/xdec_modified.json" \
  2>&1 | tee "$LOGD/r1r2_xdec_modified_partB.log" | tail -15

# --- 3. Per-legacy paradigm_readout C1 only (rows 5-8) ---
echo "=== $(date +%T) Step 3/4: 4 legacy paradigm_readout C1 (rows 5-8) ==="
for net in a1 b1 c1 e1; do
  echo "--- $net ---"
  python3 scripts/r1r2_paradigm_readout.py \
    --checkpoint "$OUTDIR/patched/${net}_ckpt.pt" \
    --config "${NET_CFG[$net]}" \
    --conditions C1_focused_native \
    --output-json "$OUTDIR/legacy/${net}_C1.json" \
    --output-fig "/tmp/task7_paradigm_${net}.png" \
    2>&1 | tee "$LOGD/${net}_paradigm_C1_partB.log" | tail -10
done

# --- 4. Merge into final 17-row matrix ---
echo "=== $(date +%T) Step 4/4: aggregate + merge into 8-column matrix ==="
python3 scripts/merge_decAprime_20k_matrix.py \
  --base-matrix results/cross_decoder_comprehensive_with_all_decoders.json \
  --paradigm-r1r2 "$OUTDIR/r1r2_paradigm.json" \
  --xdec-native "$OUTDIR/xdec_native.json" \
  --xdec-modified "$OUTDIR/xdec_modified.json" \
  --legacy-dir "$OUTDIR/legacy" \
  --output-json results/cross_decoder_comprehensive_20k_final.json \
  --output-md results/cross_decoder_comprehensive_20k_final.md \
  2>&1 | tail -30

echo "=== $(date +%T) Part B complete ==="
echo "Outputs:"
ls -la results/cross_decoder_comprehensive_20k_final.json results/cross_decoder_comprehensive_20k_final.md
