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

# ------------------------------------------------------------------
# SIGHUP hardening (Task #9 — Debugger H7 root cause fix)
# ------------------------------------------------------------------
# Problem: the canonical launch chain is
#     tmux new-session -d NAME 'bash run_sweep.sh cfg out ... 2>&1 | tee log'
# When the tmux pty later dies (WSL terminated, SSH hang-up,
# `tmux kill-server`, etc.), the kernel sends SIGHUP to every
# process in the pty's session. TWO things then kill our training:
#   a) the pipeline's bash and the external `tee` both receive
#      SIGHUP and die (default action = terminate), and
#   b) the child python3 process then tries to write to its now
#      closed stdout pipe, receives SIGPIPE, and dies silently
#      with no final log entry.
# This is the confirmed root cause (Debugger H7) of the original
# Phase 2.4 silent death.
#
# Fix: on first invocation, self-re-exec under `setsid` with TWO
# things happening atomically via the exec line:
#   (1) `setsid` moves the new process into a brand-new session
#       with no controlling terminal, so SIGHUP from the old pty
#       session cannot reach us;
#   (2) stdin is detached (< /dev/null), and stdout/stderr are
#       redirected directly to $OUTPUT_DIR/sweep.log on disk,
#       BYPASSING the caller's external-tee pipeline. Writes to
#       a regular file cannot be killed by SIGPIPE because there
#       is no pipe anywhere in the chain.
# `--wait` keeps the caller's pipeline member (the original cmd1
# bash, now replaced by setsid) alive and blocked on the child,
# so the final exit status is still propagated to the caller and
# tmux `new-session -d` keeps its pane attached to a live process
# on the happy path.
#
# We derive OUTPUT_DIR inline from `${2:-}` to keep the hardening
# block self-contained without reordering the full arg parse
# below. If args are malformed (e.g. $2 is empty) we fall through
# and the normal Usage error below fires with its message visible
# on the caller's terminal.
#
# Sentinel env var `RUN_SWEEP_DAEMONIZED=1` prevents re-exec
# recursion. Opt-out: `RUN_SWEEP_NO_DAEMONIZE=1` skips the
# hardening entirely (useful for interactive debugging where
# Ctrl-C should kill the whole job immediately).
#
# Verified end-to-end by a standalone harness: tmux kill-session
# destroys the pty mid-run, the detached child survives and runs
# to natural completion with all ticks logged to sweep.log.
# ------------------------------------------------------------------
if [ -z "${RUN_SWEEP_DAEMONIZED:-}" ] && [ -z "${RUN_SWEEP_NO_DAEMONIZE:-}" ]; then
    _RSH_OUTDIR="${2:-}"
    if [ -n "$_RSH_OUTDIR" ] && command -v setsid >/dev/null 2>&1; then
        export RUN_SWEEP_DAEMONIZED=1
        mkdir -p "$_RSH_OUTDIR"
        _RSH_LOGFILE="$_RSH_OUTDIR/sweep.log"
        # Truncate logfile so the run starts fresh, then append
        # below. Prevents stale content from a prior run leaking in.
        : > "$_RSH_LOGFILE"
        # exec replaces the current shell. setsid --wait creates a
        # new session, forks, child runs the re-execed script, parent
        # waits for exit status. FDs in the child:
        #   fd0 = /dev/null       (no interactive stdin)
        #   fd1 = /dev/null       (main-body stdout is discarded;
        #                          the internal `| tee -a "$LOGFILE"`
        #                          pipelines sprinkled throughout the
        #                          script body are the sole source of
        #                          LOGFILE writes, preventing any
        #                          doubled output)
        #   fd2 = $LOGFILE (append) (captures conda preflight stderr
        #                          and any set -x / bash error messages
        #                          that would otherwise go to /dev/null)
        # The internal tee subshells inherit fd1=/dev/null, so their
        # "stdout forward" is harmless; they still write to LOGFILE
        # via their explicit file argument. No external pipes remain
        # anywhere in the process tree → SIGPIPE is structurally
        # impossible.
        exec setsid --wait bash "$0" "$@" < /dev/null > /dev/null 2>>"$_RSH_LOGFILE"
    elif ! command -v setsid >/dev/null 2>&1; then
        # Extremely rare fallback (setsid ships with util-linux on
        # every mainstream Linux distro). Install a SIGHUP-ignoring
        # trap — less robust but better than nothing.
        echo "WARN: setsid not found; installing SIGHUP-ignore trap as fallback" >&2
        trap '' HUP
    fi
    # If _RSH_OUTDIR is empty, we fall through without detaching.
    # The Usage error from the arg-parse block below will fire and
    # the caller will see it directly on their terminal.
fi

# ------------------------------------------------------------------
# Python env preflight
# ------------------------------------------------------------------
# The tmux / remote launcher may be a bare shell with no env activated.
# We must guarantee `python3` can import torch BEFORE training starts,
# otherwise the job crashes in <1s with ModuleNotFoundError.
#
# Strategy (first match wins):
#   1. If CONDA_ENV is set, activate it (explicit override)
#   2. Else try conda env "sc_model" (project default, has torch 2.10)
#   3. If conda is unavailable or activation fails, fall back to
#      system /usr/bin/python3 (has torch via user site-packages)
#   4. Verify `python3 -c "import torch"` succeeds; exit 1 if not
# ------------------------------------------------------------------
CONDA_ENV="${CONDA_ENV:-sc_model}"
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"

if [ -f "$CONDA_SH" ]; then
    # shellcheck disable=SC1090
    source "$CONDA_SH"
    if ! conda activate "$CONDA_ENV" 2>/dev/null; then
        echo "WARN: conda env '$CONDA_ENV' not available, trying base..." >&2
        conda activate base 2>/dev/null || true
    fi
fi

# If the currently-selected python3 lacks torch, fall back to /usr/bin/python3
# (which has torch installed via pip --user at ~/.local/lib/python3.10/...)
if ! python3 -c "import torch" >/dev/null 2>&1; then
    if /usr/bin/python3 -c "import torch" >/dev/null 2>&1; then
        echo "INFO: preferred python3 lacks torch; falling back to /usr/bin/python3" >&2
        export PATH="/usr/bin:$PATH"
        hash -r
    fi
fi

# Final hard check — if torch still can't be imported, fail loudly instead of
# crashing inside scripts/train.py 5 lines later.
if ! python3 -c "import torch" >/dev/null 2>&1; then
    echo "ERROR: torch is not importable in any candidate python3 interpreter." >&2
    echo "  which python3: $(which python3 || echo none)" >&2
    echo "  CONDA_DEFAULT_ENV: ${CONDA_DEFAULT_ENV:-<none>}" >&2
    echo "  Set CONDA_ENV=<name> to force a specific conda env, or install torch." >&2
    exit 1
fi

# Report the resolved env so logs capture it
python3 -c "import torch, sys; print(f'[env] python3={sys.executable} torch={torch.__version__} cuda={torch.cuda.is_available()}')" >&2

CONFIG="${1:?Usage: $0 <config> <output_dir> [seed] [device]}"
OUTPUT_DIR="${2:?Usage: $0 <config> <output_dir> [seed] [device]}"
SEED="${3:-42}"
DEVICE="${4:-cuda}"

LABEL="$(basename "$CONFIG" .yaml)"
CHECKPOINT_DIR="${OUTPUT_DIR}/emergent_seed${SEED}"
CHECKPOINT="${CHECKPOINT_DIR}/checkpoint.pt"
LOGFILE="${OUTPUT_DIR}/sweep.log"
SUMMARY="${OUTPUT_DIR}/summary.json"

mkdir -p "$OUTPUT_DIR"

echo "================================================================" | tee -a "$LOGFILE"
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
