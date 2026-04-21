#!/usr/bin/env python3
"""Task #24 — Legacy three-regimes sweep eval wrapper.

Faithful port of the original `run_sweep.sh` analysis steps from the
`three-regimes` branch (commit 0f4cb78). Runs the two analysis scripts
that produced RESULTS.md §1's per-config metrics, then extracts the same
JSON summary fields via the same regex as the original `run_sweep.sh`.

Calls (faithful to original):
  1. scripts/legacy_sweep_eval_repr.py   # M1, M7, M10 metrics
  2. scripts/legacy_sweep_eval_evu.py    # FB-contribution
  3. Regex-extract metrics into summary.json (matches run_sweep.sh)

Differences from original (NECESSARY ports):
  - LaminarV1V2Network constructor: removed `delta_som` kwarg (current
    ctor takes only `cfg`).
  - load_state_dict(...): added strict=False (legacy ckpts have 9 dead
    feedback-operator weights from the pre-refactor EmergentFeedbackOperator
    that are unused under simple_feedback=True; verified safe in Phase A/B
    of Task #23).

No other behavior changes. Same readout windows, same trial counts, same
RNG seed (42).

Usage:
  python3 scripts/legacy_sweep_eval.py \
      --checkpoint /tmp/remote_ckpts/a1/checkpoint.pt \
      --config config/sweep/sweep_a1.yaml \
      --label a1 \
      --output /tmp/legacy_eval/a1
"""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys


_REPO = '/mnt/c/Users/User/codingproj/freshstart'
PYTHON = sys.executable


def run_step(label: str, cmd: list[str], log_path: str) -> int:
    """Run a step and tee its output to a log file. Returns exit code."""
    print(f'\n[{label}] running: {" ".join(cmd)}', flush=True)
    with open(log_path, 'w') as f:
        f.write(f'# Command: {" ".join(cmd)}\n')
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True,
                                 cwd=_REPO)
        for line in proc.stdout:
            print(line, end='')
            f.write(line)
        proc.wait()
    print(f'[{label}] exit={proc.returncode}', flush=True)
    return proc.returncode


def extract_summary(rep_log_text: str, evu_log_text: str,
                    cfg_path: str, label: str, seed: int) -> dict:
    """Same regex extraction as original run_sweep.sh."""
    def extract(pattern: str, text: str, default=None):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else default

    summary = {
        'config': cfg_path,
        'label': label,
        'seed': seed,
        # M7 — fine-discrimination linear-decoder accuracy delta
        'M7_d3': extract(r'M7 acc δ=3° delta\s+\|\s+([\-\+\d\.]+)', rep_log_text),
        'M7_d5': extract(r'M7 acc δ=5° delta\s+\|\s+([\-\+\d\.]+)', rep_log_text),
        'M7_d10': extract(r'M7 acc δ=10° delta\s+\|\s+([\-\+\d\.]+)', rep_log_text),
        'M7_d15': extract(r'M7 acc δ=15° delta\s+\|\s+([\-\+\d\.]+)', rep_log_text),
        # M10 — global mean L2/3 amplitude (ON/OFF)
        'global_amp': extract(r'M10 global mean ratio.*\|\s+([\d\.]+)', rep_log_text),
        # M1 — population bump FWHM (ON/OFF/delta)
        'fwhm_on': extract(r'PopBump FWHM ON.*\|\s+([\d\.]+)', rep_log_text),
        'fwhm_off': extract(r'PopBump FWHM OFF.*\|\s+([\d\.]+)', rep_log_text),
        'fwhm_delta': extract(r'PopBump FWHM delta.*\|\s+([\-\+\d\.]+)', rep_log_text),
        'peak_on': extract(r'stim=90,ora=90:.*ON=([\d\.]+)', rep_log_text),
        'peak_off': extract(r'stim=90,ora=90:.*OFF=([\d\.]+)', rep_log_text),
        # FB contribution — from debug_evu
        'fb_on_gap': extract(r'FB-ON gap.*:\s+([\-\+\d\.]+)', evu_log_text),
        'fb_off_gap': extract(r'FB-OFF gap.*:\s+([\-\+\d\.]+)', evu_log_text),
        'fb_contribution': extract(r'Feedback contribution:\s+([\-\+\d\.]+)', evu_log_text),
        'fb_direction': ('WIDENS' if 'WIDENS' in evu_log_text
                         else ('NARROWS' if 'NARROWS' in evu_log_text
                               else 'UNKNOWN')),
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True, help='Path to checkpoint.pt')
    ap.add_argument('--config', required=True, help='Path to sweep YAML config')
    ap.add_argument('--label', required=True, help='Label for this config (e.g. a1)')
    ap.add_argument('--output', required=True, help='Output directory for logs+summary.json')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--skip-repr', action='store_true', help='Skip M1/M7/M10 step')
    ap.add_argument('--skip-evu', action='store_true', help='Skip FB-contrib step')
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    rep_log = os.path.join(args.output, 'analysis_representation.log')
    evu_log = os.path.join(args.output, 'analysis_expected_vs_unexpected.log')
    sweep_log = os.path.join(args.output, 'sweep.log')
    summary_path = os.path.join(args.output, 'summary.json')

    rep_text = ''
    evu_text = ''

    if not args.skip_repr:
        rc = run_step('repr',
                      [PYTHON, 'scripts/legacy_sweep_eval_repr.py',
                       '--config', args.config,
                       '--checkpoint', args.checkpoint,
                       '--label', args.label,
                       '--device', args.device],
                      rep_log)
        if rc != 0:
            print(f'[repr] FAILED with exit code {rc}', flush=True)
        with open(rep_log) as f:
            rep_text = f.read()

    if not args.skip_evu:
        rc = run_step('evu',
                      [PYTHON, 'scripts/legacy_sweep_eval_evu.py',
                       '--config', args.config,
                       '--checkpoint', args.checkpoint,
                       '--label', args.label,
                       '--rng-seed', str(args.seed)],
                      evu_log)
        if rc != 0:
            print(f'[evu] FAILED with exit code {rc}', flush=True)
        with open(evu_log) as f:
            evu_text = f.read()

    summary = extract_summary(rep_text, evu_text, args.config, args.label, args.seed)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\n[summary] wrote {summary_path}')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
