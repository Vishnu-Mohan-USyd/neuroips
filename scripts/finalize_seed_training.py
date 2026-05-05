"""Task #21 — Post-training finalizer for seed-robustness train runs.

Mirrors the format of `results/training_seed43_<net>.json` (Task #4) exactly.

Given a finished `scripts/train.py` run that wrote
    /tmp/seed{SEED}_{net}_tmp/emergent_seed{SEED}/checkpoint.pt
    /tmp/seed{SEED}_{net}_tmp/emergent_seed{SEED}/training_metrics.jsonl   (if produced)
and the wrapper-supplied `logs/seed{SEED}_{net}_train.log` and `_mem.log`,
this script:

1. Reads the ckpt to extract `history.loss` (5000 stage-2 entries).
2. Parses train.log timestamps to compute Stage 1 / Stage 2 / total wall-clock.
3. Parses mem.log to compute peak GPU + RAM (MiB).
4. Copies the ckpt to `checkpoints/net_seed{SEED}_{net}.pt`.
5. If a per-100-step metrics jsonl was emitted, copies it to
   `results/training_seed{SEED}_{net}.metrics.jsonl`.
6. Writes `results/training_seed{SEED}_{net}.json` with the schema:
       task, net_name, ckpt_path, config_path, seed,
       stage1_n_steps, stage2_n_steps,
       stage2_loss_history, stage2_final_loss,
       memory_peak: {gpu_mib, ram_mib},
       wall_clock_training_seconds: {stage1, stage2, total}

Schema is byte-for-byte compatible with Task #4's training_seed43_*.json
so downstream consumers (paradigm matrix, aggregator) treat seed 43 / 44
identically.

Usage:
    python3 scripts/finalize_seed_training.py \
        --seed 44 --net r1r2 \
        --tmp-dir /tmp/seed44_r1r2_tmp \
        --train-log logs/seed44_r1r2_train.log \
        --mem-log   logs/seed44_r1r2_mem.log \
        --config-path config/sweep/sweep_rescue_1_2.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch


_REPO = Path(__file__).resolve().parent.parent

# Match training log lines like:
#   "2026-04-25 00:03:35,828 [INFO] STAGE 1: Sensory Scaffold"
#   "2026-04-25 00:04:45,168 [INFO] Stage 1 complete: loss=1.7112, acc=0.910"
#   "2026-04-25 01:00:49,594 [INFO] Stage 2 complete: loss=3.7578, ..."
_TS_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})[,.]\d{3} \[INFO\] (?P<msg>.+)$"
)


def _parse_log_timestamps(log_path: Path) -> dict:
    """Walk train.log, return ISO timestamps for the boundary events.

    Returns a dict with keys: stage1_start, stage1_end, stage2_end (datetime objs).
    Missing keys are None.
    """
    out = {"stage1_start": None, "stage1_end": None, "stage2_end": None}
    if not log_path.exists():
        return out

    with log_path.open() as f:
        for line in f:
            m = _TS_RE.match(line.rstrip("\n"))
            if not m:
                continue
            ts = datetime.strptime(m["ts"], "%Y-%m-%d %H:%M:%S")
            msg = m["msg"]
            if msg.startswith("STAGE 1") and out["stage1_start"] is None:
                # First "STAGE 1: Sensory Scaffold" line.
                out["stage1_start"] = ts
            elif msg.startswith("Stage 1 complete:"):
                out["stage1_end"] = ts
            elif msg.startswith("Stage 2 complete:") or msg.startswith("Saved to "):
                out["stage2_end"] = ts
    return out


def _parse_mem_log(mem_path: Path) -> dict:
    """Parse `logs/seed{SEED}_{net}_mem.log` lines like:
        '00:03:34 GPU=2486MiB RAM=4507MiB'
    Return peak GPU and RAM (MiB).
    """
    peak = {"gpu_mib": 0, "ram_mib": 0}
    if not mem_path.exists():
        return peak
    pat = re.compile(r"GPU=(\d+)MiB RAM=(\d+)MiB")
    with mem_path.open() as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            g, r = int(m.group(1)), int(m.group(2))
            if g > peak["gpu_mib"]:
                peak["gpu_mib"] = g
            if r > peak["ram_mib"]:
                peak["ram_mib"] = r
    return peak


def _stage_steps_from_config(cfg_path: Path) -> tuple[int, int]:
    """Light-touch YAML parse to extract stage1.n_steps and stage2.n_steps.

    Avoids importing the full src.config machinery. The sweep YAML schema
    has predictable indentation; if either field is missing we fall back to
    Task #4 defaults (2000, 5000) so the JSON still validates.
    """
    s1, s2 = 2000, 5000
    if not cfg_path.exists():
        return s1, s2
    text = cfg_path.read_text()
    # Match "stage1:" block then first "n_steps:" within ~6 lines.
    blocks = re.split(r"^\s*stage([12]):\s*$", text, flags=re.MULTILINE)
    # blocks alternates [pre, "1", body1, "2", body2, post]
    for i in range(1, len(blocks), 2):
        which = blocks[i]
        body = blocks[i + 1]
        m = re.search(r"^\s*n_steps:\s*(\d+)", body, flags=re.MULTILINE)
        if m:
            n = int(m.group(1))
            if which == "1":
                s1 = n
            elif which == "2":
                s2 = n
    return s1, s2


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--net", required=True,
                   choices=["r1r2", "a1", "b1", "c1", "e1"])
    p.add_argument("--tmp-dir", required=True,
                   help="train.py --output dir, e.g. /tmp/seed44_r1r2_tmp")
    p.add_argument("--train-log", required=True,
                   help="full Python stdout/stderr of the training run")
    p.add_argument("--mem-log", required=True,
                   help="background RAM/GPU sampler log")
    p.add_argument("--config-path", required=True,
                   help="sweep YAML used (recorded in JSON for traceability)")
    args = p.parse_args()

    seed = args.seed
    net = args.net
    tmp_dir = Path(args.tmp_dir)
    src_ckpt = tmp_dir / f"emergent_seed{seed}" / "checkpoint.pt"
    if not src_ckpt.exists():
        print(f"[finalize] FATAL: source ckpt missing: {src_ckpt}", file=sys.stderr)
        sys.exit(1)

    # Load history from the ckpt — the same source Task #4 used.
    ckpt = torch.load(src_ckpt, map_location="cpu", weights_only=False)
    loss_history: list[float] = list(ckpt.get("history", {}).get("loss", []))
    stage2_final_loss: Optional[float] = (
        float(loss_history[-1]) if loss_history else None
    )

    # Wall clock from log timestamps.
    ts = _parse_log_timestamps(Path(args.train_log))
    wc = {"stage1": None, "stage2": None, "total": None}
    if ts["stage1_start"] and ts["stage1_end"]:
        wc["stage1"] = int((ts["stage1_end"] - ts["stage1_start"]).total_seconds())
    if ts["stage1_end"] and ts["stage2_end"]:
        wc["stage2"] = int((ts["stage2_end"] - ts["stage1_end"]).total_seconds())
    if ts["stage1_start"] and ts["stage2_end"]:
        wc["total"] = int((ts["stage2_end"] - ts["stage1_start"]).total_seconds())

    # Memory peak from sampler log.
    peak = _parse_mem_log(Path(args.mem_log))

    # Stage step counts from config (recorded for traceability).
    s1_steps, s2_steps = _stage_steps_from_config(Path(args.config_path))

    # Copy ckpt to canonical path.
    dst_ckpt = _REPO / "checkpoints" / f"net_seed{seed}_{net}.pt"
    dst_ckpt.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_ckpt, dst_ckpt)
    print(f"[finalize] ckpt copied: {src_ckpt} -> {dst_ckpt}")

    # Copy per-100-step metrics jsonl (if train.py emitted it).
    # stage2_feedback.run_stage2 writes `metrics.jsonl` (not training_metrics.jsonl)
    # into its output_dir; that matches the file Task #4's training_seed43_*.metrics.jsonl
    # came from.
    src_metrics = tmp_dir / f"emergent_seed{seed}" / "metrics.jsonl"
    if src_metrics.exists():
        dst_metrics = _REPO / "results" / f"training_seed{seed}_{net}.metrics.jsonl"
        shutil.copy2(src_metrics, dst_metrics)
        print(f"[finalize] metrics jsonl copied: {src_metrics} -> {dst_metrics}")
    else:
        print(f"[finalize] no per-step metrics jsonl found at {src_metrics}; skipping")

    # Write the training_seed{seed}_{net}.json — schema-compatible with Task #4.
    out_json = {
        "task": f"Task #21 — seed-robustness train (seed {seed})",
        "net_name": net,
        "ckpt_path": str(dst_ckpt.relative_to(_REPO)),
        "config_path": args.config_path,
        "seed": seed,
        "stage1_n_steps": s1_steps,
        "stage2_n_steps": s2_steps,
        "stage2_loss_history": loss_history,
        "stage2_final_loss": stage2_final_loss,
        "memory_peak": {
            "gpu_mib": peak["gpu_mib"],
            "ram_mib": peak["ram_mib"],
        },
        "wall_clock_training_seconds": {
            "stage1": wc["stage1"],
            "stage2": wc["stage2"],
            "total": wc["total"],
        },
    }
    out_path = _REPO / "results" / f"training_seed{seed}_{net}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out_json, f, indent=2)
    print(f"[finalize] wrote {out_path}")
    print(f"[finalize] wall_clock total={wc['total']}s  stage1={wc['stage1']}s  "
          f"stage2={wc['stage2']}s  peak_gpu={peak['gpu_mib']}MiB  "
          f"peak_ram={peak['ram_mib']}MiB  stage2_final_loss={stage2_final_loss}")


if __name__ == "__main__":
    main()
