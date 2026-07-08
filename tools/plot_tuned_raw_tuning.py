#!/usr/bin/env python3
"""Plot raw aligned L2/3 tuning curves for tuned emergence checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tuned_emergence_lib import N, STEP_DEG, device, forward_seq_tuned  # noqa: E402
from validate_tuned_emergence import K, aligned_stack, build_pairs, load_net  # noqa: E402


def profile(path: str) -> tuple[list[float], list[float]]:
    net = load_net(path)
    theta_e, _, e_idx, _ = build_pairs()
    _, r_all = forward_seq_tuned(net, theta_e, 1.0)
    aligned = aligned_stack(r_all[:, K, :], e_idx)
    mean = aligned.mean(0)
    sem = aligned.std(0, unbiased=False) / math.sqrt(aligned.shape[0])
    return mean.detach().float().cpu().tolist(), sem.detach().float().cpu().tolist()


def centered_offsets() -> list[int]:
    half = N // 2
    return [i if i <= half else i - N for i in range(N)]


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot tuned raw aligned L2/3 tuning curves.")
    ap.add_argument("--sharpen", required=True)
    ap.add_argument("--dampen", required=True)
    ap.add_argument("--out-png", required=True)
    ap.add_argument("--out-csv")
    ap.add_argument("--out-json")
    args = ap.parse_args()

    import matplotlib.pyplot as plt

    sharp_mean, sharp_sem = profile(args.sharpen)
    damp_mean, damp_sem = profile(args.dampen)
    offsets = centered_offsets()
    order = sorted(range(N), key=lambda i: offsets[i])
    xs = [offsets[i] * STEP_DEG for i in order if abs(offsets[i] * STEP_DEG) <= 60]
    s_mean = [sharp_mean[i] for i in order if abs(offsets[i] * STEP_DEG) <= 60]
    s_sem = [sharp_sem[i] for i in order if abs(offsets[i] * STEP_DEG) <= 60]
    d_mean = [damp_mean[i] for i in order if abs(offsets[i] * STEP_DEG) <= 60]
    d_sem = [damp_sem[i] for i in order if abs(offsets[i] * STEP_DEG) <= 60]

    os.makedirs(os.path.dirname(os.path.abspath(args.out_png)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=180)
    ax.errorbar(xs, s_mean, yerr=s_sem, color="#c23b3b", marker="o", lw=2.0, ms=3.2, capsize=2, label="sharpen")
    ax.errorbar(xs, d_mean, yerr=d_sem, color="#2f6fb3", marker="o", lw=2.0, ms=3.2, capsize=2, label="dampen")
    ax.axvline(0, color="0.45", ls=":", lw=1.1)
    ax.set_xlabel("Orientation offset from expected (deg)")
    ax.set_ylabel("L2/3 response (a.u.)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.out_png)
    plt.close(fig)

    rows = [
        {
            "offset_deg": x,
            "sharpen_mean": sm,
            "sharpen_sem": ss,
            "dampen_mean": dm,
            "dampen_sem": ds,
        }
        for x, sm, ss, dm, ds in zip(xs, s_mean, s_sem, d_mean, d_sem, strict=True)
    ]
    if args.out_csv:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump({"rows": rows, "sharpen": args.sharpen, "dampen": args.dampen}, f, indent=2, sort_keys=True)
            f.write("\n")
    print(json.dumps({"out_png": args.out_png, "out_csv": args.out_csv, "out_json": args.out_json}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
