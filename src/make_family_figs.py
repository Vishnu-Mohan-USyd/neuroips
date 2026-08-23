#!/usr/bin/env python3
"""Family figures: both regimes under the IDENTICAL architecture (s=0.04, σ=4.0).

Two profile figures from the s=0.04 seed-8 endpoints (the joint-pass seed):
sharpening (α=0.0) and dampening (α=0.5), delivered simplified style. Curves
read from the sha-pinned ladder eval reports — no network runs, no GPU.
Run: PYTHONHASHSEED=0 python3 -B make_family_figs.py

Rendering is deterministic (SOURCE_DATE_EPOCH=0 before the matplotlib import,
fixed svg.hashsalt) and the style constants below ARE the delivered-figure
conventions (figsize 7.5x5, gray baseline #888888, dotted center line,
frameless legend, ymax = 1.16 x plotted max) — change nothing casually; the
delivered packs' provenance records them.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "family_s0p04_figs_20260822"
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent
LADDER = Path("/home/vishnu/scratch/flank_sharpening_20260819/runs/ladder_s0p04")
SOURCES = {
    "sharpening": ("eval_alpha0p0_seed8.json",
                   "fdf48fea678a0529e9461e233cc72cf82d55f4aac9f3a8fdd643fe250b7eae57"),
    "dampening": ("eval_alpha0p5_seed8.json",
                  "d01f88f9692866e3ceccf3bbbafa1c9e1a24ce8b89f14e8004c7df7d8f4daae7"),
}
VERDICT = Path("/home/vishnu/neuroips_analysis/flank_sharpening_20260819/VERDICT.md")
VERDICT_SHA = "f5640239144e5f294f3c4279d31bd3b3f0eaa8c6574ff559dfb3f6da7e706285"
BASE_COLOR = "#888888"
SHARP_COLOR = "#C44E52"
DAMP_COLOR = "#DD8452"
STATUS = ("family parity: 4/4 sharpening pass; dampening phenotype intact 4/4, "
          "activity band contested on 2/4 seeds — verdict O2, see VERDICT.md "
          "sha f5640239…")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def render(stem: str, title: str, offsets, base, adapted, color) -> dict:
    ymax = 1.16 * max(base + adapted)
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(offsets, base, "-o", ms=3, lw=1.2, color=BASE_COLOR, alpha=0.9,
            label="baseline (first stimulus)")
    ax.plot(offsets, adapted, "-o", ms=3, lw=1.2, color=color, alpha=0.95,
            label="after feedback (adapted)")
    ax.axvline(0.0, linestyle=":", color="black", linewidth=0.8)
    ax.set_xlabel("orientation relative to stimulus (°)")
    ax.set_ylabel("mean L2/3 response (a.u.)")
    ax.set_ylim(0.0, ymax)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    paths = {}
    for suffix in ("png", "pdf", "svg"):
        path = OUT / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight")
        paths[suffix] = str(path)
    plt.close(fig)
    return {"paths": paths, "ymax": ymax}


def main() -> int:
    reports = {}
    for regime, (name, sha) in SOURCES.items():
        path = LADDER / name
        assert sha256_file(path) == sha, name
        reports[regime] = json.loads(path.read_text(encoding="utf-8"))
    assert sha256_file(VERDICT) == VERDICT_SHA

    sh = reports["sharpening"]["official"]
    dm = reports["dampening"]["official"]
    offsets = sh["curves_offsets_deg"]
    assert offsets == dm["curves_offsets_deg"]
    base_diff = max(abs(a - b) for a, b in
                    zip(sh["curve_baseline_t0"], dm["curve_baseline_t0"]))

    out_sharp = render(
        "sharpening_with_surround_s0p04",
        "Sharpening (with surround inhibition)",
        offsets, sh["curve_baseline_t0"], sh["curve_adapted"], SHARP_COLOR)
    out_damp = render(
        "dampening_with_surround_s0p04",
        "Dampening (with surround inhibition)",
        offsets, dm["curve_baseline_t0"], dm["curve_adapted"], DAMP_COLOR)

    provenance = {
        "generated": "2026-08-22",
        "generator": str(OUT / "make_family_figs.py"),
        "status": STATUS,
        "network": {
            "mechanism": "feedback-recruited subtractive surround (pred_inhib) "
                         "with broadened orientation footprint — IDENTICAL "
                         "architecture in both regimes; regimes set only by "
                         "task/energy pressure (α)",
            "config": {"pred_inhib_strength": 0.04,
                       "pred_inhib_sigma_channels": 4.0,
                       "seed": 8,
                       "regimes": {"sharpening": "alpha=0.0",
                                   "dampening": "alpha=0.5"}},
            "harness_sha256": {
                "frozen_original": "cdd71a11cbd254aa452f3b60f4f9da4350fe9fd8"
                                   "5f7dcdf95cd35513435c250e",
                "s0p04_block_state": "7eb46f6c2a3b22885574b3961ce97ba9a1224259"
                                     "dc6654075cc8421b0e25d821",
                "two_line_diff_vs_frozen": [
                    '-    "pred_inhib_strength": 0.0,',
                    '-    "pred_inhib_sigma_channels": 0.65,',
                    '+    "pred_inhib_strength": 0.04,',
                    '+    "pred_inhib_sigma_channels": 4.0,',
                ],
            },
        },
        "criteria_seed8_cells": {
            "sharpening_alpha0p0": {
                "flank_ratio": sh["flank_ratio"],
                "center_ratio": sh["center_ratio"],
                "H": sh["H"],
                "vitality_pass": sh["vitality_pass"],
                "a4_s0_flank_ratio":
                    reports["sharpening"]
                    ["a4_counterfactual_s0_inference_only_sidecar"]
                    ["flank_ratio"],
                "verdict": reports["sharpening"]["criteria_verdict"],
            },
            "dampening_alpha0p5": {
                "M_auc_ratio": dm["M_auc_ratio"],
                "center_ratio": dm["center_ratio"],
                "flank_ratio": dm["flank_ratio"],
                "H": dm["H"],
                "continuation_mean_rate": dm["continuation_mean_rate"],
                "vitality_pass": dm["vitality_pass"],
                "a4_s0_M_auc_ratio":
                    reports["dampening"]
                    ["a4_counterfactual_s0_inference_only_sidecar"]
                    ["M_auc_ratio"],
                "verdict": reports["dampening"]["verdict"],
            },
        },
        "figure_data": {
            "offsets_deg": offsets,
            "sharpening": {"baseline_t0": sh["curve_baseline_t0"],
                           "adapted": sh["curve_adapted"],
                           "ymax": out_sharp["ymax"]},
            "dampening": {"baseline_t0": dm["curve_baseline_t0"],
                          "adapted": dm["curve_adapted"],
                          "ymax": out_damp["ymax"]},
            "baseline_note": (
                "each figure plots its own cell's t0 baseline; the two "
                f"cells' baselines agree to max abs diff {base_diff:.3e} "
                "(t0 is feedback-silent)"
            ),
            "ymax_rule": "1.16 x max over that figure's plotted curves "
                         "(per-figure; regimes differ ~7x in adapted scale)",
        },
        "sources": {
            "eval_reports": {r: {"path": str(LADDER / n), "sha256": s}
                             for r, (n, s) in SOURCES.items()},
            "verdict": {"path": str(VERDICT), "sha256": VERDICT_SHA},
            "study_docs": [
                "/home/vishnu/neuroips_analysis/flank_sharpening_20260819/PROTOCOL.md",
                "/home/vishnu/neuroips_analysis/flank_sharpening_20260819/DIAGNOSTIC_REPORT_PHASE4_M.md",
                "/home/vishnu/neuroips_analysis/flank_sharpening_20260819/DIAGNOSTIC_REPORT_PHASE4_LADDER.md",
                "/home/vishnu/scratch/flank_sharpening_20260819/RUN_LOG.md",
            ],
        },
        "outputs": {"sharpening": out_sharp["paths"],
                    "dampening": out_damp["paths"]},
    }
    with open(OUT / "provenance.json", "w") as fh:
        json.dump(provenance, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps({"outputs": provenance["outputs"],
                      "ymax": {"sharpening": out_sharp["ymax"],
                               "dampening": out_damp["ymax"]},
                      "baseline_max_abs_diff": base_diff}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
