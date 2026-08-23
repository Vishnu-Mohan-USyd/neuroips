#!/usr/bin/env python3
"""Deliverable figures: flank-suppressed sharpening via surround inhibition.

MAIN: seed-8 official endpoint profile (s=0.05, σ=4.0 surround), simplified
style of the delivered orientation figures. COMPANION: original α0.0
sharpening adapted curve vs the new network's adapted curve over the shared
baseline (the two networks' t0 baselines are identical to 1.2e−16 — the t0
state is feedback-silent, so the surround never acts there; asserted below).
All curves read from sha-pinned frozen JSON reports; no network runs.
Delivery is gated on validator GO — nothing here goes to the user directly.
Run: PYTHONHASHSEED=0 python3 -B make_flank_sharpening_figs.py

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
matplotlib.rcParams["svg.hashsalt"] = "flank_sharpening_20260819"
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent
SCRATCH = Path("/home/vishnu/scratch/flank_sharpening_20260819/runs/predinhib_s0p05_sig4")
ORIG_PROV = Path("/home/vishnu/neuroips_outputs/orientation_figs_20260819/provenance.json")
ORIG_PROV_SHA = "3a304d8526f24e016abacbff185254e12e918f906920f8799da43224debd91e1"
SEED_REPORTS = {
    8: ("endpoint_report.json",
        "9462b0bfeb9267b806b506e415e6809310573cbed565adc8bafc0db2e3ee9228"),
    9: ("endpoint_report_seed9.json",
        "befc6c4bf1ca38c199b7b180f453c2160064c0c197eb5a8d3614daf03a04aec8"),
    10: ("endpoint_report_seed10.json",
         "a01d944483d6dd6374e679e69fa2e1642fc80c6270283582ccd9f6476a1427b6"),
    11: ("endpoint_report_seed11.json",
         "771a6b708f4102655af747c9eeac08cf02820a44fd256de133e8f5e410fc5e8e"),
}
BASE_COLOR = "#888888"
NEW_COLOR = "#C44E52"
ORIG_COLOR = "#DD8452"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def render(stem: str, title: str, offsets, series, ymax) -> dict:
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for curve, color, label, alpha in series:
        ax.plot(offsets, curve, "-o", ms=3, lw=1.2, color=color, alpha=alpha,
                label=label)
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
    return paths


def main() -> int:
    reports = {}
    for seed, (name, sha) in SEED_REPORTS.items():
        path = SCRATCH / name
        assert sha256_file(path) == sha, name
        reports[seed] = json.loads(path.read_text(encoding="utf-8"))
    assert sha256_file(ORIG_PROV) == ORIG_PROV_SHA
    orig = json.loads(ORIG_PROV.read_text(encoding="utf-8"))

    offsets = reports[8]["official"]["curves_offsets_deg"]
    assert offsets == orig["offsets_deg"]
    new_base = reports[8]["official"]["curve_baseline_t0"]
    new_adapted = reports[8]["official"]["curve_adapted"]
    orig_base = orig["curves_plotted"]["alpha0.0"]["zero_context_curve"]
    orig_adapted = orig["curves_plotted"]["alpha0.0"]["expected_curve"]
    base_max_diff = max(abs(a - b) for a, b in zip(new_base, orig_base))
    assert base_max_diff < 1e-12, base_max_diff  # shared baseline is exact

    ymax = 1.16 * max(new_adapted + new_base + orig_adapted)
    outputs = {}
    outputs["main"] = render(
        "sharpening_with_surround",
        "Sharpening with surround inhibition",
        offsets,
        [(new_base, BASE_COLOR, "baseline (first stimulus)", 0.9),
         (new_adapted, NEW_COLOR, "after feedback (adapted)", 0.95)],
        ymax,
    )
    outputs["companion"] = render(
        "comparison_original_vs_surround",
        "Original vs with surround inhibition",
        offsets,
        [(new_base, BASE_COLOR, "baseline (first stimulus)", 0.9),
         (orig_adapted, ORIG_COLOR, "original sharpening (α=0.0)", 0.9),
         (new_adapted, NEW_COLOR, "with surround inhibition", 0.95)],
        ymax,
    )

    def crit(seed):
        r = reports[seed]
        return {
            "H": r["official"]["H"],
            "center_ratio": r["official"]["center_ratio"],
            "flank_ratio": r["official"]["flank_ratio"],
            "vitality_pass": r["official"]["vitality_pass"],
            "a4_counterfactual_s0_flank_ratio":
                r["a4_counterfactual_s0_inference_only_sidecar"]["flank_ratio"],
        }

    provenance = {
        "generated": "2026-08-19",
        "generator": str(OUT / "make_flank_sharpening_figs.py"),
        "status": "PENDING VALIDATOR GO — not user-delivered by the coder",
        "network": {
            "mechanism": "feedback-recruited subtractive surround "
                         "(pred_inhib) with broadened orientation footprint",
            "config": {"pred_inhib_strength": 0.05,
                       "pred_inhib_sigma_channels": 4.0,
                       "regime": "alpha=0.0 arm, seed 8 official; "
                                 "seeds 9/10/11 confirmation"},
            "two_line_diff_vs_frozen_harness": [
                '-    "pred_inhib_strength": 0.0,',
                '-    "pred_inhib_sigma_channels": 0.65,',
                '+    "pred_inhib_strength": 0.05,',
                '+    "pred_inhib_sigma_channels": 4.0,',
            ],
            "harness_sha256": {
                "frozen_original": "cdd71a11cbd254aa452f3b60f4f9da4350fe9fd8"
                                   "5f7dcdf95cd35513435c250e",
                "modified_copy": "9db8f975531b55a86c54791c68908708403cd4df"
                                 "72a97591ce8199b1ec25937e",
            },
            "biological_basis": (
                "feedback-recruited SOM-like broad surround with spared "
                "center (Adesnik 2012; Zhang 2014; Nurminen 2018) — "
                "primary sources read in full, DESIGN.md §5"
            ),
        },
        "criteria_all_seeds": {str(s): crit(s) for s in (8, 9, 10, 11)},
        "figure_data": {
            "offsets_deg": offsets,
            "baseline_shared": new_base,
            "baseline_note": (
                "new network's own t0 baseline; identical to the original "
                f"α0.0 baseline (max abs diff {base_max_diff:.3e}) because "
                "the t0 state is feedback-silent and the surround is "
                "feedback-driven"
            ),
            "adapted_with_surround_seed8": new_adapted,
            "adapted_original_alpha0p0": orig_adapted,
            "ymax_rule": "1.16 x max over plotted curves, shared by both "
                         "figures",
            "ymax": ymax,
        },
        "sources": {
            "seed_reports": {str(s): {"path": str(SCRATCH / n), "sha256": h}
                             for s, (n, h) in SEED_REPORTS.items()},
            "original_curves": {"path": str(ORIG_PROV),
                                "sha256": ORIG_PROV_SHA},
            "study_docs": [
                "/home/vishnu/neuroips_analysis/flank_sharpening_20260819/PROTOCOL.md",
                "/home/vishnu/neuroips_analysis/flank_sharpening_20260819/DESIGN.md",
                "/home/vishnu/neuroips_analysis/flank_sharpening_20260819/DIAGNOSTIC_REPORT.md",
                "/home/vishnu/scratch/flank_sharpening_20260819/RUN_LOG.md",
            ],
        },
        "outputs": outputs,
    }
    with open(OUT / "provenance.json", "w") as fh:
        json.dump(provenance, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps({"outputs": outputs, "ymax": ymax,
                      "base_max_diff": base_max_diff}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
