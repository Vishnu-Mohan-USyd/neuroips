#!/usr/bin/env python3
"""Phase 4 step 1: ORIGINAL α0.5 seed-8 reference values from FROZEN artifacts.

Runs BEFORE any Phase-4 retrain result is read (pins P1-P4 bars). Sources are
READ-ONLY: the frozen checkpoint (S2_plot == S2_confirm, bitwise; sha asserted)
is assayed fresh with the standard frozen convention to obtain H (absent from
the frozen assays), and every other value is cross-checked against the frozen
gate-decision artifact (must match <= 1e-6):
  center_ratio  = Cret 0.14957193609984015
  flank_ratio   = Fret 0.559041741467868
  M             = 0.3320623037521497   (whole-36-bin expected-A AUC / t0 AUC)
  rate_A        = 0.05534761527671864  (continuation final-step mean rate)
  mean_rate_t0  = 0.16667840538150916
Run: PYTHONHASHSEED=0 python3 -B phase4_reference_eval.py <out.json>
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

REPO = Path("/home/vishnu/neuroips_rnn_recreation_20260808/repo")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
import assay_emergent_task_energy_axis as assay  # noqa: E402
import tuned_emergence_lib as tuned  # noqa: E402

FROZEN_CK = Path("/home/vishnu/neuroips_runs/rnn_recreation_20260808/"
                 "S2_plot/seed_8/alpha_0p5_final.pt")
FROZEN_CK_SHA = ("156cc0f2372c6abcd42dd0798ac012d94bf2f761"
                 "f7e8a860fb5bcc8fbc70bc18")
FROZEN_GATE = Path("/home/vishnu/neuroips_runs/rnn_recreation_20260808/"
                   "S2_confirm/frozen_gate_decision_rnn.json")
# Binding assay convention (PROTOCOL "Established facts"; delivered figures):
# offsets are 5-deg orientation channels relative to the aligned stimulus.
# Center band +/-5 deg = offsets (-1, 0, 1); flank band +/-15-30 deg =
# offsets +/-(3..6). Each ratio divides the continuation-final aligned mean by
# the network's OWN literal-t0 baseline (see measure()).
CENTER_OFFSETS = (-1, 0, 1)
FLANK_OFFSETS = (-6, -5, -4, -3, 3, 4, 5, 6)
VITALITY_OFFSETS = (-2, -1, 0, 1, 2)
VITALITY_FLOOR = 0.01
PLOT_OFFSETS = tuple(range(-12, 13))
TOL = 1e-6


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def mem_available_kb() -> int:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1])
    raise RuntimeError("MemAvailable not found")


def bin_mean(curve: torch.Tensor, offsets) -> float:
    idx = [assay.OFFSETS.index(o) for o in offsets]
    return float(curve[:, idx].mean().item())


@torch.no_grad()
def measure(net, ck, device) -> dict:
    center_feedback = bool(ck.get("center_feedback", False))
    feedback_mode = tuned.resolve_feedback_mode(center_feedback,
                                                ck.get("feedback_mode"))
    theta_a, theta_b, finals = assay.matched_pairs(device)
    preds_a, rates_a = tuned.forward_seq_tuned(
        net, theta_a, 1.0, center_feedback=center_feedback,
        feedback_mode=feedback_mode)
    _, rates_b = tuned.forward_seq_tuned(
        net, theta_b, 1.0, center_feedback=center_feedback,
        feedback_mode=feedback_mode)
    # H = continuation hit rate: argmax of the step-3 prediction logits vs the
    # matched-pair final channel, over 216 histories (granularity 1/216).
    H = float((preds_a[:, 3, :].argmax(-1) == finals).float().mean())
    final_rates = rates_a[:, -1, :]
    aligned_a = assay.align_rates(final_rates, finals).to(torch.float64)
    first_a = (theta_a[:, 0] / assay.STEP_DEG).round().to(torch.long) % assay.N
    first_b = (theta_b[:, 0] / assay.STEP_DEG).round().to(torch.long) % assay.N
    # Literal-t0 baseline: pooled A+B FIRST-step responses aligned to each
    # trial's own first stimulus. The unroll evaluates L2/3 before any W_fb
    # output exists, so t0 is feedback-silent (f = 0) and the surround —
    # feedback-recruited — cannot act on it; the baseline is mechanism-free
    # by construction, not by switching anything off.
    t0 = 0.5 * (assay.align_rates(rates_a[:, 0, :], first_a).to(torch.float64)
                + assay.align_rates(rates_b[:, 0, :], first_b).to(torch.float64))
    mean_profile = aligned_a.mean(dim=0)
    vit = {str(o * 5): float(mean_profile[assay.OFFSETS.index(o)].item())
           for o in VITALITY_OFFSETS}
    plot_idx = [assay.OFFSETS.index(o) for o in PLOT_OFFSETS]
    rate_A = float(aligned_a.mean().item())
    mean_rate_t0 = float(t0.mean().item())
    return {
        "H": H,
        "center_ratio": bin_mean(aligned_a, CENTER_OFFSETS)
        / bin_mean(t0, CENTER_OFFSETS),
        "flank_ratio": bin_mean(aligned_a, FLANK_OFFSETS)
        / bin_mean(t0, FLANK_OFFSETS),
        "M_auc_ratio": rate_A / mean_rate_t0,
        "continuation_mean_rate": rate_A,
        "mean_rate_t0": mean_rate_t0,
        "vitality_band": vit,
        "vitality_pass": all(v > VITALITY_FLOOR for v in vit.values()),
        "mean_profile_max": float(mean_profile.max().item()),
        "mean_profile_min": float(mean_profile.min().item()),
        "curves_offsets_deg": [o * assay.STEP_DEG for o in PLOT_OFFSETS],
        "curve_adapted": mean_profile[plot_idx].cpu().tolist(),
        "curve_baseline_t0": t0.mean(dim=0)[plot_idx].cpu().tolist(),
    }


@torch.no_grad()
def main() -> int:
    out_path = Path(sys.argv[1])
    assert mem_available_kb() >= 25 * 1024 * 1024, "MemAvailable < 25 GB"
    device = torch.device("cuda:0")

    ck_sha = sha256_file(FROZEN_CK)
    assert ck_sha == FROZEN_CK_SHA, ck_sha

    gate = json.loads(FROZEN_GATE.read_text(encoding="utf-8"))
    seed8 = next(r for r in gate["seed_results"] if r["seed"] == 8)
    fz = seed8["metrics"]["0.5"]
    frozen = {
        "center_ratio": fz["Cret"],
        "flank_ratio": fz["Fret"],
        "M_auc_ratio": fz["M"],
        "continuation_mean_rate": fz["rate_A"],
        "mean_rate_t0": fz["mean_rate_t0"],
    }

    net, ck = assay.load_arm(FROZEN_CK, device)
    cfg = ck["tuned_net_config"]
    # Original architecture: surround OFF in the frozen reference.
    assert cfg["pred_inhib_strength"] == 0.0, cfg["pred_inhib_strength"]
    ref = measure(net, ck, device)
    del net
    torch.cuda.empty_cache()

    diffs = {k: abs(ref[k] - v) for k, v in frozen.items()}
    cross_ok = all(d <= TOL for d in diffs.values())
    report = {
        "purpose": "Phase-4 pre-registration reference values (original "
                   "alpha0.5 seed-8, no-surround architecture) — computed "
                   "BEFORE any Phase-4 retrain result was read",
        "frozen_checkpoint": str(FROZEN_CK),
        "frozen_checkpoint_sha256": ck_sha,
        "s2_confirm_s2_plot_bitwise_identical": True,
        "reference_values": {
            "center_ratio": ref["center_ratio"],
            "flank_ratio": ref["flank_ratio"],
            "H": ref["H"],
            "M_auc_ratio": ref["M_auc_ratio"],
            "continuation_mean_rate": ref["continuation_mean_rate"],
            "mean_rate_t0": ref["mean_rate_t0"],
        },
        "fresh_assay_full": ref,
        "cross_check_vs_frozen_gate_decision": {
            "source": str(FROZEN_GATE),
            "frozen_values": frozen,
            "abs_diffs": diffs,
            "tolerance": TOL,
            "pass": cross_ok,
            "note": "H absent from frozen assays; computed fresh only "
                    "(216 trials, granularity 1/216 = 0.00463)",
        },
        "M_definition": gate["M_definition"],
    }
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps({
        "reference_values": report["reference_values"],
        "cross_check_pass": cross_ok,
        "abs_diffs": diffs,
        "vitality_band": ref["vitality_band"],
        "vitality_pass": ref["vitality_pass"],
    }, indent=1))
    assert cross_ok, "cross-check vs frozen gate decision FAILED"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
