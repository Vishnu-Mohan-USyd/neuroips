#!/usr/bin/env python3
"""Phase 4 endpoint evidence: α=0.5 dampening arm under the surround architecture.

Evaluates the pre-registered bars (PROTOCOL.md Phase-4 entry + coder-appended
reference values, pinned BEFORE this retrain result was read):
  P1 dampening direction: center_ratio <= 0.35
  P2 topology preserved:  center_ratio < flank_ratio
  P3 in-family vitals:    H and M within +/-15% relative of original α0.5
     seed-8 references (H 0.194444, M 0.332062; bands loaded full-precision
     from phase4_reference_alpha0p5_seed8.json)
  P4 no collapse (A3 sense): continuation mean rate > 0.01 AND all
     |offset| <= 10 deg channels of mean aligned final profile > 0.01
A4-style s->0 counterfactual recorded as EVIDENCE, not a bar
(measurement-only sidecar, never the deliverable).
Run: PYTHONHASHSEED=0 python3 -B phase4_endpoint_eval.py <final.pt> <out.json>
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import torch

REPO = Path("/home/vishnu/neuroips_rnn_recreation_20260808/repo")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
import assay_emergent_task_energy_axis as assay  # noqa: E402
import tuned_emergence_lib as tuned  # noqa: E402
import torch.nn.functional as F  # noqa: E402

REF_JSON = Path("/home/vishnu/scratch/flank_sharpening_20260819/"
                "phase4_reference_alpha0p5_seed8.json")
# Binding assay convention (PROTOCOL "Established facts"; delivered figures):
# offsets are 5-deg orientation channels relative to the aligned stimulus.
# Center band +/-5 deg = offsets (-1, 0, 1); flank band +/-15-30 deg =
# offsets +/-(3..6). Each ratio divides the continuation-final aligned mean by
# the network's OWN literal-t0 baseline (see measure()).
CENTER_OFFSETS = (-1, 0, 1)
FLANK_OFFSETS = (-6, -5, -4, -3, 3, 4, 5, 6)
VITALITY_OFFSETS = (-2, -1, 0, 1, 2)
VITALITY_FLOOR = 0.01
RATE_FLOOR = 0.01
PLOT_OFFSETS = tuple(range(-12, 13))


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
    fpos = tuned.predictive_feedback_evidence(preds_a, center_feedback,
                                              feedback_mode)
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
        "feedback_positive_mean": float(fpos.mean().item()),
        "curves_offsets_deg": [o * assay.STEP_DEG for o in PLOT_OFFSETS],
        "curve_adapted": mean_profile[plot_idx].cpu().tolist(),
        "curve_baseline_t0": t0.mean(dim=0)[plot_idx].cpu().tolist(),
    }


@torch.no_grad()
def main() -> int:
    final_path = Path(sys.argv[1]).resolve()
    out_path = Path(sys.argv[2])
    assert mem_available_kb() >= 25 * 1024 * 1024, "MemAvailable < 25 GB"
    device = torch.device("cuda:0")
    run_dir = final_path.parent

    ref = json.loads(REF_JSON.read_text(encoding="utf-8"))["reference_values"]
    h_lo, h_hi = 0.85 * ref["H"], 1.15 * ref["H"]
    m_lo, m_hi = 0.85 * ref["M_auc_ratio"], 1.15 * ref["M_auc_ratio"]

    pre = torch.load(run_dir / "common_pretrain_final.pt", map_location="cpu",
                     weights_only=False)
    pre_lc = pre["state_dict"]["local_comp_strength_raw"].clone()
    pre_refs = pre["references"]
    del pre

    net, ck = assay.load_arm(final_path, device)
    step = int(ck["step"])
    seed = int(ck["seed"])
    alpha = float(ck["alpha"])
    cfg = ck["tuned_net_config"]
    assert alpha == 0.5, alpha
    assert cfg["pred_inhib_strength"] == 0.05 and \
        cfg["pred_inhib_sigma_channels"] == 4.0 and \
        cfg["recurrent_cell"] == "rnn_tanh"
    freeze_ok = torch.equal(ck["state_dict"]["local_comp_strength_raw"].cpu(),
                            pre_lc)
    gains = F.softplus(ck["state_dict"]["circ_raw"]).cpu().tolist()
    som_margin = gains[1] - gains[2] * gains[0]
    effective_k = gains[3] - gains[4] * max(som_margin, 0.0)

    official = measure(net, ck, device)
    del net
    torch.cuda.empty_cache()

    # A4 counterfactual (evidence sidecar, NEVER the deliverable profile):
    # rebuild the SAME trained state_dict under a config copy with the surround
    # strength zeroed. The kernel is a persistent=False buffer rebuilt from
    # config, so this is a pure inference-time s->0 switch: if the flank
    # suppression vanishes here, the surround (not some other adaptation) was
    # doing the work.
    cf_cfg = copy.deepcopy(cfg)
    cf_cfg["pred_inhib_strength"] = 0.0
    cf_net = tuned.build_tuned_from_config(cf_cfg).to(device)
    cf_net.load_state_dict(ck["state_dict"])
    cf_net.eval()
    counterfactual = measure(cf_net, ck, device)
    del cf_net
    torch.cuda.empty_cache()

    verdict = {
        "P1_center_le_0p35": official["center_ratio"] <= 0.35,
        "P2_center_lt_flank": official["center_ratio"]
        < official["flank_ratio"],
        "P3_H_in_band": h_lo <= official["H"] <= h_hi,
        "P3_M_in_band": m_lo <= official["M_auc_ratio"] <= m_hi,
        "P4_rate_above_floor": official["continuation_mean_rate"] > RATE_FLOOR,
        "P4_band_alive_A3": official["vitality_pass"],
    }
    report = {
        "checkpoint": str(final_path),
        "seed": seed,
        "alpha": alpha,
        "step": step,
        "official": official,
        "preregistered_bars": {
            "P1": "center_ratio <= 0.35",
            "P2": "center_ratio < flank_ratio",
            "P3_H_band": [h_lo, h_hi],
            "P3_M_band": [m_lo, m_hi],
            "P4": f"continuation_mean_rate > {RATE_FLOOR} AND A3 band alive",
            "reference_values": ref,
        },
        "verdict": verdict,
        "all_pass": all(verdict.values()),
        "a4_counterfactual_s0_inference_only_sidecar": {
            "label": "EVIDENCE ONLY (not a bar): trained net, "
                     "pred_inhib_strength->0 at inference; never the "
                     "deliverable profile",
            **counterfactual,
        },
        "context": {
            "effective_k": effective_k,
            "gains_softplus": gains,
            "som_margin": som_margin,
            "freeze_local_comp_matches_own_pretrain": bool(freeze_ok),
            "own_pretrain_references_data_init_identity_only_mechanism_blind":
                pre_refs,
        },
    }
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps({
        "seed": seed,
        "official": {k: official[k] for k in
                     ("H", "center_ratio", "flank_ratio", "M_auc_ratio",
                      "continuation_mean_rate", "vitality_pass")},
        "verdict": verdict,
        "all_pass": report["all_pass"],
        "a4_sidecar": {k: counterfactual[k] for k in
                       ("H", "center_ratio", "flank_ratio", "M_auc_ratio")},
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
