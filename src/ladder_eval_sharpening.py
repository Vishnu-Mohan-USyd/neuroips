#!/usr/bin/env python3
"""Per-seed endpoint evidence (multi-seed confirmation, config s=0.05/σ=4.0).

Same measurements as endpoint_eval_rung2.py; run dir (and its own
common_pretrain_final.pt) derived from the checkpoint path so it serves any
seed. Reports: criteria numbers, vitality band, A4 s->0 counterfactual
(measurement-only sidecar), full curves, context.
Run: PYTHONHASHSEED=0 python3 -B ladder_eval_sharpening.py <final.pt> <out.json> <expected_strength>

LADDER VARIANT: derived from the official endpoint_eval_seed.py with ONE functional
difference — the expected pred_inhib_strength is argv[3] instead of the
constant 0.05, asserted against the checkpoint's own config (the bars are
UNCHANGED). Used for the joint dose ladder s in {0.02, 0.03, 0.04} and the
s=0.04 multi-seed confirmation; the official files stay byte-identical to
their recorded shas.
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
    fpos = tuned.predictive_feedback_evidence(preds_a, center_feedback,
                                              feedback_mode)
    return {
        "H": H,
        "center_ratio": bin_mean(aligned_a, CENTER_OFFSETS)
        / bin_mean(t0, CENTER_OFFSETS),
        "flank_ratio": bin_mean(aligned_a, FLANK_OFFSETS)
        / bin_mean(t0, FLANK_OFFSETS),
        "vitality_band": vit,
        "vitality_pass": all(v > VITALITY_FLOOR for v in vit.values()),
        "M_mean_rate": float(final_rates.mean().item()),
        "legacy_dead_ring_exact_zero":
            float((final_rates == 0.0).float().mean(dim=1).mean().item()),
        "feedback_positive_mean": float(fpos.mean().item()),
        "curves_offsets_deg": [o * assay.STEP_DEG for o in PLOT_OFFSETS],
        "curve_adapted": aligned_a.mean(dim=0)[plot_idx].cpu().tolist(),
        "curve_baseline_t0": t0.mean(dim=0)[plot_idx].cpu().tolist(),
    }


@torch.no_grad()
def main() -> int:
    final_path = Path(sys.argv[1]).resolve()
    out_path = Path(sys.argv[2])
    expected_s = float(sys.argv[3])
    assert mem_available_kb() >= 25 * 1024 * 1024, "MemAvailable < 25 GB"
    device = torch.device("cuda:0")
    run_dir = final_path.parent

    pre = torch.load(run_dir / "common_pretrain_final.pt", map_location="cpu",
                     weights_only=False)
    pre_lc = pre["state_dict"]["local_comp_strength_raw"].clone()
    pre_refs = pre["references"]
    del pre

    net, ck = assay.load_arm(final_path, device)
    step = int(ck["step"])
    seed = int(ck["seed"])
    cfg = ck["tuned_net_config"]
    assert cfg["pred_inhib_strength"] == expected_s and \
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

    report = {
        "checkpoint": str(final_path),
        "seed": seed,
        "pred_inhib_strength": expected_s,
        "step": step,
        "official": official,
        "criteria_verdict": {
            "flank_le_0p85": official["flank_ratio"] <= 0.85,
            "flank_le_0p75_stretch": official["flank_ratio"] <= 0.75,
            "center_ge_1p15": official["center_ratio"] >= 1.15,
            "H_ge_0p95": official["H"] >= 0.95,
        },
        "vitals_catastrophe": (official["H"] < 0.9
                               or not official["vitality_pass"]),
        "a4_counterfactual_s0_inference_only_sidecar": {
            "label": "MEASUREMENT-ONLY: trained net, pred_inhib_strength->0 "
                     "at inference; never the deliverable profile",
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
                     ("H", "center_ratio", "flank_ratio", "vitality_pass")},
        "vitality_band": official["vitality_band"],
        "criteria_verdict": report["criteria_verdict"],
        "vitals_catastrophe": report["vitals_catastrophe"],
        "a4_flank_without_surround": counterfactual["flank_ratio"],
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
