"""Recompute the packaged uncapped peak-readout temporal metrics on CPU."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import assay_emergent_task_energy_axis as assay

CHECKPOINTS = ROOT / "checkpoints" / "temporal_peak"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-ratio", type=int, choices=(2, 3), default=2)
    parser.add_argument("--seeds", nargs="+", type=int, choices=(8, 9, 10), default=[8, 9, 10])
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Fresh JSON output; default outputs/temporal_peak_evaluation/init_<ratio>x/results.json.",
    )
    args = parser.parse_args()
    if args.out is None:
        args.out = ROOT / "outputs" / "temporal_peak_evaluation" / f"init_{args.init_ratio}x" / "results.json"
    return args


@torch.no_grad()
def held_out_timecourse(net, checkpoint, device):
    data = assay.train_sweep.make_generator(device, 930001)
    primary = assay.train_sweep.make_generator(device, 930002)
    auxiliary = assay.train_sweep.make_generator(device, 930003)
    references = checkpoint["references"]
    totals = {}
    for _ in range(8):
        theta, channels = assay.train_sweep.momentum_batch(
            128, 12, device, data, mismatch_prob=0.02)
        predictions, _, trace = assay.tuned.forward_seq_tuned(
            net, theta, feedback_mode=checkpoint["feedback_mode"],
            return_timecourse=True)
        rates, times = trace["rates"], trace["times"]
        early, late = assay.temporal_window_indices(times)
        remaining = [k for k in range(len(times)) if k not in early.tolist()]
        noise = torch.empty_like(rates)
        noise[:, :, early] = torch.randn(
            rates[:, :, early].shape, generator=primary, device=device,
            dtype=rates.dtype) * references["sigma_train"]
        noise[:, :, remaining] = torch.randn(
            rates[:, :, remaining].shape, generator=auxiliary, device=device,
            dtype=rates.dtype) * references["sigma_train"]
        values = [assay.temporal_current_metrics(
            net, rates[:, :, k], channels, noise[:, :, k], references, primary)
            for k in range(len(times))]
        for key in ("current_ce", "current_accuracy"):
            curve = torch.tensor([row[key] for row in values], dtype=torch.float64)
            totals[key + "_timecourse"] = totals.get(key + "_timecourse", 0) + curve / 8
        totals["next_ce"] = totals.get("next_ce", 0) + float(F.cross_entropy(
            predictions[:, :-1].reshape(-1, 36), channels[:, 1:].reshape(-1))) / 8
        totals["next_accuracy"] = totals.get("next_accuracy", 0) + float(
            (predictions[:, :-1].argmax(-1) == channels[:, 1:]).double().mean()) / 8
        integrals = assay.train_sweep.temporal_activity_integrals(net, trace)
        for key in ("on_integral", "gap_integral", "cycle_integral"):
            totals[key] = totals.get(key, 0) + float(integrals[key].double().mean()) / 8
    for key in ("current_ce", "current_accuracy"):
        curve = totals[key + "_timecourse"]
        totals[key] = float(curve[early.cpu()].mean())
        totals["late_" + key] = float(curve[late.cpu()].mean())
        totals["sustained_" + key] = float(curve[1:].mean())
        totals[key + "_timecourse"] = curve.tolist()
    totals["times"] = times.cpu().tolist()
    totals["normalized_cycle_activity"] = totals["cycle_integral"] / references["J_ref"]
    return totals


@torch.no_grad()
def selected_metrics(net, checkpoint, device):
    data = assay.train_sweep.make_generator(device, 930001)
    noisegen = assay.train_sweep.make_generator(device, 930002)
    references = checkpoint["references"]
    totals, timing, dynamic_timing = {}, [], []
    for _ in range(8):
        theta, channels = assay.train_sweep.momentum_batch(
            128, 12, device, data, mismatch_prob=.02,
        )
        predictions, _, trace = assay.tuned.forward_seq_tuned(
            net, theta, feedback_mode=checkpoint["feedback_mode"], return_timecourse=True,
        )
        selected, indices = assay.tuned.select_peak_evidence(net, trace["rates"])
        noise = torch.randn(selected.shape, generator=noisegen, device=device) * references["sigma_train"]
        current = assay.temporal_current_metrics(
            net, selected, channels, noise, references, noisegen,
        )
        for key in ("current_ce", "current_accuracy"):
            totals[key] = totals.get(key, 0) + current[key] / 8
        totals["next_ce"] = totals.get("next_ce", 0) + float(F.cross_entropy(
            predictions[:, :-1].reshape(-1, 36), channels[:, 1:].reshape(-1),
        )) / 8
        totals["next_accuracy"] = totals.get("next_accuracy", 0) + float(
            (predictions[:, :-1].argmax(-1) == channels[:, 1:]).double().mean()
        ) / 8
        integrals = assay.train_sweep.temporal_activity_integrals(net, trace)
        for key in ("on_integral", "gap_integral", "cycle_integral"):
            totals[key] = totals.get(key, 0) + float(integrals[key].double().mean()) / 8
        chosen_times = trace["times"][indices[:, 1:]]
        timing.extend(chosen_times.flatten().tolist())
        r = trace["rates"][:, 1:, 1:]
        evidence = (r @ net.readout_cos).square() + (r @ net.readout_sin).square()
        varies = evidence.amax(-1) - evidence.amin(-1) > 1e-6
        dynamic_timing.extend(chosen_times[varies].tolist())
    totals["normalized_cycle_activity"] = totals["cycle_integral"] / references["J_ref"]
    totals["selected_time_after_first"] = {
        "mean": float(np.mean(timing)),
        "median": float(np.median(timing)),
        "range": [float(min(timing)), float(max(timing))],
        "times": trace["times"].tolist(),
        "counts": np.bincount(
            np.rint(np.asarray(timing) / net.temporal_protocol["dt"]).astype(int),
            minlength=len(trace["times"]),
        ).tolist(),
        "dynamic_count": len(dynamic_timing),
        "dynamic_mean": float(np.mean(dynamic_timing)) if dynamic_timing else None,
    }
    return totals


@torch.no_grad()
def main():
    args = parse_args()
    torch.set_num_threads(2)
    device = assay.choose_device("cpu")
    selected_paths = {
        seed: CHECKPOINTS / f"init_{args.init_ratio}x" / f"seed_{seed}" / "alpha_0p3_final.pt"
        for seed in args.seeds
    }
    for checkpoint_path in selected_paths.values():
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Selected checkpoint is missing: {checkpoint_path}")
    result = {"runs": {"fitted_joint": {}}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    for seed, checkpoint_path in selected_paths.items():
        net, checkpoint = assay.load_arm(checkpoint_path, device)
        assert net.learn_temporal_kinetics and net.sst_response_gain
        assert checkpoint["step"] == checkpoint["target_steps"]
        assert checkpoint["temporal_current_window"] == "peak" and float(checkpoint["alpha"]) == .3
        assert net.temporal_protocol["dt"] == .1
        assert net.temporal_protocol["predictor_readout"] == "peak_evidence_noiseless"
        tau_e, tau_p = [float(t) for t in net.temporal_time_constants()]
        probe = assay.temporal_probe(net, checkpoint, device)
        held = held_out_timecourse(net, checkpoint, device)
        selected = selected_metrics(net, checkpoint, device)
        ceiling = checkpoint["temporal_current_ce_ceiling"]
        row = {
            "checkpoint": str(checkpoint_path.relative_to(ROOT)),
            "step": checkpoint["step"], "target_steps": checkpoint["target_steps"],
            "window": "peak", "alpha": .3, "tau_e": tau_e, "tau_p": tau_p,
            "ceiling": ceiling, "probe": probe, "held_out": held, "selected": selected,
            "objective": (
                .7 * (selected["next_ce"] + selected["current_ce"]) / (2 * math.log(36))
                + .3 * selected["normalized_cycle_activity"]
            ),
            "ceiling_feasible": selected["current_ce"] <= ceiling + 1e-6,
        }
        result["runs"]["fitted_joint"][str(seed)] = row
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({
            "init_ratio": args.init_ratio, "seed": seed,
            "tau_e": tau_e, "tau_p": tau_p,
            "current_ce": selected["current_ce"], "ceiling": ceiling,
            "current_accuracy": selected["current_accuracy"],
            "J": selected["normalized_cycle_activity"], "objective": row["objective"],
            "ceiling_feasible": row["ceiling_feasible"],
        }), flush=True)
    print("Measured", len(selected_paths), "selected endpoints:", args.out, flush=True)


if __name__ == "__main__":
    main()
