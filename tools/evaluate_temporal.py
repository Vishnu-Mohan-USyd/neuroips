"""Recompute the packaged learned-temporal metrics and kinetic counterfactuals."""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import assay_emergent_task_energy_axis as assay

CHECKPOINTS = ROOT / "checkpoints" / "temporal_v11"
CONDITIONS = {
    "joint": (.3, "early"),
    "accuracy_only": (0.0, "early"),
    "energy_only": (1.0, "early"),
    "sustained": (.3, "sustained"),
    "fast_sst": (.3, "early"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "outputs" / "temporal_v11_evaluation" / "results.json",
                        help="Write freshly computed results here; existing results are never a cache.")
    parser.add_argument("--seeds", nargs="+", type=int, choices=(8, 9, 10), default=[8, 9, 10])
    parser.add_argument("--conditions", nargs="+", choices=tuple(CONDITIONS), default=list(CONDITIONS))
    return parser.parse_args()


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



def objective(held, alpha, window, fine):
    ce = held["current_ce" if window == "early" else "sustained_current_ce"]
    return ((1-alpha)*(held["next_ce"]+ce)/(2*math.log(36))
            + alpha*fine["normalized_cycle_activity"])


@torch.no_grad()
def main():
    args = parse_args()
    torch.set_num_threads(2)
    device = assay.choose_device("cpu")
    calibration = json.loads((CHECKPOINTS / "reference_limits.json").read_text())
    selected_paths = [
        CHECKPOINTS / condition / f"seed_{seed}/alpha_{assay.alpha_slug(CONDITIONS[condition][0])}_final.pt"
        for condition in args.conditions for seed in args.seeds
    ]
    selected_paths += [ROOT / calibration[str(seed)]["source_checkpoint"] for seed in args.seeds]
    for checkpoint_path in selected_paths:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Selected checkpoint is missing: {checkpoint_path}")
    result = {"runs": {}, "references": {}, "calibration_limits": {}}
    for seed in args.seeds:
        limits = dict(calibration[str(seed)])
        if "penalty_coefficient" in limits:
            limits["original_setup_penalty_coefficient"] = limits.pop("penalty_coefficient")
        result["calibration_limits"][str(seed)] = limits
        reference_net, reference_checkpoint = assay.load_arm(
            ROOT / limits["source_checkpoint"], device)
        result["references"][str(seed)] = assay.temporal_held_out(
            reference_net, reference_checkpoint, device)
    path = args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for condition in args.conditions:
        alpha, window = CONDITIONS[condition]
        name = f"fitted_{condition}"
        rows = result["runs"].setdefault(name, {})
        for seed in args.seeds:
            checkpoint_path = CHECKPOINTS / condition / f"seed_{seed}/alpha_{assay.alpha_slug(alpha)}_final.pt"
            net, checkpoint = assay.load_arm(checkpoint_path, device)
            assert net.learn_temporal_kinetics and net.sst_response_gain and checkpoint["step"] == checkpoint["target_steps"]
            tau_e, tau_p = [float(t) for t in net.temporal_time_constants()]
            probe = assay.temporal_probe(net, checkpoint, device)
            held = held_out_timecourse(net, checkpoint, device)
            net.temporal_protocol = dict(net.temporal_protocol, dt=.025)
            fine = assay.temporal_held_out(net, checkpoint, device)
            row = {"checkpoint": str(checkpoint_path.relative_to(ROOT)),
                   "alpha": alpha, "window": window, "tau_e": tau_e, "tau_p": tau_p,
                   "penalty_coefficient": checkpoint["temporal_current_ce_penalty_coefficient"],
                   "w_sf": float(net.w_sf_effective()),
                   "w_ef": float(net.circuit_gains()[assay.tuned.CIRC_INDEX["w_ef"]]),
                   "probe": probe, "held_out": held, "fine_held_out": fine,
                   "objective": objective(held, alpha, window, fine)}
            limit = checkpoint.get("temporal_current_ce_ceiling")
            ce = held["current_ce" if window == "early" else "sustained_current_ce"]
            row["ceiling"] = limit
            if limit is not None:
                row["ceiling_feasible"] = ce <= limit + 1e-6
                row["penalized_objective"] = row["objective"] + checkpoint["temporal_current_ce_penalty_coefficient"] * max(ce-limit, 0)**2
            if name in ("fitted_joint", "fitted_fast_sst"):
                row["normal_probe_cost_fine"] = assay.temporal_probe(net, checkpoint, device, measurements=False)
                row["sst_clamped_probe_cost_fine"] = assay.temporal_probe(net, checkpoint, device, clamp=True, measurements=False)
                row["counterfactuals"] = {}
                for label, e, p in (("swap", tau_p, tau_e),
                                    ("equal_fast", min(tau_e, tau_p), min(tau_e, tau_p)),
                                    ("equal_slow", max(tau_e, tau_p), max(tau_e, tau_p))):
                    net.temporal_tau_e_raw.fill_(assay.tuned.softplus_inverse(e))
                    net.temporal_tau_p_raw.fill_(assay.tuned.softplus_inverse(p))
                    counter = assay.temporal_held_out(net, checkpoint, device)
                    net.temporal_protocol = dict(net.temporal_protocol, dt=.1)
                    counter_probe = assay.temporal_probe(net, checkpoint, device)
                    net.temporal_protocol = dict(net.temporal_protocol, dt=.025)
                    row["counterfactuals"][label] = {
                        "tau_e": e, "tau_p": p, "held_out": counter,
                        "windows": counter_probe["windows"],
                        "ceiling_feasible": counter["current_ce"] <= limit + 1e-6,
                        "objective": objective(counter, alpha, "early", counter)}
            rows[str(seed)] = row
            path.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
            early, late = [probe["windows"][k] for k in ("early", "late")]
            print(json.dumps({"condition": name, "seed": seed, "tau_e": tau_e, "tau_p": tau_p,
                              "early_center": early["preferred_ratio"], "early_flanks": early["flank_ratio"],
                              "late_center": late["preferred_ratio"], "late_flanks": late["flank_ratio"],
                              "J": fine["normalized_cycle_activity"], "objective": row["objective"]}), flush=True)
    print("Measured", sum(len(rows) for rows in result["runs"].values()),
          "of", len(args.conditions) * len(args.seeds), "selected endpoints", flush=True)


if __name__ == "__main__":
    main()

