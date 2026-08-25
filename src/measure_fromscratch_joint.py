#!/usr/bin/env python3
"""Measure the from-scratch joint run (protocol: fromscratch_joint_20260825).

Endpoint: standard frozen assay code paths — the transplant study's ncommon
(measure(), placement_hits clone, assay_arm markers, k/g), i.e. the exact
validator-verified pipeline. Comparators loaded from the transplant study's
n4_assay.json (seed 8, alpha0.5): PPP = pretrain host, TTT = two-stage arm
endpoint, both measured by this same pipeline.

Trajectory: init (reconstructed, sha-verified vs the run's logged init sha)
+ every 500-step snapshot + final: k (float64), decode (decode_A_minus_B,
plus decode_A/decode_B), mean rates, and the full profile coordinates
(M/center/flank/H). Event-log task/energy curves parsed from training.jsonl.

Writes ONLY under /home/vishnu/scratch/fromscratch_joint_20260825/.
No interpretation here — numbers only; NaN/vitality failures are reported
as measured.

PROVENANCE (this copy)
----------------------
Executed as /home/vishnu/scratch/fromscratch_joint_20260825/scripts/
measure_joint.py on reuben-ML (both regimes: no argument = alpha0.5,
``alpha0.0`` argument = the follow-up). Imports the transplant measurement
pipeline (repo copy: ``transplant/ncommon.py``) and the probe's harness
(repo copy: ``train_fromscratch_joint.py``) via the absolute reuben-ML paths
below — archived here for record/review, not standalone execution. Outputs
are the study record's results_joint*.json. This docstring section is the
only difference between this copy and the executed file (comment-only, per
the repo's packaging convention).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

BASE = Path("/home/vishnu/scratch/fromscratch_joint_20260825")
ARM = sys.argv[1] if len(sys.argv) > 1 else "alpha0.5"
SLUG = {"alpha0.5": "0p5", "alpha0.0": "0p0"}[ARM]
ALPHA = {"alpha0.5": 0.5, "alpha0.0": 0.0}[ARM]
RUN = BASE / "runs" / ("joint" if ARM == "alpha0.5" else "joint_alpha0p0") / "seed_8"
OUT = BASE / ("results_joint.json" if ARM == "alpha0.5"
              else "results_joint_alpha0p0.json")
sys.path.insert(0, "/home/vishnu/scratch/transplant_surround_20260823/scripts")
sys.path.insert(0, str(BASE / "harness"))
import ncommon as C  # noqa: E402  (transplant pipeline, read-only use)
from ncommon import assay  # noqa: E402
import train_sweep as TS  # noqa: E402  (the run's own harness copy)

INIT_SHA = "08bbb3ae58ac2bebfd46591f0934154af7e6b8fea30663c5ce5259564f5fdf5a"


@torch.no_grad()
def snapshot_row(path: Path, device, battery, theta, ch, common_raw) -> dict:
    C.mem_gate()
    prof = C.measure_path(path, device)
    summ, _ = assay.assay_arm(path, device, common_raw)
    mk = C.markers(summ)
    cep = C.ce_profile(path, device, theta, ch)
    sd = torch.load(path, map_location="cpu")["state_dict"]
    row = {
        "k": C.k_from_raw(sd["circ_raw"]),
        "g": C.g_from_raw(sd["circ_raw"]),
        "decode_A_minus_B": mk["decode_A_minus_B"],
        "decode_A": mk["decode_A"],
        "decode_B": mk["decode_B"],
        "B_minus_A_rate": mk["B_minus_A_rate"],
        "rate_A": summ.get("rate_A"),
        "rate_B": summ.get("rate_B"),
        "M_auc_ratio": prof["M_auc_ratio"],
        "center_ratio": prof["center_ratio"],
        "flank_ratio": prof["flank_ratio"],
        "H": prof["H"],
        "continuation_mean_rate": prof["continuation_mean_rate"],
        "vitality_pass": prof["vitality_pass"],
        "max_CE_A": max(cep["A"]),
        "ce_tripped_A": cep["gate_tripped_A"],
    }
    del sd, summ
    C.release()
    return row


@torch.no_grad()
def main() -> None:
    C.mem_gate()
    device = assay.choose_device("cuda:0")
    battery = C.make_battery(device)
    theta, ch = battery["theta"], C.ce_channels(battery)
    common_raw = torch.load(C.pretrain_path(8), map_location=device)[
        "state_dict"]["local_comp_strength_raw"]

    # ---- init reconstruction (sha-verified against the run's logged init) --
    TS.MODEL_CONFIG["recurrent_cell"] = "rnn_tanh"
    TS.seed_everything(8)
    net0 = TS.tuned.build_tuned_from_config(TS.MODEL_CONFIG)
    init_sha = TS.state_sha256(net0.state_dict())
    if init_sha != INIT_SHA:
        raise SystemExit(f"init reconstruction sha {init_sha} != run's "
                         f"{INIT_SHA} — STOP (report, no diagnosis)")
    init_path = RUN / "init_state_reconstructed.pt"
    run_refs = torch.load(RUN / f"alpha_{SLUG}_final.pt",
                          map_location="cpu")["references"]
    torch.save({"stage": "fromscratch_init_reconstructed", "seed": 8,
                "alpha": ALPHA, "step": 0, "state_dict": net0.state_dict(),
                "tuned_net_config": dict(TS.MODEL_CONFIG),
                "references": run_refs,
                "freeze_local_comp": True, "center_feedback": False,
                "feedback_mode": "posterior_prior_excess"}, init_path)
    del net0

    # ------------------------------------------------------- trajectory ----
    steps = [0] + list(range(500, 11001, 500))
    traj = {}
    for st in steps:
        p = (init_path if st == 0
             else RUN / (f"alpha_{SLUG}_final.pt" if st == 11000
                         else f"alpha_{SLUG}_step{st:05d}.pt"))
        traj[str(st)] = snapshot_row(p, device, battery, theta, ch, common_raw)
        print(f"step {st:5d}: k {traj[str(st)]['k']:+.4f} "
              f"decode {traj[str(st)]['decode_A_minus_B']:+.4f} "
              f"M {traj[str(st)]['M_auc_ratio']:.4f} "
              f"center {traj[str(st)]['center_ratio']:.4f} "
              f"flank {traj[str(st)]['flank_ratio']:.4f} "
              f"rate {traj[str(st)]['continuation_mean_rate']:.4f}", flush=True)

    # ------------------------------------------- endpoint + hits + E1 ------
    final_p = RUN / f"alpha_{SLUG}_final.pt"
    endpoint = dict(traj["11000"])
    endpoint["hits_A_on_y"] = C.placement_hits(final_p, device, battery)["A_on_y"]
    endpoint["registered_final_A_on_y"] = endpoint["hits_A_on_y"][3]

    # --------------------------------------- comparators (same pipeline) ---
    n4 = json.loads(Path("/home/vishnu/scratch/transplant_surround_20260823/"
                         "n4_assay.json").read_text())["per_seed"]["8"][ARM]

    def comp(cell):
        c = n4[cell]
        return {"M_auc_ratio": c["profile"]["M_auc_ratio"],
                "center_ratio": c["profile"]["center_ratio"],
                "flank_ratio": c["profile"]["flank_ratio"],
                "H": c["profile"]["H"],
                "continuation_mean_rate": c["profile"]["continuation_mean_rate"],
                "decode_A_minus_B": c["markers"]["decode_A_minus_B"],
                "decode_A": c["markers"]["decode_A"],
                "B_minus_A_rate": c["markers"]["B_minus_A_rate"],
                "registered_final_A_on_y": c["registered_final_A_on_y"],
                "k": c["k"], "E1_M": c["E1_retention"]["M"]}

    # ----------------------------------------------- event-log curves ------
    curves = {"step": [], "task": [], "energy": [], "next_ce": [],
              "k_effective": [], "gradient_norm": []}
    with (RUN / "training.jsonl").open() as fh:
        for line in fh:
            ev = json.loads(line)
            if ev.get("event") == "alpha_step":
                curves["step"].append(ev["step"])
                curves["task"].append(ev["task"])
                curves["energy"].append(ev["energy"])
                curves["next_ce"].append(ev["next_ce"])
                curves["k_effective"].append(
                    ev["effective_net_som_vip_feedback_coefficient"])
                curves["gradient_norm"].append(ev["gradient_norm"])

    k_cross = next((s for s, k in zip(curves["step"], curves["k_effective"])
                    if k < 0.0), None)
    out = {
        "run": str(RUN),
        "arm": ARM,
        "init_sha_verified": True,
        "trajectory": traj,
        "endpoint": endpoint,
        "comparator_two_stage_arm_TTT": comp("TTT"),
        "comparator_pretrain_host_PPP": comp("PPP"),
        "event_log_curves": curves,
        "k_first_negative_step_eventlog_100step_resolution": k_cross,
    }
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps({"k_first_negative_step": k_cross,
                      "endpoint": {k: endpoint[k] for k in
                                   ("M_auc_ratio", "center_ratio",
                                    "flank_ratio", "H", "decode_A_minus_B",
                                    "continuation_mean_rate", "k",
                                    "vitality_pass", "ce_tripped_A")}},
                     indent=1))


if __name__ == "__main__":
    main()
