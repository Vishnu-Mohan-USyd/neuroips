#!/usr/bin/env python3
"""N4 — all endpoints on all 94 table cells (89 distinct nets), sequential.

Per cell (DESIGN 3.1): E1 M via the frozen evaluator (dual-filename dirs);
E2 placement hits (instrumented-unroll clone, per-step bitwise-gated); E3 assay
markers (assay_arm with the seed pretrain's local_comp tensor) + chimera CE
profile (A-gate 3*ln36); flank-study profile coordinates (measure(), verbatim
core). Companions: k, g per cell.

Bookkeeping: shared cells (PPP, PRP) assayed under BOTH filenames, results
checked bitwise across the two and entered in both tables. Determinism repeat:
alpha0.5 TPT seed 8 assayed twice, exact match required. TTT consistency:
E2 A_on_y[3] vs the stored eval artifact H (recorded). Heartbeat to RUN_LOG.md
every 10 nets. Sequential one-net-at-a-time del+gc; MemAvailable gate per net.
Device cuda:0.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ncommon as C  # noqa: E402
from ncommon import assay, gates  # noqa: E402

REGIME_CELLS = list(C.CELL_IDS[1:]) + list(C.CTRL_TRAINED)  # 7 core + 3 ctrl


@torch.no_grad()
def endpoints(seed: int, arm: str, cell_id: str, device, battery, theta, ch,
              common_raw) -> dict:
    C.mem_gate()
    d = C.cell_dir(seed, arm, cell_id)
    p = d / C.SLUG_OF[arm]
    ret = gates.whole_profile_retention(d, C.ALPHA_OF[arm], device)
    prof = C.measure_path(p, device)
    hits = C.placement_hits(p, device, battery)
    summ, _ = assay.assay_arm(p, device, common_raw)
    mk = C.markers(summ)
    cep = C.ce_profile(p, device, theta, ch)
    sd = torch.load(p, map_location="cpu")["state_dict"]
    out = {"E1_retention": {k: float(v) for k, v in ret.items()},
           "profile": prof,
           "hits": hits,
           "registered_final_A_on_y": hits["A_on_y"][3],
           "markers": mk,
           "ce": cep,
           "k": C.k_from_raw(sd["circ_raw"]),
           "g": C.g_from_raw(sd["circ_raw"])}
    del sd, summ
    C.release()
    return out


def scalar_equal(a: dict, b: dict) -> bool:
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


@torch.no_grad()
def main() -> None:
    C.mem_gate()
    device = assay.choose_device("cuda:0")
    battery = C.make_battery(device)
    theta, ch = battery["theta"], C.ce_channels(battery)
    out = {"device": str(device), "ce_gate_threshold": C.CE_GATE, "per_seed": {},
           "shared_cell_checks": {}, "trip_census": {}, "determinism_repeat": {},
           "ttt_hit_vs_stored_H": {}}
    n_done = 0

    for seed in C.SEEDS:
        common_raw = torch.load(C.pretrain_path(seed), map_location=device)[
            "state_dict"]["local_comp_strength_raw"]
        ent = {a: {} for a in C.ARMS}
        # shared cells: assay under both filenames, check bitwise, enter twice
        shared_here = ("PPP", "PRP") if seed == 8 else ("PPP",)
        for cid in shared_here:
            by_file = {}
            for arm in C.ARMS:
                by_file[arm] = endpoints(seed, arm, cid, device, battery, theta,
                                         ch, common_raw)
                ent[arm][cid] = by_file[arm]
            same = scalar_equal(by_file["alpha0.0"], by_file["alpha0.5"])
            out["shared_cell_checks"][f"seed{seed}_{cid}"] = bool(same)
            n_done += 1
            if n_done % 10 == 0:
                C.heartbeat(f"N4: {n_done} nets assayed (latest seed{seed} {cid})")
        for arm in C.ARMS:
            cells = list(REGIME_CELLS)
            if seed == 8:
                cells += ["PNP", "PQP"]
            for cid in cells:
                ent[arm][cid] = endpoints(seed, arm, cid, device, battery, theta,
                                          ch, common_raw)
                n_done += 1
                if n_done % 10 == 0:
                    C.heartbeat(f"N4: {n_done} nets assayed (latest seed{seed} "
                                f"{arm} {cid})")
                if seed == 8 and arm == "alpha0.5" and cid == "TPT":
                    rep = endpoints(seed, arm, cid, device, battery, theta, ch,
                                    common_raw)
                    out["determinism_repeat"]["alpha0.5_TPT_s8"] = {
                        "exact_match": scalar_equal(ent[arm][cid], rep)}
            stored_H = C.stored_eval(seed, arm)["official"]["H"]
            out["ttt_hit_vs_stored_H"][f"seed{seed}_{arm}"] = {
                "A_on_y_t3": ent[arm]["TTT"]["registered_final_A_on_y"],
                "stored_H": stored_H,
                "equal": ent[arm]["TTT"]["registered_final_A_on_y"] == stored_H}
        out["per_seed"][str(seed)] = ent
        trips = sorted(f"{arm}:{c}" for arm in C.ARMS for c in ent[arm]
                       if ent[arm][c]["ce"]["gate_tripped_A"])
        out["trip_census"][str(seed)] = trips
        print(f"seed {seed}: done ({n_done} nets so far); trips={trips}", flush=True)
        del common_raw
        C.release()

    (C.ROOT / "n4_assay.json").write_text(json.dumps(out, indent=1, sort_keys=True))
    summary = {"n_nets_assayed": n_done,
               "trip_census": out["trip_census"],
               "shared_cell_checks": out["shared_cell_checks"],
               "determinism_repeat": out["determinism_repeat"],
               "ttt_hit_vs_stored_H_all_equal":
                   all(v["equal"] for v in out["ttt_hit_vs_stored_H"].values())}
    print(json.dumps(summary, indent=1))
    ok = (all(out["shared_cell_checks"].values())
          and out["determinism_repeat"]["alpha0.5_TPT_s8"]["exact_match"])
    C.heartbeat(f"N4 DONE: {n_done} nets assayed; trips={out['trip_census']}; "
                f"shared-cell dual-file bitwise={all(out['shared_cell_checks'].values())}; "
                f"determinism repeat exact="
                f"{out['determinism_repeat']['alpha0.5_TPT_s8']['exact_match']}")
    if not ok:
        raise SystemExit("N4 bookkeeping check FAILED (shared-cell or determinism) — STOP")


if __name__ == "__main__":
    main()
