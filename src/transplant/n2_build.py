#!/usr/bin/env python3
"""N2 — build the 60 core hybrid nets + gate chain (DESIGN 3.6).

Mirrors the frozen t2_build.py: construction audit (every key traced to its
declared source, bitwise), G5 (k consistency), G3 (PPP identity, AMENDMENT-1
dual filename with per-arm alpha metadata), then G0 STOP-first:
  - evaluator sanity: frozen whole_profile_retention on the ORIGINAL no-surround
    alpha0.5 seed-8 dir must bit-match the old anchor 0.3320623037521497;
  - TTT reconstruction bitwise per key, per regime x seed;
  - assay reproduction: flank-measure on built TTT == validator-exact anchors
    (seed 8: abs diff 0.0 required) and <=1e-6 vs stored eval officials
    (seeds 9/10/11, G4-analog);
  - E1 (whole_profile_retention M) on TTT recorded + compared to measure M.
Then G3 M lookups (both filenames bit-equal) and the PPP-direct control (EC1
analog: raw pretrain file through all path-based endpoints == built PPP,
bitwise, 4/4 seeds). Device cuda:0. STOP on any gate failure.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ncommon as C  # noqa: E402
from ncommon import assay, gates  # noqa: E402

PPP_POLICY = "per_arm_metadata"  # AMENDMENT-1, inherited verbatim


@torch.no_grad()
def main() -> None:
    C.mem_gate()
    device = assay.choose_device("cuda:0")
    if C.HYB.exists():
        shutil.rmtree(C.HYB)
    out = {"device": str(device), "ppp_policy": PPP_POLICY, "construction": {},
           "G5_k_consistency": {}, "G0": {}, "G3_PPP": {}, "EC1_ppp_direct": {},
           "verdict": {}}
    fails = []
    n_built = 0

    # ---------------------------------------------------------------- build
    for seed in C.SEEDS:
        sd_pre = torch.load(C.pretrain_path(seed), map_location="cpu")["state_dict"]
        bodies = {a: torch.load(C.arm_path(seed, a), map_location="cpu") for a in C.ARMS}
        for arm in C.ARMS:
            sd_arm = bodies[arm]["state_dict"]
            for cell_id in C.CELL_IDS:
                if cell_id == "PPP":
                    continue
                sd = C.hybrid_state_dict(cell_id, sd_pre, sd_arm)
                p = C.save_hybrid(bodies[arm], sd, C.cell_ckpt(seed, arm, cell_id))
                n_built += 1
                rl = torch.load(p, map_location="cpu")["state_dict"]
                per_key = {}
                for key in sorted(sd_pre):
                    src_name = "pretrain"
                    for letter, comp in zip(cell_id, C.COMPONENTS):
                        if key in C.COMP_TENSORS[comp]:
                            src_name = "pretrain" if letter == "P" else "arm"
                    src = sd_pre if src_name == "pretrain" else sd_arm
                    per_key[key] = bool(torch.equal(rl[key], src[key]))
                ok = all(per_key.values()) and set(rl) == set(sd_pre)
                out["construction"][f"seed{seed}_{arm}_{cell_id}"] = {
                    "all_keys_match_declared_source": ok}
                if not ok:
                    fails.append(f"construction_seed{seed}_{arm}_{cell_id}")
                gsrc = sd_pre if cell_id[2] == "P" else sd_arm
                k_cell, k_src = C.k_from_raw(rl["circ_raw"]), C.k_from_raw(gsrc["circ_raw"])
                g5ok = bool(torch.equal(rl["circ_raw"], gsrc["circ_raw"])) and k_cell == k_src
                out["G5_k_consistency"][f"seed{seed}_{arm}_{cell_id}"] = {
                    "k_cell": k_cell, "k_gains_source": k_src, "pass": g5ok}
                if not g5ok:
                    fails.append(f"G5_seed{seed}_{arm}_{cell_id}")
                del rl, sd
        # shared PPP: one network, both filenames, per-arm metadata
        for arm in C.ARMS:
            C.save_hybrid(bodies[arm], dict(sd_pre),
                          C.cell_dir(seed, arm, "PPP") / C.SLUG_OF[arm])
        n_built += 1
        rl = {a: torch.load(C.cell_dir(seed, a, "PPP") / C.SLUG_OF[a],
                            map_location="cpu")["state_dict"] for a in C.ARMS}
        ident = {a: all(torch.equal(rl[a][k], sd_pre[k]) for k in sd_pre) for a in C.ARMS}
        cross = all(torch.equal(rl["alpha0.0"][k], rl["alpha0.5"][k]) for k in sd_pre)
        out["G3_PPP"][f"seed{seed}"] = {
            "state_dict_identical_to_pretrain_both_files": ident,
            "two_files_state_dicts_bitwise_equal": bool(cross)}
        if not (all(ident.values()) and cross):
            fails.append(f"G3_weights_seed{seed}")
        out["G5_k_consistency"][f"seed{seed}_shared_PPP"] = {
            "k_cell": C.k_from_raw(rl["alpha0.0"]["circ_raw"]),
            "k_gains_source": C.k_from_raw(sd_pre["circ_raw"]),
            "pass": bool(torch.equal(rl["alpha0.0"]["circ_raw"], sd_pre["circ_raw"]))}
        del rl, sd_pre, bodies
        C.heartbeat(f"N2 build: seed {seed} core cells built ({n_built} nets total)")
    out["n_networks_built"] = n_built

    # ---------------------------------------------- G0 (STOP-first gate)
    m_orig = gates.whole_profile_retention(
        Path("/home/vishnu/neuroips_runs/rnn_recreation_20260808/S2_confirm/seed_8"),
        0.5, device)["M"]
    out["G0"]["frozen_evaluator_sanity_original_anchor"] = {
        "recomputed": m_orig, "anchor": C.FROZEN_M_ANCHOR_ORIGINAL,
        "bit_identical": m_orig == C.FROZEN_M_ANCHOR_ORIGINAL}
    if m_orig != C.FROZEN_M_ANCHOR_ORIGINAL:
        fails.append("G0_evaluator_sanity")

    for seed in C.SEEDS:
        for arm in C.ARMS:
            arm_sd = torch.load(C.arm_path(seed, arm), map_location="cpu")["state_dict"]
            p = C.cell_ckpt(seed, arm, "TTT")
            rl = torch.load(p, map_location="cpu")["state_dict"]
            wid = all(torch.equal(rl[k], arm_sd[k]) for k in arm_sd) and set(rl) == set(arm_sd)
            meas = C.measure_path(p, device)
            stored = C.stored_eval(seed, arm)["official"]  # sha-pinned artifact
            keys = [k for k in C.G0_ANCHOR_KEYS[arm] if k in stored]
            diffs = {k: abs(meas[k] - stored[k]) for k in keys}
            tol = 0.0 if seed == 8 else 1e-6
            repro_ok = all(d <= tol for d in diffs.values())
            e1 = gates.whole_profile_retention(C.cell_dir(seed, arm, "TTT"),
                                               C.ALPHA_OF[arm], device)
            ent = {"weights_bitwise_identical": bool(wid),
                   "measure_abs_diffs": diffs, "measure_repro_ok": bool(repro_ok),
                   "E1_M_frozen_evaluator": float(e1["M"]),
                   "E1_equals_measure_M": e1["M"] == meas["M_auc_ratio"],
                   "measure": {k: meas[k] for k in
                               ("H", "center_ratio", "flank_ratio", "M_auc_ratio",
                                "continuation_mean_rate", "vitality_pass")}}
            out["G0"][f"seed{seed}_{arm}"] = ent
            if not (wid and repro_ok):
                fails.append(f"G0_seed{seed}_{arm}")
            del arm_sd, rl
    if any(f.startswith("G0") for f in fails):
        out["verdict"] = {"fails": fails, "PASS": False,
                          "note": "G0 diff — STOPPED, no non-gate cell read"}
        (C.ROOT / "n2_build_gates.json").write_text(json.dumps(out, indent=1, sort_keys=True))
        raise SystemExit(f"G0 FAIL: {[f for f in fails if f.startswith('G0')]} — STOP")

    # ----------------------------------------- G3 M lookups (after G0)
    for seed in C.SEEDS:
        ms = {a: gates.whole_profile_retention(C.cell_dir(seed, a, "PPP"),
                                               C.ALPHA_OF[a], device)["M"]
              for a in C.ARMS}
        agree = ms["alpha0.0"] == ms["alpha0.5"]
        out["G3_PPP"][f"seed{seed}"]["M_lookups"] = ms
        out["G3_PPP"][f"seed{seed}"]["two_M_lookups_bit_identical"] = bool(agree)
        if not agree:
            fails.append(f"G3_M_seed{seed}")

    # ------------------------- EC1 analog: raw pretrain == built PPP
    battery = C.make_battery(device)
    theta, ch = battery["theta"], C.ce_channels(battery)
    for seed in C.SEEDS:
        raw_p = C.pretrain_path(seed)
        ppp_p = C.cell_ckpt(seed, "alpha0.0", "PPP")
        common_raw = torch.load(raw_p, map_location=device)["state_dict"][
            "local_comp_strength_raw"]
        pair = {}
        for name, p in (("raw", raw_p), ("built", ppp_p)):
            meas = C.measure_path(p, device)
            hits = C.placement_hits(p, device, battery)
            summ, _ = assay.assay_arm(p, device, common_raw)
            mk = C.markers(summ)
            cep = C.ce_profile(p, device, theta, ch)
            pair[name] = {"measure": meas, "hits": hits, "markers": mk, "ce": cep}
        same = {
            "measure": all(pair["raw"]["measure"][k] == pair["built"]["measure"][k]
                           for k in ("H", "center_ratio", "flank_ratio",
                                     "M_auc_ratio", "continuation_mean_rate")),
            "hits": pair["raw"]["hits"] == pair["built"]["hits"],
            "markers": pair["raw"]["markers"] == pair["built"]["markers"],
            "ce": pair["raw"]["ce"] == pair["built"]["ce"]}
        out["EC1_ppp_direct"][str(seed)] = {"bitwise_equal": all(same.values()),
                                            "detail": same}
        if not all(same.values()):
            fails.append(f"EC1_seed{seed}")
        del common_raw

    out["verdict"] = {"fails": fails, "PASS": not fails}
    (C.ROOT / "n2_build_gates.json").write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps({"n_networks_built": n_built,
                      "G0": {k: {kk: v[kk] for kk in
                                 ("weights_bitwise_identical", "measure_repro_ok",
                                  "E1_equals_measure_M")}
                             for k, v in out["G0"].items() if k.startswith("seed")},
                      "G0_evaluator_sanity":
                          out["G0"]["frozen_evaluator_sanity_original_anchor"],
                      "G3": {s: out["G3_PPP"][f"seed{s}"]["two_M_lookups_bit_identical"]
                             for s in C.SEEDS},
                      "EC1": {s: out["EC1_ppp_direct"][str(s)]["bitwise_equal"]
                              for s in C.SEEDS},
                      "verdict": out["verdict"]}, indent=1))
    if fails:
        raise SystemExit(f"N2 GATE FAIL: {fails} — STOP")
    C.heartbeat("N2 PASS: 60 core nets built; construction audit + G5 clean; G0 anchors "
                "exact (s8 abs diff 0.0, others <=1e-6); G3 dual-file PPP clean; "
                "EC1 raw-pretrain==built-PPP bitwise 4/4")


if __name__ == "__main__":
    main()
