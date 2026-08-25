#!/usr/bin/env python3
"""N7 — synthesis: rho tables, pre-registered classifications, strategy map,
prediction confrontation, registered questions, G6 re-verify (DESIGN 3.2-3.5,
4.4-4.6). Pure post-processing of n4/n5/n6 JSON. No GPU.

Rules implemented verbatim from DESIGN:
- 3.3 rho_m(X) = (m(X)-m(PPP))/(m(TTT)-m(PPP)) per seed per factorial; floors
  hit 0.15, decode 0.04, rate 0.008, M 0.05, center_ratio 0.05,
  flank_ratio 0.05 on |TTT-PPP|; below floor => UNREADABLE, raw reported,
  no adjudication on that coordinate.
- 3.4 bands: F = rho >= 0.75; partial = 0.25 < rho < 0.75; 0 = rho <= 0.25
  (rho < 0 reported as "0, below baseline"). Claim enters the strategy map
  only at 4/4 readable-seed agreement; CE-tripped cell => that seed
  UNRESOLVABLE for competence-dependent claims (hit/decode); verdict rests on
  untripped seeds and says so.
- 3.2 sharpening carries = rho_center AND rho_flank AND rho_hit all >= 0.75,
  4/4 readable, no trip; dampening carries = rho_M AND rho_center >= 0.75,
  4/4 readable, no trip; rho_rate DEMOTED to raw-report-only for dampening
  (companion rho_rate retained for sharpening only).
- 3.5 chimera gate: trip = max mean CE_A > 3*ln36; house rule as above.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ncommon as C  # noqa: E402

ROOT = C.ROOT
ARMS = list(C.ARMS)
SEEDS = list(C.SEEDS)

# marker -> (extractor, floor, competence_dependent)
FLOORS = {"hit": 0.15, "decode": 0.04, "rate": 0.008, "M": 0.05,
          "center": 0.05, "flank": 0.05}
COMPETENCE = {"hit", "decode"}
PRIMARIES = {"alpha0.0": ("center", "flank", "hit"), "alpha0.5": ("M", "center")}
RHO_MARKERS = {"alpha0.0": ("hit", "decode", "rate", "M", "center", "flank"),
               "alpha0.5": ("hit", "decode", "M", "center", "flank")}  # rate raw-only
ALL_CELLS = ["PPP", "TPP", "PTP", "PPT", "TTP", "TPT", "PTT", "TTT",
             "TRT", "TNT", "TQT"]
HOST_CTRLS = ["PRP", "PNP", "PQP"]  # seed 8 only


def m_of(cell: dict, marker: str) -> float:
    if marker == "hit":
        return cell["registered_final_A_on_y"]
    if marker == "decode":
        return cell["markers"]["decode_A_minus_B"]
    if marker == "rate":
        return cell["markers"]["B_minus_A_rate"]
    if marker == "M":
        return cell["E1_retention"]["M"]
    if marker == "center":
        return cell["profile"]["center_ratio"]
    if marker == "flank":
        return cell["profile"]["flank_ratio"]
    raise KeyError(marker)


def band(rho: float) -> str:
    if rho >= 0.75:
        return "F"
    if rho > 0.25:
        return "partial"
    if rho < 0.0:
        return "0_below_baseline"
    return "0"


def main() -> None:
    n4 = json.loads((ROOT / "n4_assay.json").read_text())
    n5 = json.loads((ROOT / "n5_s0.json").read_text())
    n6 = json.loads((ROOT / "n6_analyses.json").read_text())
    per_seed = n4["per_seed"]

    # ---------------------------------------------------------- G6 re-verify
    g6 = subprocess.run(["sha256sum", "-c", "--quiet", str(ROOT / "MANIFEST.sha256")],
                        capture_output=True, text=True)
    g6_ok = g6.returncode == 0
    if not g6_ok:
        print(g6.stdout + g6.stderr)
        raise SystemExit("G6 RE-VERIFY FAILED — donor files changed — STOP")

    out = {"G6_reverify_pass": True, "denominators": {}, "rho": {},
           "raw": {}, "classification": {}, "strategy_map": {},
           "prediction_confrontation": {}, "questions": {}, "hypotheses": {},
           "alignment_criticality": {}, "split_8_11_vs_9_10": {},
           "deeper": {}, "U1_host_flank_first_check": {}}

    # ------------------------------------------------- denominators + rho
    for arm in ARMS:
        out["denominators"][arm], out["rho"][arm], out["raw"][arm] = {}, {}, {}
        for s in SEEDS:
            g = per_seed[str(s)][arm]
            dens = {}
            for mk, floor in FLOORS.items():
                den = m_of(g["TTT"], mk) - m_of(g["PPP"], mk)
                dens[mk] = {"den": den, "floor": floor,
                            "readable": abs(den) >= floor}
            out["denominators"][arm][str(s)] = dens
            cells = ALL_CELLS + (HOST_CTRLS if s == 8 else [])
            rho_s, raw_s = {}, {}
            for cid in cells:
                cell = g[cid]
                tripped = bool(cell["ce"]["gate_tripped_A"])
                raw_s[cid] = {mk: m_of(cell, mk) for mk in FLOORS}
                raw_s[cid]["tripped"] = tripped
                raw_s[cid]["vitality_pass"] = cell["profile"]["vitality_pass"]
                raw_s[cid]["decode_A"] = cell["markers"]["decode_A"]
                raw_s[cid]["max_CE_A"] = max(cell["ce"]["A"])
                ent = {}
                for mk in RHO_MARKERS[arm]:
                    d = dens[mk]
                    if not d["readable"]:
                        ent[mk] = {"status": "UNREADABLE", "raw": raw_s[cid][mk]}
                        continue
                    rho = (m_of(cell, mk) - m_of(g["PPP"], mk)) / d["den"]
                    st = {"rho": rho, "band": band(rho)}
                    if tripped and mk in COMPETENCE:
                        st["status"] = "UNRESOLVABLE_TRIP"
                    ent[mk] = st
                ent["tripped"] = tripped
                rho_s[cid] = ent
            out["rho"][arm][str(s)] = rho_s
            out["raw"][arm][str(s)] = raw_s

    # --------------------------------------------------- classification 3.2/3.4
    for arm in ARMS:
        out["classification"][arm] = {}
        prim = PRIMARIES[arm]
        for cid in ALL_CELLS + HOST_CTRLS:
            seeds_here = SEEDS if cid not in HOST_CTRLS else [8]
            per = {}
            for s in seeds_here:
                e = out["rho"][arm][str(s)].get(cid)
                st = {}
                for mk in prim:
                    r = e[mk]
                    if r.get("status") == "UNREADABLE":
                        st[mk] = "UNREADABLE"
                    elif r.get("status") == "UNRESOLVABLE_TRIP":
                        st[mk] = "UNRESOLVABLE_TRIP"
                    else:
                        st[mk] = r["band"]
                st["tripped"] = e["tripped"]
                per[str(s)] = st
            n_seeds = len(seeds_here)
            untripped = [s for s in seeds_here if not per[str(s)]["tripped"]]
            readable_all = [s for s in untripped
                            if all(per[str(s)][mk] in ("F", "partial", "0",
                                                       "0_below_baseline")
                                   for mk in prim)]
            carries_seeds = [s for s in readable_all
                             if all(per[str(s)][mk] == "F" for mk in prim)]
            if len(carries_seeds) == n_seeds:
                verdict = f"CARRIES ({n_seeds}/{n_seeds})"
            else:
                per_mk = {}
                for mk in prim:
                    bands = [per[str(s)][mk] for s in seeds_here]
                    per_mk[mk] = "/".join(bands)
                trip_n = n_seeds - len(untripped)
                verdict = "; ".join(f"{mk}:{per_mk[mk]}" for mk in prim)
                if trip_n:
                    verdict += f" [{trip_n} seed(s) CE-tripped -> " \
                               f"UNRESOLVABLE for competence claims; " \
                               f"verdict rests on untripped seeds]"
                if carries_seeds and len(carries_seeds) == len(readable_all) \
                        and len(readable_all) < n_seeds:
                    verdict = f"CARRIES on all {len(readable_all)} " \
                              f"readable+untripped seeds — " + verdict
            out["classification"][arm][cid] = {"per_seed": per,
                                               "verdict": verdict}

    # ------------------------------------------------------ strategy map
    for arm in ARMS:
        rows = {}
        for cid in ALL_CELLS + (HOST_CTRLS):
            cls = out["classification"][arm][cid]
            rows[cid] = cls["verdict"]
        out["strategy_map"][arm] = rows

    # ------------------------------------------- U1 first check (PPP flank)
    for s in SEEDS:
        v = per_seed[str(s)]["alpha0.0"]["PPP"]["profile"]["flank_ratio"]
        out["U1_host_flank_first_check"][str(s)] = {
            "PPP_flank_ratio": v, "predicted_band": [0.85, 0.97],
            "in_predicted_band": 0.85 <= v <= 0.97}

    # -------------------------------------- alignment criticality A_align
    aa = {"alpha0.0": {}, "alpha0.5": {}, "per_seed_more_critical": {}}
    prim_of = {"alpha0.0": "hit", "alpha0.5": "M"}
    for s in SEEDS:
        vals = {}
        for arm in ARMS:
            mk = prim_of[arm]
            tpt = out["rho"][arm][str(s)]["TPT"][mk]
            tqt = out["rho"][arm][str(s)]["TQT"][mk]
            if "rho" in tpt and "rho" in tqt and \
                    tpt.get("status") != "UNRESOLVABLE_TRIP" and \
                    tqt.get("status") != "UNRESOLVABLE_TRIP":
                vals[arm] = tpt["rho"] - tqt["rho"]
                aa[arm][str(s)] = vals[arm]
            else:
                aa[arm][str(s)] = "UNRESOLVABLE"
        if all(isinstance(vals.get(a), float) for a in ARMS):
            aa["per_seed_more_critical"][str(s)] = vals["alpha0.0"] > vals["alpha0.5"]
        else:
            aa["per_seed_more_critical"][str(s)] = "UNRESOLVABLE"
    ok_seeds = [v for v in aa["per_seed_more_critical"].values()
                if isinstance(v, bool)]
    aa["registered_statement_sharpening_MORE_alignment_critical"] = (
        f"{sum(ok_seeds)}/{len(ok_seeds)} resolvable seeds TRUE"
        + ("" if len(ok_seeds) == 4 else
           f" ({4 - len(ok_seeds)} seed(s) unresolvable by trip)"))
    # FB premium on hit (sharpening): 1 - rho_hit(TPT) vs original 0.51-0.58
    prem = {}
    for s in SEEDS:
        t = out["rho"]["alpha0.0"][str(s)]["TPT"]["hit"]
        prem[str(s)] = 1.0 - t["rho"] if "rho" in t else "UNREADABLE"
    aa["fb_premium_on_hit_1_minus_rho_hit_TPT"] = {
        "measured": prem, "original_range": [0.51, 0.58]}
    out["alignment_criticality"] = aa

    # ------------------------------------------------ Q1..Q4 (4.4)
    q = {}
    # Q1: FB-alone flank at s=0.04 — rho_flank(PTP), rho_flank(TTP); s->0 attributes
    q1 = {"per_seed": {}}
    for s in SEEDS:
        e = out["rho"]["alpha0.0"][str(s)]
        q1["per_seed"][str(s)] = {
            "rho_flank_PTP": e["PTP"]["flank"], "rho_flank_TTP": e["TTP"]["flank"],
            "raw_flank": {c: out["raw"]["alpha0.0"][str(s)][c]["flank"]
                          for c in ("PPP", "PTP", "TTP", "TTT")}}
    q1["s0_attribution_seed8"] = {
        k: {"flank_official": v["official"]["flank_ratio"],
            "flank_s0": v["s0_counterfactual"]["flank_ratio"],
            "delta_s_minus_s0": v["delta_s_minus_s0"]["flank_ratio"]}
        for k, v in list(n5["registered"].items())
        + list(n5["extras_floored_descriptive"].items())
        if k.startswith("alpha0.0")}
    q1["note"] = ("alpha0.0 flank denominator floored on any seed => that "
                  "factorial's flank answered descriptively per 3.3")
    q["Q1_fb_alone_flank"] = q1
    # Q2: dampening GAINS lock — rho_M(TTP) + overshoot
    q2 = {}
    for s in SEEDS:
        raw = out["raw"]["alpha0.5"][str(s)]
        q2[str(s)] = {"rho_M_TTP": out["rho"]["alpha0.5"][str(s)]["TTP"]["M"],
                      "raw_M_TTP": raw["TTP"]["M"], "raw_M_PPP": raw["PPP"]["M"],
                      "overshoots_above_host": raw["TTP"]["M"] > raw["PPP"]["M"]}
    q["Q2_dampening_gains_lock"] = q2
    # Q3: softmax temperature via R vs N (dampening primaries)
    q3 = {}
    for s in SEEDS:
        e = out["rho"]["alpha0.5"][str(s)]
        row = {}
        for c in ("TPT", "TNT", "TRT"):
            row[c] = {mk: e[c][mk] for mk in ("M", "center")}
        both = all("rho" in e[c][mk] for c in ("TNT", "TRT")
                   for mk in ("M", "center"))
        row["TRT_lt_TNT_on_both_primaries"] = (
            e["TRT"]["M"]["rho"] < e["TNT"]["M"]["rho"]
            and e["TRT"]["center"]["rho"] < e["TNT"]["center"]["rho"]
            if both else "UNREADABLE")
        q3[str(s)] = row
    q["Q3_softmax_temperature_R_vs_N"] = q3
    # Q4: CE-trip census repeat
    q["Q4_trip_census"] = {
        "measured": n4["trip_census"],
        "original": "alpha0.5 PPT and PTT tripped 4/4",
        "repeat_alpha0.5_PPT": [s for s in SEEDS if f"alpha0.5:PPT"
                                in n4["trip_census"][str(s)]],
        "repeat_alpha0.5_PTT": [s for s in SEEDS if f"alpha0.5:PTT"
                                in n4["trip_census"][str(s)]],
        "new_trips_not_in_original_class": sorted(
            {f"s{s}:{t}" for s in SEEDS for t in n4["trip_census"][str(s)]
             if not t.startswith("alpha0.5:PPT")
             and not t.startswith("alpha0.5:PTT")})}
    out["questions"] = q

    # ------------------------------------------------ H-C1 / H-C2 (4.5)
    hc1 = {"per_seed": {}}
    for s in SEEDS:
        e = out["rho"]["alpha0.5"][str(s)]
        raw = out["raw"]["alpha0.5"][str(s)]
        row = {}
        for c in ("TNT", "TQT", "TRT"):
            readable = all("rho" in e[c][mk] for mk in ("M", "center"))
            row[c] = {
                "carries": (readable and not raw[c]["tripped"]
                            and e[c]["M"]["rho"] >= 0.75
                            and e[c]["center"]["rho"] >= 0.75),
                "partial_or_better": (readable and not raw[c]["tripped"]
                                      and e[c]["M"]["rho"] > 0.25
                                      and e[c]["center"]["rho"] > 0.25),
                "tripped": raw[c]["tripped"]}
        hc1["per_seed"][str(s)] = row
    hc1["confirmed"] = (
        all(hc1["per_seed"][str(s)]["TNT"]["carries"] for s in SEEDS)
        and all(hc1["per_seed"][str(s)]["TQT"]["carries"] for s in SEEDS)
        and all(hc1["per_seed"][str(s)]["TRT"]["partial_or_better"]
                for s in SEEDS))
    out["hypotheses"]["H_C1_dampening_genericity"] = hc1
    hc2 = {"per_seed": {}}
    for s in SEEDS:
        e = out["rho"]["alpha0.0"][str(s)]
        raw = out["raw"]["alpha0.0"][str(s)]
        row = {}
        for c in ("TRT", "TNT", "TQT"):
            h = e[c]["hit"]
            if "rho" not in h or h.get("status") == "UNRESOLVABLE_TRIP":
                row[c] = {"fails_to_carry_hit": "UNRESOLVABLE_TRIP"
                          if raw[c]["tripped"] else "UNREADABLE"}
            else:
                row[c] = {"rho_hit": h["rho"], "fails_to_carry_hit": h["rho"] <= 0.25}
        tqt, tpt = e["TQT"]["hit"], e["TPT"]["hit"]
        row["TQT_le_TPT_on_hit"] = (tqt["rho"] <= tpt["rho"]
                                    if "rho" in tqt and "rho" in tpt
                                    and tqt.get("status") != "UNRESOLVABLE_TRIP"
                                    else "UNRESOLVABLE")
        hc2["per_seed"][str(s)] = row
    out["hypotheses"]["H_C2_sharpening_alignment"] = hc2

    # -------------------------------------------- 8/11 vs 9/10 split (R4)
    for arm in ARMS:
        prim = PRIMARIES[arm]
        splits = {}
        for cid in ALL_CELLS:
            pat = {str(s): out["classification"][arm][cid]["per_seed"][str(s)]
                   for s in SEEDS}
            grp = {mk: {"8_11": sorted({pat["8"][mk], pat["11"][mk]}),
                        "9_10": sorted({pat["9"][mk], pat["10"][mk]})}
                   for mk in prim}
            diff = {mk: g for mk, g in grp.items()
                    if set(g["8_11"]) != set(g["9_10"])}
            if diff:
                splits[cid] = diff
        out["split_8_11_vs_9_10"][arm] = splits

    # ----------------------------------------- deeper summaries (4.1-4.3)
    orig_t0 = json.loads(Path(
        "/home/vishnu/neuroips_analysis/transplant_20260818/t0_partition_diff.json"
    ).read_text())
    d41 = {}
    for s in SEEDS:
        o = orig_t0["seeds"][str(s)]["arms"]["alpha0.0"]["tensors"]["W_fb.weight"]
        n = n6["deltas"][f"seed{s}_alpha0.0"]["W_fb.weight"]
        d41[str(s)] = {"rel_dfb_s0p04": n["rel"],
                       "rel_dfb_original": o["fro_diff"] / o["fro_pre"],
                       "smaller_at_s0p04": n["rel"] < o["fro_diff"] / o["fro_pre"]}
    out["deeper"]["Q41_alpha0p0_rel_dfb_vs_original"] = d41
    out["deeper"]["fb_geometry_summary"] = {
        k: {kk: v[kk] for kk in ("row_cos_median", "whole_matrix_inner",
                                 "E_proj_delta_fb_on_delta_hh_V5")}
        for k, v in n6["fb_geometry"].items()}
    out["deeper"]["e5_delta_hh"] = {
        k: v["SVD_delta_hh"]["e5_top5_energy"] for k, v in n6["deltas"].items()}
    d43 = {}
    for s in SEEDS:
        for arm in ARMS:
            g = n6["gains_table"][f"seed{s}_{arm}"]
            d43[f"seed{s}_{arm}"] = {
                "k": g["k"], "k_original": g["k_original_no_surround"],
                "abs_k_smaller_than_original":
                    abs(g["k"]) < abs(g["k_original_no_surround"]),
                "som_margin": g["som_margin"], "k_pretrain": g["k_pretrain"]}
    out["deeper"]["Q43_k_vs_original"] = d43

    # ------------------------------------- 4.6 prediction confrontation
    pred = {}
    cls0 = out["classification"]["alpha0.0"]
    cls5 = out["classification"]["alpha0.5"]

    def all_F(arm, cid, mks, seeds=SEEDS):
        e = out["classification"][arm][cid]["per_seed"]
        return all(e[str(s)][mk] == "F" for s in seeds for mk in mks)

    pred["sharpening_full_carry_TTT_only"] = {
        "TTT_carries": cls0["TTT"]["verdict"].startswith("CARRIES"),
        "any_other_carries": [c for c in ALL_CELLS if c != "TTT"
                              and cls0[c]["verdict"].startswith("CARRIES")]}
    tpt_hits = {str(s): out["rho"]["alpha0.0"][str(s)]["TPT"]["hit"] for s in SEEDS}
    pred["sharpening_TPT_hit_partial_0p4_0p5"] = {
        s: (v["rho"] if "rho" in v else v["status"]) for s, v in tpt_hits.items()}
    pred["sharpening_TTP_flank_partial_to_F"] = {
        str(s): out["rho"]["alpha0.0"][str(s)]["TTP"]["flank"] for s in SEEDS}
    pred["dampening_carry_TPT_TNT_TQT"] = {
        c: cls5[c]["verdict"] for c in ("TPT", "TNT", "TQT")}
    pred["dampening_TRT_partial"] = cls5["TRT"]["verdict"]
    pred["dampening_TTP_overshoot"] = q["Q2_dampening_gains_lock"]
    pred["fb_geometry_labeled"] = {
        "alpha0.0_low_row_cos_predicted": {
            k: v["row_cos_median"] for k, v in n6["fb_geometry"].items()
            if "alpha0.0" in k},
        "alpha0.0_E_proj_above_null_predicted": {
            k: v["E_proj_delta_fb_on_delta_hh_V5"]
            for k, v in n6["fb_geometry"].items() if "alpha0.0" in k},
        "alpha0.5_high_row_cos_predicted": {
            k: v["row_cos_median"] for k, v in n6["fb_geometry"].items()
            if "alpha0.5" in k},
        "alpha0.5_E_proj_near_null_predicted": {
            k: v["E_proj_delta_fb_on_delta_hh_V5"]
            for k, v in n6["fb_geometry"].items() if "alpha0.5" in k},
        "null": 5.0 / 64.0}
    out["prediction_confrontation"] = pred

    (ROOT / "n7_synth.json").write_text(json.dumps(out, indent=1, sort_keys=True,
                                                   default=str))
    print("n7_synth.json written; G6 re-verify PASS")
    C.heartbeat("N7 synthesis written (n7_synth.json); G6 re-verify PASS "
                "(48/48 donor files unchanged)")


if __name__ == "__main__":
    main()
