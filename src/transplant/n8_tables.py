#!/usr/bin/env python3
"""N8 — render TABLES.md (readable results) from n4/n5/n6/n7 JSON. No compute."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ncommon as C  # noqa: E402

ROOT = C.ROOT
SEEDS = ("8", "9", "10", "11")
ARMS = ("alpha0.0", "alpha0.5")
CELLS = ["PPP", "TPP", "PTP", "PPT", "TTP", "TPT", "PTT", "TTT",
         "TRT", "TNT", "TQT"]
HOST_CTRLS = ["PRP", "PNP", "PQP"]
RAW_COLS = ("hit", "decode", "rate", "M", "center", "flank")


def fmt(v, nd=4):
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def rho_str(e):
    if e.get("status") == "UNREADABLE":
        return "unrd"
    s = f"{e['rho']:+.3f}"
    if e.get("status") == "UNRESOLVABLE_TRIP":
        s += "*"
    return s


def main() -> None:
    n4 = json.loads((ROOT / "n4_assay.json").read_text())
    n5 = json.loads((ROOT / "n5_s0.json").read_text())
    n7 = json.loads((ROOT / "n7_synth.json").read_text())
    L = []
    A = L.append

    A("# Transplant-surround Phase 2 — results tables (generated from "
      "n4/n5/n6/n7 JSON)")
    A("")
    A("Cell ID = (CELL, FB, GAINS); P = pretrain, T = trained, R/N/Q = fresh "
      "FB controls (random / norm-matched random / rotated-trained). "
      "`*` = CE-tripped seed (competence coords UNRESOLVABLE, house rule). "
      "`unrd` = coordinate UNREADABLE (|TTT-PPP| below its floor).")
    A("")
    A("Gate chain: G6 PASS + re-verified after measurement (48/48 files "
      "unchanged); G1 8/8, G2 4/4, G0 exact (s8 abs diff 0.0 vs sha-pinned "
      "artifacts), G3 4/4, G5 all, control gates + null-edit 8/8, determinism "
      "repeat exact, EC1 4/4, shared-cell dual-file bitwise 5/5.")
    A("")

    # ---------------------------------------------------------- raw tables
    for arm in ARMS:
        A(f"## Raw markers — {arm} "
          f"({'sharpening' if arm == 'alpha0.0' else 'dampening'})")
        A("")
        for s in SEEDS:
            raw = n7["raw"][arm][s]
            A(f"### seed {s}")
            A("")
            A("| cell | hit | decode | rate | M | center | flank | trip |")
            A("|---|---|---|---|---|---|---|---|")
            for c in CELLS + (HOST_CTRLS if s == "8" else []):
                r = raw[c]
                A("| " + " | ".join(
                    [c] + [fmt(r[k]) for k in RAW_COLS]
                    + ["TRIP" if r["tripped"] else ""]) + " |")
            A("")

    # ---------------------------------------------------------- rho tables
    for arm in ARMS:
        prim = {"alpha0.0": ("center", "flank", "hit"),
                "alpha0.5": ("M", "center")}[arm]
        mks = list(n7["rho"][arm]["8"]["TTT"].keys())
        mks = [m for m in ("hit", "decode", "rate", "M", "center", "flank")
               if m in mks]
        A(f"## rho — {arm} (primaries: {', '.join(prim)}"
          + ("; rate raw-only per 3.2)" if arm == "alpha0.5" else ")"))
        A("")
        for s in SEEDS:
            dens = n7["denominators"][arm][s]
            A(f"### seed {s} — denominators: "
              + ", ".join(f"{m} {dens[m]['den']:+.4f}"
                          + ("" if dens[m]["readable"] else " (FLOORED)")
                          for m in mks))
            A("")
            A("| cell | " + " | ".join(f"rho_{m}" for m in mks) + " |")
            A("|" + "---|" * (len(mks) + 1))
            for c in CELLS + (HOST_CTRLS if s == "8" else []):
                e = n7["rho"][arm][s][c]
                A("| " + " | ".join([c] + [rho_str(e[m]) for m in mks]) + " |")
            A("")

    # ------------------------------------------------------ classification
    A("## Pre-registered classification (3.2/3.4 verbatim rules)")
    A("")
    for arm in ARMS:
        A(f"### {arm}")
        A("")
        A("| cell | verdict |")
        A("|---|---|")
        for c in CELLS + HOST_CTRLS:
            A(f"| {c} | {n7['strategy_map'][arm][c]} |")
        A("")

    # ------------------------------------------------------ prediction map
    A("## 4.6 predicted strategy map — confrontation")
    A("")
    p = n7["prediction_confrontation"]
    A("| prediction | outcome |")
    A("|---|---|")
    A(f"| sharpening: full carry TTT only | TTT carries: "
      f"{fmt(p['sharpening_full_carry_TTT_only']['TTT_carries'])}; other "
      f"carries: {p['sharpening_full_carry_TTT_only']['any_other_carries'] or 'none'}"
      " (flank unreadable 4/4 -> carry read on center+hit per 3.3 fallback) |")
    A(f"| sharpening: TPT hit partial 0.4-0.5 | rho_hit "
      + ", ".join(f"s{s} {fmt(p['sharpening_TPT_hit_partial_0p4_0p5'][s], 3)}"
                  for s in SEEDS) + " — HIT (partial 4/4, band 0.36-0.54) |")
    A("| sharpening: TTP partial-F on FLANK | rho_flank UNREADABLE 4/4 "
      "(floored); descriptively: raw flank TTP 0.754-0.814 <= TTT 0.824-0.828 "
      "on 3/4 seeds, s->0 delta -0.150 (kernel active) — descriptive HIT, "
      "rho-level UNREADABLE |")
    A(f"| dampening: carry TPT | {p['dampening_carry_TPT_TNT_TQT']['TPT']} — HIT |")
    A(f"| dampening: carry TNT (control) | {p['dampening_carry_TPT_TNT_TQT']['TNT']}"
      " — MISS (partial, not carry) |")
    A(f"| dampening: carry TQT (control) | {p['dampening_carry_TPT_TNT_TQT']['TQT']}"
      " — MISS (partial, not carry) |")
    A(f"| dampening: TRT partial | {p['dampening_TRT_partial']} — HIT |")
    A("| dampening: TTP overshoots | "
      + ", ".join(f"s{s} {fmt(p['dampening_TTP_overshoot'][s]['overshoots_above_host'])}"
                  for s in SEEDS) + " — HIT 4/4 |")
    fg = p["fb_geometry_labeled"]
    A("| FB geometry: a0.0 low row-cos | measured 0.873-0.883 — MISS "
      "(direction largely kept; original A2-c said rewritten) |")
    A(f"| FB geometry: a0.0 E_proj above null ({fmt(fg['null'], 3)}) | measured "
      + ", ".join(f"{fmt(v, 3)}" for k, v in
                  sorted(fg["alpha0.0_E_proj_above_null_predicted"].items()))
      + " — MISS (below null) |")
    A("| FB geometry: a0.5 high row-cos | measured 0.978-0.985 — HIT |")
    A(f"| FB geometry: a0.5 E_proj near null | measured "
      + ", ".join(f"{fmt(v, 3)}" for k, v in
                  sorted(fg["alpha0.5_E_proj_near_null_predicted"].items()))
      + " — MISS (3-4x above null) |")
    A("")

    # ------------------------------------------------------ questions
    A("## Registered questions (4.4) and control hypotheses (4.5)")
    A("")
    q = n7["questions"]
    A("**Q1 (FB-alone flank at s=0.04):** rho_flank UNREADABLE 4/4 (a0.0 "
      "denominators +0.0176/+0.0161/+0.0106/+0.0019, all < 0.05 floor; host "
      "flank already at TTT level — see U1). Descriptive: PTP raw flank "
      "0.979-1.068 (ABOVE baseline — FB-alone produces no flank suppression); "
      "TTP raw flank 0.754-0.814 (at-or-below TTT); s->0 seed 8: flank(s)-"
      "flank(0) = -0.150 (TTP), -0.145 (TTT), -0.063 (PTP), -0.092 (PPP) — "
      "the surround path does the flank work when f is well-placed. Placement: "
      "hit stays 0_below_baseline for PTP and TTP 4/4. Verdict: flank "
      "suppression is more transplantable than placement (prediction HIT "
      "descriptively; rho-level unreadable).")
    A("")
    A("**Q2 (dampening GAINS lock):** " + "; ".join(
        f"s{s} rho_M(TTP) {rho_str(q['Q2_dampening_gains_lock'][s]['rho_M_TTP'])}, "
        f"M {fmt(q['Q2_dampening_gains_lock'][s]['raw_M_TTP'], 3)} vs host "
        f"{fmt(q['Q2_dampening_gains_lock'][s]['raw_M_PPP'], 3)}"
        for s in SEEDS) + ". Overshoot repeats 4/4 — prediction HIT; "
      "dampening remains GAINS-locked.")
    A("")
    A("**Q3 (softmax temperature, R vs N):** prediction was TNT carries "
      "(~TPT) and TRT <= TNT. Measured: TNT partial (rho_M 0.474-0.538), and "
      "TRT ABOVE TNT on both primaries 4/4 (e.g. s8 rho_M 0.561 vs 0.478) — "
      "both clauses MISS. The magnitude-preserving qualifier is NOT what "
      "separates controls from TPT; no control reaches carry.")
    A("")
    c4 = q["Q4_trip_census"]
    A(f"**Q4 (CE-trip census):** a0.5 PPT trips {c4['repeat_alpha0.5_PPT']} "
      f"(4/4, repeat); a0.5 PTT trips {c4['repeat_alpha0.5_PTT']} (3/4 — s8 "
      "now untripped, weaker than original 4/4). NEW trips outside the "
      f"original class: {', '.join(c4['new_trips_not_in_original_class'])} "
      "(a0.0 control/FB-GAINS chimeras trip at s=0.04). Prediction partially "
      "HIT (same class repeats) with new a0.0 fragility.")
    A("")
    h1 = n7["hypotheses"]["H_C1_dampening_genericity"]
    A(f"**H-C1 (dampening genericity): REFUTED** (confirmed={h1['confirmed']}). "
      "No control carries on any seed; all three are partial on both "
      "primaries 4/4 (rho_M 0.459-0.611, rho_center 0.537-0.635) vs TPT "
      "0.927-1.027. The trained-norm random-direction FB does NOT suffice at "
      "s=0.04; the pretrain FB (task-informative) does. Dampening needs a "
      "meaningful FB direction, not just magnitude.")
    A("")
    A("**H-C2 (sharpening alignment): CONFIRMED on every resolvable seed.** "
      "All controls fail hit; measured rho_hit -0.75..-0.88 (below baseline "
      "— a wrong-direction FB actively destroys placement); TQT <= TPT on "
      "hit on all 3 resolvable seeds (s11 TQT tripped). "
      "(s8/s9 TNT tripped, s11 TQT tripped -> those cells UNRESOLVABLE there.)")
    A("")
    aa = n7["alignment_criticality"]
    A("**Alignment-criticality A_align = rho_primary(TPT) - rho_primary(TQT):** "
      + "; ".join(
          f"s{s} a0.0 {fmt(aa['alpha0.0'][s], 3) if isinstance(aa['alpha0.0'][s], float) else 'unresolvable'}"
          f" vs a0.5 {fmt(aa['alpha0.5'][s], 3)}" for s in SEEDS)
      + f". Sharpening MORE alignment-critical: "
        f"{aa['registered_statement_sharpening_MORE_alignment_critical']}.")
    prem = aa["fb_premium_on_hit_1_minus_rho_hit_TPT"]
    A("")
    A("**FB premium on hit (1 - rho_hit(TPT), a0.0):** "
      + ", ".join(f"s{s} {fmt(prem['measured'][s], 3)}" for s in SEEDS)
      + f" vs original {prem['original_range']} — premium PERSISTS "
        "(prediction HIT); the kernel did not absorb the FB's placement role.")
    A("")
    u1 = n7["U1_host_flank_first_check"]
    A("**U1 (host flank first check): prediction MISSED 4/4** — PPP "
      "flank_ratio " + ", ".join(f"s{s} {fmt(u1[s]['PPP_flank_ratio'], 4)}"
                                 for s in SEEDS)
      + " (predicted band 0.85-0.97). The pretrained host already sits at "
        "TTT-level flank suppression (0.824-0.828) => the a0.0 flank "
        "denominator floors on every seed (R1 materialized; registered "
        "fallback applied).")
    A("")

    # ------------------------------------------------------ split
    A("## 8/11 vs 9/10 split (R4)")
    A("")
    sp = n7["split_8_11_vs_9_10"]
    if not any(sp[a] for a in ARMS):
        A("No primary-coordinate band differences between {8,11} and {9,10}.")
    else:
        for arm in ARMS:
            for cid, diff in sp[arm].items():
                for mk, g in diff.items():
                    A(f"- {arm} {cid} {mk}: 8/11 bands {g['8_11']} vs 9/10 "
                      f"{g['9_10']}")
        A("")
        A("All splits are band-edge wobbles at 0/0_below_baseline or "
          "F/partial boundaries plus one trip asymmetry (a0.0 TQT s11); no "
          "systematic in-band (8,11) vs sub-band (9,10) divergence on any "
          "primary. Reported per R4, not averaged away.")
    A("")

    # ------------------------------------------------------ deeper
    A("## Deeper analyses (4.1-4.3)")
    A("")
    A("### 4.1 registered question — a0.0 relative ||Delta_fb|| vs original")
    A("")
    A("| seed | rel s=0.04 | rel original | smaller? |")
    A("|---|---|---|---|")
    for s in SEEDS:
        v = n7["deeper"]["Q41_alpha0p0_rel_dfb_vs_original"][s]
        A(f"| {s} | {fmt(v['rel_dfb_s0p04'], 3)} | "
          f"{fmt(v['rel_dfb_original'], 3)} | {fmt(v['smaller_at_s0p04'])} |")
    A("")
    A("No decrease (3/4 marginally larger, s8 marginally smaller) — the "
      "kernel did NOT absorb part of the FB rewrite (labeled prediction "
      "'modest decrease or no change': lands on 'no change').")
    A("")
    A("### 4.2 FB geometry")
    A("")
    A("| regime x seed | row-cos median | whole-matrix inner | E_proj "
      "(null 0.078) | e5(Delta_hh) |")
    A("|---|---|---|---|---|")
    for k in sorted(n7["deeper"]["fb_geometry_summary"]):
        v = n7["deeper"]["fb_geometry_summary"][k]
        A(f"| {k} | {fmt(v['row_cos_median'], 3)} | "
          f"{fmt(v['whole_matrix_inner'], 3)} | "
          f"{fmt(v['E_proj_delta_fb_on_delta_hh_V5'], 3)} | "
          f"{fmt(n7['deeper']['e5_delta_hh'][k], 3)} |")
    A("")
    A("Original e5(Delta_hh) targets reproduce (a0.5 ~0.80, a0.0 ~0.37). "
      "E_proj INVERTS the labeled prediction: the a0.5 FB micro-adjustment "
      "reads the Delta_hh top-5 subspace (0.25-0.32 >> null) while the large "
      "a0.0 FB rewrite is spread (0.047-0.054, below null).")
    A("")
    A("### 4.3 gains/k")
    A("")
    A("| regime x seed | k | k original (no-surround) | |k| smaller? | "
      "som_margin | k pretrain |")
    A("|---|---|---|---|---|---|")
    for k in sorted(n7["deeper"]["Q43_k_vs_original"]):
        v = n7["deeper"]["Q43_k_vs_original"][k]
        A(f"| {k} | {fmt(v['k'], 4)} | {fmt(v['k_original'], 4)} | "
          f"{fmt(v['abs_k_smaller_than_original'])} | "
          f"{fmt(v['som_margin'], 3)} | {fmt(v['k_pretrain'], 4)} |")
    A("")
    A("Same qualitative family both regimes (small-positive vs deep-negative "
      "k). |k| smaller than original 4/4 in a0.5 (3.26-3.50 vs 3.69-3.77) as "
      "predicted; a0.0 slightly LARGER 4/4 (+0.047..+0.054 vs +0.036..+0.040) "
      "— prediction half-MISS.")
    A("")

    # ------------------------------------------------------ s->0
    A("## s->0 counterfactual (2.4; evidence, never a bar)")
    A("")
    A(f"Selection: registered 10; rule extras (readable rho_flank >= 0.25): "
      f"{[(e['arm'], e['cell']) for e in n5['extras_selected']]}; a0.0 "
      "factorial floored -> full descriptive set (9 cells) per 3.3.")
    A("")
    for bucket in ("registered", "extras", "extras_floored_descriptive"):
        A(f"### {bucket}")
        A("")
        A("| cell | flank(s) | flank(0) | dflank | center(s) | center(0) | "
          "M(s) | M(0) |")
        A("|---|---|---|---|---|---|---|---|")
        for k in sorted(n5[bucket]):
            v = n5[bucket][k]
            o, z = v["official"], v["s0_counterfactual"]
            A(f"| {k} | {fmt(o['flank_ratio'])} | {fmt(z['flank_ratio'])} | "
              f"{v['delta_s_minus_s0']['flank_ratio']:+.4f} | "
              f"{fmt(o['center_ratio'])} | {fmt(z['center_ratio'])} | "
              f"{fmt(o['M_auc_ratio'])} | {fmt(z['M_auc_ratio'])} |")
        A("")

    # ------------------------------------------------------ trip census
    A("## CE trip census (3.5; threshold 3*ln36 = 10.7506)")
    A("")
    for s in SEEDS:
        A(f"- seed {s}: {', '.join(n4['trip_census'][s]) or 'none'}")
    A("")

    (ROOT / "TABLES.md").write_text("\n".join(L) + "\n")
    print(f"TABLES.md written ({len(L)} lines)")
    C.heartbeat("N8: TABLES.md rendered from n4/n5/n6/n7")


if __name__ == "__main__":
    main()
