"""Task #7 Part B — Merge 20k Dec A' column into the 17-row cross-decoder matrix.

Reads:
  - results/cross_decoder_comprehensive_with_all_decoders.json  (existing 7-col matrix; A, A'(5k R1+R2 only), B, C, D-raw, D-shape, E)
  - /tmp/task7_decAprime20k/r1r2_paradigm.json    (Δ_A'(20k) for R1+R2 rows 1-4 via patched ckpt)
  - /tmp/task7_decAprime20k/xdec_native.json      (Δ_A'(20k) for R1+R2 rows 9-14 via patched ckpt)
  - /tmp/task7_decAprime20k/xdec_modified.json    (Δ_A'(20k) for R1+R2 rows 15-17 via patched ckpt)
  - /tmp/task7_decAprime20k/legacy/{a1,b1,c1,e1}_C1.json  (Δ_A'(20k) for legacy rows 5-8 via patched ckpts)

Writes (per Task #7 brief):
  - results/cross_decoder_comprehensive_20k_final.json
  - results/cross_decoder_comprehensive_20k_final.md

Design choice (per Task #7 brief: "REPLACING the existing Dec A' column ... or adding new column ... your call"):
The existing matrix's `decA_prime_delta` column is R1+R2-only, 5k Adam. We REPLACE it with the new
20k Dec A' values that cover ALL 5 nets (incl. legacy a1/b1/c1/e1 — covered for the first time at 20k).
The 5k values are archived as `decA_prime_5k_delta` for reference. The new column is named
`decA_prime_20k_delta` and is populated for all 17 rows.

Sign-flip analysis: per-row sign(Δ_A) vs sign(Δ_A'_20k), with the rows-5/6 (a1/b1 HMM C1) flag
highlighted — does the 5k Dec A' positive sign on rows 5/6 PERSIST at 20k or COLLAPSE to negative
agreement with Dec A?
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


def _sign(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return 1 if x > 0 else (-1 if x < 0 else 0)


def _glyph(s):
    return {1: "+", -1: "−", 0: "0", None: "?"}.get(s, "?")


def _load_paradigm_decAp20k(path: str) -> dict[tuple[str, str], float]:
    """Read Δ_A'(20k) from `decA_acc_mean` ex/unex in a ckpt-patched paradigm JSON."""
    if not Path(path).exists():
        return {}
    label_map = {
        "C1_focused_native": "HMM C1 (focused + HMM cue)",
        "C2_routine_native": "HMM C2 (routine + HMM cue)",
        "C3_focused_neutralcue": "HMM C3 (focused + zero cue)",
        "C4_routine_neutralcue": "HMM C4 (routine + zero cue)",
    }
    with open(path) as f:
        d = json.load(f)
    out: dict[tuple[str, str], float] = {}
    for c in d.get("conditions", []):
        cid = c["id"]
        assay = label_map.get(cid, cid)
        ex = c["branches"]["ex"]
        ux = c["branches"]["unex"]
        if ex.get("decA_acc_mean") is None or ux.get("decA_acc_mean") is None:
            continue
        out[(assay, "R1+R2 (emergent_seed42)")] = float(
            ex["decA_acc_mean"] - ux["decA_acc_mean"])
    return out


def _load_xdec_decAp20k(path: str, modified: bool) -> dict[tuple[str, str], float]:
    """Read Δ_A'(20k) from `decA_delta` in a ckpt-patched cross_decoder_eval JSON."""
    if not Path(path).exists():
        return {}
    with open(path) as f:
        d = json.load(f)
    out: dict[tuple[str, str], float] = {}
    suffix = " (modified: focused+march cue)" if modified else ""
    name_map = {
        "NEW": "NEW (paired march, eval_ex_vs_unex_decC)",
        "M3R": "M3R (matched_3row_ring)",
        "HMS": "HMS (matched_hmm_ring_sequence)",
        "HMS-T": "HMS-T (matched_hmm_ring_sequence --tight-expected)",
        "P3P": "P3P (matched_probe_3pass)",
        "VCD": "VCD-test3 (v2_confidence_dissection)",
    }
    for k, r in d.get("results", {}).items():
        if r.get("decA_delta") is None:
            continue
        assay = name_map.get(k, k) + suffix
        out[(assay, "R1+R2 (emergent_seed42)")] = float(r["decA_delta"])
    return out


def _load_legacy_decAp20k(legacy_dir: str) -> dict[tuple[str, str], float]:
    """Read Δ_A'(20k) from per-legacy-net HMM C1 paradigm JSONs (patched ckpts)."""
    out: dict[tuple[str, str], float] = {}
    for sw in ("a1", "b1", "c1", "e1"):
        p = os.path.join(legacy_dir, f"{sw}_C1.json")
        if not Path(p).exists():
            continue
        with open(p) as f:
            d = json.load(f)
        conds = d.get("conditions", [])
        if not conds:
            continue
        c = conds[0]
        ex = c["branches"]["ex"]
        ux = c["branches"]["unex"]
        if ex.get("decA_acc_mean") is None or ux.get("decA_acc_mean") is None:
            continue
        out[("HMM C1 (focused + HMM cue)", f"{sw} (legacy three-regimes)")] = float(
            ex["decA_acc_mean"] - ux["decA_acc_mean"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-matrix",
                    default="results/cross_decoder_comprehensive_with_all_decoders.json")
    ap.add_argument("--paradigm-r1r2",
                    default="/tmp/task7_decAprime20k/r1r2_paradigm.json")
    ap.add_argument("--xdec-native",
                    default="/tmp/task7_decAprime20k/xdec_native.json")
    ap.add_argument("--xdec-modified",
                    default="/tmp/task7_decAprime20k/xdec_modified.json")
    ap.add_argument("--legacy-dir",
                    default="/tmp/task7_decAprime20k/legacy")
    ap.add_argument("--output-json",
                    default="results/cross_decoder_comprehensive_20k_final.json")
    ap.add_argument("--output-md",
                    default="results/cross_decoder_comprehensive_20k_final.md")
    args = ap.parse_args()

    print(f"[load] base matrix: {args.base_matrix}", flush=True)
    with open(args.base_matrix) as f:
        base = json.load(f)
    print(f"[load] base has {len(base['rows'])} rows", flush=True)

    decAp20k_by_key: dict[tuple[str, str], float] = {}
    decAp20k_by_key.update(_load_paradigm_decAp20k(args.paradigm_r1r2))
    decAp20k_by_key.update(_load_xdec_decAp20k(args.xdec_native, modified=False))
    decAp20k_by_key.update(_load_xdec_decAp20k(args.xdec_modified, modified=True))
    decAp20k_by_key.update(_load_legacy_decAp20k(args.legacy_dir))

    print(f"[load] Δ_A'(20k) populated for {len(decAp20k_by_key)} rows", flush=True)

    merged_rows = []
    signflips_A_vs_Ap20k: list[dict] = []
    persistent_signs_a1b1: dict = {}
    for r in base["rows"]:
        key = (r["assay"], r["network"])
        dA = r.get("decA_delta")
        dAp_5k = r.get("decA_prime_delta")  # archived; only R1+R2 rows in original
        dAp_20k = decAp20k_by_key.get(key)

        newrow = dict(r)
        # Archive 5k under explicit name
        newrow["decA_prime_5k_delta"] = dAp_5k
        newrow["decA_prime_20k_delta"] = dAp_20k
        # Replace primary decA_prime column with 20k (NEW canonical, all 5 nets)
        newrow["decA_prime_delta"] = dAp_20k

        merged_rows.append(newrow)

        # Sign-flip analysis: Δ_A vs Δ_A'(20k) per row
        if dA is not None and dAp_20k is not None:
            if _sign(dA) is not None and _sign(dAp_20k) is not None and _sign(dA) != _sign(dAp_20k):
                signflips_A_vs_Ap20k.append({
                    "row_idx": len(merged_rows),
                    "assay": r["assay"],
                    "network": r["network"],
                    "dA": dA,
                    "dAp_20k": dAp_20k,
                    "dAp_5k": dAp_5k,
                })

        # Headline question: rows 5/6 (a1/b1 HMM C1) — does the 5k Dec A' sign persist at 20k?
        if r["assay"] == "HMM C1 (focused + HMM cue)" and r["network"] in ("a1 (legacy three-regimes)", "b1 (legacy three-regimes)"):
            persistent_signs_a1b1[r["network"]] = {
                "dA":      {"value": dA, "sign": _glyph(_sign(dA))},
                "dAp_5k":  {"value": dAp_5k, "sign": _glyph(_sign(dAp_5k))},
                "dAp_20k": {"value": dAp_20k, "sign": _glyph(_sign(dAp_20k))},
                "sign_collapse_5k_to_20k": (
                    None if (dAp_5k is None or dAp_20k is None) else
                    "Yes (5k -> 20k changed sign)" if _sign(dAp_5k) != _sign(dAp_20k) else
                    "No (5k -> 20k same sign)"
                ),
                "matches_dA_at_20k": (
                    None if (dA is None or dAp_20k is None) else
                    "Yes" if _sign(dA) == _sign(dAp_20k) else "No"
                ),
            }

    # Per-decoder profile (all 17 rows; updated with 20k Dec A')
    def _mag_n(rows, field):
        vals = [r.get(field) for r in rows if r.get(field) is not None
                and not (isinstance(r.get(field), float) and math.isnan(r.get(field)))]
        if not vals:
            return None, None, 0
        return (
            float(sum(abs(v) for v in vals) / len(vals)),
            float(max(abs(v) for v in vals)),
            len(vals),
        )

    pdp = {}
    for dec, field in [
        ("A",          "decA_delta"),
        ("A_prime_5k", "decA_prime_5k_delta"),
        ("A_prime_20k", "decA_prime_20k_delta"),
        ("B",          "decB_delta"),
        ("C",          "decC_delta"),
        ("D_raw",      "decD_raw_delta"),
        ("D_shape",    "decD_shape_delta"),
        ("E",          "decE_delta"),
    ]:
        m, x, n = _mag_n(merged_rows, field)
        # ABC majority sign agreement (only meaningful for A/B/C decoders since ABC majority is defined over them)
        ag = None
        if field != "decA_delta":
            ag_count = 0
            n_in_match = 0
            for r in merged_rows:
                v = r.get(field)
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    continue
                # ABC majority sign (using original A, B, C)
                signs_abc = [_sign(r.get(c)) for c in ("decA_delta", "decB_delta", "decC_delta")]
                signs_abc = [s for s in signs_abc if s is not None]
                if not signs_abc:
                    continue
                # Majority sign
                pos = signs_abc.count(1)
                neg = signs_abc.count(-1)
                if pos == neg:
                    continue  # no clear majority
                maj = 1 if pos > neg else -1
                n_in_match += 1
                if _sign(v) == maj:
                    ag_count += 1
            if n_in_match > 0:
                ag = f"{ag_count}/{n_in_match}"
        pdp[dec] = {
            "n_rows_with_delta": n,
            "mean_abs_delta": m,
            "max_abs_delta":  x,
            "agrees_with_ABC_majority": ag,
        }

    # ALL-agree (A/B/C) row count for reference
    all_agree_count = 0
    for r in merged_rows:
        signs = [_sign(r.get(c)) for c in ("decA_delta", "decB_delta", "decC_delta")]
        if None in signs:
            continue
        if signs[0] == signs[1] == signs[2]:
            all_agree_count += 1

    out = {
        "task": "Task #7 Part B — 17-row cross-decoder matrix with 20k Dec A' (all 5 nets)",
        "design_choice": (
            "Replaced existing decA_prime_delta column (R1+R2-only, 5k Adam) with new "
            "decA_prime_20k_delta (all 5 nets, 20k Adam). Old 5k values archived as "
            "decA_prime_5k_delta. The new 20k Dec A' was trained per-net on each frozen "
            "fully-trained network's r_l23 with same Adam lr=1e-3, seed=42, batch 32, "
            "seq 25, 50/50 task_state — only --n-steps 5000 -> 20000."
        ),
        "sources": {
            "base_matrix": args.base_matrix,
            "paradigm_r1r2_decAp20k": args.paradigm_r1r2,
            "xdec_native_decAp20k": args.xdec_native,
            "xdec_modified_decAp20k": args.xdec_modified,
            "legacy_decAp20k_dir": args.legacy_dir,
        },
        "n_rows": len(merged_rows),
        "rows": merged_rows,
        "per_decoder_profile": pdp,
        "all_three_decoders_agree_count_ABC": all_agree_count,
        "signflips_A_vs_Aprime_20k": signflips_A_vs_Ap20k,
        "rows_5_6_a1b1_HMM_C1_persistence_test": persistent_signs_a1b1,
    }
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[json] wrote {args.output_json}", flush=True)

    # Markdown
    lines = []
    lines.append("# 17-row cross-decoder matrix with 20k Dec A' (Task #7 Part B)\n")
    lines.append("**Design choice.** This matrix REPLACES the previous Dec A' column (R1+R2-only, "
                 "5k Adam) with a new column trained at 20 000 Adam steps per-net across all 5 "
                 "nets (r1r2 + a1/b1/c1/e1). The 5k Dec A' values are archived as "
                 "`decA_prime_5k_delta` for reference. The 20k Dec A' is trained with the EXACT "
                 "same protocol as 5k (lr 1e-3 Adam, seed 42, batch 32, seq 25, 50/50 task_state) "
                 "— only `--n-steps 5000 → 20000`. Per-net 10k HMM stratified top-1 (Task #2 "
                 "convention): r1r2 0.5729, a1 0.6709, b1 0.6625, c1 (Task #7 Part A new), e1 "
                 "(Task #7 Part A new).\n")
    lines.append("**Sign-flip analysis is the headline question.** Does the 5k Dec A' positive "
                 "sign on rows 5/6 (a1/b1 HMM C1) — which was the canonical evidence for the "
                 "now-retracted 'Dec A vs Dec E dissociation' — PERSIST when Dec A' is properly "
                 "optimised at 20k steps, or does it COLLAPSE to agreement with Dec A's negative "
                 "sign? See § 'Rows 5/6 persistence test' below for the answer.\n")

    lines.append("| # | Assay | Network | n_ex | n_unex | Δ_A | Δ_A'(20k) | Δ_A'(5k) | Δ_B | Δ_C | Δ_D-raw | Δ_D-shape | Δ_E | ABC sign-agreement |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|")

    def f(x, nd=4):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return " — "
        return f"{x:+.{nd}f}"

    for i, r in enumerate(merged_rows, 1):
        sa = r.get("sign_agreement", {})
        if sa.get("all_agree") is True:
            sg = f"ALL ({_glyph(sa.get('majority'))})"
        elif sa.get("outlier") is not None:
            sg = f"{sa['outlier']}-out"
        else:
            sg = "mixed"
        lines.append(
            f"| {i} | {r['assay']} | {r['network']} | "
            f"{r['n_ex'] if r['n_ex'] is not None else '—'} | "
            f"{r['n_unex'] if r['n_unex'] is not None else '—'} | "
            f"{f(r.get('decA_delta'))} | {f(r.get('decA_prime_20k_delta'))} | "
            f"{f(r.get('decA_prime_5k_delta'))} | "
            f"{f(r.get('decB_delta'))} | {f(r.get('decC_delta'))} | "
            f"{f(r.get('decD_raw_delta'))} | {f(r.get('decD_shape_delta'))} | "
            f"{f(r.get('decE_delta'))} | {sg} |"
        )

    lines.append("\n## Rows 5/6 persistence test — does 5k Dec A' positive sign on a1/b1 HMM C1 collapse to negative at 20k?\n")
    for net, info in persistent_signs_a1b1.items():
        lines.append(f"### {net}\n")
        lines.append(f"- Δ_A (Dec A original):       {f(info['dA']['value'])} ({info['dA']['sign']})")
        lines.append(f"- Δ_A' (5k Adam, original):   {f(info['dAp_5k']['value'])} ({info['dAp_5k']['sign']})")
        lines.append(f"- Δ_A' (20k Adam, NEW):       {f(info['dAp_20k']['value'])} ({info['dAp_20k']['sign']})")
        lines.append(f"- Sign collapse 5k → 20k:     {info['sign_collapse_5k_to_20k']}")
        lines.append(f"- 20k matches Dec A sign:     {info['matches_dA_at_20k']}\n")

    lines.append("\n## Per-decoder magnitude + ABC-majority agreement profile (all 17 rows)\n")
    lines.append("| Decoder | n rows with Δ | mean \\|Δ\\| | max \\|Δ\\| | agrees with ABC majority |")
    lines.append("|---|---:|---:|---:|---|")
    for dec, p in pdp.items():
        ma = f(p["mean_abs_delta"]) if p["mean_abs_delta"] is not None else "—"
        mx = f(p["max_abs_delta"]) if p["max_abs_delta"] is not None else "—"
        ag = p["agrees_with_ABC_majority"] or "—"
        lines.append(f"| {dec} | {p['n_rows_with_delta']} | {ma} | {mx} | {ag} |")

    lines.append(f"\n**ABC ALL-agree row count:** {all_agree_count}/17.\n")

    lines.append("\n## Sign flips: Dec A vs 20k Dec A' (per-row)\n")
    if signflips_A_vs_Ap20k:
        lines.append("| Row | Assay | Network | Δ_A | Δ_A'(20k) | Δ_A'(5k) |")
        lines.append("|---:|---|---|---:|---:|---:|")
        for fl in signflips_A_vs_Ap20k:
            lines.append(
                f"| {fl['row_idx']} | {fl['assay']} | {fl['network']} | "
                f"{f(fl['dA'])} | {f(fl['dAp_20k'])} | {f(fl['dAp_5k'])} |"
            )
    else:
        lines.append("- None. 20k Dec A' agrees in sign with Dec A on all 17 rows.")

    with open(args.output_md, "w") as f_:
        f_.write("\n".join(lines) + "\n")
    print(f"[md]   wrote {args.output_md}", flush=True)

    print()
    print(f"=== summary ===")
    print(f"Δ_A'(20k) populated on {sum(1 for r in merged_rows if r.get('decA_prime_20k_delta') is not None)} of {len(merged_rows)} rows")
    print(f"Sign flips Dec A vs Dec A'(20k): {len(signflips_A_vs_Ap20k)}")
    if persistent_signs_a1b1:
        print(f"\nRows 5/6 persistence test:")
        for net, info in persistent_signs_a1b1.items():
            print(f"  {net}: Δ_A {info['dA']['sign']}  Δ_A'(5k) {info['dAp_5k']['sign']}  Δ_A'(20k) {info['dAp_20k']['sign']}  -> {info['matches_dA_at_20k']} matches Dec A at 20k")


if __name__ == "__main__":
    main()
