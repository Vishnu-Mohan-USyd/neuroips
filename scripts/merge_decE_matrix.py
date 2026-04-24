"""Task #5 — Merge Dec E column into the 17-row cross-decoder matrix.

Reads:
  - results/cross_decoder_comprehensive_withD_fbON.json  (A, B, C, D-raw, D-shape)
  - results/cross_decoder_comprehensive_decAprime.json   (A' for R1+R2 rows)
  - /tmp/task5_paradigm_R1R2_decE.json                   (Δ_E via ckpt patch)
  - /tmp/task5_xdec_native_decE.json                     (Δ_E native xdec)
  - /tmp/task5_xdec_modified_decE.json                   (Δ_E modified xdec)

Writes:
  - results/cross_decoder_comprehensive_with_all_decoders.json  (7 decoder cols)
  - results/cross_decoder_comprehensive_with_all_decoders.md

Rows are keyed by (assay, network). Dec E is populated only for R1+R2 rows
(per Task #5 brief scope); legacy rows have Δ_E = —.
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


def _load_paradigm_decE(path: str) -> dict[tuple[str, str], float]:
    """Read Δ_E from `decA_delta` in a ckpt-patched paradigm JSON."""
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


def _load_xdec_decE(path: str, modified: bool) -> dict[tuple[str, str], float]:
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


def _load_legacy_decE(legacy_dir: str) -> dict[tuple[str, str], float]:
    """Read Δ_E from per-legacy-net HMM C1 paradigm JSONs (patched ckpts)."""
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
    ap.add_argument("--base", default="results/cross_decoder_comprehensive_withD_fbON.json")
    ap.add_argument("--decaprime", default="results/cross_decoder_comprehensive_decAprime.json")
    ap.add_argument("--dece-paradigm", default="/tmp/task5_paradigm_R1R2_decE.json")
    ap.add_argument("--dece-xdec-native", default="/tmp/task5_xdec_native_decE.json")
    ap.add_argument("--dece-xdec-modified", default="/tmp/task5_xdec_modified_decE.json")
    ap.add_argument("--dece-legacy-dir", default="/tmp/task5_legacy")
    ap.add_argument("--out-json",
                    default="results/cross_decoder_comprehensive_with_all_decoders.json")
    ap.add_argument("--out-md",
                    default="results/cross_decoder_comprehensive_with_all_decoders.md")
    args = ap.parse_args()

    with open(args.base) as f:
        base = json.load(f)
    with open(args.decaprime) as f:
        ap_d = json.load(f)

    ap_by_key = {(r["assay"], r["network"]): r["decA_delta"] for r in ap_d["rows"]
                 if r.get("decA_delta") is not None}

    dece_by_key: dict[tuple[str, str], float] = {}
    dece_by_key.update(_load_paradigm_decE(args.dece_paradigm))
    dece_by_key.update(_load_xdec_decE(args.dece_xdec_native, modified=False))
    dece_by_key.update(_load_xdec_decE(args.dece_xdec_modified, modified=True))
    dece_by_key.update(_load_legacy_decE(args.dece_legacy_dir))

    merged_rows = []
    e_signflips_vs_A: list[dict] = []
    e_signflips_vs_Ap: list[dict] = []
    for r in base["rows"]:
        key = (r["assay"], r["network"])
        dA = r.get("decA_delta")
        dAp = ap_by_key.get(key)   # only populated for R1+R2 rows
        dE = dece_by_key.get(key)  # only populated for R1+R2 rows per Task #5 scope

        newrow = dict(r)
        newrow["decA_prime_delta"] = dAp
        newrow["decE_delta"] = dE
        merged_rows.append(newrow)

        if dE is not None:
            if dA is not None and _sign(dA) is not None and _sign(dE) is not None and _sign(dA) != _sign(dE):
                e_signflips_vs_A.append(
                    {"assay": r["assay"], "network": r["network"], "dA": dA, "dE": dE}
                )
            if dAp is not None and _sign(dAp) is not None and _sign(dE) is not None and _sign(dAp) != _sign(dE):
                e_signflips_vs_Ap.append(
                    {"assay": r["assay"], "network": r["network"], "dAp": dAp, "dE": dE}
                )

    # Per-decoder magnitude summaries
    def _mag(rows, field):
        vals = [r.get(field) for r in rows if r.get(field) is not None
                and not (isinstance(r.get(field), float) and math.isnan(r.get(field)))]
        if not vals:
            return None, None
        return float(sum(abs(v) for v in vals) / len(vals)), float(max(abs(v) for v in vals))

    pdp = {
        dec: {"mean_abs_delta": _mag(merged_rows, field)[0],
              "max_abs_delta":  _mag(merged_rows, field)[1]}
        for dec, field in [
            ("A", "decA_delta"),
            ("A_prime", "decA_prime_delta"),
            ("B", "decB_delta"),
            ("C", "decC_delta"),
            ("D_raw", "decD_raw_delta"),
            ("D_shape", "decD_shape_delta"),
            ("E", "decE_delta"),
        ]
    }

    out = {
        "task": "Task #5 — cross-decoder matrix with all decoders (A, A', B, C, D-raw, D-shape, E)",
        "sources": {
            "base_ABC_D": args.base,
            "dec_a_prime": args.decaprime,
            "dec_e_paradigm": args.dece_paradigm,
            "dec_e_xdec_native": args.dece_xdec_native,
            "dec_e_xdec_modified": args.dece_xdec_modified,
        },
        "n_rows": len(merged_rows),
        "rows": merged_rows,
        "per_decoder_profile": pdp,
        "e_signflips_vs_A": e_signflips_vs_A,
        "e_signflips_vs_Ap": e_signflips_vs_Ap,
    }
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[json] wrote {args.out_json}")

    # Markdown
    lines = []
    lines.append("# 17-row cross-decoder matrix (7 decoders: A, A', B, C, D-raw, D-shape, E)\n")
    lines.append("Dec E trained per-network on frozen fully-trained L2/3 with natural HMM stream "
                 "(task_p_switch per each ckpt's yaml — Markov p=0.2 for R1+R2, Bernoulli-per-batch "
                 "for legacy configs). R1+R2 Dec E ran 5000 steps; legacy a1/b1/c1 Dec E "
                 "training hit a post-training bug at step 5000 so ckpts are **step-4000-recovered** "
                 "(val_acc within ~0.02 of projected step-5000). e1 Dec E is full 5000 steps.\n")
    lines.append("A' column only populated for R1+R2 rows (Dec A' was trained on R1+R2).\n")
    lines.append("| # | Assay | Network | n_ex | n_unex | Δ_A | Δ_A' | Δ_B | Δ_C | Δ_D-raw | Δ_D-shape | Δ_E | ABC sign-agreement |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|")

    def f(x, nd=4):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return " — "
        return f"{x:+.{nd}f}"

    for i, r in enumerate(merged_rows, 1):
        sa = r["sign_agreement"]
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
            f"{f(r.get('decA_delta'))} | {f(r.get('decA_prime_delta'))} | "
            f"{f(r.get('decB_delta'))} | {f(r.get('decC_delta'))} | "
            f"{f(r.get('decD_raw_delta'))} | {f(r.get('decD_shape_delta'))} | "
            f"{f(r.get('decE_delta'))} | {sg} |"
        )

    lines.append("\n## Per-decoder magnitude profile\n")
    lines.append("| Decoder | n rows with Δ | mean \\|Δ\\| | max \\|Δ\\| |")
    lines.append("|---|---:|---:|---:|")
    for dec, p in pdp.items():
        n = sum(1 for r in merged_rows if r.get(f"dec{'_' if dec in ['A_prime','D_raw','D_shape','E'] else ''}{dec.lower() if dec!='A_prime' else 'a_prime'}_delta") is not None) if False else None
        # Simpler: count manually.
        field_map = {"A": "decA_delta", "A_prime": "decA_prime_delta", "B": "decB_delta",
                     "C": "decC_delta", "D_raw": "decD_raw_delta", "D_shape": "decD_shape_delta",
                     "E": "decE_delta"}
        n_rows = sum(1 for r in merged_rows if r.get(field_map[dec]) is not None)
        ma = f(p["mean_abs_delta"]) if p["mean_abs_delta"] is not None else "—"
        mx = f(p["max_abs_delta"]) if p["max_abs_delta"] is not None else "—"
        lines.append(f"| {dec} | {n_rows} | {ma} | {mx} |")

    lines.append("\n## Dec E sign flips vs Dec A (R1+R2 rows)\n")
    if e_signflips_vs_A:
        for fl in e_signflips_vs_A:
            lines.append(f"- **{fl['assay']}**: Δ_A {fl['dA']:+.4f} vs Δ_E {fl['dE']:+.4f}")
    else:
        lines.append("- None (Dec E sign matches Dec A on all R1+R2 rows where both are defined).")

    lines.append("\n## Dec E sign flips vs Dec A' (R1+R2 rows)\n")
    if e_signflips_vs_Ap:
        for fl in e_signflips_vs_Ap:
            lines.append(f"- **{fl['assay']}**: Δ_A' {fl['dAp']:+.4f} vs Δ_E {fl['dE']:+.4f}")
    else:
        lines.append("- None (Dec E sign matches Dec A' on all R1+R2 rows where both are defined).")

    with open(args.out_md, "w") as f_:
        f_.write("\n".join(lines) + "\n")
    print(f"[md]   wrote {args.out_md}")

    print()
    print(f"Dec E populated on {sum(1 for r in merged_rows if r.get('decE_delta') is not None)} rows (R1+R2 only per scope)")
    print(f"Sign flips Dec A → Dec E:  {len(e_signflips_vs_A)}")
    print(f"Sign flips Dec A' → Dec E: {len(e_signflips_vs_Ap)}")


if __name__ == "__main__":
    main()
