"""Task #2 — Merge per-net Dec A' values into the 17-row cross-decoder matrix.

The existing 7-column matrix `results/cross_decoder_comprehensive_with_all_decoders.json`
has `decA_prime_delta` for all 17 rows but the rows for a1/b1/c1/e1 (5/6/7/8)
were computed using **R1+R2's Dec A'** applied to legacy network r_l23. Task #2
replaces those 4 entries with values computed using **each net's own** Dec A'
(the per-net ckpts trained in Task #1: `checkpoints/decoder_a_prime_{a1,b1,c1,e1}.pt`).

Inputs:
  - base matrix:   `results/cross_decoder_comprehensive_with_all_decoders.json`
  - per-net HMM C1 paradigm JSONs, each obtained by patching the legacy ckpt
    with per-net Dec A' via `scripts/_make_decAprime_ckpt.py`, then running
    `scripts/r1r2_paradigm_readout.py --conditions C1_focused_native`:
        /tmp/task2_legacy/{a1,b1,c1,e1}_C1.json

Outputs:
  - `results/cross_decoder_comprehensive_final.{json,md}`

Added analyses (Task #2 specific):
  - Per-row sign flip Dec A' vs Dec A (Δ_A' sign ≠ Δ_A sign).
  - Per-decoder magnitude profile with Dec A' now covering all 17 rows.
  - Per-decoder ALL-agree count (how often each decoder agreed with the row's
    majority sign among A/B/C/A'/E — as a structural extension of the ABC
    sign-agreement rule kept from the prior matrix).

Read-only wrt the base JSON — never mutates it. Output is a new file.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


_LEGACY_ROW_KEY = {
    "a1": ("HMM C1 (focused + HMM cue)", "a1 (legacy three-regimes)"),
    "b1": ("HMM C1 (focused + HMM cue)", "b1 (legacy three-regimes)"),
    "c1": ("HMM C1 (focused + HMM cue)", "c1 (legacy three-regimes)"),
    "e1": ("HMM C1 (focused + HMM cue)", "e1 (legacy three-regimes)"),
}


def _sign(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return 1 if x > 0 else (-1 if x < 0 else 0)


def _glyph(s):
    return {1: "+", -1: "−", 0: "0", None: "?"}.get(s, "?")


def _load_legacy_decAprime(legacy_dir: str) -> dict[tuple[str, str], float]:
    """Read Δ_A' (actually decA_delta in the patched JSON) per legacy net."""
    out: dict[tuple[str, str], float] = {}
    for net, key in _LEGACY_ROW_KEY.items():
        p = os.path.join(legacy_dir, f"{net}_C1.json")
        if not Path(p).exists():
            print(f"[warn] missing {p} — skipping")
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
        delta = float(ex["decA_acc_mean"] - ux["decA_acc_mean"])
        out[key] = delta
        print(f"[ok] {net} HMM C1 Dec A' Δ = {delta:+.4f} (ex={ex['decA_acc_mean']:.4f}, "
              f"unex={ux['decA_acc_mean']:.4f})")
    return out


def _abc_sign_agreement(dA, dB, dC):
    """Mirror the existing convention: sign_agreement on A/B/C triple."""
    signs = {"A": _sign(dA), "B": _sign(dB), "C": _sign(dC)}
    valid = {k: v for k, v in signs.items() if v is not None}
    if len(valid) < 2:
        return {"signs": signs, "all_agree": None, "majority": None, "outlier": None}
    vals = list(valid.values())
    if len(set(vals)) == 1:
        return {"signs": signs, "all_agree": True, "majority": vals[0], "outlier": None}
    pos = sum(1 for v in vals if v == 1)
    neg = sum(1 for v in vals if v == -1)
    zer = sum(1 for v in vals if v == 0)
    if pos >= 2:
        maj = 1
    elif neg >= 2:
        maj = -1
    elif zer >= 2:
        maj = 0
    else:
        maj = None
    out = None
    if maj is not None:
        for k, v in valid.items():
            if v != maj:
                out = k
                break
    return {"signs": signs, "all_agree": False, "majority": maj, "outlier": out}


def _row_majority_sign(dA, dAp, dB, dC, dE):
    """Majority sign across 5 decoders (A, A', B, C, E). For per-decoder
    ALL-agree counts: a decoder 'agrees with majority' iff its sign == majority."""
    signs = [_sign(x) for x in (dA, dAp, dB, dC, dE)]
    valid = [s for s in signs if s is not None]
    if not valid:
        return None
    pos = sum(1 for v in valid if v == 1)
    neg = sum(1 for v in valid if v == -1)
    zer = sum(1 for v in valid if v == 0)
    # Majority rule: >= 3 (strict majority among 5) counts as majority; ties -> None.
    if pos >= 3:
        return 1
    if neg >= 3:
        return -1
    if zer >= 3:
        return 0
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base",
                    default="results/cross_decoder_comprehensive_with_all_decoders.json")
    ap.add_argument("--legacy-dir", default="/tmp/task2_legacy")
    ap.add_argument("--out-json",
                    default="results/cross_decoder_comprehensive_final.json")
    ap.add_argument("--out-md",
                    default="results/cross_decoder_comprehensive_final.md")
    args = ap.parse_args()

    with open(args.base) as f:
        base = json.load(f)
    print(f"[load] base matrix rows: {len(base['rows'])}")

    legacy_decAprime = _load_legacy_decAprime(args.legacy_dir)
    print(f"[load] per-net Dec A' values: {len(legacy_decAprime)} rows\n")

    # --- Build merged rows with updated Dec A' for legacy rows only ---
    merged_rows = []
    replacements: list[dict] = []
    for r in base["rows"]:
        key = (r["assay"], r["network"])
        newrow = dict(r)
        if key in legacy_decAprime:
            old = newrow.get("decA_prime_delta")
            new = legacy_decAprime[key]
            newrow["decA_prime_delta"] = new
            replacements.append({
                "assay": r["assay"],
                "network": r["network"],
                "decA_prime_delta_old_R1R2_patch": old,
                "decA_prime_delta_new_per_net":   new,
                "shift": (new - old) if (old is not None) else None,
            })
        merged_rows.append(newrow)

    # --- Sign-flip Dec A' vs Dec A per row ---
    sign_flips_Ap_vs_A = []
    for r in merged_rows:
        dA = r.get("decA_delta")
        dAp = r.get("decA_prime_delta")
        sA, sAp = _sign(dA), _sign(dAp)
        if sA is not None and sAp is not None and sA != sAp:
            sign_flips_Ap_vs_A.append({
                "assay": r["assay"],
                "network": r["network"],
                "decA_delta": dA,
                "decA_prime_delta": dAp,
            })

    # --- Per-decoder magnitude profile + ALL-agree counts ---
    field_map = {
        "A": "decA_delta",
        "A_prime": "decA_prime_delta",
        "B": "decB_delta",
        "C": "decC_delta",
        "D_raw": "decD_raw_delta",
        "D_shape": "decD_shape_delta",
        "E": "decE_delta",
    }

    def _mag(rows, field):
        vals = [r.get(field) for r in rows if r.get(field) is not None
                and not (isinstance(r.get(field), float) and math.isnan(r.get(field)))]
        if not vals:
            return None, None
        return float(sum(abs(v) for v in vals) / len(vals)), float(max(abs(v) for v in vals))

    pdp: dict[str, dict] = {}
    # Majority sign per row (across A/A'/B/C/E — the 5 "linear-readout" decoders;
    # D-raw/D-shape use a different input space so we exclude them from the
    # majority-sign rule to avoid confounding).
    row_majorities: list[int | None] = []
    for r in merged_rows:
        row_majorities.append(_row_majority_sign(
            r.get("decA_delta"), r.get("decA_prime_delta"),
            r.get("decB_delta"), r.get("decC_delta"), r.get("decE_delta")))

    for dec, field in field_map.items():
        mean_abs, max_abs = _mag(merged_rows, field)
        n_rows_with = sum(1 for r in merged_rows if r.get(field) is not None)
        agree_count = 0
        agree_considered = 0
        for r, maj in zip(merged_rows, row_majorities):
            s = _sign(r.get(field))
            if s is None or maj is None:
                continue
            agree_considered += 1
            if s == maj:
                agree_count += 1
        pdp[dec] = {
            "n_rows_with_delta": n_rows_with,
            "mean_abs_delta": mean_abs,
            "max_abs_delta": max_abs,
            "all_agree_count": agree_count,
            "all_agree_considered": agree_considered,
        }

    # --- Refresh sign_agreement (ABC convention) per row — unchanged since
    #     we don't touch A/B/C. Keep base row's if present.
    for r in merged_rows:
        if "sign_agreement" not in r or r["sign_agreement"] is None:
            r["sign_agreement"] = _abc_sign_agreement(
                r.get("decA_delta"), r.get("decB_delta"), r.get("decC_delta"))

    out = {
        "task": "Task #2 — 17-row cross-decoder matrix with per-net Dec A' (final)",
        "policy": (
            "Dec A' column: per-net for all 5 networks. R1+R2 rows (1-4, 9-17) use "
            "the R1+R2 Dec A' (Task #1 Part A). Legacy rows (5-8) use each legacy "
            "net's own Dec A' (Task #1 per-net training on a1/b1/c1/e1). The prior "
            "R1+R2-patch values for legacy rows are captured in "
            "`decA_prime_row_replacements` for audit."
        ),
        "sources": {
            "base": args.base,
            "legacy_dir": args.legacy_dir,
            "per_net_decAprime_ckpts": {
                net: f"checkpoints/decoder_a_prime_{net}.pt"
                for net in ("a1", "b1", "c1", "e1")
            },
        },
        "n_rows": len(merged_rows),
        "rows": merged_rows,
        "per_decoder_profile": pdp,
        "decA_prime_row_replacements": replacements,
        "sign_flips_Ap_vs_A": sign_flips_Ap_vs_A,
    }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[json] wrote {args.out_json}")

    # --- Markdown ---
    def fm(x, nd=4):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return " — "
        return f"{x:+.{nd}f}"

    lines = []
    lines.append("# 17-row cross-decoder matrix — FINAL (per-net Dec A')\n")
    lines.append("Task #2 update: rows 5-8 (HMM C1 on a1/b1/c1/e1) now use each legacy net's "
                 "own Dec A' ckpt (`checkpoints/decoder_a_prime_{net}.pt`) instead of the "
                 "R1+R2 Dec A' patched onto the legacy network. R1+R2 rows unchanged. "
                 "All other columns unchanged. See `decA_prime_row_replacements` in the "
                 "JSON for audit of the 4 swapped values.\n")
    lines.append("| # | Assay | Network | n_ex | n_unex | Δ_A | Δ_A' | Δ_B | Δ_C | Δ_D-raw | Δ_D-shape | Δ_E | ABC sign-agreement |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|")
    for i, r in enumerate(merged_rows, 1):
        sa = r.get("sign_agreement") or {}
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
            f"{fm(r.get('decA_delta'))} | {fm(r.get('decA_prime_delta'))} | "
            f"{fm(r.get('decB_delta'))} | {fm(r.get('decC_delta'))} | "
            f"{fm(r.get('decD_raw_delta'))} | {fm(r.get('decD_shape_delta'))} | "
            f"{fm(r.get('decE_delta'))} | {sg} |"
        )

    lines.append("\n## Dec A' row-level replacements (legacy rows 5-8)\n")
    lines.append("| # | Network | assay | Δ_A'_old (R1+R2 patch) | Δ_A'_new (per-net) | shift |")
    lines.append("|---|---------|-------|-----------------------:|-------------------:|------:|")
    row_num = {r["network"]: i+1 for i, r in enumerate(merged_rows) if r["assay"] == "HMM C1 (focused + HMM cue)"}
    for rep in replacements:
        n = row_num.get(rep["network"], "?")
        lines.append(
            f"| {n} | {rep['network']} | {rep['assay']} | "
            f"{fm(rep['decA_prime_delta_old_R1R2_patch'])} | "
            f"{fm(rep['decA_prime_delta_new_per_net'])} | "
            f"{fm(rep['shift'])} |"
        )

    lines.append("\n## Per-decoder magnitude profile (all 17 rows)\n")
    lines.append("| Decoder | n rows | mean |Δ| | max |Δ| | agree-with-majority | denom |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for dec, p in pdp.items():
        lines.append(
            f"| {dec} | {p['n_rows_with_delta']} | "
            f"{fm(p['mean_abs_delta']) if p['mean_abs_delta'] is not None else ' — '} | "
            f"{fm(p['max_abs_delta']) if p['max_abs_delta'] is not None else ' — '} | "
            f"{p['all_agree_count']} | {p['all_agree_considered']} |"
        )
    lines.append("\n(Majority sign per row is computed over the 5 linear-readout decoders "
                 "A/A'/B/C/E; D-raw/D-shape use a different input space and are excluded "
                 "from the majority rule but reported in the profile.)\n")

    lines.append("\n## Dec A' sign flips vs Dec A (per row)\n")
    if sign_flips_Ap_vs_A:
        for fl in sign_flips_Ap_vs_A:
            lines.append(f"- **{fl['network']} / {fl['assay']}**: "
                         f"Δ_A {fl['decA_delta']:+.4f} vs Δ_A' {fl['decA_prime_delta']:+.4f}")
    else:
        lines.append("- **None.** Dec A' sign matches Dec A on all 17 rows.")

    with open(args.out_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[md]   wrote {args.out_md}")

    # Console summary
    print()
    print(f"  n_rows          : {len(merged_rows)}")
    print(f"  rows updated    : {len(replacements)} (legacy a1/b1/c1/e1 HMM C1)")
    print(f"  sign flips A↔A' : {len(sign_flips_Ap_vs_A)}")
    for dec, p in pdp.items():
        print(f"  Dec {dec:7s}: n={p['n_rows_with_delta']:2d}  mean|Δ|={fm(p['mean_abs_delta'])}  "
              f"max|Δ|={fm(p['max_abs_delta'])}  agree={p['all_agree_count']}/{p['all_agree_considered']}")


if __name__ == "__main__":
    main()
