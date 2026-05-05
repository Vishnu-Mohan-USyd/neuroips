"""Task #21 — write /tmp/coder_seed_robustness.txt.

Generates the final coder report from the seed_robustness_matrix.json
produced by build_seed_robustness_matrix.py.

The report covers exactly what the Task #21 dispatch asked for:
  1. Per-paradigm sign-stability count across (net × seed) cells.
  2. Which (paradigm × net) combinations have unanimous signs across
     the 3 seeds vs which flip.
  3. Headline summary scoped to docs/paradigm_sign_mechanism.md (do the
     Mech 1 / Mech 2 verdicts and the dampening / sharpening signs
     survive seed perturbation?).

Wording rules (from the dispatch + memory):
  - No "Richter-style" loose grouping.
  - Plain language, concise paragraphs.
  - Numbers verified against the JSON matrix.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent

# Mech (Dec C) verdicts from docs/paradigm_sign_mechanism.md (commit 9dd1035).
MECH_DECC = {
    10: "Mech 1",                # M3R native
    15: "Mech 2 (weak)",         # M3R modified
    11: "Mech 2",                # HMS native
    12: "Mech 2",                # HMS-T native
    16: "Mech 2",                # HMS-T modified
    14: "Mech 1",                # VCD-test3 native
    17: "Mech 1",                # VCD-test3 modified
}
MECH_DECA = {
    10: "Mech 2",                # M3R native (Dec A disagrees)
    15: "Mech 2",                # M3R modified
    11: "Mech 2",                # HMS native
    12: "Mech 1",                # HMS-T native (Dec A disagrees)
    16: "Mech 2",                # HMS-T modified
    14: "Mech 1",                # VCD-test3 native
    17: "Mech 2",                # VCD-test3 modified (Dec A disagrees)
}


def _fmt(x: float) -> str:
    return f"{x:+.4f}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--matrix", default=str(_REPO / "results" / "seed_robustness_matrix.json"))
    p.add_argument("--output", default="/tmp/coder_seed_robustness.txt")
    args = p.parse_args()

    j = json.load(open(args.matrix))
    rows = j["rows"]
    summary = j["summary"]

    L: list[str] = []
    L.append("Task #21 — Seed-robustness paradigm matrix (3 seeds × 5 nets)")
    L.append("=" * 72)
    L.append("")
    L.append(
        "Scope: 3 seeds (42, 43, 44) × 5 nets (R1+R2, a1, b1, c1, e1) on the "
        "17-row cross-decoder paradigm matrix from "
        "results/cross_decoder_comprehensive_20k_final.md. Seed 42 from canonical "
        "matrix; seeds 43 + 44 from fresh paradigm-matrix runs on seed-specific ckpts."
    )
    L.append("")
    L.append(
        "Sign convention everywhere: Δ_ex_unex = decoder_acc(expected) − "
        "decoder_acc(unexpected). Dampening on expected → Δ < 0. Sharpening on "
        "expected → Δ > 0. Sign-stable across seeds means all 3 seeds have the "
        "same sign of Δ (zeros tolerated)."
    )
    L.append("")
    L.append("Source ckpts:")
    L.append("  Seed 42 (canonical):")
    L.append("    R1+R2:  results/simple_dual/emergent_seed42/checkpoint.pt")
    L.append("    a1/b1/c1/e1: /tmp/remote_ckpts/<net>/checkpoint.pt")
    L.append("  Seed 43 (Task #4): checkpoints/net_seed43_<net>.pt")
    L.append("  Seed 44 (Task #21): checkpoints/net_seed44_<net>.pt")
    L.append("")

    L.append("-" * 72)
    L.append("1. Sign-stability count")
    L.append("-" * 72)
    L.append("")
    n_total = summary["n_rows_total"]
    n_C = summary["n_sign_stable_decC"]
    n_A = summary["n_sign_stable_decA"]
    n_both = summary["n_sign_stable_both"]
    L.append(f"  Δ_decC: {n_C}/{n_total} rows sign-stable across 3 seeds.")
    L.append(f"  Δ_decA: {n_A}/{n_total} rows sign-stable across 3 seeds.")
    L.append(f"  Both:   {n_both}/{n_total} rows sign-stable on both decoders.")
    L.append("")

    L.append("-" * 72)
    L.append("2. Per-(paradigm × net) sign behaviour")
    L.append("-" * 72)
    L.append("")
    L.append("  Δ_decC — sign-flipping (paradigm × net) cells:")
    flips_C = summary["rows_unstable_decC"]
    if not flips_C:
        L.append("    NONE — all 17 (paradigm × net) cells have unanimous Δ_decC sign across seeds.")
    else:
        for r in flips_C:
            L.append(f"    Row {r['row']:>2}  {r['paradigm']:<48}  net={r['net']:<5}  signs={r['signs_decC']}")
    L.append("")
    L.append("  Δ_decA — sign-flipping (paradigm × net) cells:")
    flips_A = summary["rows_unstable_decA"]
    if not flips_A:
        L.append("    NONE — all 17 (paradigm × net) cells have unanimous Δ_decA sign across seeds.")
    else:
        for r in flips_A:
            L.append(f"    Row {r['row']:>2}  {r['paradigm']:<48}  net={r['net']:<5}  signs={r['signs_decA']}")
    L.append("")

    L.append("  Stable cells, full table (paradigm × net × seed Δ_decC):")
    L.append("")
    L.append(
        f"    {'#':>2}  {'Paradigm':<48}  {'Net':<5}  "
        f"{'s42':>9}  {'s43':>9}  {'s44':>9}  {'signs':>5}  stable?"
    )
    for r in rows:
        c = r["delta_decC"]
        stable = "yes" if r["sign_stable_decC"] else "NO "
        L.append(
            f"    {r['row']:>2}  {r['paradigm']:<48}  {r['net']:<5}  "
            f"{_fmt(c['seed42']):>9}  {_fmt(c['seed43']):>9}  {_fmt(c['seed44']):>9}  "
            f"{r['signs_decC']:>5}  {stable}"
        )
    L.append("")

    L.append("  Δ_decA full table (paradigm × net × seed):")
    L.append("")
    L.append(
        f"    {'#':>2}  {'Paradigm':<48}  {'Net':<5}  "
        f"{'s42':>9}  {'s43':>9}  {'s44':>9}  {'signs':>5}  stable?"
    )
    for r in rows:
        a = r["delta_decA"]
        stable = "yes" if r["sign_stable_decA"] else "NO "
        L.append(
            f"    {r['row']:>2}  {r['paradigm']:<48}  {r['net']:<5}  "
            f"{_fmt(a['seed42']):>9}  {_fmt(a['seed43']):>9}  {_fmt(a['seed44']):>9}  "
            f"{r['signs_decA']:>5}  {stable}"
        )
    L.append("")

    L.append("-" * 72)
    L.append("3. Headline — do Phase 4-7 mechanism verdicts survive seed perturbation?")
    L.append("-" * 72)
    L.append("")
    L.append(
        "Cross-reference: docs/paradigm_sign_mechanism.md — Phase 4-7 "
        "investigation assigns each of 8 R1+R2 paradigms to Mech 1 (V2-feedback "
        "channel-resolved gain modulation) or Mech 2 (W_rec-amplified non-V2 "
        "stim-statistics bias) under both Dec C and Dec A readouts. The doc "
        "table reports 5/8 paradigms with agreeing Dec C / Dec A verdicts on "
        "the seed-42 ckpt. Below: do the dampening / sharpening signs that "
        "underpin those verdicts survive a 3-seed perturbation?"
    )
    L.append("")
    L.append(
        f"    {'Paradigm':<28}  {'Net':<5}  {'Mech (Dec C)':<16}  "
        f"{'Δ_decC s42':>10}  {'s43':>9}  {'s44':>9}  Dec C stable?"
    )
    for row_idx in (10, 15, 11, 12, 16, 14, 17):
        r = next((rr for rr in rows if rr["row"] == row_idx), None)
        if r is None:
            continue
        c = r["delta_decC"]
        stable = "yes" if r["sign_stable_decC"] else "NO"
        # Strip "(matched_3row_ring)", etc., to keep it readable
        para_short = r["paradigm"].split(" (")[0]
        if "modified" in r["paradigm"]:
            para_short += " modified"
        else:
            para_short += " native"
        L.append(
            f"    {para_short:<28}  {r['net']:<5}  {MECH_DECC.get(row_idx,'?'):<16}  "
            f"{_fmt(c['seed42']):>10}  {_fmt(c['seed43']):>9}  {_fmt(c['seed44']):>9}  {stable}"
        )
    L.append("")
    L.append(
        f"    {'Paradigm':<28}  {'Net':<5}  {'Mech (Dec A)':<16}  "
        f"{'Δ_decA s42':>10}  {'s43':>9}  {'s44':>9}  Dec A stable?"
    )
    for row_idx in (10, 15, 11, 12, 16, 14, 17):
        r = next((rr for rr in rows if rr["row"] == row_idx), None)
        if r is None:
            continue
        a = r["delta_decA"]
        stable = "yes" if r["sign_stable_decA"] else "NO"
        para_short = r["paradigm"].split(" (")[0]
        if "modified" in r["paradigm"]:
            para_short += " modified"
        else:
            para_short += " native"
        L.append(
            f"    {para_short:<28}  {r['net']:<5}  {MECH_DECA.get(row_idx,'?'):<16}  "
            f"{_fmt(a['seed42']):>10}  {_fmt(a['seed43']):>9}  {_fmt(a['seed44']):>9}  {stable}"
        )
    L.append("")

    # Auto-generated headline sentence
    headline_rows = [10, 15, 11, 12, 16, 14, 17]  # 7 R1+R2 mechanism-scoped rows
    n_phase_47 = len(headline_rows)
    n_phase_47_C = sum(
        1 for ri in headline_rows
        if next(rr for rr in rows if rr["row"] == ri)["sign_stable_decC"]
    )
    n_phase_47_A = sum(
        1 for ri in headline_rows
        if next(rr for rr in rows if rr["row"] == ri)["sign_stable_decA"]
    )
    L.append(
        f"  Phase 4-7 paradigms — Δ_decC sign survival: {n_phase_47_C}/{n_phase_47} "
        f"unanimous across 3 seeds.  Δ_decA: {n_phase_47_A}/{n_phase_47}."
    )
    flipping_C = [
        ri for ri in headline_rows
        if not next(rr for rr in rows if rr["row"] == ri)["sign_stable_decC"]
    ]
    flipping_A = [
        ri for ri in headline_rows
        if not next(rr for rr in rows if rr["row"] == ri)["sign_stable_decA"]
    ]
    if flipping_C:
        L.append(f"  Δ_decC flips on rows {flipping_C} — Mech (Dec C) verdict NOT seed-robust on these.")
    else:
        L.append(f"  Δ_decC: all 7 mechanism-scoped paradigm signs survive seed perturbation; the Mech 1 / Mech 2 (Dec C) verdicts in docs/paradigm_sign_mechanism.md are seed-robust on the dampening / sharpening signs they rest on.")
    if flipping_A:
        L.append(f"  Δ_decA flips on rows {flipping_A} — Mech (Dec A) verdict NOT seed-robust on these.")
    else:
        L.append(f"  Δ_decA: all 7 mechanism-scoped paradigm signs survive seed perturbation; the Mech (Dec A) verdicts are seed-robust.")

    L.append("")

    L.append("-" * 72)
    L.append("Sources")
    L.append("-" * 72)
    L.append("")
    L.append(f"  Per-row JSON:   results/seed_robustness_matrix.json")
    L.append(f"  Per-row MD:     results/seed_robustness_matrix.md")
    L.append(f"  Seed 42 base:   {j['sources']['seed42']}")
    L.append(f"  Seed 43 dir:    {j['sources']['seed43']}")
    L.append(f"  Seed 44 dir:    {j['sources']['seed44']}")
    L.append("")
    L.append(f"  Cross-ref:      docs/paradigm_sign_mechanism.md")
    L.append("")

    Path(args.output).write_text("\n".join(L) + "\n")
    print(f"[txt] wrote {args.output}")


if __name__ == "__main__":
    main()
