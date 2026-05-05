"""Task #21 — Seed-robustness Part C: aggregate paradigm matrix across 3 seeds.

Builds the 17-row × 3-seed × 2-decoder seed-robustness matrix from:

  Seed 42 (canonical): results/cross_decoder_comprehensive_20k_final.json
                       (already contains Δ_A and Δ_C per row).

  Seed 43 / 44 (Task #4 / Task #21): /tmp/paradigm_matrix_seed{43,44}/
      r1r2_paradigm.json              — rows 1-4
      legacy/<net>_C1.json            — rows 5-8
      xdec_native.json                — rows 9-14
      xdec_modified.json              — rows 15-17

Outputs:
  results/seed_robustness_matrix.json — 17 rows × per-seed Δ_decC + Δ_decA + sign-stable flags
  results/seed_robustness_matrix.md   — pretty table + sign-stability summary

Sign convention everywhere: Δ_ex_unex = decoder_acc_expected − decoder_acc_unexpected
  (ex - unex), per the paradigm_sign_mechanism.md doc.

Design intent: cells where the SIGN of Δ flips across seeds are flagged as
"unstable" — that is the ground truth for the seed-robustness Q from the
Task #21 dispatch.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

_REPO = Path(__file__).resolve().parent.parent

# 17-row schema. Each entry binds:
#   (row #, paradigm name, net, source_kind, source_key)
# where source_kind is one of:
#   "paradigm_C{n}"    → r1r2_paradigm.json["conditions"][i] for matching id.
#   "legacy_C1"        → legacy/<net>_C1.json["conditions"][0]
#   "xdec_native_<S>"  → xdec_native.json["results"][<S>]
#   "xdec_modified_<S>"→ xdec_modified.json["results"][<S>]
#
# The paradigm names below are *exactly* the labels used in the Task #7
# 17-row matrix so seed 42 can be cross-referenced 1:1.
ROWS: list[dict] = [
    {"row": 1,  "paradigm": "HMM C1 (focused + HMM cue)",        "net": "r1r2", "kind": "paradigm",     "key": "C1_focused_native"},
    {"row": 2,  "paradigm": "HMM C2 (routine + HMM cue)",        "net": "r1r2", "kind": "paradigm",     "key": "C2_routine_native"},
    {"row": 3,  "paradigm": "HMM C3 (focused + zero cue)",       "net": "r1r2", "kind": "paradigm",     "key": "C3_focused_neutralcue"},
    {"row": 4,  "paradigm": "HMM C4 (routine + zero cue)",       "net": "r1r2", "kind": "paradigm",     "key": "C4_routine_neutralcue"},
    {"row": 5,  "paradigm": "HMM C1 (focused + HMM cue)",        "net": "a1",   "kind": "legacy_C1",    "key": "C1_focused_native"},
    {"row": 6,  "paradigm": "HMM C1 (focused + HMM cue)",        "net": "b1",   "kind": "legacy_C1",    "key": "C1_focused_native"},
    {"row": 7,  "paradigm": "HMM C1 (focused + HMM cue)",        "net": "c1",   "kind": "legacy_C1",    "key": "C1_focused_native"},
    {"row": 8,  "paradigm": "HMM C1 (focused + HMM cue)",        "net": "e1",   "kind": "legacy_C1",    "key": "C1_focused_native"},
    {"row": 9,  "paradigm": "NEW (paired march)",                 "net": "r1r2", "kind": "xdec_native",  "key": "NEW"},
    {"row": 10, "paradigm": "M3R (matched_3row_ring)",            "net": "r1r2", "kind": "xdec_native",  "key": "M3R"},
    {"row": 11, "paradigm": "HMS",                                "net": "r1r2", "kind": "xdec_native",  "key": "HMS"},
    {"row": 12, "paradigm": "HMS-T (tight-expected)",             "net": "r1r2", "kind": "xdec_native",  "key": "HMS-T"},
    {"row": 13, "paradigm": "P3P (matched_probe_3pass)",          "net": "r1r2", "kind": "xdec_native",  "key": "P3P"},
    {"row": 14, "paradigm": "VCD-test3",                          "net": "r1r2", "kind": "xdec_native",  "key": "VCD"},
    {"row": 15, "paradigm": "M3R (modified: focused+march cue)",  "net": "r1r2", "kind": "xdec_modified","key": "M3R"},
    {"row": 16, "paradigm": "HMS-T (modified: focused+march cue)","net": "r1r2", "kind": "xdec_modified","key": "HMS-T"},
    {"row": 17, "paradigm": "VCD (modified: focused+march cue)",  "net": "r1r2", "kind": "xdec_modified","key": "VCD"},
]


def _load_seed42_from_canonical() -> dict[int, tuple[float, float]]:
    """Read results/cross_decoder_comprehensive_20k_final.json.

    Return {row_idx (1-based) -> (Δ_A_seed42, Δ_C_seed42)}.

    Row index follows the canonical 17-row order baked into the JSON's
    `rows` list (positional). Each row entry already carries decA_delta
    and decC_delta computed from its source eval JSON.
    """
    p = _REPO / "results" / "cross_decoder_comprehensive_20k_final.json"
    if not p.exists():
        raise FileNotFoundError(
            f"missing {p}; needed for seed-42 baseline column"
        )
    j = json.load(p.open())
    rows = j.get("rows", j.get("matrix_rows", []))
    if not rows:
        raise RuntimeError(
            f"could not locate row list in {p}; keys={list(j.keys())[:8]}"
        )
    out: dict[int, tuple[float, float]] = {}
    for i, r in enumerate(rows, start=1):
        delA = r.get("decA_delta")
        delC = r.get("decC_delta")
        if delA is None or delC is None:
            continue
        out[i] = (float(delA), float(delC))
    return out


def _seed42_from_md_table(md_path: Path) -> dict[int, tuple[float, float]]:
    """Fallback parser: read the rendered MD table when the JSON dump
    nests rows under an unexpected key. Locates the canonical Δ_A and
    Δ_C columns and returns {row_idx -> (Δ_A, Δ_C)}.

    The MD table header is:
      | # | Assay | Network | n_ex | n_unex | Δ_A | Δ_A'(20k) | Δ_A'(5k) | Δ_B | Δ_C | ...
    """
    out: dict[int, tuple[float, float]] = {}
    if not md_path.exists():
        return out
    in_table = False
    header: Optional[list[str]] = None
    for line in md_path.read_text().splitlines():
        if line.startswith("| # |"):
            in_table = True
            header = [c.strip() for c in line.strip("|").split("|")]
            continue
        if in_table and line.startswith("|---"):
            continue
        if in_table:
            if not line.startswith("|"):
                in_table = False
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if not cells or not cells[0].isdigit():
                in_table = False
                continue
            idx = int(cells[0])
            if header is None:
                continue
            try:
                ia = header.index("Δ_A")
                ic = header.index("Δ_C")
            except ValueError:
                continue
            try:
                delA = float(cells[ia])
                delC = float(cells[ic])
                out[idx] = (delA, delC)
            except ValueError:
                pass
    return out


def _load_seed_paradigm_matrix(seed: int, base_dir: Path) -> dict[int, tuple[float, float]]:
    """Read the 4 JSONs produced by run_seed_paradigm_matrix.sh and
    extract Δ_A and Δ_C for each of the 17 rows at this seed.

    Returns {row_idx (1-based) -> (Δ_A_seed, Δ_C_seed)}.

    Δ_A and Δ_C are returned as float; np.nan if unavailable for that source.
    """
    r1r2_para = json.load((base_dir / "r1r2_paradigm.json").open())
    xdec_n = json.load((base_dir / "xdec_native.json").open())
    xdec_m = json.load((base_dir / "xdec_modified.json").open())
    legacy = {
        net: json.load((base_dir / "legacy" / f"{net}_C1.json").open())
        for net in ("a1", "b1", "c1", "e1")
    }

    def _delta_from_paradigm(j: dict, cond_id: str) -> tuple[float, float]:
        """Δ_A and Δ_C from a paradigm_readout-style JSON's matching condition."""
        for cond in j["conditions"]:
            if cond["id"] != cond_id:
                continue
            ex = cond["branches"]["ex"]
            un = cond["branches"]["unex"]
            delA = float(ex["decA_acc_mean"] - un["decA_acc_mean"])
            delC = float(ex["decC_acc_mean"] - un["decC_acc_mean"])
            return delA, delC
        raise KeyError(f"condition id {cond_id} missing from paradigm JSON")

    def _delta_from_xdec(j: dict, strategy: str) -> tuple[float, float]:
        r = j["results"][strategy]
        return float(r["decA_delta"]), float(r["decC_delta"])

    out: dict[int, tuple[float, float]] = {}
    for row in ROWS:
        idx = row["row"]
        kind = row["kind"]
        key = row["key"]
        net = row["net"]
        if kind == "paradigm":
            out[idx] = _delta_from_paradigm(r1r2_para, key)
        elif kind == "legacy_C1":
            out[idx] = _delta_from_paradigm(legacy[net], key)
        elif kind == "xdec_native":
            out[idx] = _delta_from_xdec(xdec_n, key)
        elif kind == "xdec_modified":
            out[idx] = _delta_from_xdec(xdec_m, key)
        else:
            raise ValueError(f"unknown kind {kind}")
    return out


def _sign(x: float) -> str:
    if abs(x) < 1e-9:
        return "0"
    return "+" if x > 0 else "−"


def _signs_stable(vals: list[float]) -> bool:
    """All non-zero values share the same sign. Zeros are tolerated.

    A value within |x| < 1e-9 is treated as zero and skipped from the
    sign agreement test (matches how paradigm_sign_mechanism.md treats
    near-zero deltas as inconclusive rather than as a flip).
    """
    s = {1 if v > 1e-9 else (-1 if v < -1e-9 else 0) for v in vals if v == v}
    s.discard(0)
    return len(s) <= 1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seed42-source", default=None,
                   help="Override seed-42 source JSON (default: "
                        "results/cross_decoder_comprehensive_20k_final.json).")
    p.add_argument("--seed43-dir", default="/tmp/paradigm_matrix_seed43")
    p.add_argument("--seed44-dir", default="/tmp/paradigm_matrix_seed44")
    p.add_argument("--output-json", default=str(_REPO / "results" / "seed_robustness_matrix.json"))
    p.add_argument("--output-md",   default=str(_REPO / "results" / "seed_robustness_matrix.md"))
    args = p.parse_args()

    # --- Load per-seed Δ tables ---
    # Seed 42: prefer canonical JSON, fall back to MD parse if JSON schema differs.
    try:
        s42 = _load_seed42_from_canonical()
        if not s42:
            raise RuntimeError("empty seed-42 row table from canonical JSON")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] seed42 JSON load failed ({e}); falling back to MD parse")
        s42 = _seed42_from_md_table(
            _REPO / "results" / "cross_decoder_comprehensive_20k_final.md"
        )

    s43 = _load_seed_paradigm_matrix(43, Path(args.seed43_dir))
    s44 = _load_seed_paradigm_matrix(44, Path(args.seed44_dir))

    # --- Build per-row matrix ---
    rows_out: list[dict] = []
    for row in ROWS:
        idx = row["row"]
        a42, c42 = s42.get(idx, (float("nan"), float("nan")))
        a43, c43 = s43[idx]
        a44, c44 = s44[idx]
        delA = [a42, a43, a44]
        delC = [c42, c43, c44]
        rows_out.append({
            "row": idx,
            "paradigm": row["paradigm"],
            "net": row["net"],
            "delta_decA": {"seed42": a42, "seed43": a43, "seed44": a44},
            "delta_decC": {"seed42": c42, "seed43": c43, "seed44": c44},
            "sign_stable_decA": _signs_stable(delA),
            "sign_stable_decC": _signs_stable(delC),
            "signs_decA": "".join(_sign(v) for v in delA),
            "signs_decC": "".join(_sign(v) for v in delC),
        })

    # --- Per-paradigm sign-stability counts ---
    n_stable_A = sum(1 for r in rows_out if r["sign_stable_decA"])
    n_stable_C = sum(1 for r in rows_out if r["sign_stable_decC"])
    n_stable_both = sum(
        1 for r in rows_out if r["sign_stable_decA"] and r["sign_stable_decC"]
    )

    # --- Headline: scope to docs/paradigm_sign_mechanism.md cells ---
    # The Phase 4-7 doc claims sign-decoder-robust dampening / sharpening on
    # specific paradigms. The sign-stability test below tells us whether
    # those signs survive seed perturbation.
    headline_paradigms = {
        # Mech 1 paradigms (Dec C native column, per paradigm_sign_mechanism.md)
        "M3R native":       {"row": 10, "mech_decC": "Mech 1"},
        "M3R modified":     {"row": 15, "mech_decC": "Mech 2 (weak)"},
        "HMS native":       {"row": 11, "mech_decC": "Mech 2"},
        "HMS-T native":     {"row": 12, "mech_decC": "Mech 2"},
        "HMS-T modified":   {"row": 16, "mech_decC": "Mech 2"},
        "VCD-test3 native": {"row": 14, "mech_decC": "Mech 1"},
        "VCD-test3 modified":{"row": 17,"mech_decC": "Mech 1"},
        "HMM C1 (R1+R2)":   {"row": 1,  "mech_decC": "Mech 1 (paired-fork V2-pred stratum)"},
        "HMM C1 (a1)":      {"row": 5,  "mech_decC": "n/a (legacy)"},
        "HMM C1 (b1)":      {"row": 6,  "mech_decC": "n/a (legacy)"},
        "HMM C1 (c1)":      {"row": 7,  "mech_decC": "n/a (legacy)"},
        "HMM C1 (e1)":      {"row": 8,  "mech_decC": "n/a (legacy)"},
    }

    # --- Write JSON ---
    out_json = {
        "task": "Task #21 — seed-robustness paradigm matrix (3 seeds × 5 nets)",
        "seeds": [42, 43, 44],
        "rows": rows_out,
        "summary": {
            "n_rows_total": len(rows_out),
            "n_sign_stable_decA": n_stable_A,
            "n_sign_stable_decC": n_stable_C,
            "n_sign_stable_both": n_stable_both,
            "rows_unstable_decC": [
                {"row": r["row"], "paradigm": r["paradigm"], "net": r["net"],
                 "signs_decC": r["signs_decC"]}
                for r in rows_out if not r["sign_stable_decC"]
            ],
            "rows_unstable_decA": [
                {"row": r["row"], "paradigm": r["paradigm"], "net": r["net"],
                 "signs_decA": r["signs_decA"]}
                for r in rows_out if not r["sign_stable_decA"]
            ],
        },
        "headline_paradigms": headline_paradigms,
        "sources": {
            "seed42": str((_REPO / "results" / "cross_decoder_comprehensive_20k_final.json").relative_to(_REPO)),
            "seed43": args.seed43_dir,
            "seed44": args.seed44_dir,
        },
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"[json] wrote {args.output_json}")

    # --- Write MD table ---
    md_lines: list[str] = []
    md_lines.append("# Seed-robustness paradigm matrix (Task #21)\n")
    md_lines.append(
        "Per-paradigm × per-net Δ_ex_unex at 3 seeds (42, 43, 44). "
        "Sign-stable means all 3 seeds agree on the sign of Δ (zeros tolerated). "
        "Δ_decC uses the shared decoder_c.pt; Δ_decA uses each ckpt's own joint-trained Dec A.\n"
    )
    md_lines.append(
        f"**Summary** — {n_stable_C}/{len(rows_out)} rows sign-stable on Δ_decC, "
        f"{n_stable_A}/{len(rows_out)} on Δ_decA, "
        f"{n_stable_both}/{len(rows_out)} on both.\n"
    )

    md_lines.append("## Δ_decC (shared decoder_c.pt)\n")
    md_lines.append("| # | Paradigm | Net | Seed 42 Δ | Seed 43 Δ | Seed 44 Δ | Signs | Sign stable? |")
    md_lines.append("|---|---|---|---:|---:|---:|:--:|:--:|")
    for r in rows_out:
        c = r["delta_decC"]
        stable = "yes" if r["sign_stable_decC"] else "**NO**"
        md_lines.append(
            f"| {r['row']} | {r['paradigm']} | {r['net']} | "
            f"{c['seed42']:+.4f} | {c['seed43']:+.4f} | {c['seed44']:+.4f} | "
            f"{r['signs_decC']} | {stable} |"
        )
    md_lines.append("")

    md_lines.append("## Δ_decA (each ckpt's joint-trained Dec A)\n")
    md_lines.append("| # | Paradigm | Net | Seed 42 Δ | Seed 43 Δ | Seed 44 Δ | Signs | Sign stable? |")
    md_lines.append("|---|---|---|---:|---:|---:|:--:|:--:|")
    for r in rows_out:
        a = r["delta_decA"]
        stable = "yes" if r["sign_stable_decA"] else "**NO**"
        md_lines.append(
            f"| {r['row']} | {r['paradigm']} | {r['net']} | "
            f"{a['seed42']:+.4f} | {a['seed43']:+.4f} | {a['seed44']:+.4f} | "
            f"{r['signs_decA']} | {stable} |"
        )
    md_lines.append("")

    if out_json["summary"]["rows_unstable_decC"]:
        md_lines.append("## Δ_decC sign flips (paradigm × net cells where sign is NOT unanimous across seeds)\n")
        for r in out_json["summary"]["rows_unstable_decC"]:
            md_lines.append(f"- Row {r['row']}: {r['paradigm']} | {r['net']} | signs across seeds = `{r['signs_decC']}`")
        md_lines.append("")
    else:
        md_lines.append("## Δ_decC sign flips\n\nAll 17 rows unanimous across the 3 seeds.\n")

    if out_json["summary"]["rows_unstable_decA"]:
        md_lines.append("## Δ_decA sign flips\n")
        for r in out_json["summary"]["rows_unstable_decA"]:
            md_lines.append(f"- Row {r['row']}: {r['paradigm']} | {r['net']} | signs across seeds = `{r['signs_decA']}`")
        md_lines.append("")
    else:
        md_lines.append("## Δ_decA sign flips\n\nAll 17 rows unanimous across the 3 seeds.\n")

    md_lines.append("## Headline paradigms (Phase 4-7 mechanism map)\n")
    md_lines.append(
        "Cross-reference to docs/paradigm_sign_mechanism.md. Each row below "
        "is a paradigm whose Dec-C sign is the basis for a Mech 1 / Mech 2 "
        "verdict in that doc. The 'sign stable?' column tells us whether "
        "the verdict survives 3-seed perturbation.\n"
    )
    md_lines.append("| Paradigm | Net | Row | Phase 4-7 Mech (Dec C) | Δ_decC seed42 | seed43 | seed44 | Sign stable? |")
    md_lines.append("|---|---|---:|---|---:|---:|---:|:--:|")
    for name, info in headline_paradigms.items():
        idx = info["row"]
        r_match = next((rr for rr in rows_out if rr["row"] == idx), None)
        if r_match is None:
            continue
        c = r_match["delta_decC"]
        stable = "yes" if r_match["sign_stable_decC"] else "**NO**"
        md_lines.append(
            f"| {name} | {r_match['net']} | {idx} | {info['mech_decC']} | "
            f"{c['seed42']:+.4f} | {c['seed43']:+.4f} | {c['seed44']:+.4f} | "
            f"{stable} |"
        )

    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_md, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"[md]   wrote {args.output_md}")


if __name__ == "__main__":
    main()
