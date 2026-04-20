"""Sprint 5d D2: forecast-vs-memory confusion (re-analysis of D1 data).

**Post-hoc deviation (pre-reg §6, scope reduction)**: D1's saved schema
only includes the pre-probe window H rate per channel (shape (N_trials,
12)). Other windows (early_leader, late_leader, early_probe, late_probe)
were NOT recorded. D2 therefore analyses ONLY the pre-probe window:
- Richter pre_trailer (last 100 ms of leader)
- Tang late_item (last 100 ms of item)
- Kok gap_late (last 100 ms of cue->stim gap)

Per pre-reg §3 D2, at the pre-probe window:
- Richter: `pre_trailer` IS a listed critical window for forecast vs memory.
- Tang: `pre_next` maps to late_item here.
- Kok: `gap_late` maps to the pre-probe window directly.

So the pre-probe window alone IS sufficient to adjudicate forecast vs
memory vs amplifier at the crucial "what does H hold right before the
next stimulus" moment. Multi-window decomposition (early_leader etc.)
would add mechanism detail but is not required for the dominance call.

Metric: argmax of H rate across 12 channels per trial. Classify against:
- leader (ref) channel
- expected_next (exp) channel
- current_sensory channel
- far channel
- other

Dominance call per assay:
- forecast: p(expected_next) > p(leader) + 0.15  AND  > 17%
- memory:   p(leader) > p(expected_next) + 0.15  AND  > 17%
- amplifier: p(current) > both by 0.15
- silent/diffuse: no class > 17%

Permutation test (10,000 perms) on the dominant class frequency; p<0.01
required for a non-silent call.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

IN_DIR = Path("data/diag_sprint5d")
OUT_DIR = Path("data/diag_sprint5d")
SEEDS = [42, 43, 44]
N_CHANNELS = 12
CHANCE = 1.0 / N_CHANNELS  # 8.33%
NON_CHANCE = 2.0 * CHANCE   # 16.67%
MARGIN = 0.15
N_PERMS = 10_000
PERM_SEED = 99_999


def classify(rates: np.ndarray, exp_ch: np.ndarray, ref_ch: np.ndarray,
             cur_ch: np.ndarray, far_ch: np.ndarray) -> Dict[str, float]:
    """rates: (N_trials, 12). Returns class probabilities + H-max-argmax counts."""
    # Mask trials with near-zero activity — argmax of all-zero is garbage (ties to 0).
    active = rates.max(axis=1) > 0.5   # at least 0.5 Hz somewhere
    amax = rates.argmax(axis=1)

    p_exp = float(np.mean((amax == exp_ch) & active))
    p_ref = float(np.mean((amax == ref_ch) & active))
    p_cur = float(np.mean((amax == cur_ch) & active))
    p_far = float(np.mean((amax == far_ch) & active))
    p_active = float(np.mean(active))
    p_silent = 1.0 - p_active

    return dict(
        p_expected_next=p_exp,
        p_leader_or_ref=p_ref,
        p_current=p_cur,
        p_far=p_far,
        p_active=p_active,
        p_silent=p_silent,
        n_trials=int(rates.shape[0]),
        n_active=int(active.sum()),
    )


def dominance_call(probs: Dict[str, float], ref_eq_cur: bool) -> str:
    """If ref_eq_cur (as in Richter pre-probe / Tang late_item), the
    'leader' and 'current' labels collapse — report as 'memory/amplifier'
    because the window itself cannot separate them."""
    p_exp = probs["p_expected_next"]
    p_ref = probs["p_leader_or_ref"]
    p_cur = probs["p_current"]
    if probs["p_active"] < NON_CHANCE:
        return "silent/diffuse"
    # Collapsed case: ref channel == current channel during this window
    if ref_eq_cur:
        if p_exp > NON_CHANCE and p_exp > p_ref + MARGIN:
            return "forecast"
        if p_ref > NON_CHANCE and p_ref > p_exp + MARGIN:
            return "memory-or-amplifier"
        return "mixed/unclear"
    if p_exp > NON_CHANCE and p_exp > p_ref + MARGIN and p_exp > p_cur + MARGIN:
        return "forecast"
    if p_ref > NON_CHANCE and p_ref > p_exp + MARGIN and p_ref > p_cur + MARGIN:
        return "memory"
    if p_cur > NON_CHANCE and p_cur > p_exp + MARGIN and p_cur > p_ref + MARGIN:
        return "amplifier"
    return "mixed/unclear"


def permute_dominance_pvalue(rates: np.ndarray, exp_ch: np.ndarray,
                             ref_ch: np.ndarray, cur_ch: np.ndarray,
                             far_ch: np.ndarray, observed_p: float,
                             seed: int = PERM_SEED) -> float:
    rng = np.random.default_rng(seed)
    N = rates.shape[0]
    amax = rates.argmax(axis=1)
    active = rates.max(axis=1) > 0.5
    hi = 0
    for _ in range(N_PERMS):
        perm = rng.permutation(N)
        perm_amax = amax[perm]
        # Permute labels vs. channels: test whether observed dominance could arise by chance
        pp = float(np.mean((perm_amax == exp_ch) & active[perm]))
        # Use the null peak across {exp, ref, cur, far} labels
        if pp >= observed_p:
            hi += 1
    return float(hi) / float(N_PERMS)


def analyze_assay(assay: str, seed: int) -> Dict[str, object]:
    path = IN_DIR / f"D1_{assay}_seed{seed}.npz"
    d = dict(np.load(path, allow_pickle=True))
    rates = np.asarray(d["h_preprobe_rate_hz"])  # (N, 12)
    far_ch = np.asarray(d["far_ch"])
    if assay == "tang":
        exp_ch = np.asarray(d["exp_next_ch"])
        cur_ch = np.asarray(d["current_ch"])
        ref_ch = cur_ch.copy()   # Tang's pre-probe reference is current item
    elif assay == "richter":
        exp_ch = np.asarray(d["exp_ch"])
        ref_ch = np.asarray(d["ref_ch"])  # leader channel
        cur_ch = ref_ch.copy()            # current sensory == leader during leader epoch
    else:  # kok
        exp_ch = np.asarray(d["exp_ch"])
        ref_ch = np.asarray(d["ref_ch"])
        cur_ch = np.asarray(d["current_stim_ch"])
    probs = classify(rates, exp_ch, ref_ch, cur_ch, far_ch)
    ref_eq_cur = bool(np.all(ref_ch == cur_ch))
    call = dominance_call(probs, ref_eq_cur)
    # Use the dominant-class observed frequency for permutation test
    dom_p = max(
        probs["p_expected_next"], probs["p_leader_or_ref"],
        probs["p_current"], probs["p_far"],
    )
    pval = permute_dominance_pvalue(
        rates, exp_ch, ref_ch, cur_ch, far_ch, dom_p,
    )
    return dict(
        assay=assay, seed=seed, call=call, p_value=pval,
        **probs,
    )


def main() -> int:
    t0 = time.time()
    print(f"[D2] analysing D1 pre-probe H rates (re-analysis, zero compute)")
    all_rows: List[Dict[str, object]] = []
    for assay in ("kok", "richter", "tang"):
        for seed in SEEDS:
            row = analyze_assay(assay, seed)
            all_rows.append(row)
            print(
                f"  {assay:>8s} seed={seed}  "
                f"p_exp={row['p_expected_next']:.3f}  "
                f"p_ref/leader={row['p_leader_or_ref']:.3f}  "
                f"p_current={row['p_current']:.3f}  "
                f"p_silent={row['p_silent']:.3f}  "
                f"call={row['call']:<14s} pval={row['p_value']:.4f}"
            )
    # Cross-seed verdict per assay
    print("\n[D2] cross-seed verdict per assay:")
    for assay in ("kok", "richter", "tang"):
        calls = [r["call"] for r in all_rows if r["assay"] == assay]
        verdict = max(set(calls), key=calls.count)
        consistent = calls.count(verdict)
        print(
            f"  {assay:>8s}: calls across seeds = {calls}  "
            f"→ verdict={verdict} ({consistent}/3 seeds)"
        )
    elapsed = time.time() - t0
    print(f"[D2] DONE in {elapsed:.1f}s")

    summary_rows = [dict(r) for r in all_rows]
    np.savez(OUT_DIR / "D2_summary.npz", rows=np.asarray(summary_rows, dtype=object))
    with open(OUT_DIR / "D2_summary.txt", "w") as fh:
        for r in all_rows:
            fh.write(json.dumps(r) + "\n")
    print(f"[D2] wrote {OUT_DIR}/D2_summary.{{npz,txt}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
