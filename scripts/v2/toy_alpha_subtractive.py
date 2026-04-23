"""Task #74 toy test — subtractive apical with bidirectional cue patterns.

Minimal, standalone numerical experiment that asks: can a simple
subtractive-apical mechanism ``r_j = softplus(basal_j(probe) -
apical_j(cue))`` produce BOTH sharpening AND dampening signatures on
one population by swapping the apical prediction pattern?

Before the Option-2 build commits to a subtractive apical with
cue-conditional predictions, we need to know the mechanism is
mechanically reachable on clean hand-crafted weights — i.e. that
the math itself admits both directions, independent of any training
artefact or architectural confound.

Setup
-----
- Toy L2/3 population: ``N=32`` units with preferred orientations
  tiled 0°–180° at 5.625° spacing (periodic, orientation not
  direction).
- Basal tuning: Gaussian on circular angle diff,
  ``b_j(probe) = exp(-Δθ² / (2σ²))`` with ``σ = 15°``.
- Three apical patterns (all bake their α coefficients into the
  returned vector so the integrator stays parameter-free):
    - **Pattern A** — ``apical_j(cue) = α · exp(-Δθ_cue² / (2σ²))``.
      Apical has the SAME shape as basal for the cued orientation,
      peaking on pref-of-cue units. Under matched (cue=probe) this
      subtracts the peak of the basal response — a "dampening-direction"
      candidate.
    - **Pattern B** — ``apical_j(cue) = α · (1 - exp(-Δθ_cue² / (2σ²)))``.
      Apical has the INVERTED shape — peaks on nonpref-of-cue, zero on
      pref-of-cue. Under matched, subtraction kills off-pref responses
      and leaves pref intact (re-scaled by ``(1+α)``) — a
      "sharpening-direction" candidate.
    - **Pattern C** — ``apical_j(cue) = α₀ + α₁ · (1 - exp(-Δθ_cue² / (2σ²)))``.
      Mixed: a uniform baseline ``α₀`` (pulls total firing down on
      every unit) plus ``α₁`` of the nonpref-pattern (sharpens
      pref/nonpref ratio). Motivated by the observation that Pattern B
      alone produces the wrong total-firing asymmetry, so adding a
      uniform floor gives the sweep room to meet all three
      sharpening criteria simultaneously.
- Integration: ``r_j = softplus(basal_j(probe) - apical_j(cue))``.
  Softplus was chosen over ReLU so the activation has a smooth floor
  (``softplus(x) → log(1 + e^x)``, ``softplus(0) = log 2 ≈ 0.693``);
  under ReLU, Pattern B mismatched collapses many units to exactly
  0 and the pref/nonpref ratio becomes degenerate.

Sweeps
------
- Pattern A: ``α ∈ {0.0, 0.3, 0.5, 1.0, 1.5}`` — 5 cells.
- Pattern B: ``α ∈ {0.0, 0.3, 0.5, 1.0, 1.5}`` — 5 cells.
- Pattern C: ``α₀ ∈ {0.0, 0.1, 0.3, 0.5}`` × ``α₁ ∈ {0.3, 0.5, 1.0, 1.5}`` — 16 cells.

Per-condition trials are run with small independent Gaussian noise
on ``basal`` so the cosine-match classifier has non-trivial
trial-level variance to score.

Metrics per (pattern, α..., condition)
---------------------------------------
1. ``mean_r``: grand mean of ``r`` across units and trials.
2. ``pref/nonpref ratio``: mean(r on units with pref within ±15° of
   probe) / mean(r on units with pref within ±15° of probe+90°),
   per trial then aggregated via median.
3. ``acc``: cosine-match readout — classify each trial by the
   argmax cosine similarity between ``r`` and each of the two ideal
   α=0 templates; report fraction correct.

Verdict rule (exactly the lead's spec, strict 3-criterion)
-----------------------------------------------------------
- Pattern A produces dampening iff
  ``mean_r_match < mean_r_mismatch`` AND
  ``pref/nonpref_match < pref/nonpref_mismatch`` AND
  ``Δacc = acc_match − acc_mismatch ≤ 0`` — reported per α.
- Pattern B produces sharpening iff
  ``mean_r_match < mean_r_mismatch`` AND
  ``pref/nonpref_match > pref/nonpref_mismatch`` AND
  ``Δacc ≥ 0`` — reported per α.
- Pattern C produces sharpening iff the same sharpening criteria are
  met per (α₀, α₁) cell.
- ``mechanism_bidirectional`` iff Pattern A produces dampening AND
  Pattern C produces sharpening at at least one cell each.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Core mechanism — self-contained, no imports from src/v2_model
# ---------------------------------------------------------------------------


def build_preferred_orientations(n_units: int = 32) -> np.ndarray:
    """Evenly tiled pref orientations over [0°, 180°)."""
    return np.linspace(0.0, 180.0, int(n_units), endpoint=False)


def angle_diff_deg(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    """Circular orientation difference on a 180° period. Returns |Δθ|
    ∈ [0°, 90°]."""
    d = np.abs(np.asarray(a) - np.asarray(b))
    return np.minimum(d, 180.0 - d)


def basal_drive(
    probe_deg: float, pref_deg: np.ndarray, sigma_deg: float = 15.0,
) -> np.ndarray:
    """``b_j(probe) = exp(-Δθ² / (2σ²))`` — Gaussian tuning on the
    circular orientation difference. Peak = 1 at pref_j = probe."""
    d = angle_diff_deg(probe_deg, pref_deg)
    return np.exp(-(d ** 2) / (2.0 * float(sigma_deg) ** 2))


def apical_pattern_A(
    cue_deg: float, pref_deg: np.ndarray, sigma_deg: float = 15.0,
    *, alpha: float = 1.0,
) -> np.ndarray:
    """Pattern A: ``α · exp(-Δθ²/(2σ²))`` — same Gaussian shape as
    basal, peaked on pref-of-cue units, scaled by ``α``."""
    return float(alpha) * basal_drive(cue_deg, pref_deg, sigma_deg)


def apical_pattern_B(
    cue_deg: float, pref_deg: np.ndarray, sigma_deg: float = 15.0,
    *, alpha: float = 1.0,
) -> np.ndarray:
    """Pattern B: ``α · (1 - exp(-Δθ²/(2σ²)))`` — inverted shape,
    peaks on nonpref-of-cue, zero on pref-of-cue, scaled by ``α``."""
    return float(alpha) * (1.0 - basal_drive(cue_deg, pref_deg, sigma_deg))


def apical_pattern_C(
    cue_deg: float, pref_deg: np.ndarray, sigma_deg: float = 15.0,
    *, alpha0: float = 0.0, alpha1: float = 1.0,
) -> np.ndarray:
    """Pattern C: ``α₀ + α₁ · (1 - exp(-Δθ²/(2σ²)))`` — uniform
    baseline ``α₀`` plus Pattern-B-shaped component scaled by ``α₁``.

    ``α₀`` provides a uniform subtractive floor that pulls total
    firing down on every unit; ``α₁`` provides the nonpref-pattern
    component that sharpens the pref/nonpref ratio. The mixture is
    motivated by Pattern B's failure to meet the total-firing
    criterion of strict Kok sharpening on its own.
    """
    return float(alpha0) + float(alpha1) * (
        1.0 - basal_drive(cue_deg, pref_deg, sigma_deg)
    )


def _softplus(x: np.ndarray) -> np.ndarray:
    """Numerically stable softplus: ``log(1 + exp(x))``.

    For ``x ≫ 0`` returns ``x`` (asymptotic); for ``x ≪ 0`` returns
    ``exp(x)`` — avoids overflow when ``x`` is large positive.
    """
    # np.logaddexp(0, x) = log(e^0 + e^x) = log(1 + e^x) = softplus(x),
    # computed stably.
    return np.logaddexp(0.0, x)


def integrate(basal: np.ndarray, apical: np.ndarray) -> np.ndarray:
    """``r_j = softplus(b_j - a_j)`` — minimal subtractive mechanism
    with a smooth non-negative floor at ``log 2 ≈ 0.693``.

    All α coefficients are expected to be baked into ``apical`` by
    the pattern functions above, so the integrator itself is
    parameter-free.
    """
    return _softplus(np.asarray(basal) - np.asarray(apical))


# ---------------------------------------------------------------------------
# Trial orchestration
# ---------------------------------------------------------------------------


def _pref_mask(
    pref_deg: np.ndarray, center_deg: float, tol_deg: float = 15.0,
) -> np.ndarray:
    return angle_diff_deg(center_deg, pref_deg) <= float(tol_deg)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def classify_by_cosine(
    r: np.ndarray, templates: dict[float, np.ndarray],
) -> float:
    """Cosine-match readout. Returns the template key (probe_deg) with
    highest cosine similarity to ``r``. Ties are broken by sorting
    order, which here is deterministic because templates is an ordered
    dict in Python 3.7+."""
    labels = list(templates.keys())
    sims = [_cosine(r, templates[lbl]) for lbl in labels]
    return float(labels[int(np.argmax(sims))])


def _run_condition(
    pref_deg: np.ndarray, *, probes_deg: list[float],
    apical_builder: Callable[[float, np.ndarray, float], np.ndarray],
    sigma_deg: float, is_matched: bool,
    n_trials_per_probe: int, noise_std: float, rng: np.random.Generator,
    templates: dict[float, np.ndarray],
) -> dict[str, Any]:
    """Run ``n_trials_per_probe`` noisy trials per probe under one
    (pattern × α..., matched/mismatched) cell.

    ``apical_builder(cue_deg, pref_deg, sigma_deg)`` returns the
    already-scaled apical vector — all α coefficients are baked in
    by the caller, keeping this runner parameter-free.
    """
    r_means: list[float] = []
    pref_nonpref_ratios: list[float] = []
    acc_flags: list[float] = []
    for probe_deg in probes_deg:
        cue_deg = (
            float(probe_deg) if is_matched
            else float((probe_deg + 90.0) % 180.0)
        )
        basal = basal_drive(float(probe_deg), pref_deg, sigma_deg)
        apical = apical_builder(float(cue_deg), pref_deg, sigma_deg)
        pref_m = _pref_mask(pref_deg, float(probe_deg))
        nonpref_m = _pref_mask(
            pref_deg, float((probe_deg + 90.0) % 180.0),
        )
        for _ in range(int(n_trials_per_probe)):
            noise = rng.normal(0.0, float(noise_std), size=pref_deg.shape)
            r = integrate(basal + noise, apical)
            r_means.append(float(r.mean()))
            pref_mean = float(r[pref_m].mean()) if pref_m.any() else 0.0
            nonpref_mean = (
                float(r[nonpref_m].mean()) if nonpref_m.any() else 0.0
            )
            pref_nonpref_ratios.append(pref_mean / (nonpref_mean + 1e-9))
            pred = classify_by_cosine(r, templates)
            acc_flags.append(1.0 if abs(pred - float(probe_deg)) < 1e-6 else 0.0)
    return {
        "mean_r": float(np.mean(r_means)),
        "pref_over_nonpref": float(np.median(pref_nonpref_ratios)),
        "acc": float(np.mean(acc_flags)),
        "n_trials": int(len(acc_flags)),
    }


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def _dampening_flags(row: dict[str, Any]) -> dict[str, bool]:
    return {
        "mean_r": row["mean_r_match"] < row["mean_r_mismatch"],
        "pref_over_nonpref": (
            row["pref_over_nonpref_match"]
            < row["pref_over_nonpref_mismatch"]
        ),
        "delta_acc": row["delta_acc"] <= 0.0,
    }


def _sharpening_flags(row: dict[str, Any]) -> dict[str, bool]:
    return {
        "mean_r": row["mean_r_match"] < row["mean_r_mismatch"],
        "pref_over_nonpref": (
            row["pref_over_nonpref_match"]
            > row["pref_over_nonpref_mismatch"]
        ),
        "delta_acc": row["delta_acc"] >= 0.0,
    }


def _one_cell(
    pref_deg: np.ndarray, *, apical_builder: Callable,
    probes_deg: list[float], sigma_deg: float, n_trials_per_probe: int,
    noise_std: float, rng: np.random.Generator,
    templates: dict[float, np.ndarray], extras: dict[str, Any],
) -> dict[str, Any]:
    """Run one matched+mismatched cell and return the row dict."""
    matched = _run_condition(
        pref_deg, probes_deg=probes_deg, apical_builder=apical_builder,
        sigma_deg=sigma_deg, is_matched=True,
        n_trials_per_probe=n_trials_per_probe,
        noise_std=noise_std, rng=rng, templates=templates,
    )
    mismatched = _run_condition(
        pref_deg, probes_deg=probes_deg, apical_builder=apical_builder,
        sigma_deg=sigma_deg, is_matched=False,
        n_trials_per_probe=n_trials_per_probe,
        noise_std=noise_std, rng=rng, templates=templates,
    )
    row = dict(extras)
    row.update({
        "mean_r_match": matched["mean_r"],
        "mean_r_mismatch": mismatched["mean_r"],
        "pref_over_nonpref_match": matched["pref_over_nonpref"],
        "pref_over_nonpref_mismatch": mismatched["pref_over_nonpref"],
        "acc_match": matched["acc"],
        "acc_mismatch": mismatched["acc"],
        "delta_acc": float(matched["acc"] - mismatched["acc"]),
        "n_trials": int(matched["n_trials"]),
    })
    return row


def run_sweep(
    *, n_units: int = 32, sigma_deg: float = 15.0,
    probes_deg: Optional[list[float]] = None,
    alphas_AB: Optional[list[float]] = None,
    alphas_C_0: Optional[list[float]] = None,
    alphas_C_1: Optional[list[float]] = None,
    n_trials_per_probe: int = 40, noise_std: float = 0.02,
    seed: int = 42,
) -> dict[str, Any]:
    """Run patterns A, B, C sweeps. Returns a nested dict with all
    rows plus per-criterion verdict flags."""
    if probes_deg is None:
        probes_deg = [45.0, 135.0]
    if alphas_AB is None:
        alphas_AB = [0.0, 0.3, 0.5, 1.0, 1.5]
    if alphas_C_0 is None:
        alphas_C_0 = [0.0, 0.1, 0.3, 0.5]
    if alphas_C_1 is None:
        alphas_C_1 = [0.3, 0.5, 1.0, 1.5]

    pref_deg = build_preferred_orientations(n_units)
    templates = {float(p): basal_drive(float(p), pref_deg, sigma_deg)
                 for p in probes_deg}
    rng = np.random.default_rng(int(seed))

    # --- Pattern A -------------------------------------------------------
    rows_A: list[dict[str, Any]] = []
    for alpha in alphas_AB:
        def _builder(cue, p, s, _a=float(alpha)):
            return apical_pattern_A(cue, p, s, alpha=_a)
        row = _one_cell(
            pref_deg, apical_builder=_builder,
            probes_deg=probes_deg, sigma_deg=sigma_deg,
            n_trials_per_probe=n_trials_per_probe,
            noise_std=noise_std, rng=rng, templates=templates,
            extras={"alpha": float(alpha)},
        )
        row["dampening_flags"] = _dampening_flags(row)
        row["sharpening_flags"] = _sharpening_flags(row)
        rows_A.append(row)

    # --- Pattern B -------------------------------------------------------
    rows_B: list[dict[str, Any]] = []
    for alpha in alphas_AB:
        def _builder(cue, p, s, _a=float(alpha)):
            return apical_pattern_B(cue, p, s, alpha=_a)
        row = _one_cell(
            pref_deg, apical_builder=_builder,
            probes_deg=probes_deg, sigma_deg=sigma_deg,
            n_trials_per_probe=n_trials_per_probe,
            noise_std=noise_std, rng=rng, templates=templates,
            extras={"alpha": float(alpha)},
        )
        row["dampening_flags"] = _dampening_flags(row)
        row["sharpening_flags"] = _sharpening_flags(row)
        rows_B.append(row)

    # --- Pattern C (mixed) -----------------------------------------------
    rows_C: list[dict[str, Any]] = []
    for a0 in alphas_C_0:
        for a1 in alphas_C_1:
            def _builder(cue, p, s, _a0=float(a0), _a1=float(a1)):
                return apical_pattern_C(
                    cue, p, s, alpha0=_a0, alpha1=_a1,
                )
            row = _one_cell(
                pref_deg, apical_builder=_builder,
                probes_deg=probes_deg, sigma_deg=sigma_deg,
                n_trials_per_probe=n_trials_per_probe,
                noise_std=noise_std, rng=rng, templates=templates,
                extras={"alpha0": float(a0), "alpha1": float(a1)},
            )
            row["dampening_flags"] = _dampening_flags(row)
            row["sharpening_flags"] = _sharpening_flags(row)
            rows_C.append(row)

    # --- Collate qualifying cells ----------------------------------------
    dampening_A = [
        row["alpha"] for row in rows_A
        if all(row["dampening_flags"].values())
    ]
    sharpening_B = [
        row["alpha"] for row in rows_B
        if all(row["sharpening_flags"].values())
    ]
    sharpening_C = [
        (row["alpha0"], row["alpha1"]) for row in rows_C
        if all(row["sharpening_flags"].values())
    ]
    # Recommended operating point: sharpening-qualifying C cell with
    # strongest (most positive) Δacc.
    recommended = None
    best_delta = -float("inf")
    for row in rows_C:
        if all(row["sharpening_flags"].values()):
            if row["delta_acc"] > best_delta:
                best_delta = row["delta_acc"]
                recommended = {
                    "alpha0": float(row["alpha0"]),
                    "alpha1": float(row["alpha1"]),
                    "delta_acc": float(row["delta_acc"]),
                    "mean_r_match": float(row["mean_r_match"]),
                    "mean_r_mismatch": float(row["mean_r_mismatch"]),
                    "pref_over_nonpref_match":
                        float(row["pref_over_nonpref_match"]),
                    "pref_over_nonpref_mismatch":
                        float(row["pref_over_nonpref_mismatch"]),
                }
    bidirectional = bool(dampening_A and sharpening_C)

    return {
        "n_units": int(n_units),
        "sigma_deg": float(sigma_deg),
        "probes_deg": [float(p) for p in probes_deg],
        "alphas_AB": [float(a) for a in alphas_AB],
        "alphas_C_0": [float(a) for a in alphas_C_0],
        "alphas_C_1": [float(a) for a in alphas_C_1],
        "n_trials_per_probe": int(n_trials_per_probe),
        "noise_std": float(noise_std),
        "pattern_A": rows_A,
        "pattern_B": rows_B,
        "pattern_C": rows_C,
        "dampening_A_at_alphas": dampening_A,
        "sharpening_B_at_alphas": sharpening_B,
        "sharpening_C_at_alpha_cells": [
            [float(a0), float(a1)] for (a0, a1) in sharpening_C
        ],
        "recommended_operating_point": recommended,
        "mechanism_bidirectional": bidirectional,
    }


# ---------------------------------------------------------------------------
# Stdout formatting
# ---------------------------------------------------------------------------


def _fmt_row_AB(row: dict[str, Any]) -> str:
    return (
        f"{row['alpha']:<6.2f} "
        f"{row['mean_r_match']:<13.4f} "
        f"{row['mean_r_mismatch']:<17.4f} "
        f"{row['pref_over_nonpref_match']:<20.4f} "
        f"{row['pref_over_nonpref_mismatch']:<23.4f} "
        f"{row['acc_match']:<11.4f} "
        f"{row['acc_mismatch']:<14.4f} "
        f"{row['delta_acc']:+.4f}"
    )


def _fmt_row_C(row: dict[str, Any]) -> str:
    return (
        f"{row['alpha0']:<6.2f} "
        f"{row['alpha1']:<6.2f} "
        f"{row['mean_r_match']:<13.4f} "
        f"{row['mean_r_mismatch']:<17.4f} "
        f"{row['pref_over_nonpref_match']:<20.4f} "
        f"{row['pref_over_nonpref_mismatch']:<23.4f} "
        f"{row['acc_match']:<11.4f} "
        f"{row['acc_mismatch']:<14.4f} "
        f"{row['delta_acc']:+.4f}"
    )


def _print_report(result: dict[str, Any]) -> None:
    print(
        f"toy_alpha_subtractive (N={result['n_units']}, "
        f"σ={result['sigma_deg']:g}°, probe ∈ "
        f"{{{', '.join(f'{p:g}°' for p in result['probes_deg'])}}})"
    )
    header_AB = (
        "α      mean_r_match   mean_r_mismatch   pref/nonpref_match  "
        "  pref/nonpref_mismatch   acc_match   acc_mismatch   Δacc"
    )
    header_C = (
        "α₀     α₁     mean_r_match   mean_r_mismatch   pref/nonpref_match  "
        "  pref/nonpref_mismatch   acc_match   acc_mismatch   Δacc"
    )
    print()
    print(
        "Pattern A — apical carries pref-pattern prediction "
        "(dampening-direction candidate)"
    )
    print(header_AB)
    for row in result["pattern_A"]:
        print(_fmt_row_AB(row))
    print()
    print(
        "Pattern B — apical carries nonpref-pattern prediction "
        "(sharpening-direction candidate)"
    )
    print(header_AB)
    for row in result["pattern_B"]:
        print(_fmt_row_AB(row))
    print()
    print(
        "Pattern C — apical = α₀ (uniform) + α₁ (nonpref-pattern)"
    )
    print(header_C)
    for row in result["pattern_C"]:
        print(_fmt_row_C(row))

    print()
    print("verdict:")
    da = result["dampening_A_at_alphas"]
    sb = result["sharpening_B_at_alphas"]
    sc = result["sharpening_C_at_alpha_cells"]
    da_str = (
        f"Y at α ∈ {{{', '.join(f'{a:g}' for a in da)}}}"
        if da else "N (no α satisfies all three dampening criteria)"
    )
    sb_str = (
        f"Y at α ∈ {{{', '.join(f'{a:g}' for a in sb)}}}"
        if sb else "N (no α satisfies all three sharpening criteria)"
    )
    sc_str = (
        "Y at (α₀, α₁) ∈ {"
        + ", ".join(f"({a0:g}, {a1:g})" for (a0, a1) in sc)
        + "}"
        if sc else "N (no cell satisfies all three sharpening criteria)"
    )
    print(
        "  Pattern A produces dampening "
        "(mean_r_match < mean_r_mismatch AND pref/nonpref_match < "
        "pref/nonpref_mismatch AND Δacc ≤ 0): " + da_str
    )
    print(
        "  Pattern B produces sharpening "
        "(mean_r_match < mean_r_mismatch AND pref/nonpref_match > "
        "pref/nonpref_mismatch AND Δacc ≥ 0): " + sb_str
    )
    print(
        "  Pattern C satisfies strict Kok sharpening "
        "(mean_r_match < mean_r_mismatch AND pref/nonpref_match > "
        "pref/nonpref_mismatch AND Δacc ≥ 0): " + sc_str
    )
    rec = result["recommended_operating_point"]
    if rec is None:
        rec_str = "none"
    else:
        rec_str = (
            f"(α₀={rec['alpha0']:g}, α₁={rec['alpha1']:g}) "
            f"with Δacc={rec['delta_acc']:+.4f}"
        )
    print(f"  recommended_operating_point: {rec_str}")
    print(
        "  mechanism_bidirectional (Pattern A dampening AND Pattern C "
        "sharpening, both satisfying all 3 criteria): "
        f"{'Y' if result['mechanism_bidirectional'] else 'N'}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-units", type=int, default=32)
    p.add_argument("--sigma-deg", type=float, default=15.0)
    p.add_argument("--n-trials-per-probe", type=int, default=40)
    p.add_argument("--noise-std", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--alphas-ab", type=float, nargs="+",
        default=[0.0, 0.3, 0.5, 1.0, 1.5],
        help="α values swept for Pattern A and Pattern B.",
    )
    p.add_argument(
        "--alphas-c-0", type=float, nargs="+",
        default=[0.0, 0.1, 0.3, 0.5],
        help="α₀ values swept for Pattern C (uniform floor).",
    )
    p.add_argument(
        "--alphas-c-1", type=float, nargs="+",
        default=[0.3, 0.5, 1.0, 1.5],
        help="α₁ values swept for Pattern C (nonpref-pattern strength).",
    )
    p.add_argument(
        "--output", type=Path,
        default=Path("logs/task74/toy_alpha_subtractive.json"),
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _cli().parse_args(argv)
    result = run_sweep(
        n_units=int(args.n_units),
        sigma_deg=float(args.sigma_deg),
        alphas_AB=list(args.alphas_ab),
        alphas_C_0=list(args.alphas_c_0),
        alphas_C_1=list(args.alphas_c_1),
        n_trials_per_probe=int(args.n_trials_per_probe),
        noise_std=float(args.noise_std),
        seed=int(args.seed),
    )
    _print_report(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"[toy] wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
