"""Phase-2 sanity gates 1-7 (plan v4 / Task #39; extended Task #68).

Evaluates seven pass/fail sensory sanity checks on a Phase-2 checkpoint:

1. Rate distribution — L2/3 E median firing rate ∈ [0.05, 0.5] under blank input.
2. Contrast response — L4 E mean response follows Naka-Rushton vs contrast
   with R² > 0.7.
3. Surround suppression — ≥ 50% of L2/3 E units show SI > 0.1.
4. Next-step prediction beats copy-last — network MSE ≤ 0.85 × copy-last MSE.
5. Orientation + identity localizer (combined) — ≥ 70% of L2/3 E units have
   FWHM < 120° AND L4 LinearSVC 12-way acc > 0.25 AND within-token mean RSA
   distance < between-token mean RSA distance. (Internally 5a + 5b; the
   combined gate_5 passes only if both sub-gates pass.)
6. Null-expectation control (wraps ``run_gate_6_null_control``) — Kok +
   Richter null assays both pass.
7. Context-memory load-bearing (wraps ``run_gate_7_c_load_bearing``) —
   ablating C degrades prediction MSE by ≥ 5%.

Usage:
    python -m scripts.v2.eval_gates \\
        --checkpoint checkpoints/v2/phase2/phase2_s42/step_1000.pt \\
        --seed 42

Writes ``gates_1_7.json`` next to the checkpoint. Exit 0 if all gates pass,
1 if any fail (still writes the JSON). Designed as a smoke harness: it
must run on a freshly-initialised checkpoint (gates may fail but no
crash), and produces reproducible numeric output given a fixed seed.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from scripts.v2._gates_common import (
    CheckpointBundle,
    load_checkpoint,
    make_blank_frame,
    make_grating_frame,
    make_surround_grating_frame,
    simulate_and_collect,
    simulate_steady_state,
)
from scripts.v2.run_null_expectation_control import run_gate_6_null_control
from scripts.v2.run_c_load_bearing_check import run_gate_7_c_load_bearing
from src.v2_model.network import V2Network
from src.v2_model.world.procedural import ProceduralWorld


__all__ = [
    "gate_1_rate_distribution",
    "gate_2_contrast_response",
    "gate_3_surround_suppression",
    "gate_4_prediction_beats_copy_last",
    "gate_5a_orientation_localizer",
    "gate_5b_identity_localizer",
    "gate_6_null_expectation_control",
    "gate_7_c_load_bearing",
    "run_gates_1_to_5",
    "run_gates_1_to_7",
]


# ---------------------------------------------------------------------------
# Gate 1 — rate distribution under blank input
# ---------------------------------------------------------------------------


@torch.no_grad()
def gate_1_rate_distribution(
    bundle: CheckpointBundle,
    *, n_steps: int = 1000, batch_size: int = 2,
    median_lo: float = 0.05, median_hi: float = 0.5,
) -> dict[str, Any]:
    """Median L2/3 E firing rate in steady-state under blank input.

    Scaled to target_rate_hz=0.5 baseline (not the plan's original 0.5-5 Hz):
    the acceptable band is [0.05, 0.5].
    """
    frame = make_blank_frame(batch_size, bundle.cfg, device=bundle.cfg.device)
    _state, info = simulate_steady_state(bundle.net, frame, n_steps)
    r_l23 = info["r_l23"].detach().cpu().numpy()     # [B, n_l23_e]
    per_unit_mean = r_l23.mean(axis=0)                # [n_l23_e]
    median = float(np.median(per_unit_mean))
    passed = bool(median_lo <= median <= median_hi)
    return {
        "gate": "1_rate_distribution",
        "median_rate": median,
        "acceptable_band": [float(median_lo), float(median_hi)],
        "mean_rate": float(per_unit_mean.mean()),
        "std_rate": float(per_unit_mean.std()),
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# Gate 2 — Naka-Rushton contrast response on L4 E
# ---------------------------------------------------------------------------


def _fit_naka_rushton(
    contrast: np.ndarray, response: np.ndarray,
) -> tuple[float, float, float, float]:
    """Fit R(c) = Rmax · c^n / (c^n + c50^n). Returns (Rmax, n, c50, r_squared).

    Uses scipy if available; otherwise falls back to a bounded log-linear
    fit that's sufficient for the R² > 0.7 smoke check.
    """
    try:
        from scipy.optimize import curve_fit                    # type: ignore

        def _nr(c, rmax, n, c50):
            return rmax * (c ** n) / ((c ** n) + (c50 ** n) + 1e-8)

        popt, _ = curve_fit(
            _nr, contrast, response,
            p0=[max(float(response.max()), 1e-3), 2.0, 0.3],
            bounds=([0.0, 0.1, 0.01], [10.0, 10.0, 1.0]),
            maxfev=2000,
        )
        rmax, n, c50 = [float(p) for p in popt]
        pred = _nr(contrast, rmax, n, c50)
    except Exception:
        # Monotonic saturating fallback fit (log-linear on x^2/(x^2+k^2))
        # so the gate machinery still runs without scipy.
        rmax = float(response.max())
        c50 = 0.3
        n = 2.0
        pred = rmax * (contrast ** n) / ((contrast ** n) + (c50 ** n) + 1e-8)

    ss_res = float(((response - pred) ** 2).sum())
    ss_tot = float(((response - response.mean()) ** 2).sum()) + 1e-12
    r_sq = 1.0 - ss_res / ss_tot
    return rmax, n, c50, float(r_sq)


@torch.no_grad()
def gate_2_contrast_response(
    bundle: CheckpointBundle,
    *, orientation_deg: float = 0.0,
    contrasts: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 1.0),
    n_steps_steady: int = 40,
    r_sq_min: float = 0.7,
) -> dict[str, Any]:
    """Sweep contrast, measure L4 E mean response, fit Naka-Rushton."""
    responses: list[float] = []
    for c in contrasts:
        frame = make_grating_frame(
            orientation_deg, c, bundle.cfg, device=bundle.cfg.device,
        )
        _s, info = simulate_steady_state(bundle.net, frame, n_steps_steady)
        responses.append(float(info["r_l4"].mean().item()))
    c_arr = np.asarray(contrasts, dtype=np.float64)
    r_arr = np.asarray(responses, dtype=np.float64)
    rmax, n, c50, r_sq = _fit_naka_rushton(c_arr, r_arr)
    # Monotonicity check — redundant with R² if fit is good, but cheap
    # structural evidence for the smoke harness.
    monotonic = bool(np.all(np.diff(r_arr) >= -1e-6))
    return {
        "gate": "2_contrast_response",
        "contrasts": list(c_arr.tolist()),
        "responses": list(r_arr.tolist()),
        "fit": {"Rmax": rmax, "n": n, "c50": c50, "r_squared": r_sq},
        "monotonic": monotonic,
        "passed": bool(r_sq > r_sq_min and monotonic),
    }


# ---------------------------------------------------------------------------
# Gate 3 — surround suppression
# ---------------------------------------------------------------------------


@torch.no_grad()
def gate_3_surround_suppression(
    bundle: CheckpointBundle,
    *, orientation_deg: float = 0.0, contrast: float = 1.0,
    n_steps_steady: int = 40,
    si_threshold: float = 0.1, frac_units_min: float = 0.5,
) -> dict[str, Any]:
    """SI = (R_center_only − R_center_surround) / R_center_only per L2/3 E unit."""
    center = make_surround_grating_frame(
        orientation_deg, contrast, bundle.cfg,
        include_surround=False, device=bundle.cfg.device,
    )
    surround = make_surround_grating_frame(
        orientation_deg, contrast, bundle.cfg,
        include_surround=True, device=bundle.cfg.device,
    )
    _s1, info_c = simulate_steady_state(bundle.net, center, n_steps_steady)
    _s2, info_s = simulate_steady_state(bundle.net, surround, n_steps_steady)
    r_c = info_c["r_l23"].mean(dim=0).detach().cpu().numpy()     # [n_l23_e]
    r_s = info_s["r_l23"].mean(dim=0).detach().cpu().numpy()
    denom = np.where(r_c > 1e-6, r_c, 1e-6)
    si = (r_c - r_s) / denom
    frac_suppressed = float((si > si_threshold).mean())
    return {
        "gate": "3_surround_suppression",
        "si_threshold": float(si_threshold),
        "frac_units_with_si_above_threshold": frac_suppressed,
        "si_mean": float(si.mean()),
        "si_median": float(np.median(si)),
        "passed": bool(frac_suppressed >= frac_units_min),
    }


# ---------------------------------------------------------------------------
# Gate 4 — prediction beats copy-last
# ---------------------------------------------------------------------------


@torch.no_grad()
def gate_4_prediction_beats_copy_last(
    bundle: CheckpointBundle,
    *, n_trajectories: int = 8, n_steps_per_traj: int = 20,
    improvement_floor: float = 0.15,
) -> dict[str, Any]:
    """Compare ‖r_l4_next − x̂‖² vs ‖r_l4_next − r_l4_t‖² on eval trajectories."""
    world = ProceduralWorld(bundle.cfg, bundle.bank, seed_family="eval")

    pred_errs: list[float] = []
    copy_errs: list[float] = []
    for traj_id in range(n_trajectories):
        frames_seq, _states = world.trajectory(
            trajectory_seed=traj_id, n_steps=n_steps_per_traj,
        )                                                    # [T, 1, H, W]
        frames_seq = frames_seq.unsqueeze(1)                 # [T, 1, 1, H, W]

        state = bundle.net.initial_state(batch_size=1)
        prev_r_l4: Tensor | None = None
        prev_x_hat: Tensor | None = None
        for t in range(frames_seq.shape[0]):
            frame = frames_seq[t]                            # [1, 1, H, W]
            x_hat, state, info = bundle.net(frame, state)
            r_l4_now = info["r_l4"]                          # [1, n_l4_e]
            if prev_r_l4 is not None and prev_x_hat is not None:
                pred_errs.append(
                    float(F.mse_loss(prev_x_hat, r_l4_now).item())
                )
                copy_errs.append(
                    float(F.mse_loss(prev_r_l4, r_l4_now).item())
                )
            prev_r_l4 = r_l4_now
            prev_x_hat = x_hat

    pred_mse = float(np.mean(pred_errs)) if pred_errs else float("nan")
    copy_mse = float(np.mean(copy_errs)) if copy_errs else float("nan")
    improvement = (
        (copy_mse - pred_mse) / copy_mse if copy_mse > 1e-12 else float("nan")
    )
    return {
        "gate": "4_prediction_beats_copy_last",
        "network_mse": pred_mse,
        "copy_last_mse": copy_mse,
        "relative_improvement": improvement,
        "improvement_floor": float(improvement_floor),
        "passed": bool(
            math.isfinite(improvement) and improvement >= improvement_floor
        ),
    }


# ---------------------------------------------------------------------------
# Gate 5a — orientation localizer
# ---------------------------------------------------------------------------


def _fwhm_from_tuning(curve: np.ndarray, orientations_deg: np.ndarray) -> float:
    """Circular FWHM in degrees of a tuning curve (180° period, 1-D)."""
    if curve.max() <= 1e-9:
        return 180.0
    c = curve - curve.min()
    half = c.max() / 2.0
    # Align the peak to index 0 via circular shift.
    peak_idx = int(np.argmax(c))
    shifted = np.roll(c, -peak_idx)
    step = 180.0 / len(c)
    above = shifted >= half
    if not above[0]:
        return 180.0
    # Walk outward from 0 in both directions until dropping below half.
    left = 0
    right = 0
    n = len(c)
    for k in range(1, n // 2 + 1):
        if above[k % n]:
            right = k
        else:
            break
    for k in range(1, n // 2 + 1):
        if above[(-k) % n]:
            left = k
        else:
            break
    return float((left + right) * step)


@torch.no_grad()
def gate_5a_orientation_localizer(
    bundle: CheckpointBundle,
    *, n_orientations: int = 12, contrast: float = 1.0,
    n_steps_steady: int = 40,
    fwhm_max_deg: float = 120.0, frac_units_min: float = 0.7,
) -> dict[str, Any]:
    """Sweep orientations; per-unit preferred orientation + FWHM."""
    orientations = np.linspace(0.0, 180.0, n_orientations, endpoint=False)
    responses: list[np.ndarray] = []
    for theta in orientations:
        frame = make_grating_frame(
            float(theta), contrast, bundle.cfg, device=bundle.cfg.device,
        )
        _s, info = simulate_steady_state(bundle.net, frame, n_steps_steady)
        r_l23 = info["r_l23"].mean(dim=0).detach().cpu().numpy()   # [n_units]
        responses.append(r_l23)
    tuning = np.stack(responses, axis=1)                            # [U, O]

    fwhms = np.array(
        [_fwhm_from_tuning(tuning[u], orientations) for u in range(tuning.shape[0])]
    )
    preferred = orientations[np.argmax(tuning, axis=1)]
    frac_tuned = float((fwhms < fwhm_max_deg).mean())
    return {
        "gate": "5a_orientation_localizer",
        "n_orientations": int(n_orientations),
        "fwhm_median_deg": float(np.median(fwhms)),
        "fwhm_mean_deg": float(fwhms.mean()),
        "frac_units_fwhm_below": frac_tuned,
        "fwhm_max_deg": float(fwhm_max_deg),
        "preferred_orientation_mean": float(preferred.mean()),
        "passed": bool(frac_tuned >= frac_units_min),
    }


# ---------------------------------------------------------------------------
# Gate 5b — identity localizer
# ---------------------------------------------------------------------------


@torch.no_grad()
def gate_5b_identity_localizer(
    bundle: CheckpointBundle,
    *, n_noise_samples: int = 40, noise_std: float = 0.01,
    n_steps_steady: int = 5, seed: int = 42,
    accuracy_floor: float = 0.25,
) -> dict[str, Any]:
    """12-way LinearSVC on L4 E + within- vs between-token RSA distance.

    Reuses ``TokenBank.verify_discriminability`` for the SVM part; adds an
    RSA check over the same L4 feature matrix.
    """
    try:
        from sklearn.model_selection import train_test_split    # type: ignore
        from sklearn.svm import LinearSVC                       # type: ignore
    except ImportError:
        return {
            "gate": "5b_identity_localizer",
            "error": "sklearn not available",
            "svm_accuracy": float("nan"),
            "within_token_mean_dist": float("nan"),
            "between_token_mean_dist": float("nan"),
            "passed": False,
        }

    tokens = bundle.bank.tokens                                  # [N, 1, H, W]
    N = tokens.shape[0]
    dev = tokens.device
    torch.manual_seed(int(seed))
    noise = noise_std * torch.randn(
        N, n_noise_samples, 1, tokens.shape[-2], tokens.shape[-1], device=dev,
    )
    frames = (tokens.unsqueeze(1) + noise).reshape(
        -1, 1, tokens.shape[-2], tokens.shape[-1],
    )
    # Run LGN+L4 only to match the identity-localizer definition.
    from src.v2_model.state import initial_state
    state = initial_state(bundle.cfg, batch_size=frames.shape[0], device=dev)
    for _ in range(int(n_steps_steady)):
        _, l4_rate, state = bundle.net.lgn_l4(frames, state)
    X = l4_rate.detach().cpu().numpy().astype(np.float32)
    y = np.repeat(np.arange(N), n_noise_samples)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=int(seed), stratify=y,
    )
    clf = LinearSVC(random_state=int(seed), max_iter=5000, dual="auto")
    clf.fit(X_tr, y_tr)
    acc = float(clf.score(X_te, y_te))

    # RSA: pairwise L2 distances, split into within- / between-token.
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    same = (y[:, None] == y[None, :])
    np.fill_diagonal(same, False)
    within = float(D[same].mean()) if same.any() else float("nan")
    between = float(D[~same].mean())

    return {
        "gate": "5b_identity_localizer",
        "svm_accuracy": acc,
        "accuracy_floor": float(accuracy_floor),
        "within_token_mean_dist": within,
        "between_token_mean_dist": between,
        "rsa_passed": bool(
            math.isfinite(within) and math.isfinite(between) and within < between
        ),
        "passed": bool(
            acc > accuracy_floor
            and math.isfinite(within)
            and math.isfinite(between)
            and within < between
        ),
    }


# ---------------------------------------------------------------------------
# Gate 6 — null-expectation control (wraps run_gate_6_null_control)
# ---------------------------------------------------------------------------


def gate_6_null_expectation_control(
    bundle: CheckpointBundle,
    *, phase: str = "phase3_kok",
    kok_kwargs: dict | None = None,
    richter_kwargs: dict | None = None,
) -> dict[str, Any]:
    """Thin wrapper around ``run_gate_6_null_control``.

    Runs the Kok + Richter null assays to verify that observed expectation
    effects are above a null-validity baseline. Returns the wrapped dict
    with an added ``gate`` label so the 7-key output is uniform.
    """
    out = run_gate_6_null_control(
        bundle, phase=phase,
        kok_kwargs=kok_kwargs, richter_kwargs=richter_kwargs,
    )
    out = dict(out)
    out["gate"] = "6_null_expectation_control"
    return out


# ---------------------------------------------------------------------------
# Gate 7 — C-load-bearing ablation (wraps run_gate_7_c_load_bearing)
# ---------------------------------------------------------------------------


def gate_7_c_load_bearing(
    bundle: CheckpointBundle,
    *, n_trajectories: int = 8, n_steps_per_traj: int = 20,
    degradation_floor: float = 0.05,
    phase: str = "phase2",
    seed_family: str = "eval",
) -> dict[str, Any]:
    """Thin wrapper around ``run_gate_7_c_load_bearing``.

    Ablates Context Memory and compares next-step prediction MSE with vs
    without C; passes iff relative degradation ≥ ``degradation_floor``.
    """
    return run_gate_7_c_load_bearing(
        bundle, n_trajectories=n_trajectories,
        n_steps_per_traj=n_steps_per_traj,
        degradation_floor=degradation_floor,
        phase=phase, seed_family=seed_family,
    )


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def run_gates_1_to_5(bundle: CheckpointBundle) -> dict[str, Any]:
    """Run sensory gates 1-5 and return a dict of per-gate results.

    Kept for backwards compatibility with callers written against the
    original 5-gate harness. New callers should prefer ``run_gates_1_to_7``.
    """
    results: dict[str, Any] = {}
    results["gate_1_rate_distribution"] = gate_1_rate_distribution(bundle)
    results["gate_2_contrast_response"] = gate_2_contrast_response(bundle)
    results["gate_3_surround_suppression"] = gate_3_surround_suppression(bundle)
    results["gate_4_next_step_prediction_beats_copy_last"] = (
        gate_4_prediction_beats_copy_last(bundle)
    )
    results["gate_5a_orientation_localizer"] = gate_5a_orientation_localizer(
        bundle
    )
    results["gate_5b_identity_localizer"] = gate_5b_identity_localizer(bundle)
    results["all_passed"] = all(
        v.get("passed", False) for v in results.values() if isinstance(v, dict)
    )
    return results


def run_gates_1_to_7(bundle: CheckpointBundle) -> dict[str, Any]:
    """Run all seven gates and return a dict with canonical 7-gate keys.

    Returns a dict with keys ``gate_1`` through ``gate_7`` (each a dict
    with at minimum a ``passed: bool`` field) plus ``all_passed: bool``.
    Gate 5 collapses 5a + 5b: ``gate_5`` passes only if both sub-gates
    pass, and its body preserves both sub-dicts under ``gate_5a`` /
    ``gate_5b`` for transparency.

    Note on phase bookkeeping: gate 6 switches ``bundle.net`` into
    ``phase3_kok`` and gate 7 into ``phase2`` (see the wrapped runners),
    so gates 1-5 are run FIRST under the checkpoint's loaded phase,
    and then 6 + 7 are run last. Callers that need a specific final
    phase should ``bundle.net.set_phase(...)`` after this returns.
    """
    results: dict[str, Any] = {}
    results["gate_1"] = gate_1_rate_distribution(bundle)
    results["gate_2"] = gate_2_contrast_response(bundle)
    results["gate_3"] = gate_3_surround_suppression(bundle)
    results["gate_4"] = gate_4_prediction_beats_copy_last(bundle)

    g5a = gate_5a_orientation_localizer(bundle)
    g5b = gate_5b_identity_localizer(bundle)
    results["gate_5"] = {
        "gate": "5_orientation_plus_identity_localizer",
        "gate_5a": g5a,
        "gate_5b": g5b,
        "passed": bool(g5a.get("passed", False) and g5b.get("passed", False)),
    }

    results["gate_6"] = gate_6_null_expectation_control(bundle)
    results["gate_7"] = gate_7_c_load_bearing(bundle)

    results["all_passed"] = all(
        bool(results[f"gate_{i}"].get("passed", False)) for i in range(1, 8)
    )
    return results


def _cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase-2 sanity gates 1-7")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument(
        "--only-1-to-5", action="store_true",
        help="Legacy mode: run only sensory gates 1-5 (gates_1_5.json).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _cli().parse_args(argv)
    bundle = load_checkpoint(args.checkpoint, seed=args.seed, device=args.device)
    if args.only_1_to_5:
        results = run_gates_1_to_5(bundle)
        out_path = args.output or (args.checkpoint.parent / "gates_1_5.json")
    else:
        results = run_gates_1_to_7(bundle)
        out_path = args.output or (args.checkpoint.parent / "gates_1_7.json")
    out_path.write_text(json.dumps(results, indent=2))
    return 0 if results["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
