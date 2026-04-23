"""Task #74 Toy Level 11 — sign-test of (β) mult gating vs (α) predictive subtraction.

Runs the Phase-2 substrate (task weights zeroed) and injects HAND-CRAFTED
cue→L23E modifications at probe time to ask: does each mechanism, given
ideal localizer-derived weights, produce the PC sign (mean_r_matched <
mean_r_mismatched)?

No learning, no training, no new tests, no commits. Pure diagnostic.

Architectural note (verified 2026-04-22 by grep):
    L23E.forward composes ``W_fb_apical`` as ``+F.linear(h_apical_input,
    _excitatory_eff(W_fb_apical_raw))`` — it is ADDITIVE EXCITATORY, not
    a subtractive prediction. Therefore toy (α) cannot be implemented by
    merely populating W_qh_gen/W_fb_apical; it requires an explicit
    subtraction hook at L23E drive. We inject this hook in the monkey-
    patched forward below.

Modes:
    baseline : pass-through (task weights still zero; no injection) —
               reproduces the Level 11 null.
    beta     : multiplicative gain on L4→L23E feedforward contribution.
               ``gain[c, u] = 1 − g0`` if unit ``u`` prefers cue ``c``'s
               expected orientation (within ±15°), else 1. Sweep
               g0 ∈ {0.1, 0.3, 0.5}.
    alpha    : additive subtractive bias at L23E drive. ``b_sub[c, :] =
               α · prob_pattern[theta_c]`` where prob_pattern is the
               cue-free localizer L23E mean response for the expected
               orientation. Sweep α ∈ {0.3, 0.5, 0.8, 1.0}.

Metrics per run:
    mean_r_matched, mean_r_mismatched, Δ-sign, asym-sign (pref/non-pref
    from localizer), bootstrap p-stats lightweight (n=200).

Reports one line per mode/sweep-value to stdout for Lead DM.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from scripts.v2._gates_common import (
    load_checkpoint, make_blank_frame, make_grating_frame,
)
from scripts.v2.train_phase3_kok_learning import (
    CUE_ORIENTATIONS_DEG, KokTiming, build_cue_tensor, cue_mapping_from_seed,
)
from src.v2_model.layers import (
    _excitatory_eff, _inhibitory_eff,
)


_NULL_EPS = 1e-4


def _sign_label(delta: float, *, correct_is_negative: bool) -> str:
    if abs(delta) < _NULL_EPS:
        return "null"
    if correct_is_negative:
        return "correct" if delta < 0 else "inverse"
    return "correct" if delta > 0 else "inverse"


# ---------------------------------------------------------------------------
# Monkey-patched L23E forward
# ---------------------------------------------------------------------------


def _install_toy_hook(net) -> None:
    """Attach a configurable forward hook to ``net.l23_e``.

    Adds attributes on the L23E instance:
      * ``_toy_mode``           : 'baseline' | 'beta' | 'alpha'
      * ``_toy_in_probe``       : bool — gate whether injection is active
      * ``_toy_cue_id``         : int — current cue context
      * ``_toy_gain_l4_per_cue``: [n_cue, n_l23_e] (beta)
      * ``_toy_b_sub_per_cue``  : [n_cue, n_l23_e] (alpha)

    Replaces ``l23_e.forward`` with a bound method that behaves
    identically to the original when ``_toy_in_probe`` is False
    (or ``_toy_mode == 'baseline'``), and otherwise applies the
    configured modification.
    """
    l23 = net.l23_e
    l23._toy_mode = "baseline"
    l23._toy_in_probe = False
    l23._toy_cue_id = 0
    l23._toy_gain_l4_per_cue: Optional[Tensor] = None
    l23._toy_b_sub_per_cue: Optional[Tensor] = None

    def toy_forward(
        self,
        l4_input: Tensor,
        l23_recurrent_input: Tensor,
        som_input: Tensor,
        pv_input: Tensor,
        h_apical_input: Tensor,
        context_bias: Tensor,
        state: Tensor,
        *,
        som_gain: Optional[Tensor] = None,
    ):
        B = state.shape[0]
        w_l4 = _excitatory_eff(self.W_l4_l23_raw) * self.mask_l4_l23
        w_rec = _excitatory_eff(self.W_rec_raw, self.mask_rec)
        w_pv = _inhibitory_eff(self.W_pv_l23_raw)
        w_som = _inhibitory_eff(self.W_som_l23_raw)
        w_fb = _excitatory_eff(self.W_fb_apical_raw)
        som_effective = (
            som_input * som_gain if som_gain is not None else som_input
        )

        ff_l4 = F.linear(l4_input, w_l4)             # [B, n_l23_e]

        active = bool(self._toy_in_probe) and self._toy_mode != "baseline"
        if active and self._toy_mode == "beta":
            gain = self._toy_gain_l4_per_cue[int(self._toy_cue_id)]
            ff_l4 = ff_l4 * gain.unsqueeze(0)        # [B, n_l23_e]

        drive = (
            ff_l4
            + F.linear(l23_recurrent_input, w_rec)
            + F.linear(som_effective, w_som)
            + F.linear(pv_input, w_pv)
            + F.linear(h_apical_input, w_fb)
            + context_bias
        )
        if active and self._toy_mode == "alpha":
            b_sub = self._toy_b_sub_per_cue[int(self._toy_cue_id)]
            drive = drive - b_sub.unsqueeze(0)       # [B, n_l23_e]

        activated = self._phi(drive - self.theta)
        rate_next = self._leak * state + (1.0 - self._leak) * activated
        return rate_next, rate_next

    import types
    l23.forward = types.MethodType(toy_forward, l23)


# ---------------------------------------------------------------------------
# Trial runners
# ---------------------------------------------------------------------------


@torch.no_grad()
def _run_probe_trial(
    bundle, *, cue_id: int, probe_orientation_deg: float,
    timing: KokTiming, noise_std: float, generator: torch.Generator,
) -> Tensor:
    cfg = bundle.cfg
    device = cfg.device
    blank = make_blank_frame(1, cfg, device=device)
    probe = make_grating_frame(
        float(probe_orientation_deg), 1.0, cfg, device=device,
    )
    q_cue = build_cue_tensor(int(cue_id), cfg.arch.n_c, device=device)

    state = bundle.net.initial_state(batch_size=1)
    cue_end = timing.cue_steps
    delay_end = cue_end + timing.delay_steps
    probe1_end = delay_end + timing.probe1_steps
    blank2_end = probe1_end + timing.blank_steps
    n_total = timing.total

    l23 = bundle.net.l23_e
    l23._toy_cue_id = int(cue_id)

    probe_rates: list[Tensor] = []
    for t in range(n_total):
        if t < cue_end:
            frame, q_t = blank, q_cue
        elif t < delay_end:
            frame, q_t = blank, None
        elif t < probe1_end:
            frame, q_t = probe, None
        elif t < blank2_end:
            frame, q_t = blank, None
        else:
            frame, q_t = probe, None
        if noise_std > 0.0:
            frame = frame + noise_std * torch.randn(
                frame.shape, generator=generator, device=device,
            )
        # Only activate the toy modification during the probe1 epoch.
        l23._toy_in_probe = (delay_end <= t < probe1_end)
        _x_hat, state, info = bundle.net(frame, state, q_t=q_t)
        if delay_end <= t < probe1_end:
            probe_rates.append(info["r_l23"][0].clone())
    l23._toy_in_probe = False
    return torch.stack(probe_rates, dim=0).mean(dim=0)


@torch.no_grad()
def _run_localizer_trial(
    bundle, *, probe_orientation_deg: float, timing: KokTiming,
    noise_std: float, generator: torch.Generator,
) -> Tensor:
    """Cue-FREE trial (toy injection disabled; toy_mode=baseline or gated off).

    We force ``_toy_in_probe = False`` throughout so the localizer sees the
    unmodified model — this is what defines "cue-free probe mean response".
    """
    cfg = bundle.cfg
    device = cfg.device
    blank = make_blank_frame(1, cfg, device=device)
    probe = make_grating_frame(
        float(probe_orientation_deg), 1.0, cfg, device=device,
    )

    state = bundle.net.initial_state(batch_size=1)
    cue_end = timing.cue_steps
    delay_end = cue_end + timing.delay_steps
    probe1_end = delay_end + timing.probe1_steps
    blank2_end = probe1_end + timing.blank_steps
    n_total = timing.total

    l23 = bundle.net.l23_e
    saved_mode = l23._toy_mode
    l23._toy_mode = "baseline"  # force no-op
    l23._toy_in_probe = False

    probe_rates: list[Tensor] = []
    try:
        for t in range(n_total):
            if t < cue_end:
                frame = blank
            elif t < delay_end:
                frame = blank
            elif t < probe1_end:
                frame = probe
            elif t < blank2_end:
                frame = blank
            else:
                frame = probe
            if noise_std > 0.0:
                frame = frame + noise_std * torch.randn(
                    frame.shape, generator=generator, device=device,
                )
            _x_hat, state, info = bundle.net(frame, state, q_t=None)
            if delay_end <= t < probe1_end:
                probe_rates.append(info["r_l23"][0].clone())
    finally:
        l23._toy_mode = saved_mode
    return torch.stack(probe_rates, dim=0).mean(dim=0)


# ---------------------------------------------------------------------------
# Assay
# ---------------------------------------------------------------------------


def _run_main_assay(
    bundle, *, cue_mapping: dict[int, float], n_trials_per_cell: int,
    timing: KokTiming, noise_std: float, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gen = torch.Generator().manual_seed(seed)
    r_mat, y_probe, y_matched = [], [], []
    for cue_id in (0, 1):
        for probe_deg in CUE_ORIENTATIONS_DEG:
            matched = int(
                abs(cue_mapping[int(cue_id)] - float(probe_deg)) < 1e-6
            )
            for _ in range(n_trials_per_cell):
                r = _run_probe_trial(
                    bundle, cue_id=int(cue_id),
                    probe_orientation_deg=float(probe_deg),
                    timing=timing, noise_std=noise_std, generator=gen,
                )
                r_mat.append(r.cpu().numpy())
                y_probe.append(
                    0 if abs(float(probe_deg) - CUE_ORIENTATIONS_DEG[0]) < 1e-6
                    else 1
                )
                y_matched.append(matched)
    return (
        np.stack(r_mat, axis=0),
        np.asarray(y_probe, dtype=np.int64),
        np.asarray(y_matched, dtype=np.int64),
    )


def _localizer_tuning(
    bundle, *, n_orients: int, n_trials: int,
    timing: KokTiming, noise_std: float, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (orient_mean_per_unit [n_orients, n_l23], orients_deg)."""
    orients = np.linspace(0.0, 180.0, n_orients, endpoint=False)
    gen = torch.Generator().manual_seed(seed + 1)
    loc = []
    loc_y = []
    for theta in orients:
        for _ in range(n_trials):
            r = _run_localizer_trial(
                bundle, probe_orientation_deg=float(theta),
                timing=timing, noise_std=noise_std, generator=gen,
            )
            loc.append(r.cpu().numpy())
            loc_y.append(float(theta))
    L = np.stack(loc, axis=0)
    L_y = np.asarray(loc_y)
    orient_mean_per_unit = np.stack([
        L[np.abs(L_y - o) < 1e-6].mean(axis=0) for o in orients
    ], axis=0)
    return orient_mean_per_unit, orients


def _unit_pref_mask(
    unit_pref_deg: np.ndarray, anchor_deg: float, tol: float = 15.0,
) -> np.ndarray:
    d = np.abs(((unit_pref_deg - anchor_deg + 90.0) % 180.0) - 90.0)
    return d <= tol


def _compute_asym(
    R: np.ndarray, y_probe: np.ndarray, y_matched: np.ndarray,
    unit_pref_deg: np.ndarray,
) -> tuple[float, float, float, float, str]:
    pref_exp, pref_unexp, nonpref_exp, nonpref_unexp = [], [], [], []
    for probe_deg in CUE_ORIENTATIONS_DEG:
        probe_cls = (
            0 if abs(float(probe_deg) - CUE_ORIENTATIONS_DEG[0]) < 1e-6 else 1
        )
        pref_m = _unit_pref_mask(unit_pref_deg, float(probe_deg))
        other = (
            CUE_ORIENTATIONS_DEG[1] if probe_cls == 0 else CUE_ORIENTATIONS_DEG[0]
        )
        nonpref_m = _unit_pref_mask(unit_pref_deg, float(other))
        sel_m = (y_probe == probe_cls) & (y_matched == 1)
        sel_u = (y_probe == probe_cls) & (y_matched == 0)
        if pref_m.any():
            pref_exp.append(float(R[np.ix_(sel_m, pref_m)].mean()))
            pref_unexp.append(float(R[np.ix_(sel_u, pref_m)].mean()))
        if nonpref_m.any():
            nonpref_exp.append(float(R[np.ix_(sel_m, nonpref_m)].mean()))
            nonpref_unexp.append(float(R[np.ix_(sel_u, nonpref_m)].mean()))
    pe = float(np.mean(pref_exp)) if pref_exp else float("nan")
    pu = float(np.mean(pref_unexp)) if pref_unexp else float("nan")
    ne = float(np.mean(nonpref_exp)) if nonpref_exp else float("nan")
    nu = float(np.mean(nonpref_unexp)) if nonpref_unexp else float("nan")
    pref_ok = pe <= pu + _NULL_EPS
    nonpref_ok = ne + _NULL_EPS >= nu
    if abs(pe - pu) < _NULL_EPS and abs(ne - nu) < _NULL_EPS:
        sign = "null"
    elif pref_ok and nonpref_ok:
        sign = "correct"
    else:
        sign = "inverse"
    return pe, pu, ne, nu, sign


def _report(
    tag: str, R: np.ndarray, y_matched: np.ndarray, y_probe: np.ndarray,
    unit_pref_deg: np.ndarray,
) -> dict:
    mean_m = float(R[y_matched == 1].mean())
    mean_u = float(R[y_matched == 0].mean())
    delta = mean_m - mean_u
    sign = _sign_label(delta, correct_is_negative=True)
    pe, pu, ne, nu, asym_sign = _compute_asym(
        R, y_probe, y_matched, unit_pref_deg,
    )
    print(
        f"{tag}: mean_r_matched={mean_m:.5f} "
        f"mean_r_mismatched={mean_u:.5f} "
        f"delta={delta:+.3e} Δsign={sign} asym_sign={asym_sign} "
        f"pref_exp={pe:.4f} pref_unexp={pu:.4f} "
        f"nonpref_exp={ne:.4f} nonpref_unexp={nu:.4f}"
    )
    return {
        "tag": tag,
        "mean_r_matched": mean_m,
        "mean_r_mismatched": mean_u,
        "delta": delta,
        "delta_sign": sign,
        "pref_exp": pe, "pref_unexp": pu,
        "nonpref_exp": ne, "nonpref_unexp": nu,
        "asym_sign": asym_sign,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-trials-per-cell", type=int, default=30)
    ap.add_argument("--noise-std", type=float, default=0.05)
    ap.add_argument("--n-localizer-orients", type=int, default=36)
    ap.add_argument("--n-localizer-trials", type=int, default=8)
    ap.add_argument(
        "--g0-sweep", type=float, nargs="+",
        default=[0.1, 0.3, 0.5],
    )
    ap.add_argument(
        "--alpha-sweep", type=float, nargs="+",
        default=[0.3, 0.5, 0.8, 1.0],
    )
    ap.add_argument("--output", type=Path, default=Path("logs/task74/toy_level11.json"))
    args = ap.parse_args()

    # --- Load substrate, zero task weights -------------------------------
    bundle = load_checkpoint(args.checkpoint, seed=args.seed, device="cpu")
    bundle.net.set_phase("phase3_kok")
    cm = bundle.net.context_memory
    with torch.no_grad():
        cm.W_qm_task.data.zero_()
        cm.W_mh_task_exc.data.zero_()
        cm.W_mh_task_inh.data.zero_()

    _install_toy_hook(bundle.net)
    l23 = bundle.net.l23_e

    timing = KokTiming()
    cue_mapping = cue_mapping_from_seed(args.seed)

    # --- Localizer (cue-free) → pref per unit + prob_pattern[orient] -----
    orient_mean, orients = _localizer_tuning(
        bundle, n_orients=args.n_localizer_orients,
        n_trials=args.n_localizer_trials, timing=timing,
        noise_std=args.noise_std, seed=args.seed,
    )
    unit_pref_idx = np.argmax(orient_mean, axis=0)
    unit_pref_deg = orients[unit_pref_idx]

    # prob_pattern[cue] = localizer mean L23E response at cue's expected
    # orientation (use the nearest localizer bin).
    def _nearest_orient_row(theta_deg: float) -> np.ndarray:
        i = int(np.argmin(
            np.abs(((orients - float(theta_deg) + 90.0) % 180.0) - 90.0)
        ))
        return orient_mean[i]
    prob_pattern_per_cue = np.stack([
        _nearest_orient_row(cue_mapping[0]),
        _nearest_orient_row(cue_mapping[1]),
    ], axis=0)                                   # [n_cue=2, n_l23_e]

    # --- Pre-compute per-cue pref masks over L23E for β gain ------------
    n_l23_e = unit_pref_deg.shape[0]
    gain_l4_template = np.ones((2, n_l23_e), dtype=np.float32)
    for c in (0, 1):
        pref_m = _unit_pref_mask(unit_pref_deg, cue_mapping[c])
        gain_l4_template[c, pref_m] = 0.0           # placeholder; overwritten
    # We keep the PREF MASK (pref_m per cue) — gain = 1 - g0 on pref units.
    pref_mask_per_cue = np.zeros((2, n_l23_e), dtype=bool)
    for c in (0, 1):
        pref_mask_per_cue[c] = _unit_pref_mask(unit_pref_deg, cue_mapping[c])

    results: list[dict] = []

    # --- Baseline (task weights zero, no injection) ---------------------
    l23._toy_mode = "baseline"
    R0, y_p0, y_m0 = _run_main_assay(
        bundle, cue_mapping=cue_mapping,
        n_trials_per_cell=args.n_trials_per_cell,
        timing=timing, noise_std=args.noise_std, seed=args.seed,
    )
    results.append(_report(
        "toy_baseline", R0, y_m0, y_p0, unit_pref_deg,
    ))

    # --- Test β: multiplicative gating ----------------------------------
    for g0 in args.g0_sweep:
        gain = np.ones((2, n_l23_e), dtype=np.float32)
        for c in (0, 1):
            gain[c, pref_mask_per_cue[c]] = 1.0 - float(g0)
        l23._toy_mode = "beta"
        l23._toy_gain_l4_per_cue = torch.tensor(
            gain, dtype=torch.float32, device=bundle.cfg.device,
        )
        l23._toy_b_sub_per_cue = None
        R, y_p, y_m = _run_main_assay(
            bundle, cue_mapping=cue_mapping,
            n_trials_per_cell=args.n_trials_per_cell,
            timing=timing, noise_std=args.noise_std, seed=args.seed,
        )
        results.append(_report(
            f"toy_beta g0={g0:.2f}", R, y_m, y_p, unit_pref_deg,
        ))

    # --- Test α: predictive subtraction ---------------------------------
    for alpha in args.alpha_sweep:
        b_sub = float(alpha) * prob_pattern_per_cue       # [2, n_l23_e]
        l23._toy_mode = "alpha"
        l23._toy_gain_l4_per_cue = None
        l23._toy_b_sub_per_cue = torch.tensor(
            b_sub, dtype=torch.float32, device=bundle.cfg.device,
        )
        R, y_p, y_m = _run_main_assay(
            bundle, cue_mapping=cue_mapping,
            n_trials_per_cell=args.n_trials_per_cell,
            timing=timing, noise_std=args.noise_std, seed=args.seed,
        )
        results.append(_report(
            f"toy_alpha alpha={alpha:.2f}", R, y_m, y_p, unit_pref_deg,
        ))

    # --- Summary ---------------------------------------------------------
    def _best_of(prefix: str):
        cand = [r for r in results if r["tag"].startswith(prefix)]
        if not cand:
            return None
        # Choose the one with most-negative delta (largest suppression of matched)
        return min(cand, key=lambda r: r["delta"])

    beta_best = _best_of("toy_beta")
    alpha_best = _best_of("toy_alpha")
    mechanisms_validated = []
    if beta_best is not None and beta_best["delta_sign"] == "correct":
        mechanisms_validated.append("beta")
    if alpha_best is not None and alpha_best["delta_sign"] == "correct":
        mechanisms_validated.append("alpha")

    print("")
    print("=== SUMMARY ===")
    if beta_best is not None:
        print(
            f"toy_beta_best={beta_best['tag']} "
            f"delta={beta_best['delta']:+.3e} "
            f"sign={beta_best['delta_sign']} "
            f"asym={beta_best['asym_sign']}"
        )
    if alpha_best is not None:
        print(
            f"toy_alpha_best={alpha_best['tag']} "
            f"delta={alpha_best['delta']:+.3e} "
            f"sign={alpha_best['delta_sign']} "
            f"asym={alpha_best['asym_sign']}"
        )
    print(f"mechanisms_validated={mechanisms_validated}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "ckpt": str(args.checkpoint),
            "seed": args.seed,
            "cue_mapping": {str(k): v for k, v in cue_mapping.items()},
            "n_trials_per_cell": args.n_trials_per_cell,
            "noise_std": args.noise_std,
            "n_localizer_orients": args.n_localizer_orients,
            "n_localizer_trials": args.n_localizer_trials,
            "g0_sweep": list(args.g0_sweep),
            "alpha_sweep": list(args.alpha_sweep),
            "unit_pref_deg_summary": {
                "mean": float(np.mean(unit_pref_deg)),
                "n_pref_cue0": int(pref_mask_per_cue[0].sum()),
                "n_pref_cue1": int(pref_mask_per_cue[1].sum()),
                "n_l23_e": int(n_l23_e),
            },
            "results": results,
            "mechanisms_validated": mechanisms_validated,
        }, f, indent=2)
    print(f"JSON: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
