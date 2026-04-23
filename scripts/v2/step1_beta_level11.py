"""Task #74 β-mechanism — Step 1: Level 11 gate with frozen hand-crafted W_q_gain.

Loads Phase-2 substrate, zeros task weights (Phase-2 learning untouched),
measures per-L23E-unit preferred orientation from a cue-free localizer,
then installs a hand-crafted ``W_q_gain`` (shape [n_cue, n_l23_e]) into
``bundle.net.l23_e.W_q_gain`` such that for cue_id ∈ {0, 1} with
expected orientation ``cue_mapping[cue_id]``:

    W_q_gain[cue_id, u] = 1.0 - g0  if unit u's preferred orientation
                                    lies within ``tol`` of the expected
                                    orientation, else 1.0
    W_q_gain[c, :]      = 1.0       for all other cue ids (unused)

Runs the Level-11 assay (4 conditions × n trials) with ``q_t = q_cue``
fed **throughout the probe epoch** (not only during cue presentation)
so that L23E's gain pathway is active where the assay measures it. The
modified probe trial never feeds ``q_t`` to L23E outside cue + probe
epochs, so delay / blank steps are exactly the same as the vanilla
eval trial.

Emits DM-ready single-line summary:
    step1_beta_level11: g0=<#> Δr=<#> Δr_sign=<correct|inverse|null>
      matches_toy=<T/F> asym_sign=<correct|inverse|null>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from scripts.v2._gates_common import (
    load_checkpoint, make_blank_frame, make_grating_frame,
)
from scripts.v2.train_phase3_kok_learning import (
    CUE_ORIENTATIONS_DEG, KokTiming, build_cue_tensor, cue_mapping_from_seed,
)


_NULL_EPS = 1e-4
_TOY_BETA_G0p3_DELTA = -0.08680          # reference from toy_level11 sweep


def _sign_label(delta: float, *, correct_is_negative: bool) -> str:
    if abs(delta) < _NULL_EPS:
        return "null"
    if correct_is_negative:
        return "correct" if delta < 0 else "inverse"
    return "correct" if delta > 0 else "inverse"


def _unit_pref_mask(unit_pref_deg: np.ndarray, anchor: float,
                    tol: float = 15.0) -> np.ndarray:
    d = np.abs(((unit_pref_deg - anchor + 90.0) % 180.0) - 90.0)
    return d <= tol


@torch.no_grad()
def _run_localizer_trial(
    bundle, *, probe_orientation_deg: float, timing: KokTiming,
    noise_std: float, generator: torch.Generator,
) -> Tensor:
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
    rates: list[Tensor] = []
    for t in range(n_total):
        if t < cue_end or cue_end <= t < delay_end:
            frame = blank
        elif delay_end <= t < probe1_end:
            frame = probe
        elif probe1_end <= t < blank2_end:
            frame = blank
        else:
            frame = probe
        if noise_std > 0.0:
            frame = frame + noise_std * torch.randn(
                frame.shape, generator=generator, device=device,
            )
        _x_hat, state, info = bundle.net(frame, state, q_t=None)
        if delay_end <= t < probe1_end:
            rates.append(info["r_l23"][0].clone())
    return torch.stack(rates, dim=0).mean(dim=0)


@torch.no_grad()
def _run_probe_trial_with_gate(
    bundle, *, cue_id: int, probe_orientation_deg: float,
    timing: KokTiming, noise_std: float, generator: torch.Generator,
) -> Tensor:
    """Mirror of ``run_kok_probe_trial`` except ``q_t`` is *also* fed
    during the probe1 epoch (in addition to the cue epoch), activating
    the β gain on L4→L23E. Delay / blank steps still receive q_t=None."""
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
    rates: list[Tensor] = []
    for t in range(n_total):
        if t < cue_end:
            frame, q_t = blank, q_cue
        elif t < delay_end:
            frame, q_t = blank, None
        elif t < probe1_end:
            frame, q_t = probe, q_cue               # β gain active here
        elif t < blank2_end:
            frame, q_t = blank, None
        else:
            frame, q_t = probe, None
        if noise_std > 0.0:
            frame = frame + noise_std * torch.randn(
                frame.shape, generator=generator, device=device,
            )
        _x_hat, state, info = bundle.net(frame, state, q_t=q_t)
        if delay_end <= t < probe1_end:
            rates.append(info["r_l23"][0].clone())
    return torch.stack(rates, dim=0).mean(dim=0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-trials-per-cell", type=int, default=30)
    ap.add_argument("--noise-std", type=float, default=0.05)
    ap.add_argument("--n-localizer-orients", type=int, default=36)
    ap.add_argument("--n-localizer-trials", type=int, default=8)
    ap.add_argument("--g0", type=float, default=0.3)
    ap.add_argument("--pref-tol-deg", type=float, default=15.0)
    ap.add_argument(
        "--output", type=Path, default=Path("logs/task74/step1_beta_level11.json"),
    )
    args = ap.parse_args()

    # --- Load substrate and zero the task weights ----------------------
    bundle = load_checkpoint(args.checkpoint, seed=args.seed, device="cpu")
    bundle.net.set_phase("phase3_kok")
    cm = bundle.net.context_memory
    with torch.no_grad():
        cm.W_qm_task.data.zero_()
        cm.W_mh_task_exc.data.zero_()
        cm.W_mh_task_inh.data.zero_()

    timing = KokTiming()
    cue_mapping = cue_mapping_from_seed(args.seed)
    n_l23_e = bundle.net.l23_e.n_units
    n_cue = bundle.net.l23_e.n_cue

    # --- Localizer (cue-free) → per-unit preferred orientation ---------
    orients = np.linspace(0.0, 180.0, args.n_localizer_orients, endpoint=False)
    gen_loc = torch.Generator().manual_seed(args.seed + 1)
    loc_rates: list[np.ndarray] = []
    loc_y: list[float] = []
    for theta in orients:
        for _ in range(args.n_localizer_trials):
            r = _run_localizer_trial(
                bundle, probe_orientation_deg=float(theta),
                timing=timing, noise_std=args.noise_std, generator=gen_loc,
            )
            loc_rates.append(r.cpu().numpy())
            loc_y.append(float(theta))
    L = np.stack(loc_rates, axis=0)
    L_y = np.asarray(loc_y)
    orient_mean = np.stack([
        L[np.abs(L_y - o) < 1e-6].mean(axis=0) for o in orients
    ], axis=0)                                   # [n_orients, n_l23]
    unit_pref_idx = np.argmax(orient_mean, axis=0)
    unit_pref_deg = orients[unit_pref_idx]

    # --- Hand-craft W_q_gain and install into L23E --------------------
    W = np.ones((n_cue, n_l23_e), dtype=np.float32)
    per_cue_n_suppressed = {}
    for c in (0, 1):
        exp_deg = cue_mapping[int(c)]
        pref_m = _unit_pref_mask(unit_pref_deg, float(exp_deg), tol=args.pref_tol_deg)
        W[c, pref_m] = 1.0 - float(args.g0)
        per_cue_n_suppressed[c] = int(pref_m.sum())
    with torch.no_grad():
        bundle.net.l23_e.W_q_gain.data.copy_(
            torch.tensor(W, dtype=bundle.net.l23_e.W_q_gain.dtype,
                         device=bundle.net.l23_e.W_q_gain.device)
        )
    installed_wqgain_sig = float(
        bundle.net.l23_e.W_q_gain[:2].mean().item()
    )

    # --- Level 11 main assay with β gate active during probe -----------
    gen = torch.Generator().manual_seed(args.seed)
    R_list: list[np.ndarray] = []
    y_probe: list[int] = []
    y_matched: list[int] = []
    for cue_id in (0, 1):
        for probe_deg in CUE_ORIENTATIONS_DEG:
            matched = int(
                abs(cue_mapping[int(cue_id)] - float(probe_deg)) < 1e-6
            )
            for _ in range(args.n_trials_per_cell):
                r = _run_probe_trial_with_gate(
                    bundle, cue_id=int(cue_id),
                    probe_orientation_deg=float(probe_deg),
                    timing=timing, noise_std=args.noise_std, generator=gen,
                )
                R_list.append(r.cpu().numpy())
                y_probe.append(
                    0 if abs(float(probe_deg) - CUE_ORIENTATIONS_DEG[0]) < 1e-6
                    else 1
                )
                y_matched.append(matched)
    R = np.stack(R_list, axis=0)
    y_probe_arr = np.asarray(y_probe, dtype=np.int64)
    y_matched_arr = np.asarray(y_matched, dtype=np.int64)

    mean_r_matched = float(R[y_matched_arr == 1].mean())
    mean_r_mismatched = float(R[y_matched_arr == 0].mean())
    delta_r = mean_r_matched - mean_r_mismatched
    r_sign = _sign_label(delta_r, correct_is_negative=True)

    # --- Pref / non-pref asymmetry (re-use localizer prefs) -----------
    pref_exp_l, pref_unexp_l = [], []
    nonpref_exp_l, nonpref_unexp_l = [], []
    for probe_deg in CUE_ORIENTATIONS_DEG:
        pc = 0 if abs(float(probe_deg) - CUE_ORIENTATIONS_DEG[0]) < 1e-6 else 1
        other = (
            CUE_ORIENTATIONS_DEG[1] if pc == 0 else CUE_ORIENTATIONS_DEG[0]
        )
        pref_m = _unit_pref_mask(unit_pref_deg, float(probe_deg))
        nonpref_m = _unit_pref_mask(unit_pref_deg, float(other))
        sel_m = (y_probe_arr == pc) & (y_matched_arr == 1)
        sel_u = (y_probe_arr == pc) & (y_matched_arr == 0)
        if pref_m.any():
            pref_exp_l.append(float(R[np.ix_(sel_m, pref_m)].mean()))
            pref_unexp_l.append(float(R[np.ix_(sel_u, pref_m)].mean()))
        if nonpref_m.any():
            nonpref_exp_l.append(float(R[np.ix_(sel_m, nonpref_m)].mean()))
            nonpref_unexp_l.append(float(R[np.ix_(sel_u, nonpref_m)].mean()))
    pe = float(np.mean(pref_exp_l))
    pu = float(np.mean(pref_unexp_l))
    ne = float(np.mean(nonpref_exp_l))
    nu = float(np.mean(nonpref_unexp_l))
    pref_ok = pe <= pu + _NULL_EPS
    nonpref_ok = ne + _NULL_EPS >= nu
    if abs(pe - pu) < _NULL_EPS and abs(ne - nu) < _NULL_EPS:
        asym_sign = "null"
    elif pref_ok and nonpref_ok:
        asym_sign = "correct"
    else:
        asym_sign = "inverse"

    matches_toy = (r_sign == "correct") and (
        abs(delta_r - _TOY_BETA_G0p3_DELTA) < 0.05
    )
    verdict = (
        "pass" if (r_sign == "correct" and asym_sign == "correct") else "fail"
    )

    line = (
        f"step1_beta_level11: g0={args.g0:.2f} "
        f"delta_r={delta_r:+.5f} Δr_sign={r_sign} "
        f"matches_toy={'T' if matches_toy else 'F'} "
        f"asym_sign={asym_sign} verdict={verdict}"
    )
    print(line)
    print(
        f"  mean_r_matched={mean_r_matched:.5f} "
        f"mean_r_mismatched={mean_r_mismatched:.5f}"
    )
    print(
        f"  pref_exp={pe:.4f} pref_unexp={pu:.4f} "
        f"nonpref_exp={ne:.4f} nonpref_unexp={nu:.4f}"
    )
    print(
        f"  installed_W_q_gain[0:2].mean={installed_wqgain_sig:.4f} "
        f"n_suppressed_cue0={per_cue_n_suppressed[0]} "
        f"n_suppressed_cue1={per_cue_n_suppressed[1]}"
    )
    print(f"  toy_reference_delta_g0p3={_TOY_BETA_G0p3_DELTA:+.5f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "ckpt": str(args.checkpoint),
            "seed": args.seed,
            "g0": args.g0,
            "pref_tol_deg": args.pref_tol_deg,
            "cue_mapping": {str(k): v for k, v in cue_mapping.items()},
            "mean_r_matched": mean_r_matched,
            "mean_r_mismatched": mean_r_mismatched,
            "delta_r": delta_r,
            "r_sign": r_sign,
            "matches_toy": bool(matches_toy),
            "pref_exp": pe, "pref_unexp": pu,
            "nonpref_exp": ne, "nonpref_unexp": nu,
            "asym_sign": asym_sign,
            "n_suppressed_per_cue": per_cue_n_suppressed,
            "verdict": verdict,
        }, f, indent=2)
    print(f"  JSON: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
