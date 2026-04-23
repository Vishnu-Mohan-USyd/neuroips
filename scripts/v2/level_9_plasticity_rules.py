"""Level 9 component validation — plasticity rules in isolation (Task #74).

Per Lead's bottom-up validation protocol. Applies each of the four rule
classes in :mod:`src.v2_model.plasticity` to hand-crafted synthetic inputs
and verifies (a) per-cell sign correctness vs the analytical formula,
(b) per-cell magnitude within tolerance, (c) no NaN/Inf. No network, no
dynamics, no training driver.

Rules tested (direct from src/v2_model/plasticity.py):
  1. UrbanczikSennRule  — Δw = lr · mean_b(ε_post ⊗ pre),  ε = apical − basal
  2. VogelsISTDPRule    — Δw = lr · mean_b((post − ρ) ⊗ pre)
                          Sign convention (code + test_plasticity_vogels.py):
                             post > ρ → ΔW > 0 (raw grows).
                             Inhibitory weights in layers.py enter as
                             ``−softplus(raw)``, so raw↑ → more-negative
                             effective weight → stronger inhibition.
  3. ThresholdHomeostasis — Δθ = lr · tanh(error/scale) · scale with
                             scale = 0.1·|ρ| + 1e-3; deadband of
                             ``deadband_fraction · |ρ|`` zeros the update.
  4. ThreeFactorRule.delta_mh (Fix J, excitatory task readout)
                             Δw = lr · mean_b(post ⊗ pre); here
                             pre = memory, post = l23e_modulator.

All rule outputs are hard-clamped by the rule implementation to ±0.01
per step (Task #62). Test inputs are sized so that analytical values stay
within the clamp band, except at the exact clamp boundary (noted in DM).

DM (one line per rule + overall):
  level9_rule1_urbanczik: verdict=<pass/fail> sign_correct=<%> mag_err_median=<#>
  level9_rule2_vogels:    verdict=<pass/fail> sign_correct=<%> mag_err_median=<#>
  level9_rule3_homeostasis: verdict=<pass/fail> sign_correct=<%> mag_err_median=<#> deadband_ok=<T/F>
  level9_rule4_fixJ:      verdict=<pass/fail> sign_correct=<%> mag_err_median=<#>
  level9_overall_verdict=<pass/fail>
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from src.v2_model.plasticity import (
    ThreeFactorRule,
    ThresholdHomeostasis,
    UrbanczikSennRule,
    VogelsISTDPRule,
)


_SIGN_TOL = 1e-9          # below this we treat "sign" as 0 (exact zero)
_MAG_REL_TOL = 0.05       # 5% per Lead's overall criterion
_CLAMP = 0.01             # rule per-step clamp (Task #62)


def _signs_equal(a: float, b: float, tol: float = _SIGN_TOL) -> bool:
    """Return True if a and b have the same sign (both positive, both
    negative, or both within ±tol of zero)."""
    if abs(a) < tol and abs(b) < tol:
        return True
    if abs(a) < tol or abs(b) < tol:
        return False
    return (a > 0) == (b > 0)


def _rel_mag_err(observed: float, expected: float) -> float:
    """Return ``|obs - exp| / max(|exp|, eps)``. If both are ~0, return 0."""
    if abs(expected) < 1e-12 and abs(observed) < 1e-12:
        return 0.0
    denom = max(abs(expected), 1e-12)
    return abs(observed - expected) / denom


def _scan_cells(
    dw: Tensor, analytical: Tensor, clamp_mask: Tensor | None = None,
) -> dict[str, Any]:
    """Element-wise compare observed vs analytical.

    Parameters
    ----------
    dw : Tensor
        Observed update tensor (same shape as ``analytical``).
    analytical : Tensor
        Closed-form expected update (pre-clamp).
    clamp_mask : Tensor | None
        Boolean tensor marking cells where analytical hits the clamp
        boundary; these are excluded from magnitude-error stats and sign
        check uses clamp-truncated expected.
    """
    if dw.shape != analytical.shape:
        raise ValueError(f"shape mismatch: {dw.shape} vs {analytical.shape}")
    # Clamp-truncated reference (what the rule is allowed to return).
    ref = torch.clamp(analytical, min=-_CLAMP, max=_CLAMP)

    n = dw.numel()
    dw_f = dw.flatten().cpu().numpy().astype(np.float64)
    ref_f = ref.flatten().cpu().numpy().astype(np.float64)

    sign_ok = 0
    mag_errs: list[float] = []
    for i in range(n):
        if _signs_equal(dw_f[i], ref_f[i]):
            sign_ok += 1
        mag_errs.append(_rel_mag_err(dw_f[i], ref_f[i]))

    mag_arr = np.asarray(mag_errs)
    sign_pct = float(sign_ok) / float(n)
    return {
        "n_cells": int(n),
        "sign_correct_pct": sign_pct,
        "mag_err_median": float(np.median(mag_arr)),
        "mag_err_max": float(np.max(mag_arr)),
        "mag_err_mean": float(np.mean(mag_arr)),
        "n_hit_clamp": int(
            (torch.abs(analytical).flatten() >= _CLAMP - 1e-9).sum().item()
        ),
        "any_nan": bool(torch.isnan(dw).any().item()),
        "any_inf": bool(torch.isinf(dw).any().item()),
    }


# ---------------------------------------------------------------------------
# Rule 1 — Urbanczik-Senn
# ---------------------------------------------------------------------------

def test_rule1_urbanczik(seed: int) -> dict[str, Any]:
    """Synthetic probe of :class:`UrbanczikSennRule`.

    Inputs per Lead's spec:
      * pre ∈ [0, 1]              shape [1, 16]
      * apical = linspace(0, 1)   shape [1, 16]
      * basal = 0.5               shape [1, 16]
      * ε = apical − basal (negative first half, positive second half)
      * w = rand                 shape [16, 16]
      * raw_prior = zeros, weight_decay = 0  (so no decay term)
      * lr = 1e-2

    Analytical:  Δw[j, i] = lr · pre[0, i] · ε[0, j]
                 ( batch=1, no averaging ).
    """
    g = torch.Generator().manual_seed(seed)
    n_pre = n_post = 16
    pre = torch.rand((1, n_pre), generator=g)
    apical = torch.linspace(0.0, 1.0, n_post).unsqueeze(0)
    basal = torch.full((1, n_post), 0.5)
    eps = apical - basal
    w = torch.rand((n_post, n_pre), generator=g)
    raw_prior = torch.zeros_like(w)
    lr = 1e-2

    rule = UrbanczikSennRule(lr=lr, weight_decay=0.0)
    dw = rule.delta(
        pre_activity=pre, apical=apical, basal=basal,
        weights=w, raw_prior=raw_prior,
    )
    # Analytical (pre-clamp). Batch=1, so no averaging issue.
    analytical = lr * eps.t() @ pre                            # [n_post, n_pre]
    stats = _scan_cells(dw, analytical)

    sign_ok = stats["sign_correct_pct"] == 1.0
    mag_ok = stats["mag_err_median"] < _MAG_REL_TOL
    no_bad = not (stats["any_nan"] or stats["any_inf"])
    # Lead also required "total Δw norm ≠ 0 at non-zero lr"
    dw_norm = float(torch.linalg.norm(dw).cpu())

    verdict = (
        "pass" if (sign_ok and mag_ok and no_bad and dw_norm > 0)
        else "fail"
    )
    return {
        "verdict": verdict,
        "dw_norm": dw_norm,
        **stats,
    }


# ---------------------------------------------------------------------------
# Rule 2 — Vogels iSTDP
# ---------------------------------------------------------------------------

def test_rule2_vogels(seed: int) -> dict[str, Any]:
    """Synthetic probe of :class:`VogelsISTDPRule`.

    Lead's dispatch states ``sign(Δraw) < 0`` when post > ρ, but the code
    (and ``tests/v2/test_plasticity_vogels.py``) follows the standard Dale
    convention where inhibitory effective weight is ``−softplus(raw)``; so
    raw↑ ⇒ stronger inhibition, i.e. post > ρ ⇒ ΔW > 0.  This probe tests
    the *code's* convention (authoritative) and flags the Lead-dispatch
    discrepancy in the DM report.

    Inputs per Lead's spec (shape only — sign test uses code convention):
      * pre (inhibitory-pop rates) = rand(8)   shape [1, 8]
      * post: first half at 2·ρ (above target), second half at 0.5·ρ.
      * ρ = 1.0, lr = 1e-2, weight_decay = 0.
      * raws init at -5.0 (informational; rule does not read them for
        the Hebbian term when decay=0).
    """
    g = torch.Generator().manual_seed(seed)
    n_pre, n_post = 8, 16
    rho = 1.0
    lr = 1e-2

    pre = torch.rand((1, n_pre), generator=g)
    post = torch.empty((1, n_post))
    post[0, : n_post // 2] = 2.0 * rho
    post[0, n_post // 2 :] = 0.5 * rho
    w = torch.full((n_post, n_pre), -5.0)

    rule = VogelsISTDPRule(lr=lr, target_rate=rho, weight_decay=0.0)
    dw = rule.delta(pre_activity=pre, post_activity=post, weights=w)

    analytical = lr * (post - rho).t() @ pre                   # [n_post, n_pre]
    stats = _scan_cells(dw, analytical)

    # Row-wise sign profile (code convention: post > ρ → Δraw > 0).
    row_signs_ok_code = 0
    row_signs_ok_lead = 0
    dw_np = dw.cpu().numpy()
    for j in range(n_post):
        post_above = post[0, j].item() > rho
        # Row must be uniformly positive if above, uniformly negative if below
        # (pre is all non-negative).
        row = dw_np[j]
        row_positive = bool((row > 0).all())
        row_negative = bool((row < 0).all())
        if post_above and row_positive:
            row_signs_ok_code += 1
        if (not post_above) and row_negative:
            row_signs_ok_code += 1
        # Lead's (inverted) convention would require the opposite row signs.
        if post_above and row_negative:
            row_signs_ok_lead += 1
        if (not post_above) and row_positive:
            row_signs_ok_lead += 1

    sign_ok = stats["sign_correct_pct"] == 1.0
    mag_ok = stats["mag_err_median"] < _MAG_REL_TOL
    no_bad = not (stats["any_nan"] or stats["any_inf"])
    verdict = "pass" if (sign_ok and mag_ok and no_bad) else "fail"

    return {
        "verdict": verdict,
        "row_sign_profile_code_convention_pct":
            float(row_signs_ok_code) / float(n_post),
        "row_sign_profile_lead_convention_pct":
            float(row_signs_ok_lead) / float(n_post),
        "note": (
            "Code convention: post>ρ → Δraw>0 (Dale −softplus(raw)). Lead "
            "dispatch stated the opposite; this probe uses the code/test "
            "convention (tests/v2/test_plasticity_vogels.py confirms)."
        ),
        **stats,
    }


# ---------------------------------------------------------------------------
# Rule 3 — Threshold homeostasis
# ---------------------------------------------------------------------------

def test_rule3_homeostasis(seed: int) -> dict[str, Any]:
    """Synthetic probe of :class:`ThresholdHomeostasis`.

    Inputs per Lead's spec (augmented with deadband check):
      * r = linspace(0, 2·ρ, n_units=16)
      * θ starts at zeros
      * ρ = 1.0, η = 1e-2
      * deadband_fraction = 0.2 (module default)

    Analytical (with bounded tanh response):
      error = r − ρ ; deadband = 0.2·|ρ| = 0.2 ; scale = 0.1·|ρ| + 1e-3
      if |error| < deadband  →  Δθ = 0
      else                    →  Δθ = lr · tanh(error/scale) · scale
    """
    n_units = 16
    rho = 1.0
    lr = 1e-2
    deadband_fraction = 0.2

    r = torch.linspace(0.0, 2.0 * rho, n_units).unsqueeze(0)   # [1, 16]
    hom = ThresholdHomeostasis(
        lr=lr, target_rate=rho, n_units=n_units,
        init_theta=0.0, deadband_fraction=deadband_fraction,
    )
    theta_before = hom.theta.detach().clone()
    hom.update(r)
    theta_after = hom.theta.detach().clone()
    d_theta = theta_after - theta_before                       # [16]

    # Analytical reference.
    scale = 0.1 * abs(rho) + 1e-3
    deadband = deadband_fraction * abs(rho)
    err = (r.mean(dim=0) - rho)
    in_band = torch.abs(err) < deadband
    err_eff = torch.where(in_band, torch.zeros_like(err), err)
    analytical = lr * torch.tanh(err_eff / scale) * scale

    stats = _scan_cells(d_theta, analytical)

    # Deadband verification: the in-band cells must have exactly Δθ = 0.
    deadband_ok = bool(
        torch.all(torch.abs(d_theta[in_band]) < 1e-12).item()
    )

    sign_ok = stats["sign_correct_pct"] == 1.0
    mag_ok = stats["mag_err_median"] < _MAG_REL_TOL
    no_bad = not (stats["any_nan"] or stats["any_inf"])
    verdict = (
        "pass" if (sign_ok and mag_ok and deadband_ok and no_bad)
        else "fail"
    )

    return {
        "verdict": verdict,
        "deadband_ok": deadband_ok,
        "scale": scale,
        "deadband_half_width": deadband,
        "n_in_deadband": int(in_band.sum().item()),
        **stats,
    }


# ---------------------------------------------------------------------------
# Rule 4 — Three-factor (Fix J) — delta_mh variant
# ---------------------------------------------------------------------------

def test_rule4_fixJ(seed: int) -> dict[str, Any]:
    """Synthetic probe of :class:`ThreeFactorRule.delta_mh` (Fix J path).

    Inputs per Lead's spec:
      * l23e_modulator = linspace(-1, 1, 16)   shape [1, 16]  (post)
      * m = tensor([1.0, 0.5, 0.0, -0.3])      shape [1, 4]   (pre)
      * eligibility_trace = 1.0 (already absorbed into the modulator by
        the caller; rule does not take it as a separate arg)
      * W init = zeros                          shape [16, 4]
      * lr = 1e-2

    Analytical (batch=1, no averaging):
      Δw[i, c] = lr · modulator[0, i] · m[0, c]
    """
    n_l23e = 16
    n_m = 4
    lr = 1e-2

    l23e_modulator = torch.linspace(-1.0, 1.0, n_l23e).unsqueeze(0)
    m = torch.tensor([[1.0, 0.5, 0.0, -0.3]])
    w = torch.zeros((n_l23e, n_m))

    rule = ThreeFactorRule(lr=lr, weight_decay=0.0)
    dw = rule.delta_mh(memory=m, probe_error=l23e_modulator, weights=w)

    analytical = lr * l23e_modulator.t() @ m                   # [16, 4]
    stats = _scan_cells(dw, analytical)

    # Spot-check Lead's explicit values.
    spot = {
        "dw[0,0]_obs": float(dw[0, 0].cpu()),
        "dw[0,0]_exp": float(torch.clamp(analytical[0, 0],
                                         -_CLAMP, _CLAMP).cpu()),
        "dw[15,3]_obs": float(dw[15, 3].cpu()),
        "dw[15,3]_exp": float(torch.clamp(analytical[15, 3],
                                          -_CLAMP, _CLAMP).cpu()),
    }

    sign_ok = stats["sign_correct_pct"] == 1.0
    mag_ok = stats["mag_err_median"] < _MAG_REL_TOL
    no_bad = not (stats["any_nan"] or stats["any_inf"])
    verdict = "pass" if (sign_ok and mag_ok and no_bad) else "fail"

    return {
        "verdict": verdict,
        "spot_checks": spot,
        **stats,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    seed = int(args.seed)
    torch.manual_seed(seed)

    r1 = test_rule1_urbanczik(seed=seed)
    r2 = test_rule2_vogels(seed=seed)
    r3 = test_rule3_homeostasis(seed=seed)
    r4 = test_rule4_fixJ(seed=seed)

    overall = (
        "pass"
        if all(r["verdict"] == "pass" for r in (r1, r2, r3, r4))
        else "fail"
    )

    summary = {
        "version": "level_9_plasticity_rules_v1",
        "seed": seed,
        "rule1_urbanczik": r1,
        "rule2_vogels": r2,
        "rule3_homeostasis": r3,
        "rule4_fixJ": r4,
        "overall_verdict": overall,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))

    print(
        f"level9_rule1_urbanczik: verdict={r1['verdict']} "
        f"sign_correct={r1['sign_correct_pct']*100:.1f}% "
        f"mag_err_median={r1['mag_err_median']:.3e}"
    )
    print(
        f"level9_rule2_vogels: verdict={r2['verdict']} "
        f"sign_correct={r2['sign_correct_pct']*100:.1f}% "
        f"mag_err_median={r2['mag_err_median']:.3e} "
        f"row_sign_code={r2['row_sign_profile_code_convention_pct']*100:.0f}% "
        f"row_sign_lead={r2['row_sign_profile_lead_convention_pct']*100:.0f}%"
    )
    print(
        f"level9_rule3_homeostasis: verdict={r3['verdict']} "
        f"sign_correct={r3['sign_correct_pct']*100:.1f}% "
        f"mag_err_median={r3['mag_err_median']:.3e} "
        f"deadband_ok={r3['deadband_ok']}"
    )
    print(
        f"level9_rule4_fixJ: verdict={r4['verdict']} "
        f"sign_correct={r4['sign_correct_pct']*100:.1f}% "
        f"mag_err_median={r4['mag_err_median']:.3e}"
    )
    print(f"level9_overall_verdict={overall}")
    print(f"[wrote] {args.output}")
    return 0 if overall == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
