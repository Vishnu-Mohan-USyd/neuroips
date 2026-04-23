"""Task #52 calibration harness — operating-point probe at blank input.

Builds a fresh :class:`V2Network`, verifies that plasticity (closed-form
scalar rules in ``plasticity.py``) and homeostasis (``ThresholdHomeostasis``)
are both inert during the forward path, runs ``N_STEPS`` synchronous-Euler
steps on a blank (zero) frame, and reports:

* Per-population rate statistics at end-of-rollout (Targets 1, 3, 4).
* End-of-rollout ``x_hat`` magnitude vs ``r_l4`` magnitude (Target 6).
* Full-state one-step Jacobian spectral radius at the final state (Target 5).

The Jacobian is computed via ``torch.autograd.functional.jacobian`` on a
pack/unpack closure that wraps the network forward. Weights are stored as
``nn.Parameter(..., requires_grad=False)`` so autograd flows through input
states without mutating any Parameter — the forward path already has no
plasticity or homeostasis side-effects (they live in external drivers).

Usage
-----
    python -m scripts.v2._calibrate_task52

Output is a single table dumped to stdout plus a JSON blob for programmatic
consumption (``--json`` flag).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.autograd.functional as AF

from src.v2_model.config import ModelConfig
from src.v2_model.network import V2Network
from src.v2_model.state import NetworkStateV2


N_STEPS: int = 500
SEED: int = 42


@dataclass(frozen=True)
class CalibReport:
    """Numerical summary of one calibration run.

    Fields are plain floats so the dict is JSON-dumpable. Shape/dtype
    invariants (e.g. that every rate tensor is finite) are asserted
    inside :func:`run_calibration`, not returned here.
    """
    # Population rate statistics (all at end-of-rollout, blank input).
    r_l4_median: float
    r_l4_max: float
    r_l23_median: float
    r_l23_max: float
    r_pv_median: float
    r_pv_max: float
    r_som_median: float
    r_som_max: float
    r_h_median: float
    r_h_max: float
    h_pv_median: float
    h_pv_max: float
    m_median: float
    m_max: float
    # Prediction-head output vs actual L4 (both at end-of-rollout).
    x_hat_median: float
    x_hat_max: float
    x_hat_l2: float
    r_l4_l2: float
    # Full-state one-step Jacobian.
    lambda_max_full: float
    lambda_max_no_memory: float


def _pack_state(state: NetworkStateV2) -> torch.Tensor:
    """Concatenate all rate tensors into a single flat vector ``[B, D]``.

    Ordering is fixed: r_l4, r_l23, r_pv, r_som, r_h, h_pv, m. Trace dicts
    and regime posterior are not part of the dynamical state (plasticity
    traces are frozen when plasticity is off; regime posterior has no
    recurrence in the forward path), so they are excluded.
    """
    return torch.cat([
        state.r_l4, state.r_l23, state.r_pv, state.r_som,
        state.r_h, state.h_pv, state.m,
    ], dim=-1)


def _state_sizes(cfg: ModelConfig) -> Tuple[int, ...]:
    """Per-component sizes in the pack order. Consumed by :func:`_unpack_state`."""
    a = cfg.arch
    return (a.n_l4_e, a.n_l23_e, a.n_l23_pv, a.n_l23_som,
            a.n_h_e, a.n_h_pv, a.n_c)


def _unpack_state(
    z: torch.Tensor,
    cfg: ModelConfig,
    template: NetworkStateV2,
) -> NetworkStateV2:
    """Inverse of :func:`_pack_state`. ``z`` shape ``[B, D]``.

    Trace dicts / regime_posterior are carried through from ``template``
    (not part of the packed vector) so the returned state is a fully
    valid :class:`NetworkStateV2`.
    """
    sizes = _state_sizes(cfg)
    parts = []
    offset = 0
    for n in sizes:
        parts.append(z[..., offset:offset + n])
        offset += n
    r_l4, r_l23, r_pv, r_som, r_h, h_pv, m = parts
    return NetworkStateV2(
        r_l4=r_l4, r_l23=r_l23, r_pv=r_pv, r_som=r_som,
        r_h=r_h, h_pv=h_pv, m=m,
        pre_traces=template.pre_traces,
        post_traces=template.post_traces,
        regime_posterior=template.regime_posterior,
    )


def _assert_plasticity_and_homeostasis_are_inert(net: V2Network) -> None:
    """Sanity-check that no forward-path code mutates weights or thresholds.

    Plasticity rules live in ``src.v2_model.plasticity`` and are only
    invoked by the Phase-2/3 training drivers — ``V2Network.forward`` does
    not call them. ``ThresholdHomeostasis`` exposes an ``update()`` method
    but forward uses only the ``theta`` buffer (read-only access). This
    function re-asserts those invariants so the calibration report is not
    quietly invalidated if someone later wires plasticity into forward.
    """
    # Every raw weight is requires_grad=False — plasticity would mutate
    # ``.data`` directly, but forward path does no such mutation.
    for p in net.parameters():
        assert p.requires_grad is False, (
            f"Parameter {p.shape} has requires_grad=True — unexpected; "
            "calibration assumes every weight is frozen"
        )
    # Homeostasis modules exist on excitatory pops; forward never calls
    # their ``update`` method. We snapshot the θ buffer before the rollout
    # and will re-check it after (see run_calibration).


def _compute_jacobian_radius(
    net: V2Network,
    state_ss: NetworkStateV2,
    cfg: ModelConfig,
    blank_x: torch.Tensor,
) -> Tuple[float, float]:
    """Full-state one-step Jacobian spectral radius at ``state_ss``.

    Parameters
    ----------
    net : V2Network
    state_ss : NetworkStateV2
        The (nearly-)steady-state state about which to linearise.
    cfg : ModelConfig
    blank_x : Tensor [B, 1, H, W]
        Blank frame (zeros). Held fixed — the Jacobian is w.r.t. state only.

    Returns
    -------
    (rho_full, rho_no_memory) : tuple[float, float]
        ``rho_full``       — ``max|eig(J)|`` of the 552×552 full-state map.
        ``rho_no_memory``  — same, but with the context-memory diagonal
                             block (which has a hard floor at exp(−dt/τ_m)
                             ≈ 0.9901 and trivially dominates when the
                             drive portion is small) set to zero to expose
                             the recurrent-weight contribution.

    Notes
    -----
    Context memory's exact-ODE leak ``exp(−dt/τ_m)`` with τ_m = 500 ms,
    dt = 5 ms gives a diagonal 0.9901 — this alone puts |eig(J)| ≈ 0.99
    architecturally, regardless of recurrent-weight init. The
    ``rho_no_memory`` variant is reported so we can distinguish "unstable
    recurrent init" from "slow memory leak".
    """
    B = state_ss.r_l4.shape[0]
    assert B == 1, f"Jacobian probe expects batch 1, got {B}"

    z0 = _pack_state(state_ss).detach().clone()
    assert z0.ndim == 2 and z0.shape[0] == 1, z0.shape

    def step_fn(z: torch.Tensor) -> torch.Tensor:
        state = _unpack_state(z, cfg, template=state_ss)
        _, next_state, _ = net(blank_x, state)
        return _pack_state(next_state)

    # jacobian returns shape [*out_shape, *in_shape]; for B=1 that is
    # [1, D, 1, D]. Squeeze the batch dims to get [D, D].
    J4 = AF.jacobian(step_fn, z0, create_graph=False, strict=True)
    D = z0.shape[-1]
    assert J4.shape == (1, D, 1, D), J4.shape
    J = J4.squeeze(0).squeeze(1)  # [D, D]

    # Full-state spectral radius (float64 for eig stability on near-zero
    # blocks — many rows are ≈0 because softplus gate silences them).
    J_f64 = J.to(torch.float64)
    eig_full = torch.linalg.eigvals(J_f64).abs()
    rho_full = float(eig_full.max().item())

    # Zero out the n_m × n_m context-memory diagonal block, keeping only
    # the "no-memory" dynamical contribution. The memory block lives at
    # the last n_c rows/cols by pack order.
    n_c = cfg.arch.n_c
    J_nm = J_f64.clone()
    start = D - n_c
    J_nm[start:, start:] = 0.0  # wipe the leak diagonal + mm_gen couplings
    eig_nm = torch.linalg.eigvals(J_nm).abs()
    rho_nm = float(eig_nm.max().item())

    assert math.isfinite(rho_full), rho_full
    assert math.isfinite(rho_nm), rho_nm
    return rho_full, rho_nm


def _apply_overrides(net: V2Network, overrides: dict) -> None:
    """Re-initialise selected raw-weight Parameters with a new init_mean.

    Each key is ``"<module>.<attr>"`` (e.g. ``"l23_e.W_l4_l23_raw"``). Value
    is a Python float — the raw-parameter is re-sampled
    ``Normal(mean, init_scale=0.1)`` using the same per-module torch.Generator
    that :class:`_BasePopulation._make_raw` uses, so the draws are
    reproducible across calls with the same overrides dict.

    Bias scalars (``prediction_head.b_pred_raw``) are handled as a special
    case — they are constant-filled, not sampled.
    """
    for key, value in overrides.items():
        mod_name, attr = key.rsplit(".", 1)
        mod = net
        for part in mod_name.split("."):
            mod = getattr(mod, part)
        param = getattr(mod, attr)
        assert isinstance(param, torch.nn.Parameter), (
            f"{key} is not an nn.Parameter"
        )
        with torch.no_grad():
            if attr == "b_pred_raw":
                param.data.fill_(float(value))
            else:
                # Re-sample Normal(mean=value, std=0.1) deterministically.
                # Python's ``hash()`` is salted between process launches
                # (PYTHONHASHSEED), so use hashlib.sha256 instead — stable
                # across Python runs. The resampled weight values must be
                # reproducible for the Jacobian spectral radius to be
                # comparable between the sweep and the verify step.
                gen = torch.Generator(device=param.device)
                digest = hashlib.sha256(key.encode("utf-8")).digest()
                stable_seed = int.from_bytes(digest[:4], "big")
                gen.manual_seed(stable_seed)
                param.data.normal_(mean=float(value), std=0.1, generator=gen)
        # Update the init-mean registry if present (Task #50 raw_prior).
        if hasattr(mod, "raw_init_means"):
            mod.raw_init_means[attr] = float(value)


def run_calibration(
    *,
    n_steps: int = N_STEPS,
    seed: int = SEED,
    device: str = "cpu",
    overrides: dict | None = None,
) -> CalibReport:
    """Build net, roll blank input ``n_steps`` steps, probe operating point.

    Parameters
    ----------
    overrides : dict | None
        Optional ``{"module.attr_raw": new_init_mean_float}`` mapping applied
        after construction to let the caller sweep without editing layers.py.

    Returns a :class:`CalibReport`. Asserts that homeostasis θ buffers and
    every weight Parameter are byte-identical before and after the rollout
    (no forward-path side-effects).
    """
    torch.manual_seed(seed)
    cfg = ModelConfig()
    # Device/dtype kept as default float32 — this probe is about dynamics,
    # not numerical regression.
    net = V2Network(cfg, token_bank=None, seed=seed, device=device)
    if overrides:
        _apply_overrides(net, overrides)
    net.eval()
    _assert_plasticity_and_homeostasis_are_inert(net)

    # Snapshot θ buffers and every Parameter for post-rollout invariance check.
    theta_l23_before = net.l23_e.theta.detach().clone()
    theta_h_before = net.h_e.theta.detach().clone()
    param_snapshot = {
        name: p.detach().clone() for name, p in net.named_parameters()
    }

    # Blank input: [B=1, 1, H, W] all zeros. With zero frame the LGN/L4
    # front-end produces a non-zero baseline via its tonic biases (if any);
    # we don't need to simulate stimulus, just measure the zero-drive
    # attractor.
    a = cfg.arch
    blank_x = torch.zeros(
        1, 1, a.grid_h, a.grid_w, device=device, dtype=torch.float32,
    )

    state = net.initial_state(batch_size=1, dtype=torch.float32)
    with torch.no_grad():
        for _ in range(n_steps):
            _, state, _ = net(blank_x, state)

    # Invariance checks — no homeostasis or plasticity happened.
    assert torch.equal(net.l23_e.theta, theta_l23_before), (
        "L23E.theta mutated during forward — homeostasis is not off!"
    )
    assert torch.equal(net.h_e.theta, theta_h_before), (
        "HE.theta mutated during forward — homeostasis is not off!"
    )
    for name, p in net.named_parameters():
        assert torch.equal(p, param_snapshot[name]), (
            f"Parameter {name} mutated during forward — plasticity is not off!"
        )

    # Get x_hat at the final state (one more forward step, purely for readout).
    with torch.no_grad():
        x_hat_next, _, info = net(blank_x, state)

    # Jacobian at the final state (requires grad path → enable autograd on z0).
    rho_full, rho_nm = _compute_jacobian_radius(net, state, cfg, blank_x)

    def _median(t: torch.Tensor) -> float:
        return float(t.flatten().median().item())

    def _maxv(t: torch.Tensor) -> float:
        return float(t.flatten().max().item())

    def _l2(t: torch.Tensor) -> float:
        return float(torch.linalg.vector_norm(t.flatten()).item())

    return CalibReport(
        r_l4_median=_median(state.r_l4),
        r_l4_max=_maxv(state.r_l4),
        r_l23_median=_median(state.r_l23),
        r_l23_max=_maxv(state.r_l23),
        r_pv_median=_median(state.r_pv),
        r_pv_max=_maxv(state.r_pv),
        r_som_median=_median(state.r_som),
        r_som_max=_maxv(state.r_som),
        r_h_median=_median(state.r_h),
        r_h_max=_maxv(state.r_h),
        h_pv_median=_median(state.h_pv),
        h_pv_max=_maxv(state.h_pv),
        m_median=_median(state.m),
        m_max=_maxv(state.m),
        x_hat_median=_median(x_hat_next),
        x_hat_max=_maxv(x_hat_next),
        x_hat_l2=_l2(x_hat_next),
        r_l4_l2=_l2(state.r_l4),
        lambda_max_full=rho_full,
        lambda_max_no_memory=rho_nm,
    )


def _evaluate_targets(r: CalibReport) -> dict[str, tuple[bool, str]]:
    """Apply Task-#52 target thresholds. Each entry: (pass, explanation)."""
    t1 = 0.01 <= r.r_l23_median <= 0.5
    t3 = r.r_h_median < r.r_l23_median
    # T4 relaxed (Lead 2026-04-19): "PV/SOM respond when L23E > 0". At
    # blank with r_l23 > 0 this reduces to "inh rate > 0", not "> 0.001".
    t4 = r.r_pv_median > 0.0 and r.r_som_median > 0.0 and r.h_pv_median > 0.0
    # T5 HARD relaxed (Lead 2026-04-19): < 1.0, not < 0.98. The context-
    # memory self-leak exp(-dt/τ_m) ≈ 0.9901 is accepted as a biologically
    # appropriate slow-integrator eigenvalue (Kok cue-delay bridging).
    t5 = r.lambda_max_full < 1.0
    # Target 6 "same order" — within one decade. Handle r_l4_l2 ≈ 0 gracefully.
    if r.r_l4_l2 > 0.0:
        ratio = r.x_hat_l2 / r.r_l4_l2
        t6 = 0.1 <= ratio <= 10.0
        t6_str = f"x_hat_l2/r_l4_l2 = {ratio:.3f}"
    else:
        t6 = r.x_hat_l2 < 1.0  # fallback: just check x_hat hasn't exploded
        t6_str = f"r_l4_l2≈0; x_hat_l2 = {r.x_hat_l2:.3g}"
    return {
        "T1 (L23E median ∈ [0.01, 0.5])": (t1, f"{r.r_l23_median:.4f}"),
        "T3 (HE mean < L23E mean)": (
            t3, f"r_h_med={r.r_h_median:.4f}, r_l23_med={r.r_l23_median:.4f}",
        ),
        "T4 (PV/SOM/HPV respond when L23E>0)": (
            t4,
            f"pv={r.r_pv_median:.4g}, som={r.r_som_median:.4g}, "
            f"hpv={r.h_pv_median:.4g}",
        ),
        "T5 HARD (λmax_full < 1.0)": (
            t5, f"λmax_full={r.lambda_max_full:.4f}",
        ),
        "T6 (|x_hat| ~ |r_l4|)": (t6, t6_str),
    }


def _format_report(r: CalibReport) -> str:
    lines = [
        "=" * 72,
        "Task #52 calibration — blank input, plasticity=0, homeostasis=0",
        "=" * 72,
        f"Rollout steps: {N_STEPS}, seed: {SEED}",
        "",
        "Population rates (median / max) at end-of-rollout:",
        f"  r_l4    : {r.r_l4_median:12.6f} / {r.r_l4_max:12.6f}",
        f"  r_l23   : {r.r_l23_median:12.6f} / {r.r_l23_max:12.6f}",
        f"  r_pv    : {r.r_pv_median:12.6f} / {r.r_pv_max:12.6f}",
        f"  r_som   : {r.r_som_median:12.6f} / {r.r_som_max:12.6f}",
        f"  r_h     : {r.r_h_median:12.6f} / {r.r_h_max:12.6f}",
        f"  h_pv    : {r.h_pv_median:12.6f} / {r.h_pv_max:12.6f}",
        f"  m       : {r.m_median:12.6f} / {r.m_max:12.6f}",
        "",
        "Prediction head:",
        f"  x_hat median / max    : {r.x_hat_median:12.6f} / {r.x_hat_max:12.6f}",
        f"  ||x_hat||_2           : {r.x_hat_l2:12.6f}",
        f"  ||r_l4||_2            : {r.r_l4_l2:12.6f}",
        "",
        "Full-state one-step Jacobian:",
        f"  ρ(J) full-state       : {r.lambda_max_full:.6f}",
        f"  ρ(J) no-memory-block  : {r.lambda_max_no_memory:.6f}",
        "",
        "Target evaluation:",
    ]
    for name, (ok, detail) in _evaluate_targets(r).items():
        mark = "PASS" if ok else "FAIL"
        lines.append(f"  [{mark}] {name}  |  {detail}")
    lines.append("=" * 72)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true",
                        help="emit JSON in addition to the text table")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-steps", type=int, default=N_STEPS)
    args = parser.parse_args()

    report = run_calibration(n_steps=args.n_steps, seed=args.seed)
    print(_format_report(report))
    if args.json:
        print()
        print(json.dumps(report.__dict__, indent=2))


if __name__ == "__main__":
    main()
