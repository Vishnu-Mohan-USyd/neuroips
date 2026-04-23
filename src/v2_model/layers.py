"""Recurrent populations for L2/3 and H (plan v4 step 9 / Task #21).

Five `nn.Module` subclasses — two excitatory (`L23E`, `HE`) and three
inhibitory (`L23PV`, `L23SOM`, `HPV`) — built on shared
`ExcitatoryPopulation` / `InhibitoryPopulation` base classes.

Dale's law via softplus
------------------------
Every incoming connection stores a raw pre-softplus tensor as
``nn.Parameter(..., requires_grad=False)``. At forward time the effective
Dale-compliant weight is computed as

    excitatory:  ``w_eff = +softplus(raw) * mask``  (non-negative)
    inhibitory:  ``w_eff = -softplus(raw) * mask``  (non-positive)

For dense connections `mask` is implicit `1.0` everywhere and omitted.
The sign-at-weight choice (``-softplus(raw)``) plus Hadamard-mask yields
exact zero on masked-off entries for both sign classes without a custom
linear op.

Mask orientation
----------------
``connectivity.generate_sparse_mask`` returns ``[n_pre, n_post]``; we
transpose-and-make-contiguous at construction and store
``[n_post, n_pre]`` so ``F.linear`` (which expects weight
``[out, in]`` i.e. ``[n_post, n_pre]``) sees the mask and weight in the
same orientation (design note #8).

H → L2/3 feedback convention
-----------------------------
``W_fb_apical`` (HE → L23E apical) and ``W_fb_som`` (HE → L23SOM) are
stored as **input projections** on the post-synaptic populations (owned
by ``L23E`` and ``L23SOM`` respectively). Rationale:

1. The plasticity rule governing each feedback weight lives on the
   *postsynaptic* side (predictive Urbanczik–Senn for L23E apical,
   homeostatic-like rule for L23SOM), so keeping the weight matrix on
   the post-synaptic population keeps the plastic-weight-to-rule mapping
   local and avoids cross-module parameter hunts during training.
2. ``HE`` stays pathway-agnostic about which target populations consume
   its activity; wiring lives in ``network.py``.

Dynamics (linear Euler)
------------------------
Every forward step integrates rate as

    drive     = Σ_k F.linear(input_k, w_eff_k)  [+ context_bias]
    activated = φ(drive − θ_or_target_rate)
    r_{t+1}   = (1 − dt/τ) · r_t + activated

where φ is ``rectified_softplus`` by default. Linear-Euler stability
requires the leak factor ``1 − dt/τ`` to stay in (0, 1]; construction
raises if ``dt_ms ≥ tau_ms``. Excitatory populations subtract a per-unit
threshold ``θ`` (maintained by a ``ThresholdHomeostasis`` submodule,
drifted externally by the training driver). Inhibitory populations
subtract a scalar Vogels ``target_rate_hz`` — the I-pop analogue of ``θ``.

Autograd
--------
All raw weights have ``requires_grad=False``. No ``@torch.no_grad()`` on
forward, so BPTT-fallback paths may flow gradients through inputs/state
while never accumulating into module parameters (design note #15).

Phase gating
-------------
All populations here hold generic weights that are plastic in Phase 2
and frozen in both Phase-3 variants. ``plastic_weight_names()`` returns
the full ``_plastic_names`` tuple in ``phase2`` and the empty list in
either Phase 3. The finer task-vs-generic split lives in
``context_memory.py``.

Out of scope
------------
- Wiring the five populations together (that is ``network.py``).
- Running plasticity rules against any of the raw weights.
- Any through-circuit forward pass.
"""

from __future__ import annotations

import math
from typing import Callable, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.v2_model.connectivity import generate_sparse_mask
from src.v2_model.plasticity import ThresholdHomeostasis
from src.v2_model.utils import rectified_softplus

__all__ = [
    "ExcitatoryPopulation", "InhibitoryPopulation", "FastInhibitoryPopulation",
    "L23E", "L23PV", "L23SOM", "HE", "HPV",
    "PhaseLiteral",
]

PhaseLiteral = Literal["phase2", "phase3_kok", "phase3_richter"]
_VALID_PHASES: tuple[str, ...] = ("phase2", "phase3_kok", "phase3_richter")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _excitatory_eff(raw: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """Dale-E effective weight ``+softplus(raw)`` (optionally mask-gated)."""
    w = F.softplus(raw)
    return w * mask if mask is not None else w


def _inhibitory_eff(raw: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """Dale-I effective weight ``-softplus(raw)`` (optionally mask-gated)."""
    w = -F.softplus(raw)
    return w * mask if mask is not None else w


def _validate_size(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0; got {value}")


def _validate_dt_tau(pop_name: str, dt_ms: float, tau_ms: float) -> None:
    """Strict ``dt_ms < tau_ms`` Euler-stability guard (leak factor ∈ (0, 1))."""
    if dt_ms <= 0.0:
        raise ValueError(f"{pop_name}: dt_ms must be > 0; got {dt_ms}")
    if tau_ms <= 0.0:
        raise ValueError(f"{pop_name}: tau_ms must be > 0; got {tau_ms}")
    if dt_ms >= tau_ms:
        raise ValueError(
            f"{pop_name}: dt_ms ({dt_ms}) must be < tau_ms ({tau_ms}) for "
            "linear-Euler stability (leak factor 1 − dt/τ must be > 0)"
        )


def _validate_batch_shape(x: Tensor, name: str, n_expected: int, B: int) -> None:
    if x.ndim != 2 or x.shape[1] != n_expected:
        raise ValueError(f"{name} must be [B, {n_expected}]; got {tuple(x.shape)}")
    if x.shape[0] != B:
        raise ValueError(f"{name} batch-size {x.shape[0]} does not match B={B}")


def _assert_spectral_radius_le(
    W: Tensor, *, leak: float, phi_prime_op: float = 0.67,
    name: str, max_radius: float = 1.4,
) -> float:
    """Raise if ``max|eig(φ'_op · W)| > max_radius`` — the per-population
    Jacobian-contribution bound under the corrected Euler update.

    Task #48: with the corrected ``r_{t+1} = leak·r_t + (1 − leak)·φ(W·r_t
    − θ)`` update, the linearised one-step Jacobian is ``J = leak·I +
    (1 − leak)·φ'(drive)·W``. Euler stability requires ``|eig(J)| < 1``
    for every eigenvalue. Since every ``eig(J) = leak + (1 − leak)·z``
    for some ``z ∈ eig(φ'·W)``, and ``|a + (1 − a)·z| ≤ 1`` iff ``|z| ≤
    1`` (for real ``a ∈ [0, 1]``), the stability condition simplifies to
    ``max|eig(φ'_op · W)| < 1``. The guard checks ``max|eig(φ'·W)|`` on
    the effective recurrent matrix (not the full Jacobian including
    leak, as in Task #42's pre-fix form).

    Parameters
    ----------
    W : Tensor
        Effective (sign-applied, mask-gated) recurrent weight of shape
        ``[n_post, n_pre]`` with ``n_post == n_pre`` (square).
    leak : float
        Retained for back-compat / reporting only. The corrected-Euler
        stability condition no longer depends on ``leak`` explicitly.
    phi_prime_op : float
        Conservative value of ``φ'`` at the operating point. For the
        rectified-softplus ``φ`` used here, ``φ'(x) = σ(x)``; at a
        typical small positive operating drive ``x ≈ 0.7`` we have
        ``σ(0.7) ≈ 0.668``. Default 0.67 is a conservative round number.
    name : str
        Human-readable identifier included in the error message.
    max_radius : float
        Strict upper bound on ``max|eig(φ'_op · W)|``. Default 1.4 —
        at ``φ'_op = 0.67`` this permits ``|eig(W)| ≤ ~2.1``, and the
        resulting full Jacobian eigenvalue ``leak + (1 − leak)·0.94 ≈
        1 − 0.06·(1 − leak)`` stays safely below 1.

    Returns
    -------
    float
        The measured spectral radius of ``φ'_op · W``.
    """
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(
            f"{name}: expected square [N, N] weight; got shape {tuple(W.shape)}"
        )
    with torch.no_grad():
        # Promote to f64 for eigenvalue stability on near-zero matrices.
        W_f64 = W.detach().to(torch.float64)
        eig = torch.linalg.eigvals(float(phi_prime_op) * W_f64)
        radius = eig.abs().max().item()
    if not math.isfinite(radius) or radius > max_radius:
        raise RuntimeError(
            f"{name}: φ'·W spectral radius {radius:.4f} "
            f"(leak={leak:.4f}, φ'_op={phi_prime_op:.2f}) "
            f"exceeds max {max_radius:.4f}; recurrent init is unstable — "
            "check init_mean / init_scale / sparsity."
        )
    return radius


# ---------------------------------------------------------------------------
# Base class (shared scaffolding for E + I bases)
# ---------------------------------------------------------------------------

class _BasePopulation(nn.Module):
    """Scaffolding common to every recurrent population.

    Subclasses register their incoming raw-weight ``nn.Parameter``s (via
    ``self._make_raw``) and declare their plastic-weight names in
    ``self._plastic_names``. The base owns: rate-integration leak factor,
    rectification φ, phase-gating API, plastic/frozen weight manifests,
    and a seeded ``torch.Generator`` used by ``_make_raw``.
    """

    def __init__(
        self,
        n_units: int,
        tau_ms: float,
        dt_ms: float,
        *,
        phi: Callable[[Tensor], Tensor] = rectified_softplus,
        init_scale: float = 0.1,
        seed: int = 0,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        _validate_size("n_units", n_units)
        _validate_dt_tau(type(self).__name__, dt_ms, tau_ms)
        if init_scale <= 0.0:
            raise ValueError(f"init_scale must be > 0; got {init_scale}")

        self.n_units = int(n_units)
        self.tau_ms = float(tau_ms)
        self.dt_ms = float(dt_ms)
        self._leak = 1.0 - self.dt_ms / self.tau_ms
        self._phi = phi
        self._init_scale = float(init_scale)
        self._seed = int(seed)
        self._device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        self._dtype = dtype
        self._phase: str = "phase2"
        self._plastic_names: tuple[str, ...] = ()
        self._gen = torch.Generator(device=self._device)
        self._gen.manual_seed(self._seed)
        # Task #50 — per-weight init-mean registry for raw-prior weight decay.
        # Populated by `_make_raw(name=...)` as each raw parameter is created.
        # The training driver reads this to anchor weight decay to the init
        # mean, avoiding anti-shrinkage for strongly negative raw inits.
        self.raw_init_means: dict[str, float] = {}

    # ---- weight-creation helper ----------------------------------------

    def _make_raw(
        self,
        shape: tuple[int, int],
        *,
        init_mean: float = 0.0,
        name: Optional[str] = None,
    ) -> nn.Parameter:
        """Return a normal-init raw-weight ``nn.Parameter(requires_grad=False)``.

        Parameters
        ----------
        shape
            (n_post, n_pre) — stored in ``F.linear``'s expected orientation.
        init_mean
            Mean of the normal init distribution. Default 0.0 keeps the
            legacy behaviour for dense feedforward / feedback / inter-area
            weights. Recurrent weights wrapped in ``softplus`` pass a
            negative value (e.g. ``-3.5``) so that
            ``softplus(raw) ≈ softplus(−3.5) ≈ 0.0298`` and each row-sum of
            the effective mask-gated recurrent matrix lands safely below
            the critical spectral radius.
        name
            If given, stash ``init_mean`` in ``self.raw_init_means[name]``.
            Consumed by the Phase-2 training driver to build the ``raw_prior``
            anchor for weight-decay (Task #50). Passing the *attribute name*
            (e.g. ``"W_l4_l23_raw"``) is the convention — the driver looks it
            up by weight-name when computing ΔW.
        """
        t = torch.empty(*shape, device=self._device, dtype=self._dtype)
        with torch.no_grad():
            t.normal_(
                mean=float(init_mean), std=self._init_scale, generator=self._gen,
            )
        if name is not None:
            self.raw_init_means[name] = float(init_mean)
        return nn.Parameter(t, requires_grad=False)

    # ---- phase gating API ----------------------------------------------

    def set_phase(self, phase: PhaseLiteral) -> None:
        """Declare the current training phase. Purely informational."""
        if phase not in _VALID_PHASES:
            raise ValueError(
                f"phase must be one of {list(_VALID_PHASES)}; got {phase!r}"
            )
        self._phase = phase

    @property
    def phase(self) -> str:
        return self._phase

    def plastic_weight_names(self) -> list[str]:
        """Names of plastic raw-weight Parameters for the current phase."""
        return list(self._plastic_names) if self._phase == "phase2" else []

    def frozen_weight_names(self) -> list[str]:
        """Names of frozen raw-weight Parameters for the current phase."""
        return [] if self._phase == "phase2" else list(self._plastic_names)


# ---------------------------------------------------------------------------
# ExcitatoryPopulation — base for L23E, HE
# ---------------------------------------------------------------------------

class ExcitatoryPopulation(_BasePopulation):
    """E-base: adds a ``ThresholdHomeostasis`` submodule managing per-unit θ.

    Subclass forward subtracts ``self.theta`` from drive before the
    rectification. ``theta`` is drifted by the training driver via
    ``self.homeostasis.update(activity)``; this module never mutates it
    inside forward.
    """

    def __init__(
        self,
        n_units: int,
        tau_ms: float,
        dt_ms: float,
        *,
        target_rate: float = 0.0,
        lr_homeostasis: float = 1e-5,
        init_theta: float = 0.0,
        phi: Callable[[Tensor], Tensor] = rectified_softplus,
        init_scale: float = 0.1,
        seed: int = 0,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(
            n_units=n_units, tau_ms=tau_ms, dt_ms=dt_ms,
            phi=phi, init_scale=init_scale, seed=seed,
            device=device, dtype=dtype,
        )
        self.homeostasis = ThresholdHomeostasis(
            lr=lr_homeostasis,
            target_rate=target_rate,
            n_units=n_units,
            init_theta=init_theta,
        )

    @property
    def theta(self) -> Tensor:
        """Current per-unit threshold vector (buffer on the homeostasis submodule)."""
        return self.homeostasis.theta


# ---------------------------------------------------------------------------
# InhibitoryPopulation — base for L23PV, L23SOM, HPV
# ---------------------------------------------------------------------------

class InhibitoryPopulation(_BasePopulation):
    """I-base: stores a scalar Vogels ``target_rate_hz`` in place of θ.

    Subclass forward subtracts ``self.target_rate_hz`` (scalar-broadcast)
    from drive before the rectification — the I-pop analogue of E's
    per-unit threshold. ``target_rate_hz`` is a plain float attribute
    (not an ``nn.Parameter``) consumed by Vogels-iSTDP training drivers.
    """

    def __init__(
        self,
        n_units: int,
        tau_ms: float,
        dt_ms: float,
        *,
        target_rate_hz: float = 1.0,
        phi: Callable[[Tensor], Tensor] = rectified_softplus,
        init_scale: float = 0.1,
        seed: int = 0,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(
            n_units=n_units, tau_ms=tau_ms, dt_ms=dt_ms,
            phi=phi, init_scale=init_scale, seed=seed,
            device=device, dtype=dtype,
        )
        if target_rate_hz < 0.0:
            raise ValueError(f"target_rate_hz must be ≥ 0; got {target_rate_hz}")
        self.target_rate_hz = float(target_rate_hz)


# ---------------------------------------------------------------------------
# L2/3 excitatory (L23E)
# ---------------------------------------------------------------------------

class L23E(ExcitatoryPopulation):
    """L2/3 predictive excitatory pyramidal population.

    Inputs per forward step:
      * ``l4_input``           [B, n_l4_e]         L4 E feedforward (dense)
      * ``l23_recurrent_input`` [B, n_units]       L2/3 recurrent (sparse mask)
      * ``som_input``          [B, n_som]          L2/3 SOM inhibition (dense)
      * ``pv_input``           [B, n_pv]           L2/3 PV inhibition (dense)
      * ``h_apical_input``     [B, n_h_e]          HE apical feedback (dense)
      * ``context_bias``       [B, n_units]        C memory additive bias
      * ``state``              [B, n_units]        previous-step rate

    Plastic incoming raw weights (all ``[n_units, n_pre]`` for F.linear):
      ``W_l4_l23_raw``, ``W_rec_raw``, ``W_pv_l23_raw``,
      ``W_som_l23_raw``, ``W_fb_apical_raw``.
    """

    def __init__(
        self,
        n_units: int = 256,
        n_l4_e: int = 128,
        n_pv: int = 16,
        n_som: int = 32,
        n_h_e: int = 64,
        tau_ms: float = 20.0,
        dt_ms: float = 5.0,
        *,
        positions: Optional[Tensor] = None,
        features: Optional[Tensor] = None,
        sparsity: float = 0.12,
        sigma_position: float = 4.0,
        sigma_feature: float = 25.0,
        target_rate: float = 0.0,
        lr_homeostasis: float = 1e-5,
        init_theta: float = 0.0,
        phi: Callable[[Tensor], Tensor] = rectified_softplus,
        init_scale: float = 0.1,
        seed: int = 0,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        w_l4_l23_init_mean: float = 1.5,
        n_cue: int = 48,
    ) -> None:
        super().__init__(
            n_units=n_units, tau_ms=tau_ms, dt_ms=dt_ms,
            target_rate=target_rate, lr_homeostasis=lr_homeostasis,
            init_theta=init_theta, phi=phi, init_scale=init_scale,
            seed=seed, device=device, dtype=dtype,
        )
        _validate_size("n_l4_e", n_l4_e)
        _validate_size("n_pv", n_pv)
        _validate_size("n_som", n_som)
        _validate_size("n_h_e", n_h_e)
        _validate_size("n_cue", n_cue)
        self.n_l4_e = int(n_l4_e)
        self.n_pv = int(n_pv)
        self.n_som = int(n_som)
        self.n_h_e = int(n_h_e)
        self.n_cue = int(n_cue)

        # Task #52 — re-calibrated init_means for plasticity=0/homeostasis=0
        # operating point at blank input. Targets (relaxed per Lead
        # 2026-04-19): T1 L23E median ∈ [0.01, 0.5], T3 HE<L23E, T4 PV/SOM
        # respond when L23E>0, T5 HARD λ(J_full)<1.0, T6 |x̂|~|r_l4|.
        #
        # Rationale for each value:
        #   W_l4_l23 = +4.0 — r_l4 at blank is ≈5e-5 (LGN DC baseline);
        #     softplus(4) ≈ 4.02, so L4→L23 drive per unit ≈ 128·4.02·5e-5
        #     ≈ 0.026 → rectified_softplus(0.026) ≈ 0.013, hitting T1.
        #   W_rec = -5.0 — softplus(-5) ≈ 6.7e-3 × 30 active ≈ 0.2 gain,
        #     safely below the spectral-radius guard.
        #   W_pv_l23 = -5.0 (weakened from -3.0) — lowers the L23↔PV loop
        #     gain. With the stronger L23→PV drive (W_l23_pv = -1.0 in
        #     FastInhibitoryPopulation init via network.py), a stronger
        #     PV→L23 feedback would push the 2-node loop past ρ=1.
        #   W_som_l23 = -5.0 — same loop-stability reasoning as PV.
        #   W_fb_apical = -5.0 — keeps H→L23 feedback weak at init;
        #     Phase-2 learning grows it.
        #
        # Target 2 (grating orientation selectivity at init) is deferred
        # to Phase-2 predictive training — untrained near-uniform L4→L23
        # weights cannot produce selective responses.
        # Task #74 Fix K (2026-04-22): W_l4_l23 is sparse orientation-biased
        # at init (mask installed post-construction by V2Network). init_mean
        # drops 4.0 → 1.5 so per-unit drive stays in biological range with
        # ~15 connections per row: softplus(1.5)≈1.70; 15 × 1.70 × r_l4≈0.2
        # ≈ 5 Hz, vs the unmasked 4.0×128 run which saturated at 30 Hz.
        # Mask is installed via ``install_l4_l23_mask`` and defaults to an
        # all-ones buffer so existing callers / tests get no behavioural
        # change until they opt in.
        self.W_l4_l23_raw = self._make_raw(
            (n_units, self.n_l4_e),
            init_mean=float(w_l4_l23_init_mean),
            name="W_l4_l23_raw",
        )
        self.register_buffer(
            "mask_l4_l23",
            torch.ones((n_units, self.n_l4_e), dtype=self._dtype,
                       device=self._device),
        )
        self.W_rec_raw = self._make_raw(
            (n_units, n_units), init_mean=-5.0, name="W_rec_raw",
        )
        # Task #74 Fix L-2 (2026-04-22): W_pv_l23_raw init_mean -6.0 → -6.5.
        # Fix L at -6.0 left L23E rate at 0.468 Hz — 0.032 Hz under the
        # 0.5 Hz biological floor — because the L23E↔PV loop compensated:
        # halving W_pv let L23E rise, which drove L23E→PV higher; PV climbed
        # from 19 → 36 Hz, absorbing ~70% of the inhibition reduction
        # (net PV→L23E inhibition 1.42 Hz vs pre-Fix-L 2.04 Hz). Loop-
        # amplification insight: single-knob PV→L23E cuts are partially
        # self-cancelling via the recurrent PV→L23→PV path. -6.5 drops the
        # softplus ratio a further 0.60× (0.22× of original -5.0), predicted
        # L23E 0.7–1.0 Hz, PV 20–28 Hz, FWHM ~50–55°, R²≈1, compression
        # 1.8–2.2×. Original Fix L rationale (-5.0 → -6.0): feedforward L4
        # drive ≈1.4 Hz was overwhelmed by 2.04 Hz of PV inhibition →
        # L23E collapse at 0.26 Hz in pre-Fix-L Level 4; halving PV
        # inhibition was meant to restore the biological band without
        # losing the Naka-Rushton divisive-normalization signature
        # (R²=1.0, gain-compression 2.26× in that run).
        self.W_pv_l23_raw = self._make_raw(
            (n_units, self.n_pv), init_mean=-6.5, name="W_pv_l23_raw",
        )
        self.W_som_l23_raw = self._make_raw(
            (n_units, self.n_som), init_mean=-5.0, name="W_som_l23_raw",
        )
        self.W_fb_apical_raw = self._make_raw(
            (n_units, self.n_h_e), init_mean=-5.0, name="W_fb_apical_raw",
        )

        # generate_sparse_mask returns [n_pre, n_post]; transpose to
        # [n_post, n_pre] for F.linear usage (design note #8).
        raw_mask = generate_sparse_mask(
            positions=positions, features=features, n_units=n_units,
            sparsity=sparsity,
            sigma_position=(sigma_position if positions is not None else None),
            sigma_feature=(sigma_feature if features is not None else None),
            seed=seed, device=self._device,
        )
        self.register_buffer(
            "mask_rec", raw_mask.t().contiguous().to(dtype=self._dtype)
        )

        # Stability guard — fail loudly if init yields super-critical recurrence.
        # Checks the FULL Jacobian ``leak·I + φ'·W_rec_eff`` (Task #42), not
        # just ``W_rec_eff`` — a too-loose guard on ``W`` alone missed the
        # leak contribution and let long trajectories (370–1200 steps)
        # explode even after Task #36's init-mean tightening.
        _assert_spectral_radius_le(
            _excitatory_eff(self.W_rec_raw, self.mask_rec),
            leak=self._leak,
            name="L23E.W_rec (softplus-gated, masked)",
            max_radius=1.4,
        )

        self._plastic_names = (
            "W_l4_l23_raw", "W_rec_raw", "W_pv_l23_raw",
            "W_som_l23_raw", "W_fb_apical_raw",
        )

        # Task #74 β-mechanism (Step 1, 2026-04-23): per-cue multiplicative
        # gain on the L4→L23E feedforward contribution. Shape ``[n_cue,
        # n_units]``, values in R+ (commonly in [0, 1] for predictive
        # suppression). When ``L23E.forward`` receives a non-None ``q_t``
        # of shape ``[B, n_cue]``, the effective L4 feedforward drive is
        # ``F.linear(l4, w_l4) * (q_t @ W_q_gain)``. At init the buffer
        # is all ones, so for any one-hot ``q_t`` the gain equals 1 and
        # forward is bit-exact to the pre-Fix path. Non-persistent buffer
        # so legacy checkpoints load without schema errors; callers that
        # want learned/hand-crafted gain install it explicitly via the
        # public ``W_q_gain`` attribute after construction.
        self.register_buffer(
            "W_q_gain",
            torch.ones(
                (self.n_cue, self.n_units),
                dtype=self._dtype, device=self._device,
            ),
            persistent=False,
        )

    def plastic_weight_names(self) -> list[str]:
        """L23E extension: W_q_gain is plastic in phase3_kok.

        Step 3 (2026-04-23): during the Kok learning sub-phase the driver
        applies the three-factor rule
        ``ΔW_q_gain[cue_id, :] = -η · cue_on · r_l23e · sign(matched)``
        clamped to ``[gain_min, gain_max]``. W_q_gain is a non-Parameter
        buffer, so the usual ``requires_grad`` gating does not apply;
        this manifest entry signals to Validators that the training
        loop may mutate the buffer under phase3_kok. Frozen everywhere
        else (phase2 substrate, phase3_richter, null_control, scan).
        """
        base = super().plastic_weight_names()
        if self._phase == "phase3_kok":
            return list(base) + ["W_q_gain"]
        return base

    # ------------------------------------------------------------------
    # Fix-K mask installer
    # ------------------------------------------------------------------

    @torch.no_grad()
    def install_l4_l23_mask(self, mask: Tensor) -> None:
        """Replace ``mask_l4_l23`` with a caller-supplied binary mask.

        Shape must match ``W_l4_l23_raw`` — [n_l23_e, n_l4_e]. Called once
        by :class:`V2Network.__init__` after constructing both L23E and
        LGN/L4; no RNG is consumed inside this layer.
        """
        if mask.shape != self.W_l4_l23_raw.shape:
            raise ValueError(
                f"install_l4_l23_mask: got {tuple(mask.shape)}, "
                f"expected {tuple(self.W_l4_l23_raw.shape)}"
            )
        self.mask_l4_l23.copy_(mask.to(dtype=self._dtype, device=self._device))

    def forward(
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
        q_t: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """One Euler step of L2/3 E dynamics. Returns ``(rate, updated_state)``.

        Task #74 Fix C-v2: ``som_gain`` is an optional per-SOM-unit
        multiplicative scale on the SOM→L23E inhibitory efficacy
        (shape ``[B, n_som]``, produced by ``ContextMemory``). When
        provided, the SOM input is multiplied elementwise by
        ``som_gain`` before the ``F.linear(…, w_som)`` projection, so
        each SOM unit's outgoing inhibition onto every L23E postsynaptic
        neuron is scaled by the same per-batch, per-SOM-unit factor.
        The gain is bounded in (0, 4.0] by design (see
        ``ContextMemory.forward``); at init it is identically 1.0, so
        this kwarg is a no-op unless Phase-3 task plasticity has
        populated ``W_mh_task_inh``. Keyword-only and optional so
        existing callers (Phase-2 drivers, unit tests) are unaffected.
        """
        B = state.shape[0]
        _validate_batch_shape(state, "state", self.n_units, B)
        _validate_batch_shape(l4_input, "l4_input", self.n_l4_e, B)
        _validate_batch_shape(l23_recurrent_input, "l23_recurrent_input",
                              self.n_units, B)
        _validate_batch_shape(som_input, "som_input", self.n_som, B)
        _validate_batch_shape(pv_input, "pv_input", self.n_pv, B)
        _validate_batch_shape(h_apical_input, "h_apical_input", self.n_h_e, B)
        _validate_batch_shape(context_bias, "context_bias", self.n_units, B)
        if som_gain is not None:
            _validate_batch_shape(som_gain, "som_gain", self.n_som, B)
        if q_t is not None:
            _validate_batch_shape(q_t, "q_t", self.n_cue, B)

        # Fix K: feedforward mask applied multiplicatively on the softplus
        # output. At Level-2-isolation test time this is the only thing that
        # gives each L23E unit an orientation-biased row of L4 afferents.
        w_l4 = _excitatory_eff(self.W_l4_l23_raw) * self.mask_l4_l23
        w_rec = _excitatory_eff(self.W_rec_raw, self.mask_rec)
        w_pv = _inhibitory_eff(self.W_pv_l23_raw)
        w_som = _inhibitory_eff(self.W_som_l23_raw)
        w_fb = _excitatory_eff(self.W_fb_apical_raw)

        som_effective = som_input * som_gain if som_gain is not None else som_input

        # Task #74 β-mechanism: per-cue multiplicative gain on the L4→L23E
        # contribution. Only applied when q_t is supplied; at default init
        # W_q_gain is all ones, so any one-hot q_t leaves ff_l4 unchanged.
        ff_l4 = F.linear(l4_input, w_l4)
        if q_t is not None:
            cue_gain = q_t @ self.W_q_gain                # [B, n_units]
            ff_l4 = ff_l4 * cue_gain

        drive = (
            ff_l4
            + F.linear(l23_recurrent_input, w_rec)
            + F.linear(som_effective, w_som)
            + F.linear(pv_input, w_pv)
            + F.linear(h_apical_input, w_fb)
            + context_bias
        )
        activated = self._phi(drive - self.theta)
        rate_next = self._leak * state + (1.0 - self._leak) * activated
        return rate_next, rate_next


# ---------------------------------------------------------------------------
# L2/3 PV (L23PV)
# ---------------------------------------------------------------------------

class L23PV(InhibitoryPopulation):
    """L2/3 parvalbumin+ inhibitory population (fast divisive stabilization).

    NOTE on default time constants. The plan table has τ_pv = dt = 5 ms,
    which fails the strict ``dt < τ`` Euler stability guard. The default
    here uses τ = 10 ms so construction works out-of-the-box; the caller
    (``network.py``) is expected to override if a different τ is chosen.
    """

    def __init__(
        self,
        n_units: int = 16,
        n_l23_e: int = 256,
        tau_ms: float = 10.0,
        dt_ms: float = 5.0,
        *,
        target_rate_hz: float = 1.0,
        phi: Callable[[Tensor], Tensor] = rectified_softplus,
        init_scale: float = 0.1,
        seed: int = 0,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(
            n_units=n_units, tau_ms=tau_ms, dt_ms=dt_ms,
            target_rate_hz=target_rate_hz, phi=phi, init_scale=init_scale,
            seed=seed, device=device, dtype=dtype,
        )
        _validate_size("n_l23_e", n_l23_e)
        self.n_l23_e = int(n_l23_e)
        self.W_l23_pv_raw = self._make_raw(
            (n_units, self.n_l23_e), name="W_l23_pv_raw",
        )
        self._plastic_names = ("W_l23_pv_raw",)

    def forward(
        self, l23e_input: Tensor, state: Tensor
    ) -> tuple[Tensor, Tensor]:
        """One Euler step of L2/3 PV dynamics."""
        B = state.shape[0]
        _validate_batch_shape(state, "state", self.n_units, B)
        _validate_batch_shape(l23e_input, "l23e_input", self.n_l23_e, B)
        # Presynaptic is excitatory, so the input weight is +softplus(raw).
        w = _excitatory_eff(self.W_l23_pv_raw)
        drive = F.linear(l23e_input, w)
        activated = self._phi(drive - self.target_rate_hz)
        rate_next = self._leak * state + (1.0 - self._leak) * activated
        return rate_next, rate_next


# ---------------------------------------------------------------------------
# L2/3 SOM (L23SOM)
# ---------------------------------------------------------------------------

class L23SOM(InhibitoryPopulation):
    """L2/3 somatostatin+ inhibitory population (apical / surround inhibition)."""

    def __init__(
        self,
        n_units: int = 32,
        n_l23_e: int = 256,
        n_h_e: int = 64,
        tau_ms: float = 20.0,
        dt_ms: float = 5.0,
        *,
        target_rate_hz: float = 1.0,
        phi: Callable[[Tensor], Tensor] = rectified_softplus,
        init_scale: float = 0.1,
        seed: int = 0,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(
            n_units=n_units, tau_ms=tau_ms, dt_ms=dt_ms,
            target_rate_hz=target_rate_hz, phi=phi, init_scale=init_scale,
            seed=seed, device=device, dtype=dtype,
        )
        _validate_size("n_l23_e", n_l23_e)
        _validate_size("n_h_e", n_h_e)
        self.n_l23_e = int(n_l23_e)
        self.n_h_e = int(n_h_e)
        # Task #52 — T29 calibration. W_l23_som = -1.0 puts SOM drive
        # ≈ 256·softplus(-1)·r_l23 = 256·0.313·0.012 = 0.96 just below the
        # target_rate_hz=1.0 threshold at blank; at typical nonzero input
        # (r_l23 > 0.013) SOM responds. Loop gain stays bounded because
        # W_som_l23 in L23E is -5.0 (weak SOM→L23 feedback).
        # H→SOM feedback kept at -5.0 (weak; Phase-2 learns).
        # Task #74 Fix M (2026-04-22): init_mean -4.5 → -4.2. Fix E's -4.5
        # set W=softplus(-4.5)=0.0110, which — *after* Fix K brought L23E
        # down from 1.66 Hz to 0.58 Hz — left SOM drive at 256·0.58·0.0110
        # ≈ 1.64 Hz (raw); phi_in = drive − target_rate_hz(1.0) = 0.64 →
        # rectified-softplus(0.64) = 0.37 Hz (Level-5 pre-Fix-M: observed
        # 0.372 Hz, under the 0.5 Hz biological floor). Fix E was calibrated
        # against the pre-Fix-K L23E substrate that no longer applies.
        # -4.2 (softplus=0.0149, 1.35× higher) predicts SOM drive 2.20 Hz,
        # phi_in 1.20, SOM rate 0.77 Hz — cleanly in band. SOM → L23E
        # knock-on: n_som(32)·softplus(W_som_l23=-5.0)(0.0067)·0.77 = 0.17
        # Hz extra inhibition; L23E drops ~0.03 Hz (0.58 → 0.55).
        #
        # Historical: Fix E (2026-04-21) moved -1.0 → -4.5 to escape r_som
        # saturation at ~129 Hz. That remedy was still the right sign but
        # overshot once Fix K shrank L23E drive. SOM selectivity remains
        # low by design (all-to-all W_l23_som produces uniform tuning at
        # init per G3 audit); Phase-2/3 plasticity is where orientation-
        # tuned SOM is expected to emerge.
        self.W_l23_som_raw = self._make_raw(
            (n_units, self.n_l23_e), init_mean=-4.2, name="W_l23_som_raw",
        )
        self.W_fb_som_raw = self._make_raw(
            (n_units, self.n_h_e), init_mean=-5.0, name="W_fb_som_raw",
        )
        # Task #74 Fix D-simpler: excitatory inputs onto L2/3 SOM are frozen
        # at init in all phases. Earlier Fix D attempts (Vogels iSTDP +
        # Turrigiano synaptic scaling on these raws) drove r_som to
        # saturation — multiple rules fighting on one weight. Freezing
        # eliminates the pathology; the Phase-3 experiment question doesn't
        # require plasticity on these synapses.
        self._plastic_names: tuple[str, ...] = ()

    def forward(
        self,
        l23e_input: Tensor,
        h_som_feedback_input: Tensor,
        state: Tensor,
        *,
        context_bias_som: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """One Euler step of L2/3 SOM dynamics.

        Task #74 Fix C: ``context_bias_som`` is an optional additive current
        from the context-memory task-specific inhibitory readout
        (``W_mh_task_inh @ m``) — the MAIN route of the Phase-3 task bias,
        providing SOM-mediated apical gain control of L23 E dendrites
        (Urbanczik & Senn 2014; Larkum 1999). Keyword-only and optional so
        existing callers (Phase-2 drivers, layer unit tests) are unaffected.
        """
        B = state.shape[0]
        _validate_batch_shape(state, "state", self.n_units, B)
        _validate_batch_shape(l23e_input, "l23e_input", self.n_l23_e, B)
        _validate_batch_shape(h_som_feedback_input, "h_som_feedback_input",
                              self.n_h_e, B)
        w_local = _excitatory_eff(self.W_l23_som_raw)
        w_fb = _excitatory_eff(self.W_fb_som_raw)
        drive = (
            F.linear(l23e_input, w_local)
            + F.linear(h_som_feedback_input, w_fb)
        )
        if context_bias_som is not None:
            _validate_batch_shape(
                context_bias_som, "context_bias_som", self.n_units, B,
            )
            drive = drive + context_bias_som
        activated = self._phi(drive - self.target_rate_hz)
        rate_next = self._leak * state + (1.0 - self._leak) * activated
        return rate_next, rate_next


# ---------------------------------------------------------------------------
# H excitatory (HE)
# ---------------------------------------------------------------------------

class HE(ExcitatoryPopulation):
    """Higher-area excitatory population (non-retinotopic latent prior).

    Sparse HE↔HE recurrence uses ``generate_sparse_mask`` with
    ``positions=None, features=None`` — a uniform-random fallback that
    still enforces exact row-wise sparsity.
    """

    def __init__(
        self,
        n_units: int = 64,
        n_l23_e: int = 256,
        n_h_pv: int = 8,
        tau_ms: float = 50.0,
        dt_ms: float = 5.0,
        *,
        sparsity: float = 0.12,
        target_rate: float = 0.0,
        lr_homeostasis: float = 1e-5,
        init_theta: float = 0.0,
        phi: Callable[[Tensor], Tensor] = rectified_softplus,
        init_scale: float = 0.1,
        seed: int = 0,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(
            n_units=n_units, tau_ms=tau_ms, dt_ms=dt_ms,
            target_rate=target_rate, lr_homeostasis=lr_homeostasis,
            init_theta=init_theta, phi=phi, init_scale=init_scale,
            seed=seed, device=device, dtype=dtype,
        )
        _validate_size("n_l23_e", n_l23_e)
        _validate_size("n_h_pv", n_h_pv)
        self.n_l23_e = int(n_l23_e)
        self.n_h_pv = int(n_h_pv)

        # Task #74 Fix N (2026-04-22): W_l23_h_raw init_mean -5.7 → -5.5.
        # T29's -5.7 was calibrated against the pre-Fix-K L23E rate (~1.66
        # Hz); post-Fix-K L23E settles at 0.56 Hz which left HE at 0.046 Hz
        # (Level 6, 0.004 Hz under the 0.05 Hz biological floor). Closed-
        # form at -5.7: drive_ff = 256·softplus(-5.7)(0.00335)·0.563 = 0.481;
        # drive_inh = -8·softplus(-5.0)(0.00667)·7.305 = -0.392; drive_net =
        # 0.089; rectified-softplus(0.089) = 0.0455 (matches observed 0.046).
        # -5.5 (softplus 1.22×) predicts drive_ff 0.59, drive_net 0.20 →
        # HE rate 0.10 Hz. T29's "HE < L23E at blank" constraint still holds
        # (0.10 ≪ 0.56). Apical-feedback onto L23E rises 0.02 → 0.04 Hz —
        # negligible, so L23E/SOM/PV balances from Fix K/L/M are preserved.
        # Self-rec = -5.0; HPV→HE = -5.0 (weakens HE↔HPV loop so HPV can
        # fire strongly without driving ρ(J) past 1.0).
        self.W_l23_h_raw = self._make_raw(
            (n_units, self.n_l23_e), init_mean=-5.5, name="W_l23_h_raw",
        )
        self.W_rec_raw = self._make_raw(
            (n_units, n_units), init_mean=-5.0, name="W_rec_raw",
        )
        self.W_pv_h_raw = self._make_raw(
            (n_units, self.n_h_pv), init_mean=-5.0, name="W_pv_h_raw",
        )

        raw_mask = generate_sparse_mask(
            positions=None, features=None, n_units=n_units,
            sparsity=sparsity, sigma_position=None, sigma_feature=None,
            seed=seed, device=self._device,
        )
        self.register_buffer(
            "mask_rec", raw_mask.t().contiguous().to(dtype=self._dtype)
        )

        # Stability guard — fail loudly if init yields super-critical recurrence.
        # Checks the FULL Jacobian ``leak·I + φ'·W_rec_eff`` (Task #42) — HE's
        # τ=50 ms, dt=5 ms gives leak=0.9, so the leak term alone already puts
        # most of the Jacobian eigenvalue budget in place; a naive ``|eig(W)|<1``
        # guard misses this.
        _assert_spectral_radius_le(
            _excitatory_eff(self.W_rec_raw, self.mask_rec),
            leak=self._leak,
            name="HE.W_rec (softplus-gated, masked)",
            max_radius=1.4,
        )

        self._plastic_names = ("W_l23_h_raw", "W_rec_raw", "W_pv_h_raw")

    def forward(
        self,
        l23_input: Tensor,
        h_recurrent_input: Tensor,
        h_pv_input: Tensor,
        context_bias: Tensor,
        state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """One Euler step of HE dynamics."""
        B = state.shape[0]
        _validate_batch_shape(state, "state", self.n_units, B)
        _validate_batch_shape(l23_input, "l23_input", self.n_l23_e, B)
        _validate_batch_shape(h_recurrent_input, "h_recurrent_input",
                              self.n_units, B)
        _validate_batch_shape(h_pv_input, "h_pv_input", self.n_h_pv, B)
        _validate_batch_shape(context_bias, "context_bias", self.n_units, B)

        w_ff = _excitatory_eff(self.W_l23_h_raw)
        w_rec = _excitatory_eff(self.W_rec_raw, self.mask_rec)
        w_pv = _inhibitory_eff(self.W_pv_h_raw)

        drive = (
            F.linear(l23_input, w_ff)
            + F.linear(h_recurrent_input, w_rec)
            + F.linear(h_pv_input, w_pv)
            + context_bias
        )
        activated = self._phi(drive - self.theta)
        rate_next = self._leak * state + (1.0 - self._leak) * activated
        return rate_next, rate_next


# ---------------------------------------------------------------------------
# H PV (HPV)
# ---------------------------------------------------------------------------

class HPV(InhibitoryPopulation):
    """Higher-area PV inhibitory population (fast divisive stabilization for HE).

    Same tau-default caveat as ``L23PV``: plan has τ = dt = 5 ms which
    fails the strict ``dt < τ`` guard, so the default here uses τ = 10 ms.
    """

    def __init__(
        self,
        n_units: int = 8,
        n_h_e: int = 64,
        tau_ms: float = 10.0,
        dt_ms: float = 5.0,
        *,
        target_rate_hz: float = 1.0,
        phi: Callable[[Tensor], Tensor] = rectified_softplus,
        init_scale: float = 0.1,
        seed: int = 0,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(
            n_units=n_units, tau_ms=tau_ms, dt_ms=dt_ms,
            target_rate_hz=target_rate_hz, phi=phi, init_scale=init_scale,
            seed=seed, device=device, dtype=dtype,
        )
        _validate_size("n_h_e", n_h_e)
        self.n_h_e = int(n_h_e)
        self.W_h_pv_raw = self._make_raw(
            (n_units, self.n_h_e), name="W_h_pv_raw",
        )
        self._plastic_names = ("W_h_pv_raw",)

    def forward(
        self, he_input: Tensor, state: Tensor
    ) -> tuple[Tensor, Tensor]:
        """One Euler step of HPV dynamics."""
        B = state.shape[0]
        _validate_batch_shape(state, "state", self.n_units, B)
        _validate_batch_shape(he_input, "he_input", self.n_h_e, B)
        w = _excitatory_eff(self.W_h_pv_raw)
        drive = F.linear(he_input, w)
        activated = self._phi(drive - self.target_rate_hz)
        rate_next = self._leak * state + (1.0 - self._leak) * activated
        return rate_next, rate_next


# ---------------------------------------------------------------------------
# FastInhibitoryPopulation — exact-ODE leak (admits τ ≤ dt)
# ---------------------------------------------------------------------------

class FastInhibitoryPopulation(_BasePopulation):
    """Single-input inhibitory population with exact-ODE leak exp(-dt/τ).

    Plan v4 assigns τ_PV = τ_HPV = 5 ms with dt = 5 ms. Linear Euler's
    leak factor (1 − dt/τ) collapses to zero at τ = dt and goes
    negative for τ < dt, so the strict ``dt < τ`` guard in
    :class:`_BasePopulation` rejects the plan's PV time constants. The
    exact homogeneous-ODE solution ``exp(-dt/τ)`` is bounded in (0, 1)
    for any positive (τ, dt) pair; using it gives a numerically stable
    update for the fast-PV regime without modifying the existing
    :class:`L23PV` / :class:`HPV` classes (which keep the strict guard
    for isolated-population testing).

    Forward logic matches :class:`L23PV` / :class:`HPV` — a single
    excitatory input stream, rectify-then-leak. Plastic weight name is
    ``W_pre_raw``.
    """

    def __init__(
        self,
        n_units: int,
        n_pre: int,
        tau_ms: float,
        dt_ms: float,
        *,
        target_rate_hz: float = 1.0,
        phi: Callable[[Tensor], Tensor] = rectified_softplus,
        init_scale: float = 0.1,
        w_pre_init_mean: float = 0.0,
        freeze_W_pre: bool = False,
        seed: int = 0,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        _validate_size("n_pre", n_pre)
        if dt_ms <= 0.0:
            raise ValueError(f"dt_ms must be > 0; got {dt_ms}")
        if tau_ms <= 0.0:
            raise ValueError(f"tau_ms must be > 0; got {tau_ms}")
        if target_rate_hz < 0.0:
            raise ValueError(f"target_rate_hz must be ≥ 0; got {target_rate_hz}")

        # Parent enforces strict dt < tau. We bypass by passing a safe tau,
        # then install the exact-ODE leak below.
        safe_tau = max(float(tau_ms), 2.0 * float(dt_ms))
        super().__init__(
            n_units=n_units, tau_ms=safe_tau, dt_ms=dt_ms,
            phi=phi, init_scale=init_scale, seed=seed,
            device=device, dtype=dtype,
        )
        self.tau_ms = float(tau_ms)
        self._leak = math.exp(-float(dt_ms) / float(tau_ms))
        self.target_rate_hz = float(target_rate_hz)
        self.n_pre = int(n_pre)
        # Task #52 — per-instance init_mean on W_pre_raw. l23_pv and h_pv
        # share this class but require different operating points under
        # plasticity=0 / homeostasis=0; the parameter lets network.py set
        # them independently.
        self.W_pre_raw = self._make_raw(
            (n_units, self.n_pre),
            init_mean=w_pre_init_mean,
            name="W_pre_raw",
        )
        # Task #74 Fix Q — ``freeze_W_pre=True`` marks this weight non-plastic
        # in every phase. Used for l23_pv where Vogels iSTDP on the E→I pre
        # synapse inverted sign and drove L23E collapse; freezing at init
        # matches the Fix D-simpler pattern used for L23SOM excitatory inputs.
        self._plastic_names = () if freeze_W_pre else ("W_pre_raw",)

    def forward(
        self, pre_input: Tensor, state: Tensor
    ) -> tuple[Tensor, Tensor]:
        """One Euler step. Pre-synaptic population is excitatory."""
        B = state.shape[0]
        _validate_batch_shape(state, "state", self.n_units, B)
        _validate_batch_shape(pre_input, "pre_input", self.n_pre, B)
        w = _excitatory_eff(self.W_pre_raw)
        drive = F.linear(pre_input, w)
        activated = self._phi(drive - self.target_rate_hz)
        rate_next = self._leak * state + (1.0 - self._leak) * activated
        return rate_next, rate_next
