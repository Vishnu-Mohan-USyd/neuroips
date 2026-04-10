"""Spiking V1 neural populations: SpikingL4Ring, SpikingPVPool,
SpikingL23Ring, SpikingSOMRing, SpikingVIPRing.

Parallel to `src/model/populations.py` (rate model). Every population
preserves the rate model's learnable parameters, sign constraints (Dale's
law), and structural invariants (e.g. L2/3 circulant W_rec with spectral
radius ≤ 0.95). The only thing that changes is how the membrane drive is
converted into an output: instead of a continuous Euler step on a rate, each
spiking population runs one step of LIF/ALIF dynamics and emits a binary
spike `z ∈ {0, 1}` plus a smoothed trace `x = α·x + z`.

Dynamics (one timestep, `dt = 1 ms`, {0,1} spike encoding)
-----------------------------------------------------------
For every LIF / ALIF population:

    v[t] = β_mem · v[t-1] + drive[t] − z[t-1] · V_thresh    (subtract reset)
    z[t] = atan_surrogate(v[t] − B[t], slope, dampen)
    x[t] = α_filter · x[t-1] + z[t]

where `β_mem = exp(-1/τ_mem)`, `α_filter = exp(-1/τ_filter)`. The effective
threshold `B[t]` is `V_thresh` for plain LIF; for ALIF it adds an adaptation
term (Bellec 2018 §2 eq. 1-2):

    B[t]   = V_thresh + β_adapt · b[t]
    b[t+1] = ρ_adapt · b[t] + (1 − ρ_adapt) · z[t]          (Bellec eq. 2)

`ρ_adapt = exp(-1/τ_adapt)`. See evidence pack §A.4 and §B.3 for the Bellec
citation and the `(1-ρ)` scaling in {0,1} spike encoding.

PV keeps the rate formulation (plan: "the only population without spikes in
the port"): it is a single-compartment leaky integrator whose output is
already a pool rate used as a divisive-normalization denominator. PV ingests
the *filtered spike traces* `x_l4` and `x_l23` instead of rate signals.

Evidence
--------
- Plan file: `/home/vishnu/.claude/plans/quirky-humming-giraffe.md`
    * Architecture mapping (lines 17-26)
    * Time constants (lines 46-55)
    * ALIF dynamics (lines 38-42)
    * Surrogate slope ~25 (line 213)
    * Subtract reset (lines 223-225)
- Rate model: `src/model/populations.py` — parameters and math verbatim
- Evidence pack: `/home/vishnu/.claude/plans/snn_port_evidence_pack.md`
    * snnTorch API (§A.3) — Leaky with reset_mechanism='subtract'
    * ALIF (§A.4) — Bellec 2018 §2 eq. 1-2
    * Firing rate targets (§A.6) — Niell & Stryker 2010, Atallah 2012
    * torch.compile (§C) — fullgraph=False, mode='default'
    * L2/3 recommendation (§A.3, §B.6) — hand-roll W_rec, do NOT use RLeaky
- Team-lead rulings (2026-04-10):
    * Ruling 1: stationary firing rate targets (untrained net)
    * Ruling 3: fixed τ_adapt = 200 ms
    * L4: ALIF threshold adaptation only; drop the explicit SSA subtraction
      from the drive that the rate model uses.

API convention
--------------
Every population is an `nn.Module` with two methods:

    init_state(batch_size, device=None, dtype=torch.float32) -> dict[str, Tensor]
    forward(<drive inputs>, state: dict) -> tuple[dict, Tensor, Tensor]

The forward return is `(new_state, z, x)` where `z` is the binary spike and
`x` is the filtered trace. PV is the sole exception: since it never spikes,
it returns `(new_state, r_pv, r_pv)` — both the "z" slot and the "x" slot
contain the same rate tensor so downstream callers don't need a special
case.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.config import ModelConfig, SpikingConfig
from src.spiking.surrogate import atan_surrogate
from src.utils import (
    InhibitoryGain,
    circular_distance_abs,
    rectified_softplus,
)


# ---------------------------------------------------------------------------
# Helper: build a circular Gaussian recurrent kernel (identical to rate model)
# ---------------------------------------------------------------------------

def _build_rec_kernel(
    n: int, sigma_raw: Tensor, gain_raw: Tensor, period: float = 180.0,
) -> Tensor:
    """Circulant Gaussian kernel with learnable σ and gain, spectral radius ≤ 0.95.

    Copied verbatim from `src/model/populations.py::_build_rec_kernel`. Kept
    as a private helper in this module so `SpikingL23Ring` is self-contained
    and does not create a cross-package dependency on the rate model's
    internals.

    Effective sigma = softplus(sigma_raw), gain = clamp(softplus(gain_raw), max=0.95).
    K is row-normalised so each row sums to 1, then scaled by `gain`, which
    gives spectral radius ≤ gain ≤ 0.95 (strictly contractive).
    """
    sigma = F.softplus(sigma_raw)
    gain = torch.clamp(F.softplus(gain_raw), max=0.95)
    step = period / n
    thetas = torch.arange(n, dtype=torch.float32, device=sigma_raw.device) * step
    dists = circular_distance_abs(
        thetas.unsqueeze(1), thetas.unsqueeze(0), period=period
    )
    K = torch.exp(-dists ** 2 / (2.0 * sigma ** 2))
    K = K / K.sum(dim=-1, keepdim=True)
    return gain * K


# ---------------------------------------------------------------------------
# L4 — ALIF ring (divisive normalization + adaptive threshold)
# ---------------------------------------------------------------------------

class SpikingL4Ring(nn.Module):
    """V1 L4 excitatory ring, ALIF version.

    Preserves the rate model's hand-set tuning (identity W_ff) and divisive
    normalization by PV. Stimulus-specific adaptation (SSA) is implemented
    *only* as an adaptive firing threshold (Bellec 2018 ALIF), not as a
    subtractive drive term — per Lead Ruling (2026-04-10).

    Cold (untrained) firing-rate regime — IMPORTANT
    ------------------------------------------------
    At the default hyperparameters (V_thresh=1.0, beta_mem=exp(-1/5)=0.8187,
    alpha_filter=exp(-1/10)=0.9048), the usable drive range of this LIF
    subtract-reset cell is approximately [V_thresh*(1-beta), 2*V_thresh*(1-beta)]
    = [0.181, 0.362]. Drives of 0.15 produce 0 Hz; drives of 0.22 jump to
    ~40 Hz; drives of 1.0 give ~673 Hz. That is, **an untrained network
    with unit-scale drive fires at 50-1000 Hz**, well above any biological
    range. This is mathematically correct LIF behavior; it is NOT a bug.
    Per the Lead ruling (2026-04-10, Option 1), Phase 1 tests do NOT gate
    on firing-rate ranges — the energy loss during Phase 2 training will
    pull rates down into the 2-15 Hz biological band. If Phase 2 fails to
    calibrate, the fallback is to add a per-population `input_gain` buffer
    (see plans/quirky-humming-giraffe.md Phase 2 fallback note).

    Drive (one step):
        drive = stimulus / (σ_norm² + r_pv_prev)           (no SSA subtract)

    Dynamics:
        v      = β_mem · v_prev + drive − z_prev · V_thresh
        B      = V_thresh + β_adapt · b_prev                (Bellec eq. 1)
        z      = atan_surrogate(v − B, slope, dampen)
        x      = α_filter · x_prev + z
        b_new  = ρ_adapt · b_prev + (1 − ρ_adapt) · z       (Bellec eq. 2)

    Constants (evidence pack §A.2):
        β_mem   = exp(-1 / τ_mem_l4)   with τ_mem_l4 = 5  → 0.8187
        α_filter= exp(-1 / τ_filter)   with τ_filter = 10 → 0.9048
        ρ_adapt = exp(-1 / τ_adapt)    with τ_adapt = 200 → 0.99501
        β_adapt = lsnn_adapt_beta      default 1.8 (Bellec 2018 sMNIST tutorial)

    State (dict keys):
        v  [B, N] — membrane potential
        z  [B, N] — spike from the current step (used as previous-step reset)
        x  [B, N] — exponential-filter trace
        b  [B, N] — ALIF adaptation state (not threshold itself)
    """

    def __init__(self, model_cfg: ModelConfig, spiking_cfg: SpikingConfig):
        super().__init__()
        self.n = model_cfg.n_orientations
        self.sigma_norm_sq = model_cfg.sigma_norm ** 2

        # SNN constants
        self.V_thresh = float(spiking_cfg.V_thresh)
        self.beta_mem = math.exp(-1.0 / spiking_cfg.tau_mem_l4)
        # spike_filter_alpha is guaranteed non-None after SpikingConfig.__post_init__
        self.alpha_filter = float(spiking_cfg.spike_filter_alpha)
        self.rho_adapt = math.exp(-1.0 / spiking_cfg.tau_adapt)
        self.beta_adapt = float(spiking_cfg.lsnn_adapt_beta)
        self.surrogate_slope = float(spiking_cfg.surrogate_slope)
        self.surrogate_dampen = float(spiking_cfg.surrogate_dampen)

        # Identity FF map — buffer, frozen. Rate model uses nn.functional.linear
        # with this matrix, so keeping the buffer preserves the exact numerical
        # structure (and lets downstream code swap in a learned W_ff later if
        # the plan changes).
        self.register_buffer("W_ff", torch.eye(self.n))

        # PV inhibition gain — carried here for convenience / parity with the
        # rate model's V1L4Ring.pv_gain even though it is actually applied at
        # the L2/3 subtractive inhibition site. Kept as a learnable parameter
        # (Dale's law: softplus-wrapped).
        self.pv_gain = InhibitoryGain(init_gain=1.0)

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> dict:
        dev = device or torch.device("cpu")
        z = lambda: torch.zeros(batch_size, self.n, device=dev, dtype=dtype)
        return {"v": z(), "z": z(), "x": z(), "b": z()}

    def forward(
        self,
        stimulus: Tensor,
        r_pv_prev: Tensor,
        state: dict,
    ) -> tuple[dict, Tensor, Tensor]:
        """One ALIF step for L4.

        Args:
            stimulus:  [B, N] population-coded, contrast-scaled input.
            r_pv_prev: [B, 1] PV pool rate from previous step.
            state: dict with keys {v, z, x, b} each [B, N].

        Returns:
            (new_state, z, x) — new_state is a new dict (tensors, not in-place);
            z is the fresh binary spike [B, N]; x is the filtered trace [B, N].
        """
        v_prev = state["v"]
        z_prev = state["z"]
        x_prev = state["x"]
        b_prev = state["b"]

        # Feedforward input (identity W_ff; keeps the F.linear call for
        # numerical parity with the rate model path).
        ff = F.linear(stimulus, self.W_ff)                       # [B, N]
        # Divisive normalization by PV (same form as rate L4). No SSA subtract.
        drive = ff / (self.sigma_norm_sq + r_pv_prev)            # [B, N]

        # LIF update with subtract reset (plan lines 223-225).
        v = self.beta_mem * v_prev + drive - z_prev * self.V_thresh

        # Adaptive threshold (Bellec eq. 1).
        B = self.V_thresh + self.beta_adapt * b_prev

        # Spike via surrogate gradient.
        z = atan_surrogate(
            v - B, slope=self.surrogate_slope, dampen=self.surrogate_dampen,
        )

        # Exponential trace filter (plan line 34).
        x = self.alpha_filter * x_prev + z

        # Adaptation state update (Bellec eq. 2 with {0,1} scaling).
        b = self.rho_adapt * b_prev + (1.0 - self.rho_adapt) * z

        new_state = {"v": v, "z": z, "x": x, "b": b}
        return new_state, z, x


# ---------------------------------------------------------------------------
# PV — rate-based leaky integrator pool (non-spiking, matches plan)
# ---------------------------------------------------------------------------

class SpikingPVPool(nn.Module):
    """PV interneuron pool — rate-based, NOT spiking.

    Per the plan, PV is the single rate-based holdover in the spiking network:
    it serves only as the divisive-normalization denominator for L4 and is not
    expected to carry spike-level information. The rate dynamics are identical
    to `src/model/populations.py::PVPool`, with one change: inputs are the
    *filtered spike traces* `x_l4` and `x_l23` instead of rates `r_l4` and
    `r_l23`. The exponential filters tau=10 already give the traces a
    rate-like steady state, so the substitution is numerically faithful.

    Drive:
        pooled = w_pv_l4 · sum(x_l4, -1) + w_pv_l23 · sum(x_l23, -1)
        r_pv_new = r_pv_prev + (1/τ_pv) · (-r_pv_prev + rectified_softplus(pooled))

    Returned tuple: `(new_state, r_pv, r_pv)` — the "z" and "x" slots both
    carry `r_pv` so downstream code can handle PV through the same `(new, z, x)`
    unpacking pattern used by the spiking populations.

    Learnable parameters (same as rate model):
        w_pv_l4_raw, w_pv_l23_raw — softplus-wrapped non-negative pooling gains
        (Dale's law: pooling weights are non-negative).
    """

    def __init__(self, model_cfg: ModelConfig):
        super().__init__()
        self.tau_pv = float(model_cfg.tau_pv)
        # Learnable non-negative pooling gains (softplus on raw params).
        # Initial values match src/model/populations.py::PVPool (0.1).
        self.w_pv_l4_raw = nn.Parameter(torch.tensor(0.1))
        self.w_pv_l23_raw = nn.Parameter(torch.tensor(0.1))

    @property
    def w_pv_l4(self) -> Tensor:
        return F.softplus(self.w_pv_l4_raw)

    @property
    def w_pv_l23(self) -> Tensor:
        return F.softplus(self.w_pv_l23_raw)

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> dict:
        dev = device or torch.device("cpu")
        return {"r_pv": torch.zeros(batch_size, 1, device=dev, dtype=dtype)}

    def forward(
        self,
        x_l4: Tensor,
        x_l23: Tensor,
        state: dict,
    ) -> tuple[dict, Tensor, Tensor]:
        """One Euler step for the PV pool.

        Args:
            x_l4:  [B, N] filtered L4 trace (this step's).
            x_l23: [B, N] filtered L2/3 trace (previous step's).
            state: dict with key "r_pv" of shape [B, 1].

        Returns:
            ({"r_pv": r_pv}, r_pv, r_pv).
        """
        r_pv_prev = state["r_pv"]

        l4_pooled = x_l4.sum(dim=-1, keepdim=True)                # [B, 1]
        l23_pooled = x_l23.sum(dim=-1, keepdim=True)              # [B, 1]
        pv_drive = self.w_pv_l4 * l4_pooled + self.w_pv_l23 * l23_pooled

        r_pv = r_pv_prev + (1.0 / self.tau_pv) * (
            -r_pv_prev + rectified_softplus(pv_drive)
        )
        return {"r_pv": r_pv}, r_pv, r_pv


# ---------------------------------------------------------------------------
# L2/3 — recurrent LIF with circulant Gaussian W_rec (hand-rolled, NOT RLeaky)
# ---------------------------------------------------------------------------

class SpikingL23Ring(nn.Module):
    """V1 L2/3 excitatory ring, recurrent LIF.

    Do NOT use `snn.RLeaky` — its built-in `W_rec` is a plain `nn.Linear`
    which would destroy the rate model's circulant Gaussian parameterization
    and the spectral-radius ≤ 0.95 contractivity guarantee (Evidence pack
    §A.3 critical caveat). Instead we run a bare LIF step and compute the
    recurrent input ourselves with the same `_build_rec_kernel(σ_rec, g_rec)`
    helper the rate model uses.

    Cold (untrained) firing-rate regime — IMPORTANT
    ------------------------------------------------
    Same LIF subtract-reset regime as SpikingL4Ring. With beta_mem =
    exp(-1/10) = 0.9048 the usable drive range is ~[0.095, 0.19] and the
    network saturates to ~880 Hz at drive = 0.1 in isolation. An untrained
    cold network will thus fire far above biological range (2-5 Hz L2/3).
    Per the Lead ruling (2026-04-10, Option 1), Phase 1 tests do NOT gate
    on firing-rate ranges — Phase 2 training's energy loss calibrates
    rates into biological bands. See SpikingL4Ring docstring for the full
    rationale and the Phase 2 `input_gain` fallback path.

    Drive (one step, matches `V1L23Ring.forward` verbatim modulo the rate→trace
    substitution):
        excitatory = x_l4 + W_rec · x_l23_prev + template_modulation
        if apical_gain is not None: excitatory = apical_gain · excitatory
        drive = excitatory − w_som · x_som − w_pv_l23 · r_pv

    Dynamics:
        v = β_mem · v_prev + drive − z_prev · V_thresh
        z = atan_surrogate(v − V_thresh, slope, dampen)
        x = α_filter · x_prev + z

    Constants:
        β_mem    = exp(-1/τ_mem_l23)  = 0.9048   (τ = 10)
        α_filter = exp(-1/τ_filter)   = 0.9048   (τ = 10)

    Dale's law / structural invariants preserved:
        * W_l4_to_l23 = identity (buffer, frozen)
        * W_rec built from σ_rec_raw, g_rec_raw (2 learnable scalars),
          spectral radius ≤ 0.95 via clamped softplus gain
        * w_som, w_pv_l23 are `InhibitoryGain` modules (softplus-wrapped,
          non-negative, subtracted at the call site)
    """

    def __init__(self, model_cfg: ModelConfig, spiking_cfg: SpikingConfig):
        super().__init__()
        self.n = model_cfg.n_orientations
        self.period = model_cfg.orientation_range

        # SNN constants
        self.V_thresh = float(spiking_cfg.V_thresh)
        self.beta_mem = math.exp(-1.0 / spiking_cfg.tau_mem_l23)
        self.alpha_filter = float(spiking_cfg.spike_filter_alpha)
        self.surrogate_slope = float(spiking_cfg.surrogate_slope)
        self.surrogate_dampen = float(spiking_cfg.surrogate_dampen)

        # Identity FF map L4 → L2/3 (frozen, buffer).
        self.register_buffer("W_l4_to_l23", torch.eye(self.n))

        # W_rec parameterization (σ_rec, g_rec) — 2 learnable scalars only,
        # identical to the rate model's V1L23Ring.
        sigma_init = math.log(math.exp(model_cfg.sigma_rec) - 1.0)  # inverse softplus
        gain_init = math.log(math.exp(model_cfg.gain_rec) - 1.0)
        self.sigma_rec_raw = nn.Parameter(torch.tensor(sigma_init))
        self.gain_rec_raw = nn.Parameter(torch.tensor(gain_init))

        # Inhibitory gains (non-negative via softplus) — Dale's law.
        self.w_som = InhibitoryGain(init_gain=1.0)
        self.w_pv_l23 = InhibitoryGain(init_gain=1.0)

        # Optional kernel cache for reuse across timesteps within one forward pass.
        self._cached_W_rec: Optional[Tensor] = None

    def cache_kernels(self) -> None:
        """Build and cache W_rec so repeated forwards in one rollout reuse it."""
        self._cached_W_rec = _build_rec_kernel(
            self.n, self.sigma_rec_raw, self.gain_rec_raw, self.period
        )

    def uncache_kernels(self) -> None:
        """Clear the cached kernel (call at end of each rollout)."""
        self._cached_W_rec = None

    @property
    def W_rec(self) -> Tensor:
        if self._cached_W_rec is not None:
            return self._cached_W_rec
        return _build_rec_kernel(
            self.n, self.sigma_rec_raw, self.gain_rec_raw, self.period
        )

    @property
    def sigma_rec(self) -> Tensor:
        return F.softplus(self.sigma_rec_raw)

    @property
    def gain_rec(self) -> Tensor:
        return F.softplus(self.gain_rec_raw)

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> dict:
        dev = device or torch.device("cpu")
        z = lambda: torch.zeros(batch_size, self.n, device=dev, dtype=dtype)
        return {"v": z(), "z": z(), "x": z()}

    def forward(
        self,
        x_l4: Tensor,
        template_modulation: Tensor,
        x_som: Tensor,
        r_pv: Tensor,
        state: dict,
        apical_gain: Optional[Tensor] = None,
    ) -> tuple[dict, Tensor, Tensor]:
        """One LIF step for L2/3 with circulant Gaussian recurrence.

        Args:
            x_l4:               [B, N] filtered L4 trace (this step's).
            template_modulation:[B, N] excitatory feedback (zeros in most modes).
            x_som:              [B, N] filtered SOM trace (this step's).
            r_pv:               [B, 1] PV pool rate (this step's).
            state: dict {v, z, x}.
            apical_gain:        [B, N] or None — multiplicative gain on the
                excitatory drive from apical dendrite feedback (centered at 1.0).

        Returns:
            (new_state, z, x).
        """
        v_prev = state["v"]
        z_prev = state["z"]
        x_prev = state["x"]

        # Feedforward from L4 (frozen identity — F.linear call kept for parity).
        ff = F.linear(x_l4, self.W_l4_to_l23)                    # [B, N]

        # Structured recurrence on the previous filtered L2/3 trace.
        W_rec = self.W_rec                                       # [N, N]
        rec = F.linear(x_prev, W_rec)                            # [B, N]

        # Excitatory drive bundle (apical gain multiplies only excitation).
        excitatory = ff + rec + template_modulation
        if apical_gain is not None:
            excitatory = apical_gain * excitatory

        # Inhibition subtracted at the call site (Dale's law).
        drive = excitatory - self.w_som(x_som) - self.w_pv_l23(r_pv)

        # LIF step with subtract reset.
        v = self.beta_mem * v_prev + drive - z_prev * self.V_thresh

        # Spike + filter.
        z = atan_surrogate(
            v - self.V_thresh,
            slope=self.surrogate_slope,
            dampen=self.surrogate_dampen,
        )
        x = self.alpha_filter * x_prev + z

        new_state = {"v": v, "z": z, "x": x}
        return new_state, z, x


# ---------------------------------------------------------------------------
# SOM — plain LIF ring
# ---------------------------------------------------------------------------

class SpikingSOMRing(nn.Module):
    """SOM inhibitory ring — plain LIF (no adaptation, no recurrence).

    Drive is provided externally by the feedback pathway. In simple_feedback
    mode this is `relu(−feedback_signal)` (the negative-part routing of the
    V2 feedback); in emergent mode it is computed by `EmergentFeedbackOperator`;
    in fixed mode it is `feedback.compute_som_drive(...)`.

    Dynamics:
        v = β_mem · v_prev + drive − z_prev · V_thresh
        z = atan_surrogate(v − V_thresh, slope, dampen)
        x = α_filter · x_prev + z
    """

    def __init__(self, model_cfg: ModelConfig, spiking_cfg: SpikingConfig):
        super().__init__()
        self.n = model_cfg.n_orientations

        self.V_thresh = float(spiking_cfg.V_thresh)
        self.beta_mem = math.exp(-1.0 / spiking_cfg.tau_mem_som)
        self.alpha_filter = float(spiking_cfg.spike_filter_alpha)
        self.surrogate_slope = float(spiking_cfg.surrogate_slope)
        self.surrogate_dampen = float(spiking_cfg.surrogate_dampen)

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> dict:
        dev = device or torch.device("cpu")
        z = lambda: torch.zeros(batch_size, self.n, device=dev, dtype=dtype)
        return {"v": z(), "z": z(), "x": z()}

    def forward(
        self,
        som_drive: Tensor,
        state: dict,
    ) -> tuple[dict, Tensor, Tensor]:
        v_prev = state["v"]
        z_prev = state["z"]
        x_prev = state["x"]

        v = self.beta_mem * v_prev + som_drive - z_prev * self.V_thresh
        z = atan_surrogate(
            v - self.V_thresh,
            slope=self.surrogate_slope,
            dampen=self.surrogate_dampen,
        )
        x = self.alpha_filter * x_prev + z

        return {"v": v, "z": z, "x": x}, z, x


# ---------------------------------------------------------------------------
# VIP — plain LIF ring (structurally identical to SOM)
# ---------------------------------------------------------------------------

class SpikingVIPRing(nn.Module):
    """VIP interneuron ring — plain LIF.

    VIP inhibits SOM (disinhibiting L2/3) — see `src/model/network.py`
    emergent feedback branch. Inactive (drive = 0) in simple_feedback mode,
    so the module still needs to exist and run a forward step.

    Dynamics match `SpikingSOMRing` exactly; kept as its own class for
    symmetry with the rate model's separate `SOMRing` and `VIPRing` and to
    allow future divergence (e.g., learned VIP-specific time constants).
    """

    def __init__(self, model_cfg: ModelConfig, spiking_cfg: SpikingConfig):
        super().__init__()
        self.n = model_cfg.n_orientations

        self.V_thresh = float(spiking_cfg.V_thresh)
        self.beta_mem = math.exp(-1.0 / spiking_cfg.tau_mem_vip)
        self.alpha_filter = float(spiking_cfg.spike_filter_alpha)
        self.surrogate_slope = float(spiking_cfg.surrogate_slope)
        self.surrogate_dampen = float(spiking_cfg.surrogate_dampen)

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> dict:
        dev = device or torch.device("cpu")
        z = lambda: torch.zeros(batch_size, self.n, device=dev, dtype=dtype)
        return {"v": z(), "z": z(), "x": z()}

    def forward(
        self,
        vip_drive: Tensor,
        state: dict,
    ) -> tuple[dict, Tensor, Tensor]:
        v_prev = state["v"]
        z_prev = state["z"]
        x_prev = state["x"]

        v = self.beta_mem * v_prev + vip_drive - z_prev * self.V_thresh
        z = atan_surrogate(
            v - self.V_thresh,
            slope=self.surrogate_slope,
            dampen=self.surrogate_dampen,
        )
        x = self.alpha_filter * x_prev + z

        return {"v": v, "z": z, "x": x}, z, x
