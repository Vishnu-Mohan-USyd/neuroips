"""V1 neural populations: V1L4Ring, PVPool, V1L23Ring, DeepTemplate, SOMRing."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.config import ModelConfig
from src.utils import (
    shifted_softplus,
    InhibitoryGain,
    circular_distance_abs,
    circular_gaussian,
)


class V1L4Ring(nn.Module):
    """V1 Layer 4 excitatory ring population.

    Hand-set tuning via identity W_ff (tuning is in the population-coded input),
    divisive normalization by PV, stimulus-specific adaptation (τ_a=200 steps).

    Stimulus arrives pre-scaled by Naka-Rushton contrast gain via generate_grating().

    Equations (one timestep, semi-implicit Euler):
        l4_input = W_ff @ stimulus            (W_ff = identity)
        l4_drive = l4_input / (σ_norm² + pv)  - adaptation
        r_l4 += dt/τ_l4 * (-r_l4 + shifted_softplus(l4_drive))
        adaptation += dt/τ_a * (-adaptation + α * r_l4)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n = cfg.n_orientations
        self.dt = cfg.dt
        self.tau_l4 = cfg.tau_l4
        self.tau_a = cfg.tau_adaptation
        self.alpha = cfg.alpha_adaptation
        self.adaptation_clamp = cfg.adaptation_clamp
        self.sigma_norm_sq = cfg.sigma_norm ** 2

        # W_ff = identity: tuning is already in the population-coded input.
        # Each L4 unit reads directly from the corresponding input channel.
        W_ff = torch.eye(self.n)
        self.register_buffer("W_ff", W_ff)

        # PV inhibition gain (learnable, frozen after Stage 1).
        # Used by L2/3 for subtractive PV inhibition; stored here for convenience.
        self.pv_gain = InhibitoryGain(init_gain=1.0)

    def forward(
        self,
        stimulus: Tensor,
        r_l4_prev: Tensor,
        r_pv_prev: Tensor,
        adaptation_prev: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """One Euler step for L4.

        Args:
            stimulus: [B, N] — already population-coded and contrast-scaled
                      via generate_grating().
            r_l4_prev: [B, N] — previous L4 rates.
            r_pv_prev: [B, 1] — previous PV rate.
            adaptation_prev: [B, N] — previous adaptation state.

        Returns:
            r_l4: Updated L4 rates [B, N].
            adaptation: Updated adaptation [B, N].
        """
        # Feedforward input (W_ff is identity)
        l4_input = F.linear(stimulus, self.W_ff)  # [B, N]

        # Divisive normalization by PV
        l4_drive = l4_input / (self.sigma_norm_sq + r_pv_prev) - adaptation_prev

        # Euler update for L4 rate
        r_l4 = r_l4_prev + (self.dt / self.tau_l4) * (
            -r_l4_prev + shifted_softplus(l4_drive)
        )

        # Euler update for adaptation (uses new r_l4)
        adaptation = adaptation_prev + (self.dt / self.tau_a) * (
            -adaptation_prev + self.alpha * r_l4
        )
        adaptation = adaptation.clamp(max=self.adaptation_clamp)

        return r_l4, adaptation


class PVPool(nn.Module):
    """PV interneuron pool for broad gain control (divisive normalization).

    Pools across ALL orientations from L4 and L2/3.
    Output shape: [B, 1].

    Equation:
        pv_drive = w_pv_l4 * r_l4.sum(-1) + w_pv_l23 * r_l23.sum(-1)
        r_pv += dt/τ_pv * (-r_pv + shifted_softplus(pv_drive))
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.dt = cfg.dt
        self.tau_pv = cfg.tau_pv

        # Learnable non-negative pooling gains (softplus on raw params)
        self.w_pv_l4_raw = nn.Parameter(torch.tensor(0.1))
        self.w_pv_l23_raw = nn.Parameter(torch.tensor(0.1))

    @property
    def w_pv_l4(self) -> Tensor:
        return F.softplus(self.w_pv_l4_raw)

    @property
    def w_pv_l23(self) -> Tensor:
        return F.softplus(self.w_pv_l23_raw)

    def forward(
        self,
        r_l4: Tensor,
        r_l23: Tensor,
        r_pv_prev: Tensor,
    ) -> Tensor:
        """One Euler step for PV pool.

        Args:
            r_l4: [B, N] — current L4 rates (new, from this timestep).
            r_l23: [B, N] — L2/3 rates from PREVIOUS timestep.
            r_pv_prev: [B, 1] — previous PV rate.

        Returns:
            r_pv: Updated PV rate [B, 1].
        """
        # Pool across all orientations
        l4_pooled = r_l4.sum(dim=-1, keepdim=True)    # [B, 1]
        l23_pooled = r_l23.sum(dim=-1, keepdim=True)  # [B, 1]

        # PV drive
        pv_drive = self.w_pv_l4 * l4_pooled + self.w_pv_l23 * l23_pooled

        # Euler update
        r_pv = r_pv_prev + (self.dt / self.tau_pv) * (
            -r_pv_prev + shifted_softplus(pv_drive)
        )

        return r_pv


# ---------------------------------------------------------------------------
# Helper: build a circular Gaussian kernel from learnable σ and gain
# ---------------------------------------------------------------------------

def _build_rec_kernel(
    n: int, sigma_raw: Tensor, gain_raw: Tensor, period: float = 180.0,
) -> Tensor:
    """Build a recurrent kernel from learnable (raw) sigma and gain.

    Effective sigma = softplus(sigma_raw), gain = softplus(gain_raw).
    Returns [n, n] kernel = gain * K(sigma).
    """
    sigma = F.softplus(sigma_raw)  # scalar, positive
    gain = F.softplus(gain_raw)    # scalar, positive
    step = period / n
    thetas = torch.arange(n, dtype=torch.float32, device=sigma_raw.device) * step
    dists = circular_distance_abs(
        thetas.unsqueeze(1), thetas.unsqueeze(0), period=period
    )
    K = torch.exp(-dists ** 2 / (2.0 * sigma ** 2))
    return gain * K


class V1L23Ring(nn.Module):
    """V1 Layer 2/3 excitatory ring population.

    Receives feedforward from L4, structured recurrence (W_rec),
    mechanism-specific template modulation, SOM inhibition, and PV inhibition.

    Constrained connectivity:
    - W_l4_to_l23: identity, registered as buffer (frozen)
    - W_rec: circular Gaussian kernel with 2 learnable params (σ_rec, gain_rec)

    L2/3 drive:
        l23_drive = W_l4_to_l23 @ r_l4 + W_rec @ r_l23_prev
                    + template_modulation - w_som * r_som - w_pv_l23 * r_pv
        r_l23 += dt/τ_l23 * (-r_l23 + shifted_softplus(l23_drive))
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n = cfg.n_orientations
        self.dt = cfg.dt
        self.tau_l23 = cfg.tau_l23
        self.period = cfg.orientation_range

        # W_l4→l23: identity, registered as buffer (frozen).
        # L2/3 inherits L4 tuning map directly.
        self.register_buffer("W_l4_to_l23", torch.eye(self.n))

        # W_rec: circular Gaussian kernel with learnable width and gain.
        # Only 2 free parameters (not a dense matrix!).
        # Initialize: sigma_rec ~ 15°, gain_rec ~ 0.3
        sigma_init = math.log(math.exp(cfg.sigma_rec) - 1.0)  # inverse softplus
        gain_init = math.log(math.exp(cfg.gain_rec) - 1.0)
        self.sigma_rec_raw = nn.Parameter(torch.tensor(sigma_init))
        self.gain_rec_raw = nn.Parameter(torch.tensor(gain_init))

        # Inhibitory gains (non-negative via softplus)
        self.w_som = InhibitoryGain(init_gain=1.0)       # SOM feature-specific
        self.w_pv_l23 = InhibitoryGain(init_gain=1.0)    # PV subtractive

    @property
    def W_rec(self) -> Tensor:
        """The effective recurrent kernel [N, N], built from σ_rec and g_rec."""
        return _build_rec_kernel(
            self.n, self.sigma_rec_raw, self.gain_rec_raw, self.period
        )

    @property
    def sigma_rec(self) -> Tensor:
        return F.softplus(self.sigma_rec_raw)

    @property
    def gain_rec(self) -> Tensor:
        return F.softplus(self.gain_rec_raw)

    def forward(
        self,
        r_l4: Tensor,
        r_l23_prev: Tensor,
        template_modulation: Tensor,
        r_som: Tensor,
        r_pv: Tensor,
    ) -> Tensor:
        """One Euler step for L2/3.

        Args:
            r_l4: [B, N] — current L4 rates.
            r_l23_prev: [B, N] — previous L2/3 rates.
            template_modulation: [B, N] — mechanism-specific excitatory input.
                Zeros for models A, B, D. Center excitation for model C.
            r_som: [B, N] — current SOM rates.
            r_pv: [B, 1] — current PV rate.

        Returns:
            r_l23: Updated L2/3 rates [B, N].
        """
        # Feedforward from L4 (frozen identity)
        ff = F.linear(r_l4, self.W_l4_to_l23)  # [B, N]

        # Structured recurrence (kernel rebuilt from 2 params)
        W_rec = self.W_rec  # [N, N]
        rec = F.linear(r_l23_prev, W_rec)  # [B, N]

        # L2/3 drive
        l23_drive = (ff + rec + template_modulation
                     - self.w_som(r_som) - self.w_pv_l23(r_pv))

        # Euler update
        r_l23 = r_l23_prev + (self.dt / self.tau_l23) * (
            -r_l23_prev + shifted_softplus(l23_drive)
        )

        return r_l23


class DeepTemplate(nn.Module):
    """Deep-V1 expectation template population.

    Computes: deep_template = gain * q_pred * π_pred

    This is a deep-layer population state. It does NOT feed into L2/3
    by default — the feedback mechanism determines how (or whether) the
    template influences superficial V1.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        # Learnable gain scalar, softplus-constrained.
        # Initialize so softplus(raw) ≈ 1.0
        raw_init = math.log(math.exp(cfg.template_gain) - 1.0)
        self.gain_raw = nn.Parameter(torch.tensor(raw_init))

    @property
    def gain(self) -> Tensor:
        return F.softplus(self.gain_raw)

    def forward(self, q_pred: Tensor, pi_pred: Tensor) -> Tensor:
        """Compute deep template.

        Args:
            q_pred: [B, N] — predicted orientation distribution (softmax from V2).
            pi_pred: [B, 1] — prediction precision (bounded scalar from V2).

        Returns:
            deep_template: [B, N] — expectation template.
        """
        return self.gain * q_pred * pi_pred


class SOMRing(nn.Module):
    """SOM inhibitory ring population.

    Feature-specific inhibition with its own rate dynamics.
    Drive is mechanism-dependent (provided externally by feedback module).

    Equation:
        r_som += dt/τ_som * (-r_som + shifted_softplus(som_drive))
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.dt = cfg.dt
        self.tau_som = cfg.tau_som

    def forward(self, som_drive: Tensor, r_som_prev: Tensor) -> Tensor:
        """One Euler step for SOM ring.

        Args:
            som_drive: [B, N] — drive from feedback mechanism.
            r_som_prev: [B, N] — previous SOM rates.

        Returns:
            r_som: Updated SOM rates [B, N].
        """
        r_som = r_som_prev + (self.dt / self.tau_som) * (
            -r_som_prev + shifted_softplus(som_drive)
        )
        return r_som
