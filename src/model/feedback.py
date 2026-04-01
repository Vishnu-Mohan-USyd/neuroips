"""Unified feedback mechanism for Models A-E.

All models share the same kernel family. Mechanism identity is imposed by
constraining specific parameters to zero. This is the most important class
in the project — the mechanism comparison lives here.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.config import ModelConfig, MechanismType
from src.utils import circular_distance_abs, shifted_softplus


def _inv_softplus(x: float) -> float:
    """Inverse of softplus: returns raw such that softplus(raw) ≈ x."""
    if x > 20.0:
        return x  # For large x, softplus(x) ≈ x
    return math.log(math.exp(x) - 1.0)


class FeedbackMechanism(nn.Module):
    """Unified feedback mechanism for Models A-E.

    Kernel family:
        A (dampening):      SOM = g_surr * K_narrow @ q_pred * pi_pred
                            Narrow positive kernel → peaks AT expected
        B (sharpening):     SOM = b_som + pi_pred * (g_surr * K_broad - g_ctr * K_narrow) @ q_pred
                            Signed DoG → minimum AT expected, maximum at flanks
        C (center-surround): SOM = g_surr * K_broad @ q_pred * pi_pred - g_ctr * K_narrow @ q_pred * pi_pred
                            center_excitation = g_ctr * K_narrow @ q_pred * pi_pred (to L2/3)
        D (adaptation):     No feedback — zero SOM
        E (predictive_error): No SOM; error = shifted_softplus(l4 - template)

    B vs C distinction:
        B: SOM is center-sparing (DoG), NO excitation to L2/3
        C: SOM is positive broad, WITH narrow excitation to L2/3
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.mechanism = cfg.mechanism
        self.n_orient = cfg.n_orientations
        self.period = cfg.orientation_range

        # Kernel infrastructure (shared by A, B, C)
        if self.mechanism not in (MechanismType.ADAPTATION_ONLY, MechanismType.PREDICTIVE_ERROR):

            if self.mechanism == MechanismType.DAMPENING:
                # Model A: narrow positive kernel, 2 params
                self.surround_gain_raw = nn.Parameter(torch.tensor(_inv_softplus(1.0)))
                self.surround_width_raw = nn.Parameter(torch.tensor(_inv_softplus(10.0)))

            elif self.mechanism == MechanismType.SHARPENING:
                # Model B: signed DoG (broad - narrow), 4 params (no baseline needed with centered q)
                self.surround_gain_raw = nn.Parameter(torch.tensor(_inv_softplus(1.0)))
                self.surround_width_raw = nn.Parameter(torch.tensor(_inv_softplus(35.0)))
                self.center_gain_raw = nn.Parameter(torch.tensor(_inv_softplus(1.5)))
                self.center_width_raw = nn.Parameter(torch.tensor(_inv_softplus(10.0)))

            else:  # CENTER_SURROUND
                # Model C: broad positive SOM + narrow excitation to L2/3, 4 params
                self.surround_gain_raw = nn.Parameter(torch.tensor(_inv_softplus(1.0)))
                self.surround_width_raw = nn.Parameter(torch.tensor(_inv_softplus(30.0)))
                self.center_gain_raw = nn.Parameter(torch.tensor(_inv_softplus(1.0)))
                self.center_width_raw = nn.Parameter(torch.tensor(_inv_softplus(10.0)))

        # Precompute orientation distances for kernel construction
        step = self.period / self.n_orient
        thetas = torch.arange(self.n_orient, dtype=torch.float32) * step
        dists = circular_distance_abs(
            thetas.unsqueeze(1), thetas.unsqueeze(0), self.period
        )
        self.register_buffer("dists_sq", dists ** 2)

    def _make_kernel(self, sigma: Tensor) -> Tensor:
        """Build a row-normalised circular Gaussian kernel [N, N] from sigma."""
        K = torch.exp(-self.dists_sq / (2.0 * sigma ** 2))
        return K / K.sum(dim=-1, keepdim=True)

    @property
    def surround_width(self) -> Tensor:
        """Effective surround width with mechanism-specific constraints."""
        raw = F.softplus(self.surround_width_raw)
        if self.mechanism == MechanismType.DAMPENING:
            return raw.clamp(max=15.0)
        elif self.mechanism == MechanismType.SHARPENING:
            # Enforce: broad sigma >= narrow sigma + 10 deg
            sigma_narrow = F.softplus(self.center_width_raw)
            return torch.clamp(raw, min=(sigma_narrow + 10.0).item())
        return raw

    @property
    def surround_gain(self) -> Tensor:
        return F.softplus(self.surround_gain_raw)

    @property
    def center_width(self) -> Tensor:
        return F.softplus(self.center_width_raw)

    @property
    def center_gain(self) -> Tensor:
        return F.softplus(self.center_gain_raw)

    def compute_som_drive(self, q_pred: Tensor, pi_pred: Tensor) -> Tensor:
        """Compute SOM drive based on mechanism type.

        Uses centered q_pred (q - 1/N) so feedback = 0 when V2 is uninformative.

        Args:
            q_pred: [B, N] — predicted orientation distribution.
            pi_pred: [B, 1] — prediction precision.

        Returns:
            som_drive: [B, N] — drive for SOM ring.
        """
        if self.mechanism in (MechanismType.ADAPTATION_ONLY, MechanismType.PREDICTIVE_ERROR):
            return torch.zeros_like(q_pred)

        N = q_pred.shape[-1]
        q_centered = q_pred - 1.0 / N

        if self.mechanism == MechanismType.DAMPENING:
            # Model A: narrow positive kernel → peaks AT expected
            K = self._make_kernel(self.surround_width)  # row-normalised
            return self.surround_gain * (K @ q_centered.unsqueeze(-1)).squeeze(-1) * pi_pred

        elif self.mechanism == MechanismType.SHARPENING:
            # Model B: signed DoG → minimum at expected, maximum at flanks
            K_broad = self._make_kernel(self.surround_width)   # row-normalised
            K_narrow = self._make_kernel(self.center_width)     # row-normalised
            broad = self.surround_gain * (K_broad @ q_centered.unsqueeze(-1)).squeeze(-1)
            narrow = self.center_gain * (K_narrow @ q_centered.unsqueeze(-1)).squeeze(-1)
            return pi_pred * (broad - narrow)  # no baseline needed with centered q

        else:  # CENTER_SURROUND
            # Model C: broad positive SOM (surround - center)
            K_surround = self._make_kernel(self.surround_width)  # row-normalised
            K_center = self._make_kernel(self.center_width)      # row-normalised
            surround = self.surround_gain * (K_surround @ q_centered.unsqueeze(-1)).squeeze(-1) * pi_pred
            center = self.center_gain * (K_center @ q_centered.unsqueeze(-1)).squeeze(-1) * pi_pred
            return surround - center

    def compute_center_excitation(self, q_pred: Tensor, pi_pred: Tensor) -> Tensor:
        """Compute center excitation for L2/3 (only nonzero for Model C).

        Uses centered q_pred and clamps non-negative (excitation only).

        Args:
            q_pred: [B, N] — predicted orientation distribution.
            pi_pred: [B, 1] — prediction precision.

        Returns:
            center_excitation: [B, N] — excitatory input to L2/3 drive (>= 0).
        """
        if self.mechanism != MechanismType.CENTER_SURROUND:
            return torch.zeros_like(q_pred)

        N = q_pred.shape[-1]
        q_centered = q_pred - 1.0 / N
        K_center = self._make_kernel(self.center_width)
        raw_excitation = self.center_gain * (K_center @ q_centered.unsqueeze(-1)).squeeze(-1) * pi_pred
        return F.relu(raw_excitation)  # Clamp non-negative — excitation only

    def compute_error_signal(self, r_l4: Tensor, deep_template: Tensor) -> Tensor:
        """Compute prediction error for Model E, or pass-through for others.

        For Model E: error = shifted_softplus(l4 - template).
        For all others: returns r_l4 unchanged.

        Args:
            r_l4: [B, N] — current L4 rates.
            deep_template: [B, N] — deep-V1 expectation template.

        Returns:
            l4_to_l23: [B, N] — either raw L4 or rectified error.
        """
        if self.mechanism != MechanismType.PREDICTIVE_ERROR:
            return r_l4  # pass through unchanged

        return shifted_softplus(r_l4 - deep_template)
