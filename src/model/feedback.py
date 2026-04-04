"""Feedback mechanisms for the laminar V1-V2 model.

Two feedback systems:
1. FeedbackMechanism (fixed): Models A-E with hardcoded kernel shapes.
2. EmergentFeedbackOperator: Learned circulant kernel via basis functions.

Mechanism identity is imposed by constraining specific parameters to zero
(fixed mode) or emerges from training objectives (emergent mode).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.config import ModelConfig, MechanismType
from src.utils import circular_distance_abs, circular_distance, shifted_softplus


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

        # Kernel cache (populated by cache_kernels(), cleared by uncache_kernels())
        self._cached_surround_kernel: Tensor | None = None
        self._cached_center_kernel: Tensor | None = None

    def cache_kernels(self) -> None:
        """Build and cache feedback kernels for reuse across timesteps."""
        if self.mechanism in (MechanismType.ADAPTATION_ONLY, MechanismType.PREDICTIVE_ERROR):
            return
        self._cached_surround_kernel = self._make_kernel(self.surround_width)
        if self.mechanism in (MechanismType.SHARPENING, MechanismType.CENTER_SURROUND):
            self._cached_center_kernel = self._make_kernel(self.center_width)

    def uncache_kernels(self) -> None:
        """Clear cached kernels after the forward pass."""
        self._cached_surround_kernel = None
        self._cached_center_kernel = None

    def _make_kernel(self, sigma: Tensor) -> Tensor:
        """Build a row-normalised circular Gaussian kernel [N, N] from sigma."""
        K = torch.exp(-self.dists_sq / (2.0 * sigma ** 2))
        return K / K.sum(dim=-1, keepdim=True)

    def _get_surround_kernel(self) -> Tensor:
        """Return cached surround kernel or build one."""
        if self._cached_surround_kernel is not None:
            return self._cached_surround_kernel
        return self._make_kernel(self.surround_width)

    def _get_center_kernel(self) -> Tensor:
        """Return cached center kernel or build one."""
        if self._cached_center_kernel is not None:
            return self._cached_center_kernel
        return self._make_kernel(self.center_width)

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
            K = self._get_surround_kernel()
            return self.surround_gain * (K @ q_centered.unsqueeze(-1)).squeeze(-1) * pi_pred

        elif self.mechanism == MechanismType.SHARPENING:
            # Model B: signed DoG → minimum at expected, maximum at flanks
            K_broad = self._get_surround_kernel()
            K_narrow = self._get_center_kernel()
            broad = self.surround_gain * (K_broad @ q_centered.unsqueeze(-1)).squeeze(-1)
            narrow = self.center_gain * (K_narrow @ q_centered.unsqueeze(-1)).squeeze(-1)
            return pi_pred * (broad - narrow)  # no baseline needed with centered q

        else:  # CENTER_SURROUND
            # Model C: broad positive SOM (surround - center)
            K_surround = self._get_surround_kernel()
            K_center = self._get_center_kernel()
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
        K_center = self._get_center_kernel()
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


class EmergentFeedbackOperator(nn.Module):
    """Learned feedback operator via circulant basis functions.

    Instead of hardcoded kernel shapes (dampening, sharpening, center-surround),
    this operator learns two profiles (inhibitory and excitatory) as linear
    combinations of fixed circular basis functions.

    Basis functions (K ~ 7):
        Even: narrow (sigma=5), medium (sigma=15), broad (sigma=30),
              very broad (sigma=60), Mexican hat (narrow - broad), constant.
        Odd: sin-like for tuning shift detection.

    Learnable parameters:
        alpha_inh [K]: weights for SOM (inhibitory) pathway

    The scientific result is a phase diagram: under different loss weights,
    the operator converges to different kernel shapes that can be classified
    post-hoc against the known mechanism templates.
    """

    def __init__(self, cfg: ModelConfig, delta_som: bool = False):
        super().__init__()
        N = cfg.n_orientations
        self.n_orient = N
        self.period = cfg.orientation_range
        self.delta_som = delta_som

        # Build fixed circular basis functions
        basis = self._build_basis(N, cfg.orientation_range)  # [K, N]
        self.register_buffer("basis", basis)
        K = basis.shape[0]

        # Learnable weights for inhibitory (SOM) profile only.
        # Small non-zero init to avoid dead ReLU gradient (relu(0) has grad 0).
        # At 0.01, feedback output is ~0.01 * pi — effectively off but gradient
        # can flow. During burn-in (feedback_scale=0), this is fully zeroed out.
        self.alpha_inh = nn.Parameter(torch.full((K,), 0.01))

        # Delta-SOM baseline: softplus(baseline + field) - softplus(baseline)
        # removes the constant bias from softplus, so zero field → zero drive.
        if self.delta_som:
            self.som_baseline = nn.Parameter(torch.tensor(0.0))

        # Cache for circulant matrix (populated by cache_kernels)
        self._cached_inh_circulant: Tensor | None = None

    def _build_basis(self, N: int, period: float) -> Tensor:
        """Build circular basis functions over orientation space.

        Args:
            N: Number of orientation channels (e.g. 36).
            period: Orientation range in degrees (e.g. 180).

        Returns:
            Basis matrix [K, N] where K ~ 7 basis functions.
        """
        step = period / N
        thetas = torch.arange(N, dtype=torch.float32) * step  # [N]

        # Unsigned circular distances from channel 0
        dists = circular_distance_abs(
            thetas.unsqueeze(0), thetas[0:1].unsqueeze(1), period
        ).squeeze(0)  # [N]

        bases = []

        # Even Gaussians at different widths (4 bases)
        for sigma in [5.0, 15.0, 30.0, 60.0]:
            g = torch.exp(-dists ** 2 / (2 * sigma ** 2))
            g = g / g.sum()  # normalize to sum=1
            bases.append(g)

        # Mexican hat: narrow - broad (1 basis)
        narrow = torch.exp(-dists ** 2 / (2 * 10.0 ** 2))
        broad = torch.exp(-dists ** 2 / (2 * 30.0 ** 2))
        mh = narrow / narrow.sum() - broad / broad.sum()
        bases.append(mh)

        # Constant / global gain (1 basis)
        bases.append(torch.ones(N) / N)

        # Odd basis: sin-like for tuning shifts (1 basis)
        signed_dists = circular_distance(
            thetas.unsqueeze(0), thetas[0:1].unsqueeze(1), period
        ).squeeze(0)  # [N], signed
        odd1 = torch.sin(signed_dists * math.pi / (period / 2))  # one cycle over +/- period/2
        odd1 = odd1 / (odd1.abs().sum() + 1e-8)
        bases.append(odd1)

        return torch.stack(bases)  # [K, N]

    def get_profiles(self) -> Tensor:
        """Return the current inhibitory (SOM) kernel profile.

        Returns:
            K_inh: [N] inhibitory (SOM) kernel profile.
        """
        K_inh = (self.alpha_inh.unsqueeze(-1) * self.basis).sum(dim=0)  # [N]
        return K_inh

    def _to_circulant(self, profile: Tensor) -> Tensor:
        """Convert a 1D profile [N] to a circulant matrix [N, N].

        Row i of the circulant matrix is profile shifted by i positions.
        Entry (i, j) = profile[(j - i) % N].

        Args:
            profile: [N] kernel profile (centered at channel 0).

        Returns:
            Circulant matrix [N, N].
        """
        N = profile.shape[0]
        indices = torch.arange(N, device=profile.device)
        return profile[(indices.unsqueeze(0) - indices.unsqueeze(1)) % N]

    def cache_kernels(self) -> None:
        """Build and cache circulant matrix for reuse across timesteps."""
        K_inh = self.get_profiles()
        self._cached_inh_circulant = self._to_circulant(K_inh)

    def uncache_kernels(self) -> None:
        """Clear cached circulant matrix after the forward pass."""
        self._cached_inh_circulant = None

    def forward(self, q_pred: Tensor, pi_eff: Tensor) -> Tensor:
        """Compute SOM drive from learned inhibitory profile.

        Args:
            q_pred: [B, N] -- predicted orientation distribution.
            pi_eff: [B, 1] -- effective precision (after warmup scaling).

        Returns:
            som_drive: [B, N] -- non-negative drive for SOM ring.
        """
        N = q_pred.shape[-1]
        q_centered = q_pred - 1.0 / N  # zero-mean so feedback=0 when uninformative

        # Use cached or compute circulant matrix
        if self._cached_inh_circulant is not None:
            inh_circulant = self._cached_inh_circulant
        else:
            K_inh = self.get_profiles()
            inh_circulant = self._to_circulant(K_inh)

        # Circular convolution: circulant @ q_centered
        inh_field = (inh_circulant @ q_centered.unsqueeze(-1)).squeeze(-1)  # [B, N]

        # Scale by precision. Use softplus (not relu) to keep output non-negative
        # while preserving gradient flow at zero (relu has zero gradient at 0,
        # which causes dead weights when alpha is pushed to zero by L1 sparsity).
        if self.delta_som:
            # Delta-SOM: softplus(baseline + field) - softplus(baseline)
            # Removes constant bias so zero field → zero drive.
            som_drive = pi_eff * (F.softplus(self.som_baseline + inh_field) - F.softplus(self.som_baseline))
        else:
            som_drive = pi_eff * F.softplus(inh_field)

        return som_drive
