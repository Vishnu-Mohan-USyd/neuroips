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

    def compute_center_excitation(
        self,
        q_pred: Tensor,
        pi_pred: Tensor,
        gate_signal: Tensor | None = None,
    ) -> Tensor:
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
        self.center_support_enabled = cfg.emergent_center_support_enabled
        self.center_support_gain = cfg.emergent_center_support_gain
        self.center_support_sigma = cfg.emergent_center_support_sigma
        self.center_support_cue_gated = cfg.emergent_center_support_cue_gated
        self.recurrent_gain_enabled = cfg.emergent_recurrent_gain_enabled
        self.recurrent_gain_beta = cfg.emergent_recurrent_gain_beta
        self.recurrent_gain_sigma = cfg.emergent_recurrent_gain_sigma
        self.recurrent_gain_mode = cfg.emergent_recurrent_gain_mode
        self.recurrent_gain_flank_beta = cfg.emergent_recurrent_gain_flank_beta
        self.recurrent_gain_sigma_surround = cfg.emergent_recurrent_gain_sigma_surround
        self.recurrent_gain_cue_gated = cfg.emergent_recurrent_gain_cue_gated
        self.flank_som_enabled = cfg.emergent_flank_som_enabled
        self.flank_som_gain = cfg.emergent_flank_som_gain
        self.flank_som_sigma_center = cfg.emergent_flank_som_sigma_center
        self.flank_som_sigma_surround = cfg.emergent_flank_som_sigma_surround
        self.flank_som_cue_gated = cfg.emergent_flank_som_cue_gated
        self.flank_shunt_enabled = cfg.emergent_flank_shunt_enabled
        self.flank_shunt_gain = cfg.emergent_flank_shunt_gain
        self.flank_shunt_sigma_center = cfg.emergent_flank_shunt_sigma_center
        self.flank_shunt_sigma_surround = cfg.emergent_flank_shunt_sigma_surround
        self.flank_shunt_cue_gated = cfg.emergent_flank_shunt_cue_gated
        self.flank_shunt_source = cfg.emergent_flank_shunt_source
        self.som_regime_gate_enabled = cfg.som_regime_gate_enabled
        self.som_regime_gate_target = cfg.som_regime_gate_target
        self.apical_gain_enabled = cfg.apical_gain_enabled
        self.apical_gain_beta = cfg.apical_gain_beta
        self.apical_gain_sigma = cfg.apical_gain_sigma
        self.apical_gain_cue_gated = cfg.apical_gain_cue_gated
        self.pi_max = cfg.pi_max

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
        self._cached_center_support_circulant: Tensor | None = None
        self._cached_recurrent_gain_circulant: Tensor | None = None
        self._cached_recurrent_gain_surround_circulant: Tensor | None = None
        self._cached_flank_som_circulant: Tensor | None = None
        self._cached_flank_shunt_circulant: Tensor | None = None
        self._cached_flank_shunt_center_circulant: Tensor | None = None
        self._cached_apical_gain_circulant: Tensor | None = None

        step = cfg.orientation_range / N
        thetas = torch.arange(N, dtype=torch.float32) * step
        center_dists = circular_distance_abs(
            thetas.unsqueeze(0), thetas[0:1].unsqueeze(1), cfg.orientation_range
        ).squeeze(0)
        self.register_buffer(
            "center_support_dists_sq", center_dists ** 2, persistent=False
        )

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
        if self.center_support_enabled and self.center_support_gain > 0.0:
            center_profile = self._make_center_support_profile()
            self._cached_center_support_circulant = self._to_circulant(center_profile)
        if self.recurrent_gain_enabled and (
            self.recurrent_gain_beta > 0.0 or self.recurrent_gain_flank_beta > 0.0
        ):
            recurrent_profile = self._make_recurrent_gain_profile()
            self._cached_recurrent_gain_circulant = self._to_circulant(recurrent_profile)
            if self.recurrent_gain_mode == "signed_center_surround":
                surround_profile = self._make_recurrent_gain_surround_profile()
                self._cached_recurrent_gain_surround_circulant = self._to_circulant(surround_profile)
        if self.flank_som_enabled and self.flank_som_gain > 0.0:
            flank_profile = self._make_flank_som_profile()
            self._cached_flank_som_circulant = self._to_circulant(flank_profile)
        if self.flank_shunt_enabled and self.flank_shunt_gain > 0.0:
            shunt_profile = self._make_flank_shunt_profile()
            self._cached_flank_shunt_circulant = self._to_circulant(shunt_profile)
            center_profile = self._make_flank_shunt_center_profile()
            self._cached_flank_shunt_center_circulant = self._to_circulant(center_profile)
        if self.apical_gain_enabled and self.apical_gain_beta > 0.0:
            apical_profile = self._make_apical_gain_profile()
            self._cached_apical_gain_circulant = self._to_circulant(apical_profile)

    def uncache_kernels(self) -> None:
        """Clear cached circulant matrix after the forward pass."""
        self._cached_inh_circulant = None
        self._cached_center_support_circulant = None
        self._cached_recurrent_gain_circulant = None
        self._cached_recurrent_gain_surround_circulant = None
        self._cached_flank_som_circulant = None
        self._cached_flank_shunt_circulant = None
        self._cached_flank_shunt_center_circulant = None
        self._cached_apical_gain_circulant = None

    def _make_center_support_profile(self) -> Tensor:
        """Build a narrow non-negative Gaussian profile centered at channel 0."""
        sigma = max(self.center_support_sigma, 1e-6)
        profile = torch.exp(-self.center_support_dists_sq / (2.0 * sigma ** 2))
        return profile / profile.sum()

    def _get_center_support_circulant(self) -> Tensor:
        """Return cached or on-the-fly center-support circulant matrix."""
        if self._cached_center_support_circulant is not None:
            return self._cached_center_support_circulant
        return self._to_circulant(self._make_center_support_profile())

    def _make_recurrent_gain_profile(self) -> Tensor:
        """Build the center-focused recurrent-gain profile centered at channel 0."""
        sigma = max(self.recurrent_gain_sigma, 1e-6)
        profile = torch.exp(-self.center_support_dists_sq / (2.0 * sigma ** 2))
        return profile / profile.sum()

    def _get_recurrent_gain_circulant(self) -> Tensor:
        """Return cached or on-the-fly recurrent-gain circulant matrix."""
        if self._cached_recurrent_gain_circulant is not None:
            return self._cached_recurrent_gain_circulant
        return self._to_circulant(self._make_recurrent_gain_profile())

    def _make_recurrent_gain_surround_profile(self) -> Tensor:
        """Build the broader surround profile for signed recurrent modulation."""
        sigma_center = max(self.recurrent_gain_sigma, 1e-6)
        sigma_surround = max(self.recurrent_gain_sigma_surround, sigma_center + 1e-6)
        profile = torch.exp(-self.center_support_dists_sq / (2.0 * sigma_surround ** 2))
        return profile / profile.sum()

    def _get_recurrent_gain_surround_circulant(self) -> Tensor:
        """Return cached or on-the-fly recurrent surround circulant matrix."""
        if self._cached_recurrent_gain_surround_circulant is not None:
            return self._cached_recurrent_gain_surround_circulant
        return self._to_circulant(self._make_recurrent_gain_surround_profile())

    def _make_flank_dog_profile(self, sigma_center: float, sigma_surround: float) -> Tensor:
        """Build a broad-minus-narrow DoG profile for center-spared flank effects.

        The returned profile is signed. After convolving with ``q_centered`` and
        rectifying, it yields a non-negative field that is near zero at the
        predicted center and positive on nearby flanks.
        """
        sigma_center = max(sigma_center, 1e-6)
        sigma_surround = max(sigma_surround, sigma_center + 1e-6)
        center = torch.exp(-self.center_support_dists_sq / (2.0 * sigma_center ** 2))
        surround = torch.exp(-self.center_support_dists_sq / (2.0 * sigma_surround ** 2))
        center = center / center.sum()
        surround = surround / surround.sum()
        return surround - center

    def _make_flank_som_profile(self) -> Tensor:
        """Build the flank-SOM DoG profile."""
        return self._make_flank_dog_profile(
            self.flank_som_sigma_center,
            self.flank_som_sigma_surround,
        )

    def _get_flank_som_circulant(self) -> Tensor:
        """Return cached or on-the-fly flank-SOM circulant matrix."""
        if self._cached_flank_som_circulant is not None:
            return self._cached_flank_som_circulant
        return self._to_circulant(self._make_flank_som_profile())

    def _make_flank_shunt_profile(self) -> Tensor:
        """Build the flank-only shunting DoG profile."""
        return self._make_flank_dog_profile(
            self.flank_shunt_sigma_center,
            self.flank_shunt_sigma_surround,
        )

    def _make_flank_shunt_center_profile(self) -> Tensor:
        """Build a narrow center profile for center-recruited shunt sources."""
        sigma = max(self.flank_shunt_sigma_center, 1e-6)
        profile = torch.exp(-self.center_support_dists_sq / (2.0 * sigma ** 2))
        return profile / profile.sum()

    def _get_flank_shunt_circulant(self) -> Tensor:
        """Return cached or on-the-fly flank-shunt circulant matrix."""
        if self._cached_flank_shunt_circulant is not None:
            return self._cached_flank_shunt_circulant
        return self._to_circulant(self._make_flank_shunt_profile())

    def _get_flank_shunt_center_circulant(self) -> Tensor:
        """Return cached or on-the-fly center-recruitment circulant matrix."""
        if self._cached_flank_shunt_center_circulant is not None:
            return self._cached_flank_shunt_center_circulant
        return self._to_circulant(self._make_flank_shunt_center_profile())

    def _make_apical_gain_profile(self) -> Tensor:
        """Build a narrow non-negative apical gain profile centered at channel 0."""
        sigma = max(self.apical_gain_sigma, 1e-6)
        profile = torch.exp(-self.center_support_dists_sq / (2.0 * sigma ** 2))
        return profile / profile.sum()

    def _get_apical_gain_circulant(self) -> Tensor:
        """Return cached or on-the-fly apical-gain circulant matrix."""
        if self._cached_apical_gain_circulant is not None:
            return self._cached_apical_gain_circulant
        return self._to_circulant(self._make_apical_gain_profile())

    def forward(
        self,
        q_pred: Tensor,
        pi_eff: Tensor,
        som_regime_gate: Tensor | None = None,
    ) -> Tensor:
        """Compute SOM drive from learned inhibitory profile.

        Args:
            q_pred: [B, N] -- predicted orientation distribution.
            pi_eff: [B, 1] -- effective precision (after warmup scaling).
            som_regime_gate: Optional [B, 1] scalar gate that scales only the
                learned inhibitory field before the SOM nonlinearity.

        Returns:
            som_drive: [B, N] -- non-negative drive for SOM ring.
        """
        if getattr(self, "_analysis_force_zero", False):
            return torch.zeros_like(q_pred)

        if som_regime_gate is None or not self.som_regime_gate_enabled:
            gate = torch.ones(
                q_pred.shape[0], 1, device=q_pred.device, dtype=q_pred.dtype
            )
        else:
            assert som_regime_gate.shape == (q_pred.shape[0], 1), (
                f"som_regime_gate shape {som_regime_gate.shape} must be ({q_pred.shape[0]}, 1)"
            )
            if self.som_regime_gate_target != "alpha_inh":
                raise ValueError(
                    f"Unsupported som_regime_gate_target={self.som_regime_gate_target!r}. "
                    "Expected 'alpha_inh'."
                )
            gate = som_regime_gate.to(device=q_pred.device, dtype=q_pred.dtype)

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
        inh_field = inh_field * gate

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

    def compute_center_excitation(
        self,
        q_pred: Tensor,
        pi_eff: Tensor,
        gate_signal: Tensor | None = None,
    ) -> Tensor:
        """Compute optional narrow center support for L2/3 in emergent mode.

        The support term is derived from the predicted orientation distribution
        rather than the raw sensory input. It stays non-negative, spatially
        narrow, and weak by construction. When cue gating is enabled, the
        branch uses a carried cue trace (for example VIP state) so support can
        persist into the probe window after the raw cue disappears.

        Args:
            q_pred: [B, N] predicted orientation distribution.
            pi_eff: [B, 1] effective precision after warmup scaling.
            gate_signal: Optional [B, N] persistent cue trace used only as a
                multiplicative gate.

        Returns:
            center_excitation: [B, N] excitatory input to L2/3 drive (>= 0).
        """
        if getattr(self, "_analysis_force_zero", False):
            return torch.zeros_like(q_pred)

        if not self.center_support_enabled or self.center_support_gain <= 0.0:
            return torch.zeros_like(q_pred)

        if gate_signal is not None:
            assert gate_signal.shape == q_pred.shape, (
                f"gate_signal shape {gate_signal.shape} must match q_pred shape {q_pred.shape}"
            )

        N = q_pred.shape[-1]
        q_centered = q_pred - 1.0 / N
        support_circulant = self._get_center_support_circulant()
        support_field = (support_circulant @ q_centered.unsqueeze(-1)).squeeze(-1)
        center_exc = self.center_support_gain * pi_eff * F.relu(support_field)

        if self.center_support_cue_gated:
            if gate_signal is None:
                return torch.zeros_like(center_exc)
            support_gate = gate_signal.clamp_min(0.0).amax(dim=-1, keepdim=True)
            center_exc = center_exc * support_gate

        return center_exc

    def compute_recurrent_gain(
        self,
        q_pred: Tensor,
        pi_eff: Tensor,
        gate_signal: Tensor | None = None,
    ) -> Tensor:
        """Compute a bounded multiplicative gain on the L2/3 recurrent term.

        The gain is prediction-driven, spatially narrow, non-negative, and
        optionally gated by a persistent cue trace (for example VIP state).
        In ``positive`` mode the branch is a narrow non-negative center gain.
        In ``signed_center_surround`` mode it becomes a bounded center-positive,
        flank-negative modulation built from a narrow center field and a
        flank-only broad-minus-narrow surround field. The output is always
        bounded so that the recurrent multiplier ``1 + gain`` remains positive.
        """
        if getattr(self, "_analysis_force_zero", False):
            return torch.zeros_like(q_pred)

        if not self.recurrent_gain_enabled:
            return torch.zeros_like(q_pred)

        if self.recurrent_gain_mode not in {"positive", "signed_center_surround"}:
            raise ValueError(
                f"Unsupported emergent_recurrent_gain_mode={self.recurrent_gain_mode!r}. "
                "Expected 'positive' or 'signed_center_surround'."
            )

        if (
            self.recurrent_gain_mode == "positive"
            and self.recurrent_gain_beta <= 0.0
        ):
            return torch.zeros_like(q_pred)
        if (
            self.recurrent_gain_mode == "signed_center_surround"
            and self.recurrent_gain_beta <= 0.0
            and self.recurrent_gain_flank_beta <= 0.0
        ):
            return torch.zeros_like(q_pred)

        if gate_signal is not None:
            assert gate_signal.shape == q_pred.shape, (
                f"gate_signal shape {gate_signal.shape} must match q_pred shape {q_pred.shape}"
            )

        q_centered = q_pred - 1.0 / q_pred.shape[-1]
        precision_scale = (pi_eff / max(self.pi_max, 1e-6)).clamp(0.0, 1.0)
        center_circulant = self._get_recurrent_gain_circulant()
        center_raw = (center_circulant @ q_centered.unsqueeze(-1)).squeeze(-1)
        center_field = F.relu(center_raw)
        center_field = center_field / (center_field.amax(dim=-1, keepdim=True) + 1e-8)

        if self.recurrent_gain_mode == "positive":
            if self.recurrent_gain_cue_gated:
                if gate_signal is None:
                    return torch.zeros_like(center_field)
                gate_profile = gate_signal.clamp_min(0.0)
                gate_profile = gate_profile / (gate_profile.amax(dim=-1, keepdim=True) + 1e-8)
                center_field = center_field * gate_profile
            return self.recurrent_gain_beta * precision_scale * center_field

        surround_circulant = self._get_recurrent_gain_surround_circulant()
        surround_raw = (surround_circulant @ q_centered.unsqueeze(-1)).squeeze(-1)
        flank_field = F.relu(surround_raw - center_raw)
        flank_field = flank_field / (flank_field.amax(dim=-1, keepdim=True) + 1e-8)

        if self.recurrent_gain_cue_gated:
            if gate_signal is None:
                return torch.zeros_like(center_field)
            gate_strength = gate_signal.clamp_min(0.0).amax(dim=-1, keepdim=True)
            center_field = center_field * gate_strength
            flank_field = flank_field * gate_strength

        gain_field = (
            self.recurrent_gain_beta * center_field
            - self.recurrent_gain_flank_beta * flank_field
        ) * precision_scale
        min_gain = -0.95 + 1e-6
        max_gain = max(self.recurrent_gain_beta, 0.0)
        return gain_field.clamp(min=min_gain, max=max_gain)

    def compute_apical_gain(
        self,
        q_pred: Tensor,
        pi_eff: Tensor,
        gate_signal: Tensor | None = None,
    ) -> Tensor:
        """Compute a bounded multiplicative gain on the feedforward excitatory term.

        This is a prediction-coupled apical-style gain branch: narrow,
        non-negative, optionally persistent-cue gated, and bounded in [0, beta].
        """
        if getattr(self, "_analysis_force_zero", False):
            return torch.zeros_like(q_pred)

        if not self.apical_gain_enabled or self.apical_gain_beta <= 0.0:
            return torch.zeros_like(q_pred)

        if gate_signal is not None:
            assert gate_signal.shape == q_pred.shape, (
                f"gate_signal shape {gate_signal.shape} must match q_pred shape {q_pred.shape}"
            )

        q_centered = q_pred - 1.0 / q_pred.shape[-1]
        gain_circulant = self._get_apical_gain_circulant()
        gain_field = (gain_circulant @ q_centered.unsqueeze(-1)).squeeze(-1)
        gain_field = F.relu(gain_field)
        gain_field = gain_field / (gain_field.amax(dim=-1, keepdim=True) + 1e-8)

        if self.apical_gain_cue_gated:
            if gate_signal is None:
                return torch.zeros_like(gain_field)
            gate_profile = gate_signal.clamp_min(0.0)
            gate_profile = gate_profile / (gate_profile.amax(dim=-1, keepdim=True) + 1e-8)
            gain_field = gain_field * gate_profile

        precision_scale = (pi_eff / max(self.pi_max, 1e-6)).clamp(0.0, 1.0)
        return self.apical_gain_beta * precision_scale * gain_field

    def compute_flank_som_boost(
        self,
        q_pred: Tensor,
        pi_eff: Tensor,
        gate_signal: Tensor | None = None,
    ) -> Tensor:
        """Compute a cue-gated, prediction-driven SOM supplement on the flanks.

        This branch is intentionally local and biologically simple: it adds
        extra SOM drive only where a broad-minus-narrow predicted feature
        profile is positive. The center is spared because the DoG field is
        negative there and the branch is rectified before application.

        Args:
            q_pred: [B, N] predicted orientation distribution.
            pi_eff: [B, 1] effective precision after warmup scaling.
            gate_signal: Optional [B, N] persistent cue trace (for example VIP).

        Returns:
            flank_boost: [B, N] non-negative SOM supplement, near-zero at center.
        """
        if getattr(self, "_analysis_force_zero", False):
            return torch.zeros_like(q_pred)

        if not self.flank_som_enabled or self.flank_som_gain <= 0.0:
            return torch.zeros_like(q_pred)

        if gate_signal is not None:
            assert gate_signal.shape == q_pred.shape, (
                f"gate_signal shape {gate_signal.shape} must match q_pred shape {q_pred.shape}"
            )

        q_centered = q_pred - 1.0 / q_pred.shape[-1]
        flank_circulant = self._get_flank_som_circulant()
        flank_field = (flank_circulant @ q_centered.unsqueeze(-1)).squeeze(-1)
        flank_field = F.relu(flank_field)

        if self.flank_som_cue_gated:
            if gate_signal is None:
                return torch.zeros_like(flank_field)
            # Use the cue trace as an amplitude gate rather than a pointwise
            # mask. A center-peaked cue should permit an off-center flank
            # profile; multiplying by the cue profile itself would zero the
            # flanks exactly.
            gate_strength = gate_signal.clamp_min(0.0).amax(dim=-1, keepdim=True)
            flank_field = flank_field * gate_strength

        precision_scale = (pi_eff / max(self.pi_max, 1e-6)).clamp(0.0, 1.0)
        return self.flank_som_gain * precision_scale * flank_field

    def compute_flank_shunt(
        self,
        q_pred: Tensor,
        pi_eff: Tensor,
        gate_signal: Tensor | None = None,
        winner_proxy: Tensor | None = None,
    ) -> Tensor:
        """Compute a bounded divisive shunt field on total excitatory drive.

        The field is center-spared and flank-positive by construction. It is
        applied later in the circuit as a divisive modulation on ``ff + rec``
        (after any positive excitatory gain, before subtractive inhibition).

        In ``prediction_direct`` mode, the shunt source is the predicted feature
        profile itself. In ``center_recruited`` mode, the source is a narrow
        winner proxy recruited from local L2/3 activity and confined to the
        predicted center before being spread through the same flank DoG kernel.
        When ``winner_proxy`` is omitted in center-recruited mode, the function
        falls back to a narrow predicted-center proxy so standalone analysis
        helpers remain compatible.

        Args:
            q_pred: [B, N] predicted orientation distribution.
            pi_eff: [B, 1] effective precision after warmup scaling.
            gate_signal: Optional [B, N] persistent cue trace (for example VIP).
            winner_proxy: Optional [B, N] winner activity used only in
                ``center_recruited`` mode.

        Returns:
            flank_shunt: [B, N] bounded non-negative shunt field.
        """
        if getattr(self, "_analysis_force_zero", False):
            return torch.zeros_like(q_pred)

        if not self.flank_shunt_enabled or self.flank_shunt_gain <= 0.0:
            return torch.zeros_like(q_pred)

        if gate_signal is not None:
            assert gate_signal.shape == q_pred.shape, (
                f"gate_signal shape {gate_signal.shape} must match q_pred shape {q_pred.shape}"
            )
        if winner_proxy is not None:
            assert winner_proxy.shape == q_pred.shape, (
                f"winner_proxy shape {winner_proxy.shape} must match q_pred shape {q_pred.shape}"
            )

        if self.flank_shunt_source not in {"prediction_direct", "center_recruited"}:
            raise ValueError(
                f"Unsupported emergent_flank_shunt_source={self.flank_shunt_source!r}. "
                "Expected 'prediction_direct' or 'center_recruited'."
            )

        q_centered = q_pred - 1.0 / q_pred.shape[-1]
        shunt_circulant = self._get_flank_shunt_circulant()
        if self.flank_shunt_source == "prediction_direct":
            shunt_source = q_centered
        else:
            center_circulant = self._get_flank_shunt_center_circulant()
            center_field = (center_circulant @ q_centered.unsqueeze(-1)).squeeze(-1)
            center_field = F.relu(center_field)
            center_field = center_field / (center_field.amax(dim=-1, keepdim=True) + 1e-8)
            if winner_proxy is None:
                shunt_source = center_field
            else:
                shunt_source = F.relu(winner_proxy) * center_field
                shunt_source = shunt_source / (shunt_source.amax(dim=-1, keepdim=True) + 1e-8)

        shunt_field = (shunt_circulant @ shunt_source.unsqueeze(-1)).squeeze(-1)
        shunt_field = F.relu(shunt_field)
        shunt_field = shunt_field / (shunt_field.amax(dim=-1, keepdim=True) + 1e-8)

        if self.flank_shunt_cue_gated:
            if gate_signal is None:
                return torch.zeros_like(shunt_field)
            gate_strength = gate_signal.clamp_min(0.0).amax(dim=-1, keepdim=True)
            shunt_field = shunt_field * gate_strength

        precision_scale = (pi_eff / max(self.pi_max, 1e-6)).clamp(0.0, 1.0)
        flank_shunt = self.flank_shunt_gain * precision_scale * shunt_field
        assert torch.isfinite(flank_shunt).all(), "flank shunt must remain finite"
        return flank_shunt
