"""Feedback mechanisms for the laminar V1-V2 model.

Two feedback systems:
1. FeedbackMechanism (fixed): Models A-E with hardcoded kernel shapes.
2. EmergentFeedbackOperator: Learned circulant kernel via direct channel weights.

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
    """Learned feedback operator via direct channel-wise circulant kernels.

    Each pathway (SOM, VIP, apical) has one learnable weight per orientation
    channel offset, giving full 5° resolution (N=36 weights per pathway).
    The kernel profile is used as row 0 of a circulant matrix — shifting it
    to each channel center via circular convolution.

    Learnable parameters:
        alpha_inh [N]: direct channel weights for SOM (inhibitory) pathway
        alpha_vip [N]: direct channel weights for VIP (disinhibitory) pathway
        alpha_apical [N]: direct channel weights for apical gain pathway
    """

    def __init__(self, cfg: ModelConfig, delta_som: bool = False):
        super().__init__()
        N = cfg.n_orientations
        self.n_orient = N
        self.period = cfg.orientation_range
        self.delta_som = delta_som

        # Learnable weights — one per orientation channel offset [N].
        # Small non-zero init to avoid dead ReLU gradient (relu(0) has grad 0).
        # At 0.01, feedback output is ~0.01 * pi — effectively off but gradient
        # can flow. During burn-in (feedback_scale=0), this is fully zeroed out.
        self.alpha_inh = nn.Parameter(torch.full((N,), 0.01))

        # VIP pathway: separate learnable weights.
        # Init at 0.01 — NOT zero, because rectified_softplus has zero gradient
        # at 0, which would permanently kill the VIP pathway. The L1 penalty
        # will push alpha_vip back to zero if the task doesn't need disinhibition.
        self.alpha_vip = nn.Parameter(torch.full((N,), 0.01))

        # Apical gain pathway: multiplicative modulation of L2/3 excitatory
        # drive at the predicted channel. Constrained to [1-max, 1+max] via
        # tanh. Biologically: active apical dendrites in L2/3 pyramidal cells
        # receive top-down feedback in layer 1, modulating gain of feedforward
        # drive (multiplicative, not additive).
        self.alpha_apical = nn.Parameter(torch.full((N,), 0.01))
        self.max_apical_gain = cfg.max_apical_gain

        # Delta-SOM baseline: softplus(baseline + field) - softplus(baseline)
        # removes the constant bias from softplus, so zero field → zero drive.
        if self.delta_som:
            self.som_baseline = nn.Parameter(torch.tensor(0.0))
            # SOM tonic baseline: a small positive floor that ensures SOM always
            # provides SOME inhibition at every channel. This is critical for VIP
            # disinhibition to work — without it, the SOM kernel learns to spare
            # the predicted channel (negative center), leaving r_som = 0 there,
            # so VIP has nothing to disinhibit. The tonic floor creates a positive
            # SOM drive everywhere, and VIP can selectively reduce it at center.
            # Init at -3.0 → softplus(-3) ≈ 0.049 (very small positive floor).
            # Must be small enough not to kill L2/3, but large enough to give
            # VIP something to disinhibit. The optimizer adjusts it during training.
            self.som_tonic = nn.Parameter(torch.tensor(-3.0))
        # VIP always uses delta-style (bias-corrected)
        self.vip_baseline = nn.Parameter(torch.tensor(0.0))

        # Cache for circulant matrices (populated by cache_kernels)
        self._cached_inh_circulant: Tensor | None = None
        self._cached_vip_circulant: Tensor | None = None
        self._cached_apical_circulant: Tensor | None = None

    def get_profiles(self) -> Tensor:
        """Return the current inhibitory (SOM) kernel profile.

        Returns:
            K_inh: [N] inhibitory (SOM) kernel profile (direct channel weights).
        """
        return self.alpha_inh

    def get_vip_profile(self) -> Tensor:
        """Return the current VIP (disinhibitory) kernel profile.

        Returns:
            K_vip: [N] VIP kernel profile (direct channel weights).
        """
        return self.alpha_vip

    def get_apical_profile(self) -> Tensor:
        """Return the current apical gain kernel profile.

        Returns:
            K_apical: [N] apical gain kernel profile (direct channel weights).
        """
        return self.alpha_apical

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

    def compute_simple_feedback(self, q_pred: Tensor) -> Tensor:
        """Simple additive feedback: 36-weight kernel convolved with centered prediction.

        Bypasses SOM/VIP/apical pathways entirely. Uses alpha_apical as the
        single feedback kernel, applied via circulant convolution.

        Args:
            q_pred: [B, N] prediction distribution from V2.

        Returns:
            modulation: [B, N] additive modulation signal for L2/3 excitatory drive.
        """
        N = q_pred.shape[-1]
        q_centered = q_pred - 1.0 / N

        if self._cached_apical_circulant is not None:
            K_circulant = self._cached_apical_circulant
        else:
            K = self.alpha_apical  # [N]
            K_circulant = self._to_circulant(K)  # [N, N]

        modulation = (K_circulant @ q_centered.unsqueeze(-1)).squeeze(-1)  # [B, N]
        return modulation

    def cache_kernels(self) -> None:
        """Build and cache circulant matrices for reuse across timesteps."""
        K_inh = self.get_profiles()
        self._cached_inh_circulant = self._to_circulant(K_inh)
        K_vip = self.get_vip_profile()
        self._cached_vip_circulant = self._to_circulant(K_vip)
        K_apical = self.get_apical_profile()
        self._cached_apical_circulant = self._to_circulant(K_apical)

    def uncache_kernels(self) -> None:
        """Clear cached circulant matrices after the forward pass."""
        self._cached_inh_circulant = None
        self._cached_vip_circulant = None
        self._cached_apical_circulant = None

    def forward(self, q_pred: Tensor, pi_eff: Tensor, r_l4: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """Compute SOM drive, VIP drive, and apical gain from learned profiles.

        Args:
            q_pred: [B, N] -- predicted orientation distribution.
            pi_eff: [B, 1] -- effective precision (after warmup scaling).
            r_l4: [B, N] or None -- L4 firing rates for coincidence gating of
                apical gain. When provided, apical gain is gated by the
                element-wise coincidence of top-down (apical_field) and
                bottom-up (centered r_l4) signals. When None, falls back
                to pure top-down apical modulation (backward compat).

        Returns:
            (som_drive, vip_drive, apical_gain): each [B, N].
                som_drive: non-negative drive for SOM ring.
                vip_drive: drive for VIP ring (delta-style, can be negative).
                apical_gain: multiplicative gain for L2/3 excitatory drive,
                    centered at 1.0, range [1-max_apical_gain, 1+max_apical_gain].
        """
        N = q_pred.shape[-1]
        q_centered = q_pred - 1.0 / N  # zero-mean so feedback=0 when uninformative

        # --- SOM pathway ---
        if self._cached_inh_circulant is not None:
            inh_circulant = self._cached_inh_circulant
        else:
            K_inh = self.get_profiles()
            inh_circulant = self._to_circulant(K_inh)

        inh_field = (inh_circulant @ q_centered.unsqueeze(-1)).squeeze(-1)  # [B, N]

        if self.delta_som:
            # Delta modulation (can be positive or negative)
            delta = F.softplus(self.som_baseline + inh_field) - F.softplus(self.som_baseline)
            # Tonic positive floor: ensures SOM always has some drive at every
            # channel, so VIP disinhibition has something to act on.
            tonic = F.softplus(self.som_tonic)  # guaranteed positive
            som_drive = pi_eff * (tonic + delta)
        else:
            som_drive = pi_eff * F.softplus(inh_field)

        # --- VIP pathway ---
        if self._cached_vip_circulant is not None:
            vip_circulant = self._cached_vip_circulant
        else:
            K_vip = self.get_vip_profile()
            vip_circulant = self._to_circulant(K_vip)

        vip_field = (vip_circulant @ q_centered.unsqueeze(-1)).squeeze(-1)  # [B, N]
        # Always delta-style for VIP (bias-corrected)
        vip_drive = pi_eff * (F.softplus(self.vip_baseline + vip_field) - F.softplus(self.vip_baseline))

        # --- Apical gain pathway (coincidence-gated) ---
        if self._cached_apical_circulant is not None:
            apical_circulant = self._cached_apical_circulant
        else:
            K_apical = self.get_apical_profile()
            apical_circulant = self._to_circulant(K_apical)

        apical_field = (apical_circulant @ q_centered.unsqueeze(-1)).squeeze(-1)  # [B, N]

        if r_l4 is not None:
            # Coincidence gate: top-down × bottom-up
            # Center L4 so uniform/zero input → zero basal field
            basal_field = r_l4 - r_l4.mean(dim=-1, keepdim=True)  # [B, N]
            # Multiplicative coincidence: zero if either side is zero/negative
            coincidence = F.relu(apical_field) * F.relu(basal_field)  # [B, N]
            apical_gain = 1.0 + self.max_apical_gain * torch.tanh(pi_eff * coincidence)
        else:
            # Fallback: pure top-down (backward compat for direct calls without r_l4)
            apical_gain = 1.0 + self.max_apical_gain * torch.tanh(pi_eff * apical_field)

        return som_drive, vip_drive, apical_gain
