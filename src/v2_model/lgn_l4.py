"""Fixed LGN + V1 L4 front end (never trained).

Per v4 spec §Architecture table:
  * LGN: 32×32 × N_ori DoG + Gabor bank, ON/OFF + orientation energy, τ=20 ms, never plastic.
  * V1 L4 E: 128 units, τ=10 ms, never plastic.
  * V1 L4 PV: 16 units, τ=5 ms (→ instantaneous at dt=5 ms), never plastic.

Scope (Task #9 / Phase 1 step 4):
  * Spatial LGN filters only — DoG ON/OFF + quadrature-pair Gabor energy.
  * No temporal integration of LGN (requires NetworkStateV2 extension — deferred).
  * No stimulus-specific adaptation (requires NetworkStateV2 extension — deferred).
  * L4 PV is INSTANTANEOUS (τ_pv / dt = 5 / 5 = 1 → tracks feedforward drive each step,
    matches the V1L4Ring+PVPool pattern in `src/model/populations.py` at τ_pv = dt).
  * All parameters are `register_buffer`s so the module has zero
    `nn.Parameter`s — the frozen-core contract is enforced by construction.

Retinotopic layout (chosen to satisfy the architecture counts 128 / 16):
  * L4 E: 4×4 retinotopic grid × N_ori (=8) orientations × 1 phase = 128.
    Linear index = (retino_i * retino_side + retino_j) * N_ori + ori.
  * L4 PV: 4×4 retinotopic grid × 1 unit per cell = 16.
    Each PV cell pools its 8 co-located L4 E orientations.

Module API:
  class LGNL4FrontEnd(nn.Module):
      __init__(cfg: ModelConfig)
      forward(frames: Tensor [B, 1, H, W],
              state: NetworkStateV2)
          -> (lgn_feature_map [B, 2+N_ori, H, W],
              l4_e_rate       [B, n_l4_e=128],
              updated_state   : NetworkStateV2 with r_l4 replaced).

Determinism: forward depends only on `frames` and `state`; no RNG, no dropout,
all convolutions + pooling are deterministic.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.v2_model.config import ModelConfig
from src.v2_model.state import NetworkStateV2
from src.v2_model.utils import rectified_softplus

__all__ = ["LGNL4FrontEnd"]


# ---------------------------------------------------------------------------
# Filter-bank builders (module-private, run once in __init__)
# ---------------------------------------------------------------------------

def _make_dog_kernel(
    kernel_size: int,
    sigma_center_px: float,
    sigma_surround_px: float,
) -> Tensor:
    """Build a DC-balanced Difference-of-Gaussians kernel.

    Returns:
        [K, K] float32 tensor. Mean is removed so uniform inputs produce zero
        response (DC-balanced: no response to mean luminance).
    """
    half = kernel_size // 2
    coords = torch.arange(-half, half + 1, dtype=torch.float32)
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    r2 = x * x + y * y
    g_c = torch.exp(-r2 / (2.0 * sigma_center_px * sigma_center_px))
    g_s = torch.exp(-r2 / (2.0 * sigma_surround_px * sigma_surround_px))
    g_c = g_c / g_c.sum()
    g_s = g_s / g_s.sum()
    dog = g_c - g_s
    return dog - dog.mean()  # enforce DC balance


def _make_gabor_kernel(
    kernel_size: int,
    theta_rad: float,
    sigma_px: float,
    wavelength_px: float,
    phase: float,
    gamma: float,
) -> Tensor:
    """Build a single 2D Gabor kernel (DC-balanced).

    Gabor(x, y) = exp(-(x'² + γ² y'²) / (2σ²)) · cos(2π x' / λ + φ)

    where (x', y') is (x, y) rotated by θ.

    Returns:
        [K, K] float32 tensor with mean subtracted (DC-free).
    """
    half = kernel_size // 2
    coords = torch.arange(-half, half + 1, dtype=torch.float32)
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    cos_t, sin_t = math.cos(theta_rad), math.sin(theta_rad)
    x_rot = x * cos_t + y * sin_t
    y_rot = -x * sin_t + y * cos_t
    envelope = torch.exp(-(x_rot * x_rot + (gamma * gamma) * (y_rot * y_rot))
                         / (2.0 * sigma_px * sigma_px))
    carrier = torch.cos(2.0 * math.pi * x_rot / wavelength_px + phase)
    gabor = envelope * carrier
    return gabor - gabor.mean()  # DC-balance (odd phase is already ~0)


def _build_filter_banks(
    kernel_size: int,
    n_ori: int,
    dog_sigma_center_px: float,
    dog_sigma_surround_px: float,
    gabor_sigma_px: float,
    gabor_wavelength_px: float,
    gabor_gamma: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build DoG (ON/OFF) + Gabor (even / odd) banks as conv2d weight tensors.

    Returns:
        dog_kernel: [2, 1, K, K]  — channel 0 is ON, channel 1 is OFF (= -ON).
        gabor_even: [N_ori, 1, K, K]
        gabor_odd:  [N_ori, 1, K, K]
    """
    dog = _make_dog_kernel(kernel_size, dog_sigma_center_px, dog_sigma_surround_px)
    dog_on = dog.unsqueeze(0).unsqueeze(0)       # [1, 1, K, K]
    dog_off = -dog.unsqueeze(0).unsqueeze(0)     # [1, 1, K, K]
    dog_kernel = torch.cat([dog_on, dog_off], dim=0)  # [2, 1, K, K]

    thetas = [float(k) * math.pi / float(n_ori) for k in range(n_ori)]
    even_list = [
        _make_gabor_kernel(kernel_size, t, gabor_sigma_px, gabor_wavelength_px,
                           phase=0.0, gamma=gabor_gamma)
        for t in thetas
    ]
    odd_list = [
        _make_gabor_kernel(kernel_size, t, gabor_sigma_px, gabor_wavelength_px,
                           phase=math.pi / 2.0, gamma=gabor_gamma)
        for t in thetas
    ]
    gabor_even = torch.stack(even_list, dim=0).unsqueeze(1)  # [N_ori, 1, K, K]
    gabor_odd = torch.stack(odd_list, dim=0).unsqueeze(1)    # [N_ori, 1, K, K]
    return dog_kernel, gabor_even, gabor_odd


# ---------------------------------------------------------------------------
# Fixed LGN + L4 front end
# ---------------------------------------------------------------------------

class LGNL4FrontEnd(nn.Module):
    """Fixed DoG + Gabor LGN bank, L4 E divisive-normalized tuning, L4 PV pool.

    Never trained — all filter kernels, divisive-norm constants, and time
    ratios are registered as buffers. `list(self.parameters())` is empty.

    Attributes (buffers):
        dog_kernel:   [2, 1, K, K]      ON/OFF DoG.
        gabor_even:   [N_ori, 1, K, K]  Gabor even (cos).
        gabor_odd:    [N_ori, 1, K, K]  Gabor odd  (sin).
        sigma_norm_sq: scalar           semi-saturation constant for divisive norm.

    Attributes (hyperparameters, held as plain Python ints/floats):
        n_ori, grid_h, grid_w, n_l4_e, n_l4_pv, retino_side, pool_h, pool_w,
        kernel_size, dt_over_tau_l4_e, energy_eps.
    """

    # Filter-bank geometry — chosen to fit the 32×32 retinotopic grid.
    KERNEL_SIZE: int = 11
    DOG_SIGMA_CENTER_PX: float = 1.0
    DOG_SIGMA_SURROUND_PX: float = 2.0
    GABOR_SIGMA_PX: float = 2.0
    GABOR_WAVELENGTH_PX: float = 4.0
    GABOR_GAMMA: float = 0.5

    # Divisive-norm + Gabor-energy constants — placeholder values.
    # Phase-2 tuning may adjust sigma_norm_sq; until then, 1.0 matches the
    # v1 default (see src/config.py:47). Energy floor prevents sqrt-of-zero.
    SIGMA_NORM_SQ: float = 1.0
    ENERGY_EPS: float = 1e-8

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()

        a = cfg.arch
        t = cfg.tau

        # Architectural counts (asserted consistent with 4×4 retinotopy).
        self.n_ori: int = int(a.n_orientations)
        self.grid_h: int = int(a.grid_h)
        self.grid_w: int = int(a.grid_w)
        self.n_l4_e: int = int(a.n_l4_e)
        self.n_l4_pv: int = int(a.n_l4_pv)

        retino_area = self.n_l4_pv
        retino_side = int(round(math.sqrt(retino_area)))
        if retino_side * retino_side != retino_area:
            raise ValueError(
                f"n_l4_pv={self.n_l4_pv} is not a perfect square; "
                f"current layout requires a square retinotopic grid."
            )
        if retino_side * retino_side * self.n_ori != self.n_l4_e:
            raise ValueError(
                f"n_l4_e={self.n_l4_e} must equal retino_side² × n_ori "
                f"({retino_side * retino_side} × {self.n_ori})."
            )
        if self.grid_h % retino_side != 0 or self.grid_w % retino_side != 0:
            raise ValueError(
                f"LGN grid {self.grid_h}x{self.grid_w} must be divisible by "
                f"retino_side={retino_side} for pooling to tile cleanly."
            )
        self.retino_side: int = retino_side
        self.pool_h: int = self.grid_h // retino_side
        self.pool_w: int = self.grid_w // retino_side

        # Timescale: dt/τ for L4 E (semi-implicit Euler).
        # dt and τ both in ms; we only ever need the ratio.
        self.dt_over_tau_l4_e: float = float(t.dt_ms) / float(t.tau_l4_e_ms)

        # Build + register filter banks as buffers (no gradient).
        dog, g_even, g_odd = _build_filter_banks(
            kernel_size=self.KERNEL_SIZE,
            n_ori=self.n_ori,
            dog_sigma_center_px=self.DOG_SIGMA_CENTER_PX,
            dog_sigma_surround_px=self.DOG_SIGMA_SURROUND_PX,
            gabor_sigma_px=self.GABOR_SIGMA_PX,
            gabor_wavelength_px=self.GABOR_WAVELENGTH_PX,
            gabor_gamma=self.GABOR_GAMMA,
        )
        self.register_buffer("dog_kernel", dog)
        self.register_buffer("gabor_even", g_even)
        self.register_buffer("gabor_odd", g_odd)
        self.register_buffer(
            "sigma_norm_sq",
            torch.tensor(self.SIGMA_NORM_SQ, dtype=torch.float32),
        )

        # Frozen-by-construction invariant: no nn.Parameters.
        assert len(list(self.parameters())) == 0, (
            "LGNL4FrontEnd must hold zero nn.Parameters — all state should be "
            "registered as buffers."
        )

    # ------------------------------------------------------------------
    # L4 unit metadata (derivable from filter-bank layout; no RNG)
    # ------------------------------------------------------------------
    #
    # L4 E unit linear index j ∈ [0, n_l4_e) decomposes as
    #   retino_flat_j = j // n_ori      ∈ [0, retino_side²)
    #   ori_bin_j     = j %  n_ori      ∈ [0, n_ori)
    # with
    #   retino_i_j    = retino_flat_j // retino_side
    #   retino_j_j    = retino_flat_j %  retino_side
    #   preferred_orient_deg_j = ori_bin_j × (180 / n_ori)
    # These properties are used by Fix-K to build the orientation-biased
    # feedforward mask on W_l4_l23 without a runtime probe.

    @property
    def pref_orient_deg_l4(self) -> Tensor:
        """Per-L4-unit preferred orientation (deg, circular on 180°)."""
        idx = torch.arange(self.n_l4_e, dtype=torch.float32)
        bin_per_unit = idx % float(self.n_ori)
        return bin_per_unit * (180.0 / float(self.n_ori))

    @property
    def retino_ij_l4(self) -> tuple[Tensor, Tensor]:
        """Per-L4-unit (row, col) indices into the retinotopic grid."""
        idx = torch.arange(self.n_l4_e, dtype=torch.long)
        retino_flat = idx // int(self.n_ori)
        retino_i = retino_flat // int(self.retino_side)
        retino_j = retino_flat %  int(self.retino_side)
        return retino_i, retino_j

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        frames: Tensor,
        state: NetworkStateV2,
    ) -> tuple[Tensor, Tensor, NetworkStateV2]:
        """One timestep of fixed LGN → L4 E (with L4 PV divisive norm).

        Args:
            frames: [B, 1, H, W] grayscale image at current timestep.
                Expected H=grid_h, W=grid_w. Values typically in [-1, 1] or [0, 1];
                DC-balanced filters make mean luminance irrelevant.
            state: NetworkStateV2 with previous r_l4 of shape [B, n_l4_e].

        Returns:
            lgn_feature_map: [B, 2 + N_ori, H, W] — channel order is
                [DoG_ON, DoG_OFF, Gabor_energy_ori_0, ..., Gabor_energy_ori_{N-1}].
                DoG channels are rectified (ReLU); Gabor channels are quadrature
                energy sqrt(even² + odd²), always non-negative.
            l4_e_rate: [B, n_l4_e] — updated L4 E rates.
            updated_state: NetworkStateV2 with `r_l4` replaced by `l4_e_rate`.

        Shapes / invariants:
            Assumes frames.shape == (B, 1, grid_h, grid_w). Raises on mismatch.
            Assumes state.r_l4.shape == (B, n_l4_e).
        """
        if frames.dim() != 4 or frames.shape[1] != 1:
            raise ValueError(
                f"frames must be [B, 1, H, W]; got shape {tuple(frames.shape)}."
            )
        B, _, H, W = frames.shape
        if H != self.grid_h or W != self.grid_w:
            raise ValueError(
                f"frames spatial shape {(H, W)} != expected "
                f"{(self.grid_h, self.grid_w)}."
            )
        if state.r_l4.shape != (B, self.n_l4_e):
            raise ValueError(
                f"state.r_l4 shape {tuple(state.r_l4.shape)} != ({B}, {self.n_l4_e})."
            )

        pad = self.KERNEL_SIZE // 2

        # --- LGN bank: DoG (ON/OFF) + Gabor energy --------------------
        # DoG ON/OFF, rectified to ReLU (ON responds to bright centers,
        # OFF to dark centers; mutually exclusive for a given pixel).
        dog_raw = F.conv2d(frames, self.dog_kernel, padding=pad)       # [B, 2, H, W]
        dog_rect = F.relu(dog_raw)                                      # [B, 2, H, W]

        # Gabor quadrature pair → orientation energy (phase-invariant).
        g_even = F.conv2d(frames, self.gabor_even, padding=pad)         # [B, N_ori, H, W]
        g_odd = F.conv2d(frames, self.gabor_odd, padding=pad)           # [B, N_ori, H, W]
        gabor_energy = torch.sqrt(
            g_even * g_even + g_odd * g_odd + self.ENERGY_EPS
        )                                                               # [B, N_ori, H, W]

        lgn_feat = torch.cat([dog_rect, gabor_energy], dim=1)
        # Contract: channel layout is [DoG_ON, DoG_OFF, E_ori_0, ..., E_ori_{N-1}].

        # --- L4 E feedforward input: retinotopic × ori pooling --------
        # Avg-pool the Gabor energy onto a retino_side × retino_side grid,
        # then flatten (retino_i, retino_j, ori) → 128 units.
        pooled = F.avg_pool2d(
            gabor_energy, kernel_size=(self.pool_h, self.pool_w)
        )  # [B, N_ori, retino_side, retino_side]
        # Reorder so the fast-changing axis is `ori` (matches PV
        # broadcast via repeat_interleave).
        pooled = pooled.permute(0, 2, 3, 1).contiguous()  # [B, R, R, N_ori]
        l4_input = pooled.reshape(B, self.n_l4_e)          # [B, 128]

        # --- L4 PV: instantaneous pool across ori per retinotopic cell -
        r_pv_l4 = l4_input.reshape(B, self.n_l4_pv, self.n_ori).sum(dim=-1)
        # [B, n_l4_pv=16]. Broadcast back to the 128-unit E layout.
        r_pv_broadcast = r_pv_l4.repeat_interleave(self.n_ori, dim=-1)  # [B, 128]

        # --- L4 E divisive drive + semi-implicit Euler update ---------
        l4_drive = l4_input / (self.sigma_norm_sq + r_pv_broadcast)
        r_l4_new = state.r_l4 + self.dt_over_tau_l4_e * (
            -state.r_l4 + rectified_softplus(l4_drive)
        )

        updated_state = state._replace(r_l4=r_l4_new)
        return lgn_feat, r_l4_new, updated_state
