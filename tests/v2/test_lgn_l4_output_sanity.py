"""Biologically-motivated output-sanity checks for `LGNL4FrontEnd`.

Per v4 spec: the LGN front end must behave like a fixed ON/OFF + orientation-
energy detector. These tests assert the invariants that define that behavior:

  (a) DC balance — a uniform gray frame produces ~zero LGN response, because
      DoG and Gabor kernels are mean-subtracted.
  (b) Non-negativity — Gabor energy is a quadrature sum (always ≥0);
      DoG ON/OFF are ReLU'd; L4 E rate is `rectified_softplus` (≥0).
  (c) Phase invariance — shifting a grating in space produces the same
      Gabor-energy channel activation (quadrature-pair is phase-invariant).
  (d) Orientation selectivity — the N_ori Gabor channel with the strongest
      mean response aligns with the grating's orientation (within one channel
      bin).
  (e) Divisive normalization — doubling the input contrast raises L4 E by
      less than 2× (divisive PV normalization).
  (f) First-step update — with zero initial r_l4, after one step
      `r_l4 = (dt/τ) · rectified_softplus(l4_drive)` (analytic check).
"""

from __future__ import annotations

import math

import pytest
import torch

from src.v2_model.config import ModelConfig
from src.v2_model.lgn_l4 import LGNL4FrontEnd
from src.v2_model.state import initial_state
from src.v2_model.utils import rectified_softplus


# ---------------------------------------------------------------------------
# Stimulus helpers (test-private)
# ---------------------------------------------------------------------------

def _make_grating(
    h: int, w: int, theta_rad: float,
    wavelength_px: float = 4.0, phase: float = 0.0, contrast: float = 0.5,
) -> torch.Tensor:
    """Build a [1, 1, h, w] oriented grating centered on (h/2, w/2)."""
    coords_y = torch.arange(h, dtype=torch.float32) - (h - 1) / 2.0
    coords_x = torch.arange(w, dtype=torch.float32) - (w - 1) / 2.0
    y, x = torch.meshgrid(coords_y, coords_x, indexing="ij")
    x_rot = x * math.cos(theta_rad) + y * math.sin(theta_rad)
    grating = contrast * torch.cos(2.0 * math.pi * x_rot / wavelength_px + phase)
    return grating.unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# (a) DC balance
# ---------------------------------------------------------------------------

def test_uniform_frame_gives_small_lgn_response(cfg, batch_size) -> None:
    """DC-balanced filters → uniform gray produces near-zero LGN output
    at interior locations.

    We restrict the check to the interior (outside the kernel-half-size
    boundary), because conv2d uses zero-padding: at image boundaries the
    kernel reaches outside the image and sees 0s that are not equal to the
    uniform-gray mean, so the mean-subtracted kernel does not fully cancel.
    In the interior, the kernel stays fully inside the uniform-gray image
    and the mean-subtracted filter gives exactly zero response (up to float
    rounding).
    """
    front = LGNL4FrontEnd(cfg)
    state = initial_state(cfg, batch_size=batch_size)
    frames = 0.5 * torch.ones(batch_size, 1, cfg.arch.grid_h, cfg.arch.grid_w)

    lgn_feat, _, _ = front(frames, state)
    pad = front.KERNEL_SIZE // 2
    interior = lgn_feat[..., pad:-pad, pad:-pad]

    # Gabor energy has a sqrt(eps) floor, so the min is not exactly 0.
    assert interior.abs().max().item() < 1e-3, (
        f"uniform frame produced interior max |LGN| = "
        f"{interior.abs().max().item():.3e}"
    )


# ---------------------------------------------------------------------------
# (b) Non-negativity
# ---------------------------------------------------------------------------

def test_dog_channels_non_negative(cfg, batch_size) -> None:
    """DoG channels are ReLU'd: always ≥ 0."""
    front = LGNL4FrontEnd(cfg)
    state = initial_state(cfg, batch_size=batch_size)
    frames = torch.randn(batch_size, 1, cfg.arch.grid_h, cfg.arch.grid_w)

    lgn_feat, _, _ = front(frames, state)
    dog_channels = lgn_feat[:, :2]
    assert (dog_channels >= 0).all()


def test_gabor_energy_non_negative(cfg, batch_size) -> None:
    """Gabor energy channels are sqrt(e²+o²): always ≥ 0."""
    front = LGNL4FrontEnd(cfg)
    state = initial_state(cfg, batch_size=batch_size)
    frames = torch.randn(batch_size, 1, cfg.arch.grid_h, cfg.arch.grid_w)

    lgn_feat, _, _ = front(frames, state)
    gabor_channels = lgn_feat[:, 2:]
    assert (gabor_channels >= 0).all()


def test_l4_rate_non_negative(cfg, batch_size) -> None:
    """L4 E rate is rectified_softplus(drive): always ≥ 0."""
    front = LGNL4FrontEnd(cfg)
    state = initial_state(cfg, batch_size=batch_size)
    frames = torch.randn(batch_size, 1, cfg.arch.grid_h, cfg.arch.grid_w)

    _, r_l4, _ = front(frames, state)
    assert (r_l4 >= 0).all()


# ---------------------------------------------------------------------------
# (c) Phase invariance
# ---------------------------------------------------------------------------

def test_gabor_energy_is_phase_invariant(cfg) -> None:
    """Two gratings of the same orientation but different phase produce the
    same per-channel mean Gabor energy (up to float tolerance).

    This is the defining property of a quadrature-pair energy model.
    """
    front = LGNL4FrontEnd(cfg)
    state = initial_state(cfg, batch_size=1)

    theta = 0.0
    grating_even = _make_grating(cfg.arch.grid_h, cfg.arch.grid_w, theta, phase=0.0)
    grating_odd = _make_grating(cfg.arch.grid_h, cfg.arch.grid_w, theta, phase=math.pi / 2)

    lgn_even, _, _ = front(grating_even, state)
    lgn_odd, _, _ = front(grating_odd, state)

    # Per-channel mean Gabor energy (skip DoG channels).
    mean_even = lgn_even[0, 2:].mean(dim=(-1, -2))
    mean_odd = lgn_odd[0, 2:].mean(dim=(-1, -2))

    # Tolerance: boundary effects make this not exactly identical, but the
    # per-channel means agree to ~5% of the peak energy.
    peak = mean_even.max().item()
    assert torch.allclose(mean_even, mean_odd, atol=0.05 * peak, rtol=0.0), (
        f"phase-invariance violated: even={mean_even.tolist()}, odd={mean_odd.tolist()}"
    )


# ---------------------------------------------------------------------------
# (d) Orientation selectivity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target_k", [0, 2, 4, 6])
def test_orientation_selectivity_peaks_at_matching_channel(cfg, target_k) -> None:
    """A grating at θ_k drives the k-th Gabor channel the strongest.

    For N_ori = 8, channels are spaced π/8 = 22.5°. Gratings are drawn
    at the centers of each channel. The argmax over channel-mean energy
    must equal the channel index.
    """
    front = LGNL4FrontEnd(cfg)
    state = initial_state(cfg, batch_size=1)

    theta_target = float(target_k) * math.pi / cfg.arch.n_orientations
    frame = _make_grating(cfg.arch.grid_h, cfg.arch.grid_w,
                          theta_target, wavelength_px=front.GABOR_WAVELENGTH_PX)

    lgn_feat, _, _ = front(frame, state)
    gabor_mean_per_ch = lgn_feat[0, 2:].mean(dim=(-1, -2))  # [N_ori]
    peak_ch = int(gabor_mean_per_ch.argmax().item())
    assert peak_ch == target_k, (
        f"target ori={target_k} (θ={math.degrees(theta_target):.1f}°), "
        f"peak channel={peak_ch}, means={gabor_mean_per_ch.tolist()}"
    )


# ---------------------------------------------------------------------------
# (e) Divisive normalization (sublinear contrast scaling)
# ---------------------------------------------------------------------------

def test_l4_rate_grows_sublinearly_with_contrast(cfg) -> None:
    """Doubling contrast produces less than 2× the L4 E sum.

    With divisive normalization `l4_drive = input / (σ² + pv)` and `pv` pooling
    across the same input, doubling the input doubles both numerator and the
    input-dependent part of the denominator → strictly sub-linear scaling.
    """
    front = LGNL4FrontEnd(cfg)
    state0 = initial_state(cfg, batch_size=1)

    g_low = _make_grating(cfg.arch.grid_h, cfg.arch.grid_w, 0.0, contrast=0.25)
    g_high = g_low * 2.0  # doubled contrast, same pattern

    _, r_low, _ = front(g_low, state0)
    _, r_high, _ = front(g_high, state0)

    # Sums are strictly positive for this oriented grating and doubled input.
    sum_low = r_low.sum().item()
    sum_high = r_high.sum().item()
    assert sum_low > 0
    assert sum_high > sum_low, "doubling contrast failed to raise activity."
    assert sum_high < 2.0 * sum_low, (
        f"L4 rate grew super-linearly: low={sum_low:.4f}, high={sum_high:.4f}. "
        f"Divisive normalization should make this sub-linear."
    )


# ---------------------------------------------------------------------------
# (f) Analytic first-step Euler update
# ---------------------------------------------------------------------------

def test_first_step_matches_analytic_euler(cfg) -> None:
    """With r_l4=0 initial state, one step gives r_l4 = (dt/τ) · rectified_softplus(drive).

    This nails down the semi-implicit Euler formula in `forward`.
    """
    front = LGNL4FrontEnd(cfg)
    state0 = initial_state(cfg, batch_size=1)
    frame = _make_grating(cfg.arch.grid_h, cfg.arch.grid_w, 0.0, contrast=0.5)

    _, r_l4, _ = front(frame, state0)

    # Re-derive the drive manually using the same pipeline.
    import torch.nn.functional as F
    pad = front.KERNEL_SIZE // 2
    g_even = F.conv2d(frame, front.gabor_even, padding=pad)
    g_odd = F.conv2d(frame, front.gabor_odd, padding=pad)
    gabor_energy = torch.sqrt(g_even * g_even + g_odd * g_odd + front.ENERGY_EPS)
    pooled = F.avg_pool2d(gabor_energy,
                          kernel_size=(front.pool_h, front.pool_w))
    pooled = pooled.permute(0, 2, 3, 1).contiguous()
    l4_input = pooled.reshape(1, cfg.arch.n_l4_e)
    r_pv_l4 = l4_input.reshape(1, cfg.arch.n_l4_pv, cfg.arch.n_orientations).sum(dim=-1)
    r_pv_broadcast = r_pv_l4.repeat_interleave(cfg.arch.n_orientations, dim=-1)
    drive = l4_input / (front.sigma_norm_sq + r_pv_broadcast)
    expected = front.dt_over_tau_l4_e * rectified_softplus(drive)  # r_l4_prev = 0

    torch.testing.assert_close(r_l4, expected, atol=1e-6, rtol=1e-6)
