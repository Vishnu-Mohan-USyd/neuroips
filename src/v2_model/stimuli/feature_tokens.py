"""Identity tokens for the Richter-like assay (plan v4 step 11 / Task #25).

A bank of 12 32×32 textured identity tokens simultaneously satisfying:
  1. Matched mean luminance                   (|Δ| < 1e-4)
  2. Matched RMS contrast                     (±5 %)
  3. Matched radial SF power spectrum         (±5 %)
  4. Matched orientation-energy histogram     (±5 %)
     through the fixed LGN/Gabor bank.
  5. Matched total Sobel edge energy          (±5 %)
AND remaining linearly discriminable by a 12-way LinearSVC on L4 E rate
features (chance = 1/12 ≈ 0.083; required floor ≥ 0.25).

Construction. Each token is a sum of 16 oriented Gabor stamps placed on a
fixed 4×4 position grid (spacing = 8 px). The 16 positions are grouped
into 8 point-symmetric diagonal pairs ``(i, j) ↔ (3-i, 3-j)``; each of the
8 orientations (k·π/8) is assigned to exactly one pair per token, so every
token contains two stamps of every orientation and same-orientation stamps
sit at ≥24 px separation (negligible quadrature cross-terms). What differs
across tokens is the (orientation → pair) permutation, yielding distinct
retinotopic orientation maps (strong discriminability signal) while the
spatially-summed orientation histogram is invariant up to residual pair-
class bias from LGN zero-padding at grid edges.

A short Adam loop balances per-(token, orientation) stamp amplitudes
against the cross-token-mean LGN Gabor histogram, collapsing that residual
below 5 % before the final RMS + luminance normalisation. The best-seen
token bank is kept (the loss is non-convex). Matched-luminance reduces to
the arithmetic identity ``mean(token - mean(token) + 0.5) = 0.5``.

``verify_discriminability`` is deferred (needs sklearn + ``LGNL4FrontEnd``)
so unit tests that don't use it don't require sklearn.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.v2_model.config import ModelConfig


__all__ = ["TokenBank"]


# --- Geometry constants (module-level so tests can introspect) -----------
N_TOKENS: int = 12
N_ORI: int = 8
GRID_SIZE: int = 32
STAMP_SIZE: int = 9
STAMP_SIGMA_PX: float = 2.0
STAMP_WAVELENGTH_PX: float = 4.0
POSITION_GRID: int = 4                 # 4x4 = 16 stamp positions per token
POSITION_SPACING: int = GRID_SIZE // POSITION_GRID  # = 8 px
POSITION_OFFSET: int = POSITION_SPACING // 2        # = 4 px (first-cell centre)

TARGET_MEAN: float = 0.5
TARGET_RMS: float = 0.08

# Adam loop on per-(token, orientation) stamp amplitudes drives the Gabor-bank
# orientation-energy histogram to match across tokens within BALANCE_TOL.
BALANCE_MAX_ITER: int = 500
BALANCE_TOL: float = 0.03        # 5 % spec, 2 % margin
BALANCE_LR: float = 0.02


def _make_gabor_stamp(
    size: int, theta_rad: float, sigma_px: float, wavelength_px: float,
) -> Tensor:
    """DC-balanced oriented Gabor patch ``[size, size]``."""
    half = size // 2
    coords = torch.arange(-half, half + 1, dtype=torch.float32)
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    cos_t, sin_t = math.cos(theta_rad), math.sin(theta_rad)
    x_rot = x * cos_t + y * sin_t
    y_rot = -x * sin_t + y * cos_t
    envelope = torch.exp(-(x_rot * x_rot + y_rot * y_rot)
                         / (2.0 * sigma_px * sigma_px))
    carrier = torch.cos(2.0 * math.pi * x_rot / wavelength_px)
    stamp = envelope * carrier
    return stamp - stamp.mean()


def _paste_stamp(canvas: Tensor, stamp: Tensor, y_c: int, x_c: int) -> None:
    """Add ``stamp`` into ``canvas`` centred at ``(y_c, x_c)`` in-place,
    clipping any portion that falls outside the canvas."""
    S = stamp.shape[0]
    half = S // 2
    H, W = canvas.shape
    y0, y1 = y_c - half, y_c + half + 1
    x0, x1 = x_c - half, x_c + half + 1
    sy0, sy1 = max(0, -y0), S - max(0, y1 - H)
    sx0, sx1 = max(0, -x0), S - max(0, x1 - W)
    ty0, ty1 = max(0, y0), min(H, y1)
    tx0, tx1 = max(0, x0), min(W, x1)
    canvas[ty0:ty1, tx0:tx1] += stamp[sy0:sy1, sx0:sx1]


def _orientation_energy_histogram(
    tokens: Tensor, gabor_even: Tensor, gabor_odd: Tensor,
) -> Tensor:
    """Per-token space-summed Gabor-quadrature energy ``[N, N_ori]``."""
    pad = gabor_even.shape[-1] // 2
    e = F.conv2d(tokens, gabor_even, padding=pad)
    o = F.conv2d(tokens, gabor_odd, padding=pad)
    energy = torch.sqrt(e * e + o * o + 1e-12)
    return energy.sum(dim=(-1, -2))


def _radial_sf_power(token: Tensor, n_bins: int = 16) -> Tensor:
    """Radial (1D) mean of ``|FFT(token)|²`` over ``n_bins`` concentric annuli."""
    H, W = token.shape
    fft = torch.fft.fft2(token)
    power = torch.fft.fftshift(fft.real * fft.real + fft.imag * fft.imag)
    cy, cx = H // 2, W // 2
    y, x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32) - cy,
        torch.arange(W, dtype=torch.float32) - cx,
        indexing="ij",
    )
    r = torch.sqrt(x * x + y * y)
    bin_idx = torch.clip(
        (r / (float(min(H, W)) / 2.0) * n_bins).long(), 0, n_bins - 1,
    )
    out = torch.zeros(n_bins, dtype=torch.float32)
    count = torch.zeros(n_bins, dtype=torch.float32)
    out.scatter_add_(0, bin_idx.flatten(), power.flatten())
    count.scatter_add_(0, bin_idx.flatten(), torch.ones_like(power.flatten()))
    return out / torch.clip(count, min=1.0)


def _edge_energy(token: Tensor) -> Tensor:
    """Scalar Σ √(G_x² + G_y²) using 3×3 Sobel gradients."""
    sx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
    ).view(1, 1, 3, 3)
    sy = sx.transpose(-1, -2).contiguous()
    t = token.unsqueeze(0).unsqueeze(0)
    gx = F.conv2d(t, sx, padding=1)
    gy = F.conv2d(t, sy, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-12).sum()


def _build_lgn_gabor_bank(
    n_ori: int = N_ORI, kernel_size: int = 11,
    sigma_px: float = 2.0, wavelength_px: float = 4.0, gamma: float = 0.5,
) -> tuple[Tensor, Tensor]:
    """Even + odd Gabor banks matching ``LGNL4FrontEnd`` geometry."""
    from src.v2_model.lgn_l4 import _make_gabor_kernel
    thetas = [k * math.pi / n_ori for k in range(n_ori)]
    evens = torch.stack([
        _make_gabor_kernel(kernel_size, t, sigma_px, wavelength_px,
                           phase=0.0, gamma=gamma) for t in thetas
    ]).unsqueeze(1)
    odds = torch.stack([
        _make_gabor_kernel(kernel_size, t, sigma_px, wavelength_px,
                           phase=math.pi / 2.0, gamma=gamma) for t in thetas
    ]).unsqueeze(1)
    return evens, odds


class TokenBank(nn.Module):
    """Bank of 12 textured identity tokens for the Richter-like assay.

    ``tokens`` is a buffer ``[12, 1, 32, 32]`` float32 in [0, 1], built once
    from the supplied seed. No parameters, no plasticity.
    """

    def __init__(self, cfg: ModelConfig, seed: int = 0) -> None:
        super().__init__()
        self.cfg = cfg
        self.seed = int(seed)
        self.register_buffer("tokens", self._build_tokens(self.seed))

    @staticmethod
    def _build_tokens(seed: int) -> Tensor:
        """Three stages: enumerate positions + pairs; pre-compute per-
        (token, orientation) stamp templates (boundary-clipping resolved
        up front); Adam-optimise amplitudes against the cross-token-mean
        LGN orientation-energy histogram, keeping the best-seen bank."""
        stamps = torch.stack([
            _make_gabor_stamp(
                STAMP_SIZE, k * math.pi / N_ORI,
                STAMP_SIGMA_PX, STAMP_WAVELENGTH_PX,
            )
            for k in range(N_ORI)
        ])                                                   # [N_ORI, S, S]

        # 4×4 position grid indexed by (i, j) ∈ {0..3}²; linear index
        # p_idx = i * POSITION_GRID + j. Pair each cell with its
        # point-symmetric partner (i, j) ↔ (3-i, 3-j). 16 cells / 2 = 8 pairs.
        positions = [
            (POSITION_OFFSET + i * POSITION_SPACING,
             POSITION_OFFSET + j * POSITION_SPACING)
            for i in range(POSITION_GRID)
            for j in range(POSITION_GRID)
        ]                                                    # 16 (y, x)
        diagonal_pairs: list[tuple[int, int]] = []
        seen: set[int] = set()
        for i in range(POSITION_GRID):
            for j in range(POSITION_GRID):
                a = i * POSITION_GRID + j
                b = (POSITION_GRID - 1 - i) * POSITION_GRID + (POSITION_GRID - 1 - j)
                if a in seen or b in seen:
                    continue
                diagonal_pairs.append((a, b))
                seen.update({a, b})
        assert len(diagonal_pairs) == N_ORI, (
            f"expected {N_ORI} diagonal pairs, got {len(diagonal_pairs)}"
        )

        rng = torch.Generator().manual_seed(seed)
        assignments: list[list[int]] = []
        for _ in range(N_TOKENS):
            assignments.append(
                torch.randperm(N_ORI, generator=rng).tolist()
            )

        # --- Precompute per-(token, orientation) stamp templates ---------
        # templates[t, k] == sum of the TWO ori-k stamps pasted at token t's
        # assigned pair (with boundary clipping resolved here once).
        templates = torch.zeros(
            N_TOKENS, N_ORI, GRID_SIZE, GRID_SIZE, dtype=torch.float32,
        )
        half = STAMP_SIZE // 2
        for t in range(N_TOKENS):
            for ori_idx, pair_idx in enumerate(assignments[t]):
                a_pos, b_pos = diagonal_pairs[pair_idx]
                for pos_idx in (a_pos, b_pos):
                    y_c, x_c = positions[pos_idx]
                    y0, y1 = y_c - half, y_c + half + 1
                    x0, x1 = x_c - half, x_c + half + 1
                    sy0, sy1 = max(0, -y0), STAMP_SIZE - max(0, y1 - GRID_SIZE)
                    sx0, sx1 = max(0, -x0), STAMP_SIZE - max(0, x1 - GRID_SIZE)
                    ty0, ty1 = max(0, y0), min(GRID_SIZE, y1)
                    tx0, tx1 = max(0, x0), min(GRID_SIZE, x1)
                    templates[t, ori_idx, ty0:ty1, tx0:tx1] += (
                        stamps[ori_idx, sy0:sy1, sx0:sx1]
                    )

        # --- Differentiable token synthesis (vectorised) -----------------
        def _synthesise(amp: Tensor) -> Tensor:
            raw = (amp.unsqueeze(-1).unsqueeze(-1) * templates).sum(dim=1)
            raw = raw - raw.mean(dim=(-1, -2), keepdim=True)
            std = raw.std(dim=(-1, -2), keepdim=True).clamp(min=1e-8)
            raw = raw * (TARGET_RMS / std) + TARGET_MEAN
            # Defensive clip + mean-shift (no-op for typical Gabor sums but
            # guarantees the returned tokens are exactly in [0, 1]).
            raw = torch.clamp(raw, 0.0, 1.0)
            raw = raw - raw.mean(dim=(-1, -2), keepdim=True) + TARGET_MEAN
            raw = torch.clamp(raw, 0.0, 1.0)
            return raw.unsqueeze(1)                          # [N, 1, H, W]

        # --- Adam-optimise amplitudes ------------------------------------
        evens, odds = _build_lgn_gabor_bank()
        amplitudes = torch.ones(
            N_TOKENS, N_ORI, dtype=torch.float32, requires_grad=True,
        )
        optimizer = torch.optim.Adam([amplitudes], lr=BALANCE_LR)
        best_err: float = float("inf")
        best_tokens: Tensor | None = None
        for _iter in range(BALANCE_MAX_ITER):
            optimizer.zero_grad()
            tokens_4d = _synthesise(amplitudes)
            ori_hist = _orientation_energy_histogram(tokens_4d, evens, odds)
            target = ori_hist.mean(dim=0, keepdim=True).detach()
            rel = (ori_hist - target) / target.clamp(min=1e-8)
            loss = (rel * rel).mean()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                current_err = float(
                    ((ori_hist - target) / target.clamp(min=1e-8))
                    .abs().max().item()
                )
            if current_err < best_err:
                best_err = current_err
                best_tokens = tokens_4d.detach().clone()
            if current_err < BALANCE_TOL:
                break

        assert best_tokens is not None
        return best_tokens.contiguous()                      # [N, 1, H, W]

    def verify_discriminability(
        self, lgn_l4,
        n_noise_samples: int = 200, noise_std: float = 0.01,
        seed: int = 42, n_steps: int = 5,
    ) -> float:
        """12-way LinearSVC accuracy on L4 E rate features (chance ≈ 0.083).

        Generates ``n_noise_samples`` noise-augmented copies of each token,
        runs ``n_steps`` Euler steps of ``lgn_l4`` from zero state, fits a
        LinearSVC on a 70/30 stratified split of the resulting 128-d L4 E
        rates, and returns held-out accuracy.
        """
        from sklearn.svm import LinearSVC                    # type: ignore
        from sklearn.model_selection import train_test_split
        from src.v2_model.state import initial_state

        dev = self.tokens.device
        torch.manual_seed(seed)
        noise = noise_std * torch.randn(
            N_TOKENS, n_noise_samples, 1, GRID_SIZE, GRID_SIZE, device=dev,
        )
        frames = (self.tokens.unsqueeze(1) + noise).reshape(
            -1, 1, GRID_SIZE, GRID_SIZE,
        )
        state = initial_state(self.cfg, batch_size=frames.shape[0], device=dev)
        with torch.no_grad():
            for _ in range(n_steps):
                _, l4_rate, state = lgn_l4(frames, state)

        X = l4_rate.detach().cpu().numpy().astype(np.float32)
        y = np.repeat(np.arange(N_TOKENS), n_noise_samples)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, random_state=seed, stratify=y,
        )
        clf = LinearSVC(random_state=seed, max_iter=5000, dual="auto")
        clf.fit(X_tr, y_tr)
        return float(clf.score(X_te, y_te))

    def balance_statistics(self) -> dict[int, dict[str, object]]:
        """Per-token ``{mean_lum, rms_contrast, sf_power_radial{,_sum},
        total_edge_energy, orientation_energy_hist}`` diagnostic dicts."""
        evens, odds = _build_lgn_gabor_bank()
        ori_hist = _orientation_energy_histogram(self.tokens, evens, odds)
        out: dict[int, dict[str, object]] = {}
        for t in range(N_TOKENS):
            tok = self.tokens[t, 0]
            sf_rad = _radial_sf_power(tok)
            out[t] = {
                "mean_lum": float(tok.mean().item()),
                "rms_contrast": float(tok.std().item()),
                "sf_power_radial_sum": float(sf_rad.sum().item()),
                "sf_power_radial": sf_rad.tolist(),
                "total_edge_energy": float(_edge_energy(tok).item()),
                "orientation_energy_hist": ori_hist[t].tolist(),
            }
        return out
