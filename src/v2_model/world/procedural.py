"""Procedural synthetic world with hidden regime (Phase 1 step 12 / Task #27).

Synthetic training/eval environment for the laminar predictive circuit. Each
frame is a 32×32 scene rendered from six latent variables

    z        ∈ {0..11}            identity  (token index into ``TokenBank``)
    θ        ∈ [0°, 360°)          orientation  (rotation applied to token)
    p        ∈ [0, 31]²            spatial position  (rendered-token centre)
    c        ∈ [0.1, 1.0]          contrast  (scales around mean-luminance 0.5)
    o        ∈ {0, 1}              occluder on/off
    g        ∈ {CW, CCW, loH, hiH} **hidden** regime  (not observable from a
                                   single frame; must be inferred from history)

Regime persistence is P(g_{t+1}=g_t)=0.98; on a switch, the new regime is
drawn uniformly over the other allowed regimes. Conditional on the freshly
sampled regime g_{t+1}, one of two dynamics fires:

* **Smooth drift** (prob ``1 − P_jump[g]``): z unchanged, θ drifts with
  regime-specific mean (+5° CW, −5° CCW, N(0, 5°) for hazard regimes), p
  drifts by integer N(0, 1) px (bounded to the grid), c drifts by N(0, 0.05)
  (bounded to [0.1, 1.0]), o unchanged.
* **Cause switch** (prob ``P_jump[g]``): z, θ, p, c, o re-sampled
  independently from their priors.

``P_jump`` is 0.05 for CW/CCW/low-hazard and 0.30 for high-hazard — so
``high-hazard`` is the only regime that routinely breaks long continuity,
and the single-frame pixel content carries no information about g (verified
by ``test_procedural_world_regime_inference``). The circuit must infer g
from a rolling history of Δθ statistics + jump rate → this is what makes
context memory C load-bearing for Phase 2 Gate 7.

Train/eval seed split: train family uses seeds 42+i, eval family uses
9000+i. ``held_out_regime`` (train only) removes one regime from the
transition support so eval-time presentation of the held-out regime is
genuinely out-of-distribution.

Rendering is via ``torch.nn.functional.affine_grid`` + ``grid_sample`` on
the deviation ``token − 0.5`` (so the grey-background outside the rotated
token falls out for free via zero-padding).
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from src.v2_model.config import ModelConfig
from src.v2_model.stimuli.feature_tokens import N_TOKENS, TokenBank


__all__ = [
    "JUMP_PROBS",
    "REGIMES",
    "REGIME_PERSIST_PROB",
    "ProceduralWorld",
    "WorldState",
]


# --- Regime constants -----------------------------------------------------
REGIMES: tuple[str, str, str, str] = (
    "CW-drift", "CCW-drift", "low-hazard", "high-hazard",
)
REGIME_PERSIST_PROB: float = 0.98
JUMP_PROBS: dict[str, float] = {
    "CW-drift":    0.05,
    "CCW-drift":   0.05,
    "low-hazard":  0.05,
    "high-hazard": 0.30,
}

# --- Seed families --------------------------------------------------------
SEED_BASE_TRAIN: int = 42
SEED_BASE_EVAL: int = 9000

# --- Latent priors and drift magnitudes ----------------------------------
GRID_SIZE: int = 32
TARGET_MEAN: float = 0.5
CONTRAST_MIN: float = 0.1
CONTRAST_MAX: float = 1.0
OCCLUDER_P: float = 0.2
OCCLUDER_SIZE: int = 10
OCCLUDER_Y0: int = GRID_SIZE - OCCLUDER_SIZE    # 22
OCCLUDER_X0: int = GRID_SIZE - OCCLUDER_SIZE    # 22

DRIFT_STEP_DEG: float = 5.0                     # cfg.regime.drift_step_deg
POSITION_DRIFT_SIGMA_PX: float = 1.0
CONTRAST_DRIFT_SIGMA: float = 0.05


# --- Ground-truth trajectory state ---------------------------------------

class WorldState(NamedTuple):
    """Ground-truth latents of a single procedural-world step.

    Fields
    ------
    z           Identity / token index (0..11).
    theta       Orientation in degrees in [0, 360).
    position    (y, x) integer pixel center of the rendered token (0..31).
    contrast    Contrast in [0.1, 1.0].
    occluder    0 or 1 (10×10 grey patch at fixed corner when 1).
    regime      One of ``REGIMES``. **Hidden** — not observable from the
                rendered frame.
    step_idx    0-indexed step number; incremented by :meth:`step`.
    """
    z: int
    theta: float
    position: tuple[int, int]
    contrast: float
    occluder: int
    regime: str
    step_idx: int


# --- RNG helpers (every draw goes through an explicit generator) ---------

def _uniform_int(low: int, high: int, rng: torch.Generator) -> int:
    """Integer uniform on ``[low, high)``."""
    return int(torch.randint(low, high, (1,), generator=rng).item())


def _uniform_float(low: float, high: float, rng: torch.Generator) -> float:
    """Float uniform on ``[low, high)``."""
    u = torch.rand(1, generator=rng).item()
    return float(low + (high - low) * u)


def _gaussian(sigma: float, rng: torch.Generator) -> float:
    """Scalar ``N(0, sigma)``."""
    return float(sigma * torch.randn(1, generator=rng).item())


def _bernoulli(p: float, rng: torch.Generator) -> int:
    """0/1 draw with ``P(1) = p``."""
    return int(torch.rand(1, generator=rng).item() < p)


# --- Rendering ------------------------------------------------------------

def _render_frame(token: Tensor, state: WorldState) -> Tensor:
    """Render a single frame from ``token`` (the bank entry for z) + latents.

    Uses ``affine_grid`` + ``grid_sample`` on the deviation ``token − 0.5``
    so out-of-canvas rotation/translation naturally resolves to background
    grey. Output shape ``[1, 32, 32]``, float32, values in [0, 1].
    """
    dev = (token - TARGET_MEAN).unsqueeze(0)                     # [1, 1, H, W]
    theta_rad = math.radians(state.theta)
    cos_t, sin_t = math.cos(theta_rad), math.sin(theta_rad)
    p_y, p_x = state.position
    # affine_grid (align_corners=False) maps normalised (-1, 1) to pixel
    # (-0.5, H-0.5): pixel k sits at (2k+1)/H − 1.
    dx_norm = (2.0 * p_x + 1.0) / GRID_SIZE - 1.0
    dy_norm = (2.0 * p_y + 1.0) / GRID_SIZE - 1.0
    affine = torch.tensor(
        [[cos_t,  sin_t, -cos_t * dx_norm - sin_t * dy_norm],
         [-sin_t, cos_t,  sin_t * dx_norm - cos_t * dy_norm]],
        dtype=torch.float32,
    ).unsqueeze(0)
    grid = F.affine_grid(
        affine, [1, 1, GRID_SIZE, GRID_SIZE], align_corners=False,
    )
    rot_dev = F.grid_sample(
        dev, grid, mode="bilinear", padding_mode="zeros", align_corners=False,
    ).squeeze(0)                                                 # [1, H, W]
    frame = TARGET_MEAN + float(state.contrast) * rot_dev
    if int(state.occluder) == 1:
        frame[:, OCCLUDER_Y0:OCCLUDER_Y0 + OCCLUDER_SIZE,
              OCCLUDER_X0:OCCLUDER_X0 + OCCLUDER_SIZE] = TARGET_MEAN
    return frame.clamp(0.0, 1.0)


# --- Dynamics helpers -----------------------------------------------------

def _wrap_deg(theta: float) -> float:
    """Wrap to [0, 360)."""
    return float(theta % 360.0)


def _drift_theta(regime: str, rng: torch.Generator) -> float:
    """Regime-conditioned Δθ in degrees (per spec §Synthetic training world)."""
    if regime == "CW-drift":
        return DRIFT_STEP_DEG
    if regime == "CCW-drift":
        return -DRIFT_STEP_DEG
    return _gaussian(DRIFT_STEP_DEG, rng)               # hazard regimes


def _sample_prior_latents(
    rng: torch.Generator,
) -> tuple[int, float, tuple[int, int], float, int]:
    """Draw (z, θ, p, c, o) independently from priors (used on cause-switch)."""
    z = _uniform_int(0, N_TOKENS, rng)
    theta = _uniform_float(0.0, 360.0, rng)
    p_y = _uniform_int(0, GRID_SIZE, rng)
    p_x = _uniform_int(0, GRID_SIZE, rng)
    c = _uniform_float(CONTRAST_MIN, CONTRAST_MAX, rng)
    o = _bernoulli(OCCLUDER_P, rng)
    return z, theta, (p_y, p_x), c, o


# --- Procedural world ----------------------------------------------------

class ProceduralWorld:
    """Procedural world with hidden regime (see module docstring).

    Parameters
    ----------
    cfg
        :class:`ModelConfig` — used for ``cfg.regime.drift_step_deg``.
    token_bank
        :class:`TokenBank` providing the 12 identity tokens that the ``z``
        latent indexes.
    seed_family
        ``"train"`` (seeds 42+i) or ``"eval"`` (seeds 9000+i). Train-family
        trajectories and eval-family trajectories at the same trajectory
        seed are statistically distinguishable.
    held_out_regime
        Optional regime name to exclude from the transition support in
        training mode. Eval mode accesses all four regimes; passing a
        ``held_out_regime`` with ``seed_family="eval"`` raises.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        token_bank: TokenBank,
        seed_family: str,
        held_out_regime: Optional[str] = None,
    ) -> None:
        if seed_family not in ("train", "eval"):
            raise ValueError(
                f"seed_family must be 'train' or 'eval'; got {seed_family!r}"
            )
        if held_out_regime is not None:
            if held_out_regime not in REGIMES:
                raise ValueError(
                    f"held_out_regime={held_out_regime!r} not in {REGIMES}"
                )
            if seed_family != "train":
                raise ValueError(
                    "held_out_regime is only legal with seed_family='train'"
                )
        self.cfg = cfg
        self.token_bank = token_bank
        self.seed_family = seed_family
        self._seed_base = (
            SEED_BASE_TRAIN if seed_family == "train" else SEED_BASE_EVAL
        )
        self.held_out_regime = held_out_regime
        if held_out_regime is not None:
            self._regimes: tuple[str, ...] = tuple(
                r for r in REGIMES if r != held_out_regime
            )
        else:
            self._regimes = REGIMES
        self._rng: Optional[torch.Generator] = None

    # --- Lifecycle -------------------------------------------------------

    def reset(self, trajectory_seed: int) -> WorldState:
        """Draw initial latents + regime uniformly over the allowed set.

        Creates (or reseeds) this instance's :class:`torch.Generator` from
        ``seed_base + trajectory_seed``. All subsequent :meth:`step` calls
        draw from this same generator — so ``reset(k)`` followed by N calls
        to :meth:`step` gives a reproducible trajectory.
        """
        self._rng = torch.Generator().manual_seed(
            self._seed_base + int(trajectory_seed),
        )
        z, theta, pos, c, o = _sample_prior_latents(self._rng)
        regime = self._regimes[_uniform_int(0, len(self._regimes), self._rng)]
        return WorldState(
            z=z, theta=theta, position=pos, contrast=c, occluder=o,
            regime=regime, step_idx=0,
        )

    def step(self, state: WorldState) -> tuple[Tensor, WorldState, dict]:
        """Advance one step; returns ``(frame, next_state, info)``.

        Regime transition fires first (0.98 persistence, uniform over the
        allowed-set \\ {current} on a switch). Then, conditional on the new
        regime, one of {smooth drift, cause switch} determines how the
        other latents update. ``info`` carries the ground-truth latents +
        ``is_jump`` flag; the network never sees ``regime``.
        """
        rng = self._require_rng()

        # Regime transition (0.98 persistence)
        if torch.rand(1, generator=rng).item() < REGIME_PERSIST_PROB:
            new_regime = state.regime
        else:
            candidates = [r for r in self._regimes if r != state.regime]
            # candidates is non-empty: len(self._regimes) ≥ 3 by construction
            new_regime = candidates[_uniform_int(0, len(candidates), rng)]

        # Jump / drift decision (conditional on new_regime)
        is_jump = (
            torch.rand(1, generator=rng).item() < JUMP_PROBS[new_regime]
        )

        if is_jump:
            z_n, theta_n, pos_n, c_n, o_n = _sample_prior_latents(rng)
        else:
            z_n = state.z
            theta_n = _wrap_deg(state.theta + _drift_theta(new_regime, rng))
            eps_py = round(_gaussian(POSITION_DRIFT_SIGMA_PX, rng))
            eps_px = round(_gaussian(POSITION_DRIFT_SIGMA_PX, rng))
            p_y = max(0, min(GRID_SIZE - 1, state.position[0] + eps_py))
            p_x = max(0, min(GRID_SIZE - 1, state.position[1] + eps_px))
            pos_n = (p_y, p_x)
            eps_c = _gaussian(CONTRAST_DRIFT_SIGMA, rng)
            c_n = float(max(
                CONTRAST_MIN,
                min(CONTRAST_MAX, state.contrast + eps_c),
            ))
            o_n = state.occluder

        next_state = WorldState(
            z=z_n, theta=theta_n, position=pos_n, contrast=c_n,
            occluder=o_n, regime=new_regime, step_idx=state.step_idx + 1,
        )
        frame = self.render(next_state)
        info = {
            "z": z_n, "theta": theta_n, "position": pos_n,
            "contrast": c_n, "occluder": o_n, "regime": new_regime,
            "is_jump": is_jump, "step_idx": next_state.step_idx,
        }
        return frame, next_state, info

    def trajectory(
        self, trajectory_seed: int, n_steps: int,
    ) -> tuple[Tensor, list[WorldState]]:
        """Generate a full trajectory from a fresh :meth:`reset`.

        Returns ``(frames, states)`` where ``frames`` has shape
        ``[n_steps, 1, 32, 32]`` and ``states[i]`` is the :class:`WorldState`
        used to render ``frames[i]``. ``states[0]`` is the reset state.
        """
        if n_steps < 1:
            raise ValueError(f"n_steps must be ≥ 1; got {n_steps}")
        state = self.reset(trajectory_seed)
        frames: list[Tensor] = [self.render(state)]
        states: list[WorldState] = [state]
        for _ in range(n_steps - 1):
            frame, state, _info = self.step(state)
            frames.append(frame)
            states.append(state)
        return torch.stack(frames), states

    # --- Rendering -------------------------------------------------------

    def render(self, state: WorldState) -> Tensor:
        """Render a single frame ``[1, 32, 32]`` from a :class:`WorldState`."""
        token = self.token_bank.tokens[int(state.z)]             # [1, H, W]
        return _render_frame(token, state)

    # --- Internal --------------------------------------------------------

    def _require_rng(self) -> torch.Generator:
        """``self._rng`` after :meth:`reset` has been called."""
        if self._rng is None:
            raise RuntimeError(
                "ProceduralWorld.step() called before reset(); "
                "call reset(trajectory_seed) first"
            )
        return self._rng
