"""Shared helpers for Phase-2 gate evaluation harnesses.

Small utilities used by ``eval_gates.py``,
``run_null_expectation_control.py`` and ``run_c_load_bearing_check.py``:

* Checkpoint load that reconstructs a :class:`V2Network` from a state_dict
  saved by ``train_phase2_predictive._save_checkpoint``.
* Frame synthesis helpers — blank grey, oriented grating (sinusoidal),
  centre-masked grating for surround-suppression assays.
* A ``simulate_steady_state`` helper that runs the network for a fixed
  number of steps against a constant frame and returns the final info
  bundle (end-of-run rates).

All helpers are forward-only (no autograd, no plasticity) and batch-aware.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor

from src.v2_model.config import ModelConfig
from src.v2_model.network import V2Network
from src.v2_model.state import NetworkStateV2
from src.v2_model.stimuli.feature_tokens import TokenBank


__all__ = [
    "CheckpointBundle",
    "load_checkpoint",
    "make_blank_frame",
    "make_grating_frame",
    "make_surround_grating_frame",
    "simulate_steady_state",
    "simulate_and_collect",
]


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------


@dataclass
class CheckpointBundle:
    """Holds a loaded network + its config + token bank + metadata."""
    cfg: ModelConfig
    net: V2Network
    bank: TokenBank
    meta: dict


def load_checkpoint(
    path: Path, *, seed: int = 42, device: str = "cpu",
) -> CheckpointBundle:
    """Reconstruct a :class:`V2Network` from a Phase-2 checkpoint file.

    Parameters
    ----------
    path : Path
        ``.pt`` file as written by
        ``scripts.v2.train_phase2_predictive._save_checkpoint`` (keys:
        ``step``, ``state_dict``, ``phase``, ``frozen_sha``).
    seed : int
        Network construction seed. Must match the training seed because
        masks + random init depend on it; the checkpoint only holds
        state_dict values, not seeds.
    device : str
        Device string for the reconstructed network.

    Returns
    -------
    CheckpointBundle
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelConfig(seed=seed, device=device)
    bank = TokenBank(cfg, seed=0)
    net = V2Network(cfg, token_bank=bank, seed=seed, device=device)
    net.load_state_dict(ckpt["state_dict"])
    net.set_phase(ckpt.get("phase", "phase2"))
    net.eval()
    meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    return CheckpointBundle(cfg=cfg, net=net, bank=bank, meta=meta)


# ---------------------------------------------------------------------------
# Frame synthesis
# ---------------------------------------------------------------------------


def make_blank_frame(
    batch_size: int, cfg: ModelConfig,
    *, value: float = 0.5, device: str = "cpu",
) -> Tensor:
    """``[B, 1, H, W]`` uniform-grey frame (TARGET_MEAN ≈ 0.5)."""
    H, W = cfg.arch.grid_h, cfg.arch.grid_w
    return torch.full(
        (batch_size, 1, H, W), fill_value=float(value),
        dtype=torch.float32, device=device,
    )


def make_grating_frame(
    orientation_deg: float, contrast: float, cfg: ModelConfig,
    *, spatial_freq: float = 0.15, batch_size: int = 1, device: str = "cpu",
) -> Tensor:
    """Sinusoidal grating frame ``[B, 1, H, W]`` in [0, 1].

    ``orientation_deg`` is the orientation of the grating's wavefronts;
    the wave travels perpendicular to that direction. Luminance is
    ``0.5 + 0.5 · contrast · sin(2π f (x cosθ + y sinθ))``.
    """
    H, W = cfg.arch.grid_h, cfg.arch.grid_w
    ys = torch.arange(H, dtype=torch.float32, device=device) - (H - 1) / 2.0
    xs = torch.arange(W, dtype=torch.float32, device=device) - (W - 1) / 2.0
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    theta = math.radians(float(orientation_deg))
    proj = xx * math.cos(theta) + yy * math.sin(theta)
    grating = 0.5 + 0.5 * float(contrast) * torch.sin(
        2.0 * math.pi * float(spatial_freq) * proj,
    )
    grating = grating.clamp(0.0, 1.0).unsqueeze(0).unsqueeze(0)
    return grating.expand(batch_size, 1, H, W).contiguous()


def make_surround_grating_frame(
    orientation_deg: float, contrast: float, cfg: ModelConfig,
    *, center_radius: int = 6, include_surround: bool = True,
    spatial_freq: float = 0.15, batch_size: int = 1, device: str = "cpu",
) -> Tensor:
    """Grating windowed to a central disc (surround optionally grey).

    ``center_radius`` is in pixels. If ``include_surround`` is False, the
    outer region is set to 0.5 (grey); if True, the full frame is
    gratinged. The pair (False / True) powers the surround-suppression
    index SI = (R_center_only − R_center_surround) / R_center_only.
    """
    full = make_grating_frame(
        orientation_deg, contrast, cfg,
        spatial_freq=spatial_freq, batch_size=batch_size, device=device,
    )
    H, W = cfg.arch.grid_h, cfg.arch.grid_w
    ys = torch.arange(H, dtype=torch.float32, device=device) - (H - 1) / 2.0
    xs = torch.arange(W, dtype=torch.float32, device=device) - (W - 1) / 2.0
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    r = torch.sqrt(yy * yy + xx * xx)
    center_mask = (r <= float(center_radius)).to(full.dtype)
    if include_surround:
        return full
    # center-only: grey surround
    grey = torch.full_like(full, 0.5)
    return grey * (1.0 - center_mask) + full * center_mask


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


@torch.no_grad()
def simulate_steady_state(
    net: V2Network, frame: Tensor, n_steps: int,
    *, state: Optional[NetworkStateV2] = None,
) -> tuple[NetworkStateV2, dict]:
    """Run ``n_steps`` forward passes against a constant ``frame``.

    Returns the final ``(state, info)`` tuple from the last step. The
    initial state defaults to ``net.initial_state(batch_size=frame.shape[0])``.
    """
    B = frame.shape[0]
    if state is None:
        state = net.initial_state(batch_size=B)
    info: dict = {}
    for _ in range(int(n_steps)):
        _x_hat, state, info = net(frame, state)
    return state, info


@torch.no_grad()
def simulate_and_collect(
    net: V2Network, frames: list[Tensor], *,
    state: Optional[NetworkStateV2] = None,
    collect_keys: tuple[str, ...] = ("r_l4", "r_l23", "r_h"),
) -> tuple[NetworkStateV2, dict[str, Tensor]]:
    """Run the network through a list of frames, collecting per-step tensors.

    ``collect_keys`` picks which info entries to stack across steps.
    Returned dict maps key → ``[T, B, ...]`` tensor.
    """
    if not frames:
        raise ValueError("frames must be non-empty")
    B = frames[0].shape[0]
    if state is None:
        state = net.initial_state(batch_size=B)
    buffers: dict[str, list[Tensor]] = {k: [] for k in collect_keys}
    x_hat_list: list[Tensor] = []
    for f in frames:
        x_hat, state, info = net(f, state)
        x_hat_list.append(x_hat.detach())
        for k in collect_keys:
            buffers[k].append(info[k].detach())
    collected = {k: torch.stack(v, dim=0) for k, v in buffers.items()}
    collected["x_hat_next"] = torch.stack(x_hat_list, dim=0)
    return state, collected
