"""Bridge API: keep src/analysis/* usable against the v2 network.

Per v4 spec §Critical files and prior scope-impact analysis: current analysis
modules (decoding, tuning_curves, rsa, suppression_profile, v2_probes, plotting,
bias_analysis) consume dict-keyed activation bundles. Rather than rewrite them
all, v2 exposes the same dict shape via `extract_activations`.

Keys returned (populated once the v2 network lands in network.py):
  - r_l4, r_l23, r_pv, r_som         — V1 rates
  - r_h, h_pv                         — H rates
  - m                                 — context-memory state
  - regime_posterior                  — [B, n_regimes] latent-regime posterior
  - feature_map                       — [B, C, H, W] LGN/Gabor output (analysis code
                                        can reduce to r_l4 surrogate for plots)

This module is a scaffold: real extraction logic lands once `network.py` exists.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def extract_activations(net: Any, stim: Tensor) -> dict[str, Tensor]:
    """Run the v2 network on a stimulus batch and return activations by key.

    Rolls out ``net`` for ``T`` steps starting from :meth:`V2Network.initial_state`
    and stacks per-step populations along a time axis.

    Args:
        net: A :class:`V2Network` instance.
        stim: Stimulus tensor, shape ``[B, T, 1, H, W]`` (video).

    Returns:
        Dict keyed by population name with stacked ``[B, T, ...]`` tensors:
          * ``r_l4``, ``r_l23``, ``r_pv``, ``r_som``   — V1 rates
          * ``r_h``, ``h_pv``                          — H rates
          * ``m``                                       — context-memory state
          * ``regime_posterior``                        — [B, T, n_regimes] (frozen
                                                         since the forward does
                                                         not update it)
          * ``feature_map``                             — [B, T, C, H, W] LGN output

    Raises:
        ValueError: If ``stim`` is not ``[B, T, 1, H, W]``.
    """
    if stim.ndim != 5 or stim.shape[2] != 1:
        raise ValueError(
            f"stim must be [B, T, 1, H, W]; got shape {tuple(stim.shape)}"
        )
    B, T, _, _, _ = stim.shape
    state = net.initial_state(batch_size=B, device=stim.device, dtype=stim.dtype)

    bufs: dict[str, list[Tensor]] = {
        "r_l4": [], "r_l23": [], "r_pv": [], "r_som": [],
        "r_h": [], "h_pv": [], "m": [], "feature_map": [],
        "regime_posterior": [],
    }
    for t in range(T):
        _x_hat, state, info = net(stim[:, t], state)
        bufs["r_l4"].append(info["r_l4"])
        bufs["r_l23"].append(info["r_l23"])
        bufs["r_pv"].append(info["r_pv"])
        bufs["r_som"].append(info["r_som"])
        bufs["r_h"].append(info["r_h"])
        bufs["h_pv"].append(info["h_pv"])
        bufs["m"].append(info["m"])
        bufs["feature_map"].append(info["lgn_feature_map"])
        bufs["regime_posterior"].append(state.regime_posterior)

    return {k: torch.stack(v, dim=1) for k, v in bufs.items()}


def extract_readout_data(activations: dict[str, Tensor], readout_fn: Any) -> Tensor:
    """Readout over trial-structured activation dicts.

    Matches the signature of `src/training/trainer.py::extract_readout_data` so
    existing analysis pipelines can be swapped over without signature changes.

    Args:
        activations: Dict from `extract_activations`.
        readout_fn: Callable applied to a selected activation tensor.

    Raises:
        NotImplementedError: Scaffold only.
    """
    raise NotImplementedError(
        "extract_readout_data is a scaffold; implement once trial-structured "
        "paradigms (Kok/Richter) are defined (Task #15+)."
    )
