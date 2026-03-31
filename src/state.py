"""NetworkState: recurrent state container for the laminar V1-V2 model."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor


class NetworkState(NamedTuple):
    """Full recurrent state of the laminar V1-V2 network.

    All tensors have batch dimension first: [B, ...].
    """
    r_l4: Tensor          # [B, 36] — V1 Layer 4 excitatory rates
    r_l23: Tensor         # [B, 36] — V1 Layer 2/3 excitatory rates
    r_pv: Tensor          # [B, 1]  — PV pool rate
    r_som: Tensor         # [B, 36] — SOM ring rates
    adaptation: Tensor    # [B, 36] — L4 adaptation state
    h_v2: Tensor          # [B, 16] — V2 GRU hidden state
    deep_template: Tensor # [B, 36] — Deep-V1 expectation template


def initial_state(batch_size: int, n_orientations: int = 36,
                  v2_hidden_dim: int = 16, device: torch.device | None = None) -> NetworkState:
    """Create a zero-initialized NetworkState."""
    dev = device or torch.device("cpu")
    return NetworkState(
        r_l4=torch.zeros(batch_size, n_orientations, device=dev),
        r_l23=torch.zeros(batch_size, n_orientations, device=dev),
        r_pv=torch.zeros(batch_size, 1, device=dev),
        r_som=torch.zeros(batch_size, n_orientations, device=dev),
        adaptation=torch.zeros(batch_size, n_orientations, device=dev),
        h_v2=torch.zeros(batch_size, v2_hidden_dim, device=dev),
        deep_template=torch.zeros(batch_size, n_orientations, device=dev),
    )
