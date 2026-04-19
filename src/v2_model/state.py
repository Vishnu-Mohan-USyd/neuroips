"""V2 NetworkState NamedTuple (scaffold).

Per v4 spec §Architecture: state carries rate vectors for every population
(L4 E, L2/3 E/PV/SOM, H E/PV, C memory) plus the plasticity-rule bookkeeping
(pre/post synaptic traces) and the inferred latent-regime posterior.

Trace containers are `dict[str, Tensor]` keyed by the plastic connection name
(e.g. "W_l23_rec", "W_l23_h", "W_h_l23", "W_hm_gen"); population this dict
lazily as plasticity modules come online in later tasks.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor

from src.v2_model.config import ModelConfig


class NetworkStateV2(NamedTuple):
    """Full recurrent state of the v2 laminar predictive circuit.

    Batch dim first: [B, ...]. All tensors on the same device and dtype.
    """
    # V1 L4 (fixed front end)
    r_l4: Tensor                # [B, n_l4_e]   — L4 excitatory rates
    # (L4 PV pool rate omitted from named state — implicit in divisive norm)

    # V1 L2/3 (plastic Phase 2)
    r_l23: Tensor               # [B, n_l23_e]  — L2/3 excitatory rates
    r_pv: Tensor                # [B, n_l23_pv] — L2/3 PV rates
    r_som: Tensor               # [B, n_l23_som] — L2/3 SOM rates

    # Higher area H
    r_h: Tensor                 # [B, n_h_e]    — H excitatory rates
    h_pv: Tensor                # [B, n_h_pv]   — H PV rates

    # Context memory C
    m: Tensor                   # [B, n_c]      — context-memory state

    # Plasticity bookkeeping (keyed by plastic-connection name)
    pre_traces: dict[str, Tensor]
    post_traces: dict[str, Tensor]

    # Latent regime posterior over the 4 regime categories
    regime_posterior: Tensor    # [B, n_regimes]


def initial_state(
    cfg: ModelConfig,
    batch_size: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> NetworkStateV2:
    """Zero-initialise all rate vectors; empty trace dicts; uniform regime posterior.

    Args:
        cfg: Model config (supplies population sizes).
        batch_size: Batch dimension B.
        device: Torch device (falls back to `cfg.device`).
        dtype: Torch dtype (default float32).

    Returns:
        NetworkStateV2 with all rate tensors at zero and regime_posterior uniform
        (1/n_regimes). pre_traces / post_traces are empty dicts — plasticity
        modules fill them on first call.
    """
    dev = device if device is not None else torch.device(cfg.device)
    a = cfg.arch
    n_reg = cfg.regime.n_regimes

    def _z(*shape: int) -> Tensor:
        return torch.zeros(*shape, device=dev, dtype=dtype)

    return NetworkStateV2(
        r_l4=_z(batch_size, a.n_l4_e),
        r_l23=_z(batch_size, a.n_l23_e),
        r_pv=_z(batch_size, a.n_l23_pv),
        r_som=_z(batch_size, a.n_l23_som),
        r_h=_z(batch_size, a.n_h_e),
        h_pv=_z(batch_size, a.n_h_pv),
        m=_z(batch_size, a.n_c),
        pre_traces={},
        post_traces={},
        regime_posterior=torch.full(
            (batch_size, n_reg), 1.0 / n_reg, device=dev, dtype=dtype
        ),
    )
