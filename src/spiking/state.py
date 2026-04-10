"""SpikingNetworkState: recurrent state container for the spiking V1-V2 model.

Parallel to `src/state.py` (rate model). Each spiking population owns three
state tensors per timestep:
    v_*  — membrane potential
    z_*  — binary spike (0/1) emitted at this step
    x_*  — exponentially filtered spike trace (the downstream interface —
           decoders, losses, V2 input, and analysis all consume `x_*`)

Adaptive populations (L4 ALIF, V2 LSNN) add one more tensor:
    adapt_l4 / b_v2 — threshold adaptation state

PV stays rate-based (divisive normalization pool) and keeps its `r_pv` field
unchanged. The deep_template is pure computation (algebraic) and also unchanged.

Evidence / rationale
--------------------
Field list taken verbatim from the Phase 1 port plan's `SpikingNetworkState`
specification:

    plans/quirky-humming-giraffe.md lines 100-128 (SpikingNetworkState Design)

Shapes and dtype conventions follow the rate model:
    src/state.py lines 11-23 (NetworkState)

Population sizes:
    n_orientations = 36 (plan architecture table, line 19-24)
    v2_hidden_dim  = 80 (plan line 25 — LSNN 80 neurons: 40 LIF_exc + 20 ALIF_exc + 20 LIF_inh)
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor


class SpikingNetworkState(NamedTuple):
    """Full recurrent state of the spiking laminar V1-V2 network.

    All tensors have batch dimension first: [B, ...].

    Total: 19 fields
        - L4 (ALIF):      4 (v_l4, z_l4, x_l4, adapt_l4)
        - PV (rate):      1 (r_pv)
        - L2/3 (RLIF):    3 (v_l23, z_l23, x_l23)
        - SOM (LIF):      3 (v_som, z_som, x_som)
        - VIP (LIF):      3 (v_vip, z_vip, x_vip)
        - V2 (LSNN):      4 (v_v2, z_v2, x_v2, b_v2)
        - Shared:         1 (deep_template)
    """

    # ---- L4 (ALIF) ----
    v_l4: Tensor          # [B, 36] — membrane potential
    z_l4: Tensor          # [B, 36] — spike (binary)
    x_l4: Tensor          # [B, 36] — filtered spike trace (downstream)
    adapt_l4: Tensor      # [B, 36] — SSA adaptation state (tau_a = 200)

    # ---- PV (rate-based, unchanged) ----
    r_pv: Tensor          # [B, 1]  — divisive normalization pool rate

    # ---- L2/3 (Recurrent LIF) ----
    v_l23: Tensor         # [B, 36] — membrane potential
    z_l23: Tensor         # [B, 36] — spike (binary)
    x_l23: Tensor         # [B, 36] — filtered trace (main network output)

    # ---- SOM (LIF) ----
    v_som: Tensor         # [B, 36] — membrane potential
    z_som: Tensor         # [B, 36] — spike (binary)
    x_som: Tensor         # [B, 36] — filtered trace

    # ---- VIP (LIF) ----
    v_vip: Tensor         # [B, 36] — membrane potential
    z_vip: Tensor         # [B, 36] — spike (binary)
    x_vip: Tensor         # [B, 36] — filtered trace (zero in simple_feedback mode)

    # ---- V2 (LSNN) ----
    v_v2: Tensor          # [B, 80] — membrane potential
    z_v2: Tensor          # [B, 80] — spike (binary)
    x_v2: Tensor          # [B, 80] — filtered trace (readout heads read this)
    b_v2: Tensor          # [B, 80] — adaptive threshold state (nonzero only for ALIF fraction)

    # ---- Shared (algebraic, unchanged) ----
    deep_template: Tensor # [B, 36] — deep-V1 expectation template


def initial_spiking_state(
    batch_size: int,
    n_orientations: int = 36,
    v2_hidden_dim: int = 80,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> SpikingNetworkState:
    """Create a zero-initialized SpikingNetworkState.

    Args:
        batch_size: Number of trials.
        n_orientations: Number of orientation channels for V1 populations.
            Default 36 (plan architecture).
        v2_hidden_dim: Number of V2 LSNN neurons. Default 80 (plan line 25).
        device: Device to place tensors on. Default CPU.
        dtype: Floating dtype for all state tensors. Default float32.

    Returns:
        SpikingNetworkState with every field initialized to zeros.
    """
    dev = device or torch.device("cpu")

    def z36() -> Tensor:
        return torch.zeros(batch_size, n_orientations, device=dev, dtype=dtype)

    def z80() -> Tensor:
        return torch.zeros(batch_size, v2_hidden_dim, device=dev, dtype=dtype)

    return SpikingNetworkState(
        # L4
        v_l4=z36(),
        z_l4=z36(),
        x_l4=z36(),
        adapt_l4=z36(),
        # PV
        r_pv=torch.zeros(batch_size, 1, device=dev, dtype=dtype),
        # L2/3
        v_l23=z36(),
        z_l23=z36(),
        x_l23=z36(),
        # SOM
        v_som=z36(),
        z_som=z36(),
        x_som=z36(),
        # VIP
        v_vip=z36(),
        z_vip=z36(),
        x_vip=z36(),
        # V2
        v_v2=z80(),
        z_v2=z80(),
        x_v2=z80(),
        b_v2=z80(),
        # Shared
        deep_template=z36(),
    )
