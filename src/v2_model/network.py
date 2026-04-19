"""V2 laminar predictive circuit — top-level wiring (plan v4 step 13 / Task #29).

Wires the five component modules built in Tasks #9–#23 into a single
runnable forward step:

* Fixed LGN + V1 L4 front end (``LGNL4FrontEnd``) — frame → L4 E rate.
* L2/3 recurrent populations ``L23E`` / ``L23PV`` / ``L23SOM``.
* Higher-area recurrent populations ``HE`` / ``HPV``.
* Context memory ``ContextMemory`` — leaky hidden-regime integrator.
* ``PredictionHead`` — linear readout → next-step L4 E prediction.

Synchronous Euler integration
------------------------------
Every recurrent population reads *pre-update* state and returns its
*new* rate for the next step. The only exception is ``LGNL4FrontEnd``,
which samples the current-step frame (external input, not a recurrent
variable) but integrates its own dynamics against ``state.r_l4``. This
yields a clean "atomic swap" at the end of each forward call:

    state_t  →  (every pop applies pre-update sync-Euler)  →  state_{t+1}

Recurrent cycles (L23E↔PV, L23E↔SOM, L23↔H, HE↔HPV, etc.) are resolved
by the pre-update convention — no pop reads a same-step sibling's fresh
rate.

ContextMemory routing
---------------------
:class:`ContextMemory` exposes a single ``n_out`` output bias; the
design note in ``context_memory.py`` makes explicit that the caller
chooses its target. We route the bias to **L2/3 apical** (primary
predictive locus; ``n_out = n_l23_e = 256``). The HE ``context_bias``
slot is filled with a zero tensor — ``HE`` accepts the field per its
signature contract but the HE apical pathway is not context-modulated
in this wiring. A second readout can be added cheaply if a future
plan revision re-routes context to H as well.

Prediction head wiring
----------------------
The prediction head predicts *next-step* L4 rate. It consumes:

* ``h_rate`` = the just-computed ``r_h_new`` (end-of-step H state),
* ``c_bias`` = pre-update memory ``state.m`` (caller-side readout of
  the memory before this step — the prediction depends on the context
  the memory held entering the step),
* ``l23_apical_summary`` = the just-computed ``r_l23_new`` (end-of-step
  L2/3 E rate supplies the apical compartment summary).

These feed the Urbanczik–Senn apical-basal comparison at the next step:
``ε_{t+1} = r_l4_{t+1}(actual) − x̂_{t+1}(predicted at step t)``.

Memory budget (expected — sanity-checked in tests)
---------------------------------------------------
Roughly 0.36 M Parameters total across L23E (92k) + HE (28k) + H↔M
(27k + 13k + LGN front-end has zero Parameters). All weights are
``nn.Parameter(..., requires_grad=False)`` — plasticity is driven
externally by ``plasticity.py`` rules.

Frozen-core contract
---------------------
:meth:`frozen_sensory_core_sha` returns a SHA-256 hash per buffer on
:class:`LGNL4FrontEnd`. A training driver compares hashes before/after
a plasticity epoch to assert the LGN/L4 front end was never mutated.
Because LGN/L4 has zero ``nn.Parameter``s by construction, any non-trivial
change to its buffers would be a logic error.
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor

from src.v2_model.config import ModelConfig
from src.v2_model.context_memory import ContextMemory
from src.v2_model.layers import (
    FastInhibitoryPopulation,
    HE,
    L23E,
    L23SOM,
    PhaseLiteral,
)
from src.v2_model.lgn_l4 import LGNL4FrontEnd
from src.v2_model.prediction_head import PredictionHead
from src.v2_model.state import NetworkStateV2, initial_state as _zero_state


__all__ = ["V2Network"]


class V2Network(nn.Module):
    """Laminar predictive circuit: LGN/L4 + L2/3 + H + C + prediction head.

    Parameters
    ----------
    cfg : ModelConfig
        V2 model config. Population sizes and time constants come from
        ``cfg.arch`` / ``cfg.tau`` / ``cfg.conn`` / ``cfg.plasticity``.
    token_bank : Any
        Optional identity-token bank (``TokenBank`` from
        ``src.v2_model.stimuli.feature_tokens``). Held by the network for
        downstream stimulus generation; not consumed in the forward step.
        Pass ``None`` when the caller constructs frames externally.
    seed : int
        Base seed. Each child module is instantiated with ``seed + k`` so
        no two weight tensors share initial RNG state.
    device, dtype
        Standard PyTorch placement/precision. Defaults: CPU / float32.
    """

    # Per-module seed offsets — keeps weight inits independent across pops.
    _SEED_OFFSET_LGN_L4: int = 0
    _SEED_OFFSET_L23_E: int = 1
    _SEED_OFFSET_L23_PV: int = 2
    _SEED_OFFSET_L23_SOM: int = 3
    _SEED_OFFSET_H_E: int = 4
    _SEED_OFFSET_H_PV: int = 5
    _SEED_OFFSET_CONTEXT: int = 6
    _SEED_OFFSET_PREDICTION: int = 7

    def __init__(
        self,
        cfg: ModelConfig,
        token_bank: Optional[Any] = None,
        seed: int = 42,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_bank = token_bank
        self._seed = int(seed)
        self._device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        self._dtype = dtype
        self._phase: str = "phase2"

        a = cfg.arch
        t = cfg.tau
        p = cfg.plasticity
        conn = cfg.conn

        # -- LGN + L4 (frozen, no Parameters) ------------------------------
        self.lgn_l4 = LGNL4FrontEnd(cfg).to(device=self._device, dtype=self._dtype)

        # -- L2/3 E (principal predictive pyramidal population) -----------
        self.l23_e = L23E(
            n_units=a.n_l23_e,
            n_l4_e=a.n_l4_e,
            n_pv=a.n_l23_pv,
            n_som=a.n_l23_som,
            n_h_e=a.n_h_e,
            tau_ms=t.tau_l23_e_ms,
            dt_ms=t.dt_ms,
            sparsity=conn.sparsity,
            sigma_position=conn.sigma_r_px,
            sigma_feature=conn.sigma_theta_deg,
            # Mouse V1 L2/3 baseline rate (Niell & Stryker 2008). Must be
            # reachable from typical drive; else θ runs away trying to raise
            # an unattainable rate.
            target_rate=0.5,
            lr_homeostasis=p.lr_homeostasis,
            seed=self._seed + self._SEED_OFFSET_L23_E,
            device=self._device,
            dtype=self._dtype,
        )

        # -- L2/3 PV (fast, τ=5ms — exact-ODE leak) -----------------------
        self.l23_pv = FastInhibitoryPopulation(
            n_units=a.n_l23_pv,
            n_pre=a.n_l23_e,
            tau_ms=t.tau_l23_pv_ms,
            dt_ms=t.dt_ms,
            target_rate_hz=p.target_rate_hz,
            seed=self._seed + self._SEED_OFFSET_L23_PV,
            device=self._device,
            dtype=self._dtype,
        )

        # -- L2/3 SOM (surround / apical inhibition) ----------------------
        self.l23_som = L23SOM(
            n_units=a.n_l23_som,
            n_l23_e=a.n_l23_e,
            n_h_e=a.n_h_e,
            tau_ms=t.tau_l23_som_ms,
            dt_ms=t.dt_ms,
            target_rate_hz=p.target_rate_hz,
            seed=self._seed + self._SEED_OFFSET_L23_SOM,
            device=self._device,
            dtype=self._dtype,
        )

        # -- H E (higher-area latent prior) -------------------------------
        self.h_e = HE(
            n_units=a.n_h_e,
            n_l23_e=a.n_l23_e,
            n_h_pv=a.n_h_pv,
            tau_ms=t.tau_h_ms,
            dt_ms=t.dt_ms,
            sparsity=conn.sparsity,
            # Lower target than L2/3 — higher area rates are sparser under
            # the current sparse L23→HE input. Target 1.0 is unreachable and
            # drives θ downward without bound (Debugger #37 confirmed root
            # cause of Phase-2 |ε| divergence).
            target_rate=0.1,
            lr_homeostasis=p.lr_homeostasis,
            seed=self._seed + self._SEED_OFFSET_H_E,
            device=self._device,
            dtype=self._dtype,
        )

        # -- H PV (fast, τ=5ms — shares tau with L23 PV) ------------------
        self.h_pv = FastInhibitoryPopulation(
            n_units=a.n_h_pv,
            n_pre=a.n_h_e,
            tau_ms=t.tau_l23_pv_ms,
            dt_ms=t.dt_ms,
            target_rate_hz=p.target_rate_hz,
            seed=self._seed + self._SEED_OFFSET_H_PV,
            device=self._device,
            dtype=self._dtype,
        )

        # -- Context memory C (targets L2/3 apical; n_out = n_l23_e) -------
        # Kok cue + Richter leader are sized placeholder (n_c, n_h_e); real
        # task-specific dimensions are task-driver-wired in Phase 3.
        self.context_memory = ContextMemory(
            n_m=a.n_c,
            n_h=a.n_h_e,
            n_cue=a.n_c,
            n_leader=a.n_h_e,
            n_out=a.n_l23_e,
            tau_m_ms=t.tau_c_ms,
            dt_ms=t.dt_ms,
            seed=self._seed + self._SEED_OFFSET_CONTEXT,
            device=self._device,
            dtype=self._dtype,
        )

        # -- Prediction head (x̂_{t+1}) -----------------------------------
        self.prediction_head = PredictionHead(
            n_l4_e=a.n_l4_e,
            n_h_e=a.n_h_e,
            n_c_bias=a.n_c,
            n_l23_apical=a.n_l23_e,
            seed=self._seed + self._SEED_OFFSET_PREDICTION,
            device=self._device,
            dtype=self._dtype,
        )

    # ---------------------------------------------------------------------
    # Phase propagation
    # ---------------------------------------------------------------------

    def set_phase(self, phase: PhaseLiteral) -> None:
        """Propagate phase to every child with a ``set_phase`` method.

        Pure book-keeping — no weight is mutated. Training drivers
        consult :meth:`plastic_weight_names` after switching.
        """
        self._phase = phase
        for child in self.children():
            fn = getattr(child, "set_phase", None)
            if callable(fn):
                fn(phase)

    @property
    def phase(self) -> str:
        return self._phase

    def plastic_weight_names(self) -> list[tuple[str, str]]:
        """Aggregated plastic-weight manifest across all sub-modules.

        Returns
        -------
        list[tuple[str, str]]
            Pairs ``(module_name, raw_weight_name)`` for every weight the
            current phase's plasticity rules may mutate. Module names
            match the ``nn.Module`` child attribute names
            (``l23_e``, ``l23_pv``, …). The ``lgn_l4`` front end is
            never plastic and is omitted.
        """
        out: list[tuple[str, str]] = []
        for name, child in self.named_children():
            if name == "lgn_l4":
                continue
            fn = getattr(child, "plastic_weight_names", None)
            if not callable(fn):
                continue
            out.extend((name, w) for w in fn())
        return out

    # ---------------------------------------------------------------------
    # Frozen-core SHA
    # ---------------------------------------------------------------------

    def frozen_sensory_core_sha(self) -> dict[str, str]:
        """SHA-256 hex digest of every LGN/L4 buffer.

        The LGN/L4 front end is frozen by construction (zero
        ``nn.Parameter``s, all filters as buffers). Callers can store a
        snapshot of this dict before any training epoch and compare
        after to assert non-mutation.
        """
        assert len(list(self.lgn_l4.parameters())) == 0, (
            "LGN/L4 must hold zero nn.Parameters (frozen-by-construction)"
        )
        digests: dict[str, str] = {}
        for name, buf in self.lgn_l4.named_buffers():
            arr = buf.detach().contiguous().cpu().numpy().tobytes()
            digests[name] = hashlib.sha256(arr).hexdigest()
        return digests

    # ---------------------------------------------------------------------
    # State helpers
    # ---------------------------------------------------------------------

    def initial_state(
        self,
        batch_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> NetworkStateV2:
        """Zero-initial :class:`NetworkStateV2` of the right shapes."""
        dev = device if device is not None else self._device
        return _zero_state(self.cfg, batch_size, device=dev, dtype=dtype)

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------

    def forward(
        self,
        x_t: Tensor,
        state: NetworkStateV2,
        q_t: Optional[Tensor] = None,
        leader_t: Optional[Tensor] = None,
    ) -> tuple[Tensor, NetworkStateV2, dict[str, Tensor]]:
        """One synchronous-Euler step.

        Parameters
        ----------
        x_t : Tensor [B, 1, H, W]
            Current-step frame (retinotopic grayscale image).
        state : NetworkStateV2
            Pre-update state entering this step. Must be batch-aligned
            with ``x_t``.
        q_t, leader_t : Tensor | None
            Optional Kok cue / Richter leader streams for the context
            memory (forwarded untouched). Defaults to ``None`` (ignored).

        Returns
        -------
        x_hat_next : Tensor [B, n_l4_e]
            Predicted next-step L4 E rate.
        next_state : NetworkStateV2
            State after the sync-Euler update.
        info : dict[str, Tensor]
            Bundle of intermediate tensors useful for logging / analysis:
            ``lgn_feature_map`` [B, 2+N_ori, H, W], the end-of-step
            rate vectors, the context readout ``b_l23``, and the
            predicted rate ``x_hat_next``.
        """
        if x_t.ndim != 4 or x_t.shape[1] != 1:
            raise ValueError(
                f"x_t must be [B, 1, H, W]; got shape {tuple(x_t.shape)}"
            )
        B = x_t.shape[0]
        if state.r_l4.shape[0] != B:
            raise ValueError(
                f"state batch size {state.r_l4.shape[0]} does not match "
                f"frames batch {B}"
            )

        # --- LGN + L4 (frame → r_l4_new) ---------------------------------
        lgn_feat, r_l4_new, _ = self.lgn_l4(x_t, state)

        # --- Context memory C (m_t, r_h_t → m_next, b_l23) ----------------
        m_next, b_l23 = self.context_memory(
            state.m, state.r_h, q_t=q_t, leader_t=leader_t,
        )

        # HE context_bias is not routed in this wiring (see module
        # docstring): supply an explicit zero tensor so HE's forward
        # shape-guard passes.
        zero_h_context = torch.zeros(
            B, self.h_e.n_units, device=state.r_h.device, dtype=state.r_h.dtype,
        )

        # --- Recurrent pops (strict sync-Euler: pre-update state only) ----
        r_l23_new, _ = self.l23_e(
            l4_input=state.r_l4,
            l23_recurrent_input=state.r_l23,
            som_input=state.r_som,
            pv_input=state.r_pv,
            h_apical_input=state.r_h,
            context_bias=b_l23,
            state=state.r_l23,
        )
        r_pv_new, _ = self.l23_pv(state.r_l23, state.r_pv)
        r_som_new, _ = self.l23_som(
            l23e_input=state.r_l23,
            h_som_feedback_input=state.r_h,
            state=state.r_som,
        )
        r_h_new, _ = self.h_e(
            l23_input=state.r_l23,
            h_recurrent_input=state.r_h,
            h_pv_input=state.h_pv,
            context_bias=zero_h_context,
            state=state.r_h,
        )
        h_pv_new, _ = self.h_pv(state.r_h, state.h_pv)

        # --- Prediction head: x̂_{t+1} = g(r_h_new, m_t, r_l23_new) --------
        x_hat_next = self.prediction_head(
            h_rate=r_h_new,
            c_bias=state.m,
            l23_apical_summary=r_l23_new,
        )

        next_state = NetworkStateV2(
            r_l4=r_l4_new,
            r_l23=r_l23_new,
            r_pv=r_pv_new,
            r_som=r_som_new,
            r_h=r_h_new,
            h_pv=h_pv_new,
            m=m_next,
            pre_traces=state.pre_traces,
            post_traces=state.post_traces,
            regime_posterior=state.regime_posterior,
        )

        info = {
            "lgn_feature_map": lgn_feat,
            "r_l4": r_l4_new,
            "r_l23": r_l23_new,
            "r_pv": r_pv_new,
            "r_som": r_som_new,
            "r_h": r_h_new,
            "h_pv": h_pv_new,
            "m": m_next,
            "b_l23": b_l23,
            "x_hat_next": x_hat_next,
        }
        return x_hat_next, next_state, info
