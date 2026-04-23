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
:class:`ContextMemory` exposes two output quantities (Task #74 Fix C-v2):

* ``b_l23_exc`` — [B, n_l23_e] = W_mh_gen·m + task_exc_gain·W_mh_task_exc·m.
  Routed to **L2/3 E apical** (secondary predictive-bias additive term);
  the task-specific contribution is deliberately small
  (``task_exc_gain=0.1``).
* ``som_gain`` — [B, n_l23_som] = softplus(W_mh_task_inh·m + 0.5413
  ).clamp(max=4.0). A per-SOM-unit **multiplicative gain** applied to
  the SOM→L23E inhibitory synapses via the new ``som_gain`` kwarg on
  :class:`L23E.forward`. This is the MAIN route for Phase-3 task bias:
  the readout scales the efficacy of SOM-mediated apical inhibition of
  L23E dendrites rather than adding to SOM drive directly. Biologically
  interpreted as cholinergic / noradrenergic modulation of GABAergic
  transmission (Disney & Aoki 2008; Pfeffer 2013). At init (W=0) the
  gain is identically 1, so Phase-2 dynamics are exactly preserved.

The HE ``context_bias`` slot remains zero — HE apical is not
context-modulated in this wiring.

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


# ---------------------------------------------------------------------------
# Fix K helper — build sparse orientation-biased L4→L23E mask.
# ---------------------------------------------------------------------------


def _build_l4_l23_mask(
    *,
    n_l23_e: int,
    lgn_l4: LGNL4FrontEnd,
    sparsity: float,
    sigma_theta_deg: float,
    retino_radius_cells: int,
    device: torch.device | str | None,
    dtype: torch.dtype,
) -> Tensor:
    """Deterministic top-k orientation-like-to-like retinotopic mask.

    For each L23E unit i we assign a target preferred orientation and
    retinotopic cell, compute a score S[i, j] = Gaussian(Δθ, σ_θ) gated by
    Chebyshev retino-distance ≤ ``retino_radius_cells``, and keep the
    top-k entries with k = round(sparsity · n_l4_e). No RNG: ties broken
    by argsort stability.

    Shape: [n_l23_e, n_l4_e]. Values ∈ {0, 1}. Every row sums to exactly k.

    Target assignment (fixed tiling; requires n_l23_e divisible by
    retino_side²). For ``n_l23_e = 256`` and ``retino_side = 4`` this
    gives 16 orientation slots per retino cell (11.25° spacing).
    """
    n_l4_e = int(lgn_l4.n_l4_e)
    retino_side = int(lgn_l4.retino_side)
    n_retino_cells = retino_side * retino_side
    if n_l23_e % n_retino_cells != 0:
        raise ValueError(
            f"Fix-K mask: n_l23_e={n_l23_e} must be divisible by "
            f"retino_side²={n_retino_cells}."
        )
    n_ori_slots = n_l23_e // n_retino_cells         # e.g. 16

    # Per-L4-unit metadata (from filter-bank layout, no probe required).
    l4_pref_deg = lgn_l4.pref_orient_deg_l4.to(
        device=device if device is not None else torch.device("cpu")
    )                                                # [n_l4_e]
    l4_ri, l4_rj = lgn_l4.retino_ij_l4
    l4_ri = l4_ri.to(device=l4_pref_deg.device)
    l4_rj = l4_rj.to(device=l4_pref_deg.device)

    # Per-L23E-unit target metadata.
    idx = torch.arange(n_l23_e, dtype=torch.long, device=l4_pref_deg.device)
    retino_flat_tgt = idx // n_ori_slots
    orient_bin_tgt = idx %  n_ori_slots
    target_ri = retino_flat_tgt // retino_side
    target_rj = retino_flat_tgt %  retino_side
    target_deg = orient_bin_tgt.to(torch.float32) * (180.0 / float(n_ori_slots))

    # Pairwise circular Δθ on 180°: Δ = min(|a−b|, 180 − |a−b|).
    dth = (target_deg.unsqueeze(1) - l4_pref_deg.unsqueeze(0)).abs()
    dth = torch.minimum(dth, 180.0 - dth)             # [n_l23_e, n_l4_e]
    orient_score = torch.exp(-(dth ** 2) / (2.0 * float(sigma_theta_deg) ** 2))

    # Chebyshev retino distance ≤ r_ret.
    dri = (target_ri.unsqueeze(1) - l4_ri.unsqueeze(0)).abs()
    drj = (target_rj.unsqueeze(1) - l4_rj.unsqueeze(0)).abs()
    retino_gate = (
        (dri <= int(retino_radius_cells))
        & (drj <= int(retino_radius_cells))
    ).to(torch.float32)

    score = orient_score * retino_gate                # [n_l23_e, n_l4_e]

    k = max(1, int(round(float(sparsity) * n_l4_e)))
    # Pick top-k per row. torch.topk is deterministic for float32 with
    # stable ordering on ties.
    _, top_idx = torch.topk(score, k=k, dim=1, largest=True, sorted=False)
    mask = torch.zeros_like(score)
    mask.scatter_(1, top_idx, 1.0)
    return mask.to(dtype=dtype)


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
            # Fix K: W_l4_l23 init_mean dropped 4.0 → 1.5 to keep per-unit
            # drive in biological range once the sparse orientation-biased
            # mask is installed below.
            w_l4_l23_init_mean=conn.w_l4_l23_init_mean,
            # Task #74 β-mechanism (Step 1): per-cue multiplicative gain
            # on L4→L23E. ``n_cue = a.n_c`` matches the one-hot cue
            # tensor emitted by ``build_cue_tensor``. W_q_gain defaults to
            # ones (no-op) so Phase-2 checkpoints remain bit-exact.
            n_cue=a.n_c,
        )
        # Fix K: install sparse orient-biased retinotopic mask on W_l4_l23.
        mask = _build_l4_l23_mask(
            n_l23_e=a.n_l23_e,
            lgn_l4=self.lgn_l4,
            sparsity=conn.l4_l23_mask_sparsity,
            sigma_theta_deg=conn.l4_l23_sigma_theta_deg,
            retino_radius_cells=conn.l4_l23_retino_radius_cells,
            device=self._device,
            dtype=self._dtype,
        )
        self.l23_e.install_l4_l23_mask(mask)

        # -- L2/3 PV (fast, τ=5ms — exact-ODE leak) -----------------------
        self.l23_pv = FastInhibitoryPopulation(
            n_units=a.n_l23_pv,
            n_pre=a.n_l23_e,
            tau_ms=t.tau_l23_pv_ms,
            dt_ms=t.dt_ms,
            target_rate_hz=p.target_rate_hz,
            # Task #52 — T29 init_mean. W_pre = -1.0 puts PV drive
            # ≈ 256·softplus(-1)·r_l23 = 256·0.313·0.012 ≈ 0.96 just below
            # the target_rate_hz=1.0 threshold at blank; for r_l23 > 0.013
            # PV crosses threshold and responds. The L23↔PV loop stays
            # stable because W_pv_l23 in L23E is -5.0 (weak feedback).
            w_pre_init_mean=-1.0,
            # Task #74 Fix Q — freeze L23E→L23PV at init. Applying Vogels
            # iSTDP (an I→E rule) to this E→I synapse inverted the sign of
            # the homeostatic term and drove L23E monotone collapse; the
            # cleanest intervention is to freeze this weight end-to-end in
            # Phase-2, matching the Fix D-simpler pattern for L23SOM
            # excitatory inputs.
            freeze_W_pre=True,
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
        # Task #74 Fix Q' — same E→I+Vogels sign-inversion structural bug as
        # Fix Q on l23_pv. ``h_pv.W_pre_raw`` is the HE→HPV excitatory synapse;
        # Vogels iSTDP applied here inverts sign and is anti-homeostatic even
        # if HPV rates are currently stable. Freeze at init end-to-end.
        self.h_pv = FastInhibitoryPopulation(
            n_units=a.n_h_pv,
            n_pre=a.n_h_e,
            freeze_W_pre=True,
            tau_ms=t.tau_l23_pv_ms,
            dt_ms=t.dt_ms,
            target_rate_hz=p.target_rate_hz,
            # Task #52 — T29 init_mean. HE rate ≈ 0.005 at blank is much
            # lower than L23E (0.012), so HPV needs a much stronger W_pre
            # to cross threshold: softplus(3)·64·0.005 ≈ 1.00 matches the
            # target_rate_hz=1.0 threshold. Loop stability maintained by
            # W_pv_h = -5.0 in HE (weak HPV→HE feedback).
            w_pre_init_mean=3.0,
            seed=self._seed + self._SEED_OFFSET_H_PV,
            device=self._device,
            dtype=self._dtype,
        )

        # -- Context memory C (targets L2/3 apical + L2/3 SOM) -------------
        # Kok cue + Richter leader are sized placeholder (n_c, n_h_e); real
        # task-specific dimensions are task-driver-wired in Phase 3.
        # Task #74 Fix C: n_out_som enables the SOM-route readout
        # (W_mh_task_inh) — preserved for state-dict compatibility but the
        # Fix-J redesign (Task #74, 2026-04-21) makes the SOM-gain path
        # inert: training drivers no longer update W_mh_task_inh, so
        # ``som_gain = softplus(0·m + 0.5413)`` stays identically 1.0.
        # task_exc_gain raised from 0.1 → 1.0 so the Phase-3 task bias
        # actually drives L23E modulation through W_mh_task_exc (previously
        # the 0.1× multiplier made this path near-silent, which is why the
        # SOM-gain route was introduced in Fix C-v2). Fix J reverts to the
        # additive L23E path with unit gain + L23E-space differential
        # modulator — see train_phase3_kok_learning.py.
        self.context_memory = ContextMemory(
            n_m=a.n_c,
            n_h=a.n_h_e,
            n_cue=a.n_c,
            n_leader=a.n_h_e,
            n_out=a.n_l23_e,
            n_out_som=a.n_l23_som,
            task_exc_gain=1.0,
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

        # --- Context memory C (m_t, r_h_t → m_next, b_l23_exc, som_gain) --
        # Task #74 Fix C-v2: task readout split is
        #   b_l23_exc → L23 E apical (additive, scaled by task_exc_gain)
        #   som_gain  → per-SOM-unit multiplicative gain on SOM→L23E
        #               synapses. At init (W=0) som_gain == 1 everywhere —
        #               the L23E forward is a strict no-op vs Phase-2.
        m_next, b_l23_exc, som_gain = self.context_memory(
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
            context_bias=b_l23_exc,
            state=state.r_l23,
            som_gain=som_gain,
            # Task #74 β-mechanism (Step 1, 2026-04-23): route the cue
            # one-hot (when present) into L23E so W_q_gain can scale the
            # L4 feedforward contribution. q_t is None outside the cue /
            # gated-probe epochs, in which case L23E.forward skips the
            # multiplication entirely (no-op).
            q_t=q_t,
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
            # Task #74 Fix C-v2: expose both task-readout components for the
            # training drivers + diagnostics (effective-drive computation).
            # "b_l23" is retained as an alias for b_l23_exc for backward
            # compatibility with any caller reading the legacy key.
            # "som_gain" is the per-SOM-unit multiplicative gain on the
            # SOM→L23E synapses (softplus(W_mh_task_inh·m + 0.5413).clamp(max=4)),
            # replacing the old additive ``b_l23_inh`` route.
            "b_l23": b_l23_exc,
            "b_l23_exc": b_l23_exc,
            "som_gain": som_gain,
            "x_hat_next": x_hat_next,
        }
        return x_hat_next, next_state, info
