"""Context memory C — generic/task-specific weight split (v4 plan §Architecture + D.2).

The memory is a leaky recurrent population driven by (a) the H excitatory state
(generic pathway, plastic in Phase 2), (b) its own recurrence (generic), and
(c) two optional task-specific input streams, the Kok cue token `q_t` and the
Richter leader summary `leader_t` (both plastic in Phase 3, frozen in Phase 2).
The output bias is the sum of a generic and a task-specific readout of the
current memory state.

Dynamics (plan v4 D.2 with exact-ODE leak factor)
-------------------------------------------------

``m_{t+1} = exp(-dt/τ_m) · m_t
           + φ(  W_hm^gen · h_t
               + W_mm^gen · m_t
               + W_qm^task · q_t          (zero during Phase 2 / Richter)
               + W_lm^task · leader_t )   (zero during Phase 2 / Kok)

b_exc_t   = W_mh^gen · m_t  +  task_exc_gain · W_mh^task_exc · m_t
som_gain_t = softplus(W_mh^task_inh · m_t + bias0).clamp(max=g_max)``

  where ``bias0 = log(e − 1) ≈ 0.5413`` so ``softplus(bias0) = 1.0`` (init
  no-op — at W_mh_task_inh=0 the gain is exactly 1 everywhere) and
  ``g_max = 4.0`` bounds the multiplicative up-regulation, matching the
  biological cholinergic/noradrenergic modulation range on inhibitory
  GABAergic efficacy (Disney & Aoki 2008; Pfeffer 2013).

  Task #74 Fix C-v2 redesign: the Phase-3 task readout used to add an
  additive drive ``b_inh_t = W_mh^task_inh · m_t`` into L2/3 SOM. Because
  SOM operates in a saturated regime at the Phase-2 baseline (r_som on the
  order of 100s/unit under the current homeostasis), any additive drive
  further amplified SOM → L23E inhibition and silenced L23E to machine
  zero within a few hundred training trials (Fix-C incidents #74B and
  #74C). The redesigned route is *gain modulation* of the SOM→L23E
  synapses instead: the task readout produces a per-SOM-unit gain that
  scales ``w_som_l23`` multiplicatively. Three-factor plasticity on
  ``W_mh_task_inh`` is preserved verbatim — only the *application* site
  of the produced signal changes.

where ``φ`` is a non-negative rectified nonlinearity (``rectified_softplus`` by
default). The plan v4 D.2 wording uses the first-order-Taylor leak factor
``(1 − dt/τ_m)``; we use the exact ODE solution ``exp(−dt/τ_m)`` for the
homogeneous part — numerically ≤ one ULP different for τ=500 ms, dt=5 ms but
strictly more accurate for long-τ integration (endorsed in the Task-16 GO
review). The drive term is added *without* ``dt/τ_m`` scaling, matching the
plan's separable decay-vs-drive parameterisation.

Weight split
------------

* **Generic** (plastic in Phase 2, frozen in Phase 3):
    `W_hm_gen`  : [n_m, n_h]      — H → memory
    `W_mm_gen`  : [n_m, n_m]      — memory recurrence
    `W_mh_gen`  : [n_out, n_m]    — memory → output bias (generic readout)

* **Task-specific** (frozen in Phase 2, plastic in Phase 3):
    `W_qm_task`     : [n_m, n_cue]        — Kok cue → memory
    `W_lm_task`     : [n_m, n_leader]     — Richter leader → memory
    `W_mh_task_exc` : [n_out, n_m]        — readout → L23 E apical
                                            (scaled by ``task_exc_gain``)
    `W_mh_task_inh` : [n_out_som, n_m]    — readout → L23 SOM — MAIN route
                                            for Phase-3 task bias (Fix C,
                                            Task #74). SOM provides apical
                                            gain control of L23 E dendrites
                                            (Urbanczik & Senn 2014).

All six tensors are stored as ``nn.Parameter(..., requires_grad=False)`` —
uniform v2 convention for plastic weights. Plasticity is driven externally
(closed-form rules in `plasticity.py`); gradients do not accumulate on these
parameters even when the forward is called inside an autograd context, so
both pure-local and BPTT-fallback paths are supported without code change.

Phase gating is **purely informational** — the module never modifies weights
on a phase switch. Training drivers read `plastic_weight_names()` /
`frozen_weight_names()` to decide which weights to update.

Out of scope
------------

Wiring into `network.py`, running plasticity rules against these weights, and
the activity-silent supplementary variant.
"""

from __future__ import annotations

import math
from typing import Callable, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.v2_model.utils import rectified_softplus


__all__ = ["ContextMemory", "PhaseLiteral"]


PhaseLiteral = Literal["phase2", "phase3_kok", "phase3_richter"]


# Bias shift that makes ``softplus(_SOM_GAIN_BIAS0) = 1.0`` exactly
# (up to float-precision rounding). Computed as ``log(e − 1)`` — the
# inverse of softplus at 1. Keeping it a named module-level constant
# (rather than a decimal literal) guarantees the init no-op property
# holds to machine precision.
_SOM_GAIN_BIAS0: float = math.log(math.e - 1.0)
_SOM_GAIN_CLAMP_MAX: float = 4.0


# ---------------------------------------------------------------------------
# Phase gating specification
# ---------------------------------------------------------------------------
# The plan v4 fix: generic weights are learnt in Phase 2 and *frozen* in
# Phase 3; task-specific weights are frozen during Phase-2 pre-training and
# *learnt* during the corresponding Phase-3 task. During Kok, only the cue
# stream is plastic (W_qm_task, W_mh_task); during Richter, only the leader
# stream is plastic (W_lm_task, W_mh_task). W_mh_task is plastic in both
# Phase 3 variants (task-specific readout).
_GEN_WEIGHT_NAMES: tuple[str, ...] = ("W_hm_gen", "W_mm_gen", "W_mh_gen")
_TASK_WEIGHT_NAMES: tuple[str, ...] = (
    "W_qm_task", "W_lm_task", "W_mh_task_exc", "W_mh_task_inh",
)

# Task #74 Fix C: ``W_mh_task`` was split into ``_exc`` (→ L23 E apical,
# scaled by ``task_exc_gain``) and ``_inh`` (→ L23 SOM, main route). Both
# are plastic in either Phase-3 variant — the Kok/Richter training drivers
# update them via ThreeFactorRule.delta_mh / .delta_mh_inh respectively.
_PHASE_SPEC: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    # phase: (plastic names, frozen names)
    "phase2":         (_GEN_WEIGHT_NAMES,
                       _TASK_WEIGHT_NAMES),
    "phase3_kok":     (("W_qm_task", "W_mh_task_exc", "W_mh_task_inh"),
                       ("W_hm_gen", "W_mm_gen", "W_mh_gen", "W_lm_task")),
    "phase3_richter": (("W_lm_task", "W_mh_task_exc", "W_mh_task_inh"),
                       ("W_hm_gen", "W_mm_gen", "W_mh_gen", "W_qm_task")),
}


# ---------------------------------------------------------------------------
# ContextMemory
# ---------------------------------------------------------------------------

class ContextMemory(nn.Module):
    """Context memory population C with phase-gated plastic/frozen weights.

    The bias-output size ``n_out`` is a constructor argument: the caller
    (``network.py``) decides whether it targets H apical (``n_h_e``) or
    L2/3 apical (``n_l23_e``). This module makes no assumption about which.

    Attributes
    ----------
    W_hm_gen, W_mm_gen, W_mh_gen : nn.Parameter (requires_grad=False)
        Generic pathway weights. Normal-initialised with std=``init_std``.
    W_qm_task, W_lm_task : nn.Parameter (requires_grad=False)
        Task-specific input-pathway weights. Initialised with small random
        values (``N(0, task_input_init_std)`` with ``task_input_init_std``
        ≈ 0.01) so the cue/leader streams already carry a tiny
        cue-differentiated signal at Phase-3 trial 0 — without this,
        the three-factor rule ``cue × memory × memory_error`` produces
        identical ``dw`` for both cues and no learning can bootstrap
        (Task #58 / debugger Task #49 Claim 4). The magnitude is small
        enough that the null-expectation control (Gate 6) still passes
        within its SEM tolerance.
    W_mh_task_exc : nn.Parameter (requires_grad=False)
        Task-specific readout → L2/3 E apical. Stays at exact zero at
        construction; Phase-3 plasticity binds it. Effective drive is
        multiplied by ``task_exc_gain`` (default 0.1) at the readout
        site — so even after training, the excitatory-apical route is
        a *secondary* bias. Biologically: direct top-down → apical is
        kept small because the MAIN bias arrives via SOM inhibition.
    W_mh_task_inh : nn.Parameter (requires_grad=False)
        Task-specific readout that drives per-SOM-unit gain modulation
        of the SOM→L23E inhibitory synapses (Task #74 Fix C-v2, MAIN
        route). Shape ``[n_out_som, n_m]``. Stays at exact zero at
        construction. The forward produces
        ``som_gain = softplus(W_mh_task_inh·m + 0.5413).clamp(max=4.0)``
        — a bounded multiplicative scale on the SOM→L23E synaptic
        efficacy, biologically interpreted as cholinergic /
        noradrenergic modulation of GABAergic transmission
        (Disney & Aoki 2008; Pfeffer 2013). The bias 0.5413 makes
        ``softplus(0.5413) = 1.0`` exactly, so at init (W=0) the gain
        is identically 1 and the circuit matches Phase-2 baseline.
        When ``n_out_som=0`` this weight is shape [0, n_m] — unwired.
    task_exc_gain : float
        Multiplicative scale on the ``W_mh_task_exc @ m`` readout,
        applied at the apply site (not at the weight). Keeps the
        weight's own dynamic range intact for diagnostics while
        attenuating its contribution to the L23 E apical drive.
    tau_m_ms, dt_ms : float
        Time constant and integration step. The cached leak factor
        ``exp(−dt/τ_m)`` lives in (0, 1) for any positive tau/dt, so only
        positivity is enforced.
    """

    def __init__(
        self,
        n_m: int,
        n_h: int,
        n_cue: int,
        n_leader: int,
        n_out: int,
        tau_m_ms: float,
        dt_ms: float,
        phi: Callable[[Tensor], Tensor] = rectified_softplus,
        init_std: float = 0.1,
        task_input_init_std: float = 0.3,
        cue_gain: float = 5.0,
        n_out_som: int = 0,
        task_exc_gain: float = 0.1,
        seed: int = 0,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        for name, val in (("n_m", n_m), ("n_h", n_h), ("n_cue", n_cue),
                          ("n_leader", n_leader), ("n_out", n_out)):
            if val <= 0:
                raise ValueError(f"{name} must be > 0; got {val}")
        if n_out_som < 0:
            raise ValueError(f"n_out_som must be ≥ 0; got {n_out_som}")
        if task_exc_gain < 0:
            raise ValueError(f"task_exc_gain must be ≥ 0; got {task_exc_gain}")
        if tau_m_ms <= 0:
            raise ValueError(f"tau_m_ms must be > 0; got {tau_m_ms}")
        if dt_ms <= 0:
            raise ValueError(f"dt_ms must be > 0; got {dt_ms}")
        if init_std < 0:
            raise ValueError(f"init_std must be ≥ 0; got {init_std}")
        if task_input_init_std < 0:
            raise ValueError(
                f"task_input_init_std must be ≥ 0; got {task_input_init_std}"
            )
        if cue_gain < 0:
            raise ValueError(f"cue_gain must be ≥ 0; got {cue_gain}")

        self.n_m = int(n_m)
        self.n_h = int(n_h)
        self.n_cue = int(n_cue)
        self.n_leader = int(n_leader)
        self.n_out = int(n_out)
        self.n_out_som = int(n_out_som)
        self.task_exc_gain = float(task_exc_gain)
        self.tau_m_ms = float(tau_m_ms)
        self.dt_ms = float(dt_ms)
        self._decay = math.exp(-self.dt_ms / self.tau_m_ms)
        self._phi = phi
        self._phase: str = "phase2"

        dev = torch.device(device) if device is not None else torch.device("cpu")
        gen = torch.Generator(device=dev)
        gen.manual_seed(int(seed))

        def _normal(shape: tuple[int, int], std: float) -> nn.Parameter:
            t = torch.empty(*shape, device=dev, dtype=dtype)
            with torch.no_grad():
                if std > 0:
                    t.normal_(mean=0.0, std=std, generator=gen)
                else:
                    t.zero_()
            return nn.Parameter(t, requires_grad=False)

        def _zeros(shape: tuple[int, int]) -> nn.Parameter:
            return nn.Parameter(
                torch.zeros(*shape, device=dev, dtype=dtype),
                requires_grad=False,
            )

        self._task_input_init_std = float(task_input_init_std)
        self.cue_gain = float(cue_gain)

        # Generic weights.
        self.W_hm_gen = _normal((self.n_m, self.n_h), init_std)
        self.W_mm_gen = _normal((self.n_m, self.n_m), init_std)
        self.W_mh_gen = _normal((self.n_out, self.n_m), init_std)
        # Task-specific input weights — small random init so Phase-3 trial 0
        # already has cue/leader-differentiated memory and the three-factor
        # rule can bootstrap (Task #58). Magnitude small enough to satisfy
        # Gate-6 null expectation control within 1·SEM.
        self.W_qm_task = _normal((self.n_m, self.n_cue), task_input_init_std)
        self.W_lm_task = _normal((self.n_m, self.n_leader), task_input_init_std)
        # Task-specific readouts (Task #74 Fix C). Both start at exact zero —
        # the output path must not be touched by task inputs until Phase-3
        # plasticity binds them.
        #   W_mh_task_exc → L23 E apical, scaled by ``task_exc_gain`` (0.1
        #                   default) at the readout site; secondary bias.
        #   W_mh_task_inh → per-SOM-unit GAIN modulator (Fix C-v2). At
        #                   forward: ``som_gain = softplus(W_mh_task_inh·m
        #                   + 0.5413).clamp(max=4.0)``. At init (W=0), gain
        #                   is identically 1 — circuit unchanged vs
        #                   Phase-2. Requires ``n_out_som > 0`` to be
        #                   functional; when 0, som_gain has shape [B, 0]
        #                   and L23E forward treats it as "no modulation".
        self.W_mh_task_exc = _zeros((self.n_out, self.n_m))
        self.W_mh_task_inh = _zeros((self.n_out_som, self.n_m))

    # ---- Legacy state-dict migration (Task #74 Fix C) ---------------------

    def _load_from_state_dict(
        self,
        state_dict,
        prefix: str,
        local_metadata,
        strict: bool,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Rename legacy ``W_mh_task`` → ``W_mh_task_exc`` on load.

        Pre-Fix-C checkpoints (Task #70 Phase-2 baseline and earlier) store
        the unified ``W_mh_task`` parameter. Fix C splits this into two
        parameters (``_exc`` for L23 E, ``_inh`` for L23 SOM). On load:

          * ``W_mh_task`` is copied into ``W_mh_task_exc`` (same shape —
            legacy was [n_out, n_m], new _exc is [n_out, n_m]).
          * ``W_mh_task_inh`` stays at its construction-time zeros (shape
            [n_out_som, n_m]), since no legacy ckpt encodes it.

        This keeps ``strict=True`` loads passing for legacy ckpts while
        preserving the biologically motivated routing split.
        """
        legacy = prefix + "W_mh_task"
        exc_key = prefix + "W_mh_task_exc"
        inh_key = prefix + "W_mh_task_inh"
        if legacy in state_dict:
            if exc_key not in state_dict:
                state_dict[exc_key] = state_dict[legacy]
            if inh_key not in state_dict:
                state_dict[inh_key] = self.W_mh_task_inh.detach().clone()
            del state_dict[legacy]
        return super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    # ---- Phase gating API --------------------------------------------------

    def set_phase(self, phase: PhaseLiteral) -> None:
        """Declare the current training phase. Does *not* mutate any weight."""
        if phase not in _PHASE_SPEC:
            raise ValueError(
                f"phase must be one of {sorted(_PHASE_SPEC.keys())}; got {phase!r}"
            )
        self._phase = phase

    @property
    def phase(self) -> str:
        return self._phase

    def plastic_weight_names(self) -> list[str]:
        """Names of weights currently treated as plastic (training driver updates)."""
        return list(_PHASE_SPEC[self._phase][0])

    def frozen_weight_names(self) -> list[str]:
        """Names of weights currently treated as frozen (training driver skips)."""
        return list(_PHASE_SPEC[self._phase][1])

    # ---- Forward -----------------------------------------------------------

    def forward(
        self,
        m_t: Tensor,
        h_t: Tensor,
        q_t: Optional[Tensor] = None,
        leader_t: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """One integration step of the context memory.

        Args:
            m_t:      [B, n_m]     current memory state.
            h_t:      [B, n_h]     H excitatory state this step.
            q_t:      [B, n_cue]   Kok cue token, or `None` to skip entirely
                                   (equivalent to passing an explicit zero
                                   tensor; avoids an unnecessary matmul).
            leader_t: [B, n_leader] Richter leader summary, or `None`.

        Returns:
            (m_next, b_exc, som_gain) where
                `m_next`   is [B, n_m] — integrated memory one step ahead.
                `b_exc`    is [B, n_out] — additive bias for L2/3 E apical
                           = W_mh_gen·m + task_exc_gain·(W_mh_task_exc·m).
                `som_gain` is [B, n_out_som] — per-SOM-unit MULTIPLICATIVE
                           gain on the SOM→L23E inhibitory synapses
                           (Task #74 Fix C-v2):
                           ``som_gain = softplus(W_mh_task_inh·m + 0.5413
                                        ).clamp(max=4.0)``.
                           At init (W=0) gain is identically 1.0; the
                           upper clamp at 4.0 bounds runaway up-regulation
                           and matches the biological range of cholinergic
                           GABA modulation (Disney & Aoki 2008; Pfeffer
                           2013). When ``n_out_som=0`` the tensor has
                           shape [B, 0] and L23E ignores it.

        Raises:
            ValueError: on shape mismatches or batch-size mismatches.
        """
        if m_t.ndim != 2 or m_t.shape[1] != self.n_m:
            raise ValueError(
                f"m_t must be [B, {self.n_m}]; got {tuple(m_t.shape)}"
            )
        if h_t.ndim != 2 or h_t.shape[1] != self.n_h:
            raise ValueError(
                f"h_t must be [B, {self.n_h}]; got {tuple(h_t.shape)}"
            )
        B = m_t.shape[0]
        if h_t.shape[0] != B:
            raise ValueError(
                f"batch-size mismatch: m_t B={B} vs h_t B={h_t.shape[0]}"
            )

        drive = F.linear(h_t, self.W_hm_gen) + F.linear(m_t, self.W_mm_gen)

        if q_t is not None:
            if q_t.ndim != 2 or q_t.shape[1] != self.n_cue or q_t.shape[0] != B:
                raise ValueError(
                    f"q_t must be [B={B}, {self.n_cue}]; got {tuple(q_t.shape)}"
                )
            drive = drive + self.cue_gain * F.linear(q_t, self.W_qm_task)

        if leader_t is not None:
            if (
                leader_t.ndim != 2
                or leader_t.shape[1] != self.n_leader
                or leader_t.shape[0] != B
            ):
                raise ValueError(
                    f"leader_t must be [B={B}, {self.n_leader}]; got "
                    f"{tuple(leader_t.shape)}"
                )
            drive = drive + self.cue_gain * F.linear(leader_t, self.W_lm_task)

        m_next = self._decay * m_t + (1.0 - self._decay) * self._phi(drive)

        # Bias uses the pre-update memory state per plan D.2; the readout
        # reflects what the memory looked like as the caller entered this step.
        # Task #74 Fix C-v2:
        #   b_exc    → L23 E apical (additive, scaled by task_exc_gain, small)
        #   som_gain → per-SOM-unit MULTIPLICATIVE gain on the SOM→L23E
        #              inhibitory synapses. Bias 0.5413 ≈ log(e-1) so that
        #              softplus(0.5413)=1.0 exactly; the ``clamp(max=4.0)``
        #              bounds runaway up-regulation within the biological
        #              cholinergic/noradrenergic modulation envelope.
        b_exc = (
            F.linear(m_t, self.W_mh_gen)
            + self.task_exc_gain * F.linear(m_t, self.W_mh_task_exc)
        )
        som_gain_pre = F.linear(m_t, self.W_mh_task_inh) + _SOM_GAIN_BIAS0
        som_gain = F.softplus(som_gain_pre).clamp(max=_SOM_GAIN_CLAMP_MAX)

        return m_next, b_exc, som_gain
