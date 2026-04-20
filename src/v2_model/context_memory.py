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

b_t     = W_mh^gen · m_t  +  W_mh^task · m_t``

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
    `W_qm_task` : [n_m, n_cue]    — Kok cue → memory         (zero-init)
    `W_lm_task` : [n_m, n_leader] — Richter leader → memory  (zero-init)
    `W_mh_task` : [n_out, n_m]    — task-specific readout    (zero-init)

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
_TASK_WEIGHT_NAMES: tuple[str, ...] = ("W_qm_task", "W_lm_task", "W_mh_task")

_PHASE_SPEC: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    # phase: (plastic names, frozen names)
    "phase2":         (_GEN_WEIGHT_NAMES,
                       _TASK_WEIGHT_NAMES),
    "phase3_kok":     (("W_qm_task", "W_mh_task"),
                       ("W_hm_gen", "W_mm_gen", "W_mh_gen", "W_lm_task")),
    "phase3_richter": (("W_lm_task", "W_mh_task"),
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
    W_mh_task : nn.Parameter (requires_grad=False)
        Task-specific output-path readout. Stays at exact zero at
        construction so the output bias is untouched by the task stream
        until Phase-3 plasticity binds it — this keeps the null control
        strictly conservative on the output side.
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
        seed: int = 0,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        for name, val in (("n_m", n_m), ("n_h", n_h), ("n_cue", n_cue),
                          ("n_leader", n_leader), ("n_out", n_out)):
            if val <= 0:
                raise ValueError(f"{name} must be > 0; got {val}")
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
        # Task-specific readout stays at exact zero — the output path must
        # not be touched by task inputs until Phase-3 plasticity binds it.
        self.W_mh_task = _zeros((self.n_out, self.n_m))

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
    ) -> tuple[Tensor, Tensor]:
        """One integration step of the context memory.

        Args:
            m_t:      [B, n_m]     current memory state.
            h_t:      [B, n_h]     H excitatory state this step.
            q_t:      [B, n_cue]   Kok cue token, or `None` to skip entirely
                                   (equivalent to passing an explicit zero
                                   tensor; avoids an unnecessary matmul).
            leader_t: [B, n_leader] Richter leader summary, or `None`.

        Returns:
            (m_next, b_t) where
                `m_next` is [B, n_m] — the integrated memory one step ahead.
                `b_t`    is [B, n_out] — additive bias to be injected at the
                         caller-chosen target (H apical or L2/3 apical).

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
        b_t = F.linear(m_t, self.W_mh_gen) + F.linear(m_t, self.W_mh_task)

        return m_next, b_t
