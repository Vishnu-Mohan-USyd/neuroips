"""Spiking V2 context module (LSNN — Bellec et al. 2018, NeurIPS).

Parallel to `src/model/v2_context.py::V2ContextModule` (GRU-based) but
implemented as a Long Short-term Memory SNN: 80 leaky-integrate-and-fire
neurons split into three sub-populations, with a subset carrying ALIF
adaptive thresholds. Readout heads consume the exponentially-filtered spike
trace `x_v2` and produce exactly the same tensors the rate model's V2 head
produces, so callers can swap the two modules interchangeably.

Architecture (evidence pack §B.6, final cite-able spec)
-------------------------------------------------------
    SpikingV2Context (80 neurons):
        input_proj  : Linear(input_dim -> 80)      # same input_dim as rate GRU
        W_rec       : Linear(80 -> 80, bias=False) # full-rank recurrent, NO Dale split
        neuron composition:
          [0:40]   = LIF excitatory   (tau_mem=20, V_thresh=1)
          [40:60]  = ALIF excitatory  (tau_mem=20, V_thresh=1, beta_adapt=1.8, tau_a=200)
          [60:80]  = LIF inhibitory   (tau_mem=20, V_thresh=1)
        readout heads (from filtered trace x_v2, alpha_filter=exp(-1/10)=0.9048):
            head_mu       : Linear(80 -> 36) + softmax       (emergent)
            head_feedback : Linear(80 -> 36) raw             (emergent)
            head_q        : Linear(80 -> 36) + softmax       (fixed)
            head_state    : Linear(80 -> 3)  raw             (fixed)
            head_pi       : Linear(80 -> 1)  + softplus + clamp(max=pi_max)
        surrogate: ATan(slope=25, dampen=0.3)

Why this is a **5x downscale of Bellec 2018 TIMIT** (page 4 of the paper):
    |                 | Bellec TIMIT | Plan V2 | Ratio |
    | Total           | 400          | 80      | 5x    |
    | LIF excitatory  | 200          | 40      | 5x    |
    | ALIF (excitatory)| 100         | 20      | 5x    |
    | LIF inhibitory  | 100          | 20      | 5x    |
    | E:I ratio       | 3:1          | 3:1     | match |
    | Adaptive frac.  | 25%          | 25%     | match |

Dale's law on W_rec
-------------------
Bellec 2018 TIMIT does **not** enforce column-sign constraints on W_rec, and
the LSNN-official reference implementation also keeps W_rec as a plain dense
matrix. Lead Ruling 2 (2026-04-10) explicitly states: *skip Dale's law on V2
W_rec*. Evidence pack §B.6 ("Dale's law on W_rec: Optional for initial pass
... Recommendation: skip Dale's law on W_rec for initial implementation").
If the three-regime result fails to emerge, revisit.

The 3:1 E:I ratio is still imposed implicitly through *which neurons* are
used as excitatory vs inhibitory by downstream populations in
`SpikingLaminarV1V2Network`.

ALIF dynamics (Bellec 2018 eq. 1-2, {0,1} spike encoding)
---------------------------------------------------------
For every neuron i in [0, 80):

    v[t] = beta_mem * v[t-1] + drive[t] - z[t-1] * V_thresh    (subtract reset)
    B[t] = V_thresh + beta_adapt * b[t-1]                       (eq. 1; b=0 for LIF neurons)
    z[t] = atan_surrogate(v[t] - B[t], slope, dampen)
    x[t] = alpha_filter * x[t-1] + z[t]
    b[t] = rho_adapt * b[t-1] + (1 - rho_adapt) * z[t]          (eq. 2)
           * adapt_mask                                          (b stays 0 for LIF slices)

Drive aggregation:
    drive = input_proj(v2_input) + W_rec(x_prev)

where `x_prev` is the filtered-trace recurrence input. Using the filtered
trace (not the raw spike) as the recurrent input matches snnTorch's
`RLeaky` convention and smooths the recurrence so that a single spike does
not propagate as a unit impulse on the next step — important for stability
on long BPTT rollouts. The filtered trace also matches what the readout
heads consume, keeping the module internally consistent.

API parity with rate `V2ContextModule`
--------------------------------------
Constructor takes `(model_cfg, spiking_cfg)` — mirrors the spiking
populations in `src/spiking/populations.py`. Same `v2_input_mode` branching,
same input concatenation order, same `feedback_mode` two-head split
('emergent' vs 'fixed'). The forward return tuple has the same element
order as `V2ContextModule.forward` with one change: the final element is a
`dict` (the full LSNN state) rather than a single hidden-state tensor, so
callers can round-trip all four state variables (v, z, x, b) across steps.

Evidence
--------
- `/home/vishnu/.claude/plans/snn_port_evidence_pack.md`
    - §B.1-B.7   LSNN architecture, beta_adapt, tau_a, tau_mem, Dale ruling
    - §A.4       Bellec eq. 1-2 derivation with {0,1} encoding
    - §A.5       ATan surrogate as pure tensor ops for torch.compile
- `/home/vishnu/.claude/plans/quirky-humming-giraffe.md`
    - line 25    "Custom ALIF (40 LIF + 20 ALIF exc + 20 LIF inh)" = 80 neurons
    - lines 38-42 ALIF dynamics
- Bellec et al. 2018, NeurIPS, "Long short-term memory and learning-to-learn
  in networks of spiking neurons", §2 eq. 1-2 (page 2), §4 TIMIT (page 4).
- LSNN-official reference implementation:
    * bin/tutorial_sequential_mnist_with_LSNN.py (beta=1.8, dampening=0.3)
    * lsnn/spiking_models.py ALIF class (dampening=0.3)
- Lead rulings (2026-04-10):
    * Ruling 2: skip Dale's law on V2 W_rec
    * Ruling 3: fix tau_adapt = 200 ms
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.config import ModelConfig, SpikingConfig
from src.spiking.surrogate import atan_surrogate


class SpikingV2Context(nn.Module):
    """LSNN-based V2 context inference module.

    Drop-in spiking replacement for `V2ContextModule` (rate GRU). Consumes
    filtered spike traces from V1 L4 and L2/3 instead of rates, runs one
    LSNN step with ALIF-fraction adaptation, and emits the same readout
    tensors (`mu_pred`/`q_pred`, `pi_pred`, `feedback_signal`/`state_logits`)
    plus the LSNN state dict for the next step.

    Args:
        model_cfg:    Rate-model config (supplies `n_orientations`,
            `v2_input_mode`, `pi_max`, `feedback_mode`).
        spiking_cfg:  Spiking config (supplies `n_lsnn_*`, `tau_mem_v2`,
            `tau_adapt`, `lsnn_adapt_beta`, `spike_filter_alpha`, surrogate
            hyperparameters).
    """

    def __init__(self, model_cfg: ModelConfig, spiking_cfg: SpikingConfig):
        super().__init__()

        # ---- Parameters ripped from the configs ----
        n = model_cfg.n_orientations
        self.n_orientations = n
        self.v2_input_mode = model_cfg.v2_input_mode
        self.pi_max = float(model_cfg.pi_max)
        self.feedback_mode = model_cfg.feedback_mode

        # LSNN sizes (Bellec 2018 TIMIT 5x downscale — evidence pack §B.2).
        self.n_v2 = int(spiking_cfg.n_lsnn_neurons)
        self.n_exc = int(spiking_cfg.n_lsnn_exc)
        self.n_adapt = int(spiking_cfg.n_lsnn_adaptive)
        self.n_inh = int(spiking_cfg.n_lsnn_inh)

        # Neuron-slice boundaries: [0:40] LIF exc, [40:60] ALIF exc, [60:80] LIF inh.
        exc_end = self.n_exc
        adapt_end = self.n_exc + self.n_adapt
        inh_end = self.n_v2
        self.slice_exc = slice(0, exc_end)
        self.slice_adapt = slice(exc_end, adapt_end)
        self.slice_inh = slice(adapt_end, inh_end)

        # Input dimension must match the rate V2ContextModule.
        if self.v2_input_mode == "l23":
            input_dim = n + n + 2            # L2/3 + cue + task_state
        elif self.v2_input_mode == "l4":
            input_dim = n + n + 2            # L4 + cue + task_state
        elif self.v2_input_mode == "l4_l23":
            input_dim = n + n + n + 2        # L4 + L2/3 + cue + task_state
        else:
            raise ValueError(f"Unknown v2_input_mode: {self.v2_input_mode}")
        self.input_dim = int(input_dim)

        # ---- SNN dynamics constants ----
        self.V_thresh = float(spiking_cfg.V_thresh)
        self.beta_mem = math.exp(-1.0 / spiking_cfg.tau_mem_v2)           # 0.9512 at tau=20
        self.alpha_filter = float(spiking_cfg.spike_filter_alpha)          # 0.9048 at tau=10
        self.rho_adapt = math.exp(-1.0 / spiking_cfg.tau_adapt)            # 0.99501 at tau=200
        self.beta_adapt = float(spiking_cfg.lsnn_adapt_beta)               # 1.8 (Bellec sMNIST)
        self.surrogate_slope = float(spiking_cfg.surrogate_slope)
        self.surrogate_dampen = float(spiking_cfg.surrogate_dampen)

        # ---- Layers ----
        # Input projection: raw Linear (no Dale split; nothing about the stimulus
        # / cue / task_state inputs implies a sign constraint on their fan-out).
        self.input_proj = nn.Linear(self.input_dim, self.n_v2)

        # Recurrent projection: plain Linear(80, 80), bias=False. Per Lead Ruling 2
        # and evidence pack §B.6, Dale's law is NOT enforced on W_rec. The 3:1 E:I
        # neuron split is structural — it only matters for how downstream layers
        # read from the LSNN, not for the intrinsic recurrence.
        self.W_rec = nn.Linear(self.n_v2, self.n_v2, bias=False)

        # Readout heads (fed from the filtered trace x_v2 — see forward).
        if self.feedback_mode == "emergent":
            self.head_mu = nn.Linear(self.n_v2, n)             # -> softmax -> mu_pred
            self.head_feedback = nn.Linear(self.n_v2, n)       # -> raw feedback [B, N]
        else:
            # Legacy 'fixed' mode: orientation distribution + 3-state logits.
            self.head_q = nn.Linear(self.n_v2, n)              # -> softmax -> q_pred
            self.head_state = nn.Linear(self.n_v2, 3)          # -> raw logits [B, 3]

        self.head_pi = nn.Linear(self.n_v2, 1)                 # -> softplus + clamp
        nn.init.constant_(self.head_pi.bias, 0.0)

        # ---- Static tensors (registered as buffers for device tracking) ----
        # adapt_mask marks the ALIF slice [40:60]. Used to zero out b updates on
        # the LIF sub-populations so b stays identically zero there (which makes
        # B[t] = V_thresh on non-adaptive neurons — plain LIF).
        adapt_mask = torch.zeros(self.n_v2)
        adapt_mask[self.slice_adapt] = 1.0
        self.register_buffer("adapt_mask", adapt_mask)

    # --------------------------------------------------------------------
    # State
    # --------------------------------------------------------------------

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> dict:
        """Zero-initialized LSNN state for a fresh rollout.

        Returns:
            dict with keys {v, z, x, b}, each shape [batch_size, n_v2].
        """
        dev = device or torch.device("cpu")
        z = lambda: torch.zeros(batch_size, self.n_v2, device=dev, dtype=dtype)
        return {"v": z(), "z": z(), "x": z(), "b": z()}

    # --------------------------------------------------------------------
    # Forward
    # --------------------------------------------------------------------

    def forward(
        self,
        x_l4: Tensor,
        x_l23_prev: Tensor,
        cue: Tensor,
        task_state: Tensor,
        state: dict,
    ) -> tuple:
        """One LSNN step.

        Args:
            x_l4:         [B, N] filtered L4 spike trace (current step).
            x_l23_prev:   [B, N] filtered L2/3 spike trace (previous step).
            cue:          [B, N] cue input (zeros by default).
            task_state:   [B, 2] task-relevance state.
            state: dict   {v, z, x, b}, each [B, n_v2].

        Returns (feedback_mode == "emergent"):
            mu_pred:        [B, N]  softmax orientation prior.
            pi_pred:        [B, 1]  precision in [0, pi_max].
            feedback_signal:[B, N]  raw additive feedback (no activation).
            new_state:      dict    full LSNN state for next step.

        Returns (feedback_mode == "fixed"):
            q_pred:         [B, N]  softmax orientation distribution.
            pi_pred:        [B, 1]  precision in [0, pi_max].
            state_logits:   [B, 3]  raw CW/CCW/neutral logits.
            new_state:      dict    full LSNN state for next step.
        """
        v_prev = state["v"]
        z_prev = state["z"]
        x_prev = state["x"]
        b_prev = state["b"]

        # ---- Assemble V2 input vector (same concat as rate V2ContextModule) ----
        if self.v2_input_mode == "l23":
            v2_input = torch.cat([x_l23_prev, cue, task_state], dim=-1)
        elif self.v2_input_mode == "l4":
            v2_input = torch.cat([x_l4, cue, task_state], dim=-1)
        else:  # l4_l23
            v2_input = torch.cat([x_l4, x_l23_prev, cue, task_state], dim=-1)

        # ---- Drive: feedforward + filtered recurrence ----
        # Using filtered trace x_prev (not raw z_prev) on the recurrent path:
        #   (a) matches snnTorch RLeaky convention,
        #   (b) gives the recurrence a smooth short-horizon memory (instead of
        #       injecting a single unit impulse per spike), which is what we
        #       want for a context / prior readout module,
        #   (c) keeps the signal consumed by the readout heads (also x) and the
        #       signal consumed by W_rec in the same representational space.
        drive = self.input_proj(v2_input) + self.W_rec(x_prev)             # [B, 80]

        # ---- LIF subtract-reset membrane update ----
        v = self.beta_mem * v_prev + drive - z_prev * self.V_thresh        # [B, 80]

        # ---- Adaptive threshold (Bellec eq. 1) ----
        # For LIF sub-populations b_prev is identically zero (enforced below),
        # so B_thresh = V_thresh there. For the ALIF slice [40:60] the
        # threshold is elevated by beta_adapt * b_prev.
        B_thresh = self.V_thresh + self.beta_adapt * b_prev                # [B, 80]

        # ---- Surrogate-gradient spike ----
        z = atan_surrogate(
            v - B_thresh,
            slope=self.surrogate_slope,
            dampen=self.surrogate_dampen,
        )                                                                   # [B, 80]

        # ---- Adaptation state update (Bellec eq. 2, {0,1} encoding) ----
        # Update all rows uniformly, then mask out the LIF sub-populations so
        # their b stays zero across steps (B_thresh stays at V_thresh -> plain LIF).
        b = self.rho_adapt * b_prev + (1.0 - self.rho_adapt) * z           # [B, 80]
        b = b * self.adapt_mask                                             # [B, 80] * [80]

        # ---- Exponential trace filter (readout-facing signal) ----
        x = self.alpha_filter * x_prev + z                                  # [B, 80]

        # ---- Readout heads, computed from the freshly-updated trace ----
        pi_pred = torch.clamp(
            F.softplus(self.head_pi(x)), max=self.pi_max
        )                                                                   # [B, 1]

        new_state = {"v": v, "z": z, "x": x, "b": b}

        if self.feedback_mode == "emergent":
            mu_pred = F.softmax(self.head_mu(x), dim=-1)                    # [B, N]
            feedback_signal = self.head_feedback(x)                         # [B, N]
            return mu_pred, pi_pred, feedback_signal, new_state
        else:
            q_pred = F.softmax(self.head_q(x), dim=-1)                      # [B, N]
            state_logits = self.head_state(x)                               # [B, 3]
            return q_pred, pi_pred, state_logits, new_state
