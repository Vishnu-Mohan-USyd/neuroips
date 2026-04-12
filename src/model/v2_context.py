"""V2 context inference module: GRU-based latent-state inference."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.config import ModelConfig


class V2ContextModule(nn.Module):
    """V2 context inference module.

    NOT another orientation ring — a latent-state inference module.
    Inputs: L4 activity and/or L2/3 activity (previous step) + cue + task_state.

    Two modes controlled by cfg.feedback_mode:

    'fixed' (legacy): Outputs q_pred [B, N], pi_pred [B, 1],
        state_logits [B, 3], h_v2 [B, H].

    'emergent' (learned prior): Outputs mu_pred [B, N] (softmax
        orientation prior distribution), pi_pred [B, 1], h_v2 [B, H].
        mu_pred IS the prior — V2 directly outputs a full orientation
        distribution, enabling genuine prestimulus priors during ISI.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        n = cfg.n_orientations
        self.v2_input_mode = cfg.v2_input_mode
        self.hidden_dim = cfg.v2_hidden_dim
        self.pi_max = cfg.pi_max
        self.feedback_mode = cfg.feedback_mode
        self.use_per_regime_feedback = cfg.use_per_regime_feedback

        # Input dimension depends on mode
        if self.v2_input_mode == 'l23':
            input_dim = n + n + 2        # L2/3 + cue + task_state
        elif self.v2_input_mode == 'l4':
            input_dim = n + n + 2        # L4 + cue + task_state
        elif self.v2_input_mode == 'l4_l23':
            input_dim = n + n + n + 2    # L4 + L2/3 + cue + task_state
        else:
            raise ValueError(f"Unknown v2_input_mode: {self.v2_input_mode}")

        self.gru = nn.GRUCell(input_dim, cfg.v2_hidden_dim)

        # Output heads depend on feedback_mode
        if self.feedback_mode == 'emergent':
            # Learned prior: full orientation distribution
            self.head_mu = nn.Linear(cfg.v2_hidden_dim, n)  # -> softmax -> mu_pred [B, N]
            if self.use_per_regime_feedback:
                # Task #9 / Fix 2 / Network_both: per-regime feedback heads.
                # One Linear per task regime (focused vs routine), gated at
                # forward time by `task_state[:, 0]` (focused) and
                # `task_state[:, 1]` (routine). The blended feedback signal is
                #     fb = task_state[:,0:1] * head_feedback_focused(h_v2)
                #        + task_state[:,1:2] * head_feedback_routine(h_v2)
                # so when task_state is one-hot exactly one head supplies the
                # signal; under a mixture (e.g. soft per-presentation routing)
                # the blend is convex. Motivation, per debugger evidence for
                # Fix 2: a single shared head must serve two opposing
                # objectives (focused: amplify sensory; routine: suppress
                # sensory) — separating the heads removes that conflict and
                # lets each regime learn its own additive feedback projection.
                #
                # We deliberately do NOT also construct `self.head_feedback`
                # here so that any stale code path referencing the legacy
                # name fails loudly instead of silently using random init.
                self.head_feedback_focused = nn.Linear(cfg.v2_hidden_dim, n)  # [B, N] raw
                self.head_feedback_routine = nn.Linear(cfg.v2_hidden_dim, n)  # [B, N] raw
                # Init for symmetry: copy focused weights into routine so
                # both heads start identical. The optimizer then pushes them
                # apart under the per-presentation Markov task_state routing.
                with torch.no_grad():
                    self.head_feedback_routine.weight.copy_(
                        self.head_feedback_focused.weight
                    )
                    self.head_feedback_routine.bias.copy_(
                        self.head_feedback_focused.bias
                    )
            else:
                # Legacy (Network_mm): single shared feedback head.
                self.head_feedback = nn.Linear(cfg.v2_hidden_dim, n)  # [B, N] raw
        else:
            # Legacy: full orientation distribution + state logits
            self.head_q = nn.Linear(cfg.v2_hidden_dim, n)        # -> softmax -> q_pred
            self.head_state = nn.Linear(cfg.v2_hidden_dim, 3)    # -> raw logits

        # Precision head (shared by both modes)
        self.head_pi = nn.Linear(cfg.v2_hidden_dim, 1)       # -> softplus + clamp
        nn.init.constant_(self.head_pi.bias, 0.0)

        # Rescue 3: VIP drive head. Drives VIP interneurons from V2 context.
        # Only instantiated when use_vip=True. Output is raw (no activation),
        # VIPRing applies rectified_softplus internally.
        if getattr(cfg, 'use_vip', False):
            self.head_vip = nn.Linear(cfg.v2_hidden_dim, n)  # [B, N]

    def forward(
        self,
        r_l4: Tensor,
        r_l23_prev: Tensor,
        cue: Tensor,
        task_state: Tensor,
        h_v2_prev: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None]:
        """One step of V2 context inference.

        Args:
            r_l4: [B, N] -- current L4 rates (stable, pre-feedback).
            r_l23_prev: [B, N] -- L2/3 rates from PREVIOUS timestep.
            cue: [B, N] -- cue input (zeros by default).
            task_state: [B, 2] -- task relevance state.
            h_v2_prev: [B, H] -- previous GRU hidden state.

        Returns (feedback_mode == 'emergent'):
            mu_pred: [B, N] -- predicted orientation prior (softmax, sums to 1).
            pi_pred: [B, 1] -- prediction precision in [0, pi_max].
            feedback_signal: [B, N] -- raw additive feedback signal (no activation).
            h_v2: [B, H] -- updated GRU hidden state.
            vip_drive: [B, N] or None -- raw VIP drive (only when use_vip=True).

        Returns (feedback_mode == 'fixed'):
            q_pred: [B, N] -- predicted orientation distribution (sums to 1).
            pi_pred: [B, 1] -- prediction precision in [0, pi_max].
            state_logits: [B, 3] -- raw logits for CW/CCW/neutral.
            h_v2: [B, H] -- updated GRU hidden state.
            vip_drive: [B, N] or None -- raw VIP drive (only when use_vip=True).
        """
        if self.v2_input_mode == 'l23':
            v2_input = torch.cat([r_l23_prev, cue, task_state], dim=-1)
        elif self.v2_input_mode == 'l4':
            v2_input = torch.cat([r_l4, cue, task_state], dim=-1)
        elif self.v2_input_mode == 'l4_l23':
            v2_input = torch.cat([r_l4, r_l23_prev, cue, task_state], dim=-1)
        h_v2 = self.gru(v2_input, h_v2_prev)  # [B, H]

        pi_pred = torch.clamp(F.softplus(self.head_pi(h_v2)), max=self.pi_max)  # [B, 1]

        # Rescue 3: VIP drive from head_vip (None when use_vip=False).
        vip_drive = self.head_vip(h_v2) if hasattr(self, 'head_vip') else None

        if self.feedback_mode == 'emergent':
            mu_pred = F.softmax(self.head_mu(h_v2), dim=-1)  # [B, N]
            if self.use_per_regime_feedback:
                # Per-regime feedback gated by task_state. task_state is [B, 2]
                # with column 0 = focused, column 1 = routine (one-hot under
                # the Markov per-presentation routing used by simple_dual,
                # but the convex-combination form supports soft mixtures).
                fb_focused = self.head_feedback_focused(h_v2)     # [B, N] raw
                fb_routine = self.head_feedback_routine(h_v2)     # [B, N] raw
                feedback_signal = (
                    task_state[:, 0:1] * fb_focused
                    + task_state[:, 1:2] * fb_routine
                )                                                  # [B, N] raw
            else:
                # Legacy (Network_mm): single shared head, ignores task_state.
                feedback_signal = self.head_feedback(h_v2)         # [B, N] raw
            return mu_pred, pi_pred, feedback_signal, h_v2, vip_drive
        else:
            q_pred = F.softmax(self.head_q(h_v2), dim=-1)  # [B, N]
            state_logits = self.head_state(h_v2)             # [B, 3]
            return q_pred, pi_pred, state_logits, h_v2, vip_drive
