"""HMM-based transition sequence generator for CW/CCW/neutral orientation sequences."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import torch
from torch import Tensor


class HMMState(IntEnum):
    """Latent states of the orientation HMM."""
    CW = 0
    CCW = 1
    NEUTRAL = 2


@dataclass
class SequenceMetadata:
    """Metadata for a generated stimulus sequence."""
    orientations: Tensor      # [B, T] true orientations in degrees
    states: Tensor            # [B, T] latent HMM state indices
    contrasts: Tensor         # [B, T] contrast values
    is_ambiguous: Tensor      # [B, T] bool: whether presentation is ambiguous
    task_states: Tensor       # [B, T, 2] task state vectors
    cues: Tensor              # [B, T, N] cue input (zeros by default)


def build_transition_matrix(
    p_self: float = 0.95,
) -> Tensor:
    """Build the 3x3 HMM state transition matrix.

    States: CW, CCW, NEUTRAL.
    p_self = probability of staying in the same state.
    Remaining probability split equally among other states.

    Returns:
        Transition matrix [3, 3] where T[i, j] = P(state_t+1=j | state_t=i).
    """
    p_switch = (1.0 - p_self) / 2.0
    T = torch.full((3, 3), p_switch)
    T[0, 0] = p_self
    T[1, 1] = p_self
    T[2, 2] = p_self
    return T


def sample_state_sequence(
    n_steps: int,
    p_self: float = 0.95,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample a sequence of HMM latent states.

    Args:
        n_steps: Length of sequence.
        p_self: Self-transition probability.
        generator: Optional RNG for reproducibility.

    Returns:
        State indices [n_steps] with values in {0, 1, 2}.
    """
    T = build_transition_matrix(p_self)
    states = torch.empty(n_steps, dtype=torch.long)

    # Uniform initial state
    states[0] = torch.randint(3, (1,), generator=generator).item()

    for t in range(1, n_steps):
        probs = T[states[t - 1]]
        states[t] = torch.multinomial(probs, 1, generator=generator).item()

    return states


def generate_orientation_sequence(
    n_steps: int,
    p_self: float = 0.95,
    p_transition_cw: float = 0.80,
    p_transition_ccw: float = 0.80,
    n_anchors: int = 12,
    jitter_range: float = 7.5,
    transition_step: float = 15.0,
    period: float = 180.0,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Generate a sequence of orientations driven by an HMM.

    The HMM controls latent state (CW/CCW/neutral). Given the state:
      - CW: next orientation = prev + transition_step (with prob p_transition_cw)
      - CCW: next orientation = prev - transition_step (with prob p_transition_ccw)
      - NEUTRAL: next orientation drawn uniformly from anchors

    Orientations are drawn from n_anchors canonical positions with continuous
    jitter of +/- jitter_range degrees.

    Args:
        n_steps: Number of presentations.
        p_self: HMM self-transition probability.
        p_transition_cw: Prob of +step given CW state.
        p_transition_ccw: Prob of -step given CCW state.
        n_anchors: Number of canonical anchor orientations.
        jitter_range: Max jitter around anchors (degrees).
        transition_step: Step size for CW/CCW transitions (degrees).
        period: Orientation period.
        generator: Optional RNG.

    Returns:
        orientations: [n_steps] orientation in degrees (continuous, with jitter).
        states: [n_steps] latent state indices.
    """
    states = sample_state_sequence(n_steps, p_self, generator)

    anchor_step = period / n_anchors
    anchors = torch.arange(n_anchors, dtype=torch.float32) * anchor_step  # [n_anchors]

    orientations = torch.empty(n_steps, dtype=torch.float32)

    # Initial: random anchor + jitter
    idx = torch.randint(n_anchors, (1,), generator=generator).item()
    jitter = (torch.rand(1, generator=generator).item() * 2 - 1) * jitter_range
    orientations[0] = (anchors[idx] + jitter) % period

    for t in range(1, n_steps):
        state = states[t].item()
        prev = orientations[t - 1].item()

        if state == HMMState.CW:
            if torch.rand(1, generator=generator).item() < p_transition_cw:
                base = (prev + transition_step) % period
            else:
                base = anchors[torch.randint(n_anchors, (1,), generator=generator).item()].item()
        elif state == HMMState.CCW:
            if torch.rand(1, generator=generator).item() < p_transition_ccw:
                base = (prev - transition_step) % period
            else:
                base = anchors[torch.randint(n_anchors, (1,), generator=generator).item()].item()
        else:
            base = anchors[torch.randint(n_anchors, (1,), generator=generator).item()].item()

        jitter = (torch.rand(1, generator=generator).item() * 2 - 1) * jitter_range
        orientations[t] = (base + jitter) % period

    return orientations, states


class HMMSequenceGenerator:
    """Full-featured HMM sequence generator for training/evaluation.

    3-state HMM: CW, CCW, NEUTRAL.
    Generates sequences of oriented gratings with transition structure.

    Features:
    - 12 canonical orientations (15 deg spacing) with continuous jitter +/-7.5 deg
    - CW: next = theta + 15 deg (p=0.80), else uniform
    - CCW: next = theta - 15 deg (p=0.80), else uniform
    - Neutral: uniform next
    - State self-transition: p_self=0.95
    - Variable contrast (configurable range)
    - 10-20% ambiguous presentations (orientation mixtures)
    - Occasional transition reliability drops (p_transition -> 0.5)
    - Both task states interleaved (~50/50)
    - Cue input null by default
    """

    def __init__(
        self,
        n_orientations: int = 36,
        p_self: float = 0.95,
        p_transition_cw: float = 0.80,
        p_transition_ccw: float = 0.80,
        n_anchors: int = 12,
        jitter_range: float = 7.5,
        transition_step: float = 15.0,
        period: float = 180.0,
        contrast_range: tuple[float, float] = (0.15, 1.0),
        ambiguous_fraction: float = 0.15,
        ambiguous_offset: float = 15.0,
        reliability_drop_prob: float = 0.05,
        reliability_drop_value: float = 0.50,
        cue_dim: int = 2,
    ):
        self.n_orientations = n_orientations
        self.p_self = p_self
        self.p_transition_cw = p_transition_cw
        self.p_transition_ccw = p_transition_ccw
        self.n_anchors = n_anchors
        self.jitter_range = jitter_range
        self.transition_step = transition_step
        self.period = period
        self.contrast_range = contrast_range
        self.ambiguous_fraction = ambiguous_fraction
        self.ambiguous_offset = ambiguous_offset
        self.reliability_drop_prob = reliability_drop_prob
        self.reliability_drop_value = reliability_drop_value
        self.cue_dim = cue_dim

    def generate(
        self,
        batch_size: int,
        seq_length: int,
        generator: torch.Generator | None = None,
    ) -> SequenceMetadata:
        """Generate a full batch of stimulus sequences with all metadata.

        Args:
            batch_size: Number of sequences.
            seq_length: Presentations per sequence.
            generator: Optional RNG for reproducibility.

        Returns:
            SequenceMetadata with orientations, states, contrasts, is_ambiguous,
            task_states, and cues.
        """
        all_oris = []
        all_states = []

        for _ in range(batch_size):
            # Occasional reliability drops: per-sequence coin flip
            p_cw = self.p_transition_cw
            p_ccw = self.p_transition_ccw
            if torch.rand(1, generator=generator).item() < self.reliability_drop_prob:
                p_cw = self.reliability_drop_value
                p_ccw = self.reliability_drop_value

            oris, sts = generate_orientation_sequence(
                n_steps=seq_length,
                p_self=self.p_self,
                p_transition_cw=p_cw,
                p_transition_ccw=p_ccw,
                n_anchors=self.n_anchors,
                jitter_range=self.jitter_range,
                transition_step=self.transition_step,
                period=self.period,
                generator=generator,
            )
            all_oris.append(oris)
            all_states.append(sts)

        orientations = torch.stack(all_oris)   # [B, T]
        states = torch.stack(all_states)        # [B, T]

        # Random contrasts
        lo, hi = self.contrast_range
        contrasts = lo + (hi - lo) * torch.rand(batch_size, seq_length, generator=generator)

        # Ambiguous presentations: mark ~ambiguous_fraction of positions
        is_ambiguous = torch.rand(batch_size, seq_length, generator=generator) < self.ambiguous_fraction

        # For ambiguous stimuli, reduce contrast to low level
        # (the actual mixing happens at stimulus generation time)
        contrasts[is_ambiguous] = torch.clamp(contrasts[is_ambiguous], max=0.20)

        # Task states: ~50/50 interleaved, one per sequence
        # [1, 0] = orientation-relevant, [0, 1] = orientation-irrelevant
        task_relevant = torch.rand(batch_size, generator=generator) < 0.5  # [B]
        task_states = torch.zeros(batch_size, seq_length, 2)
        for b in range(batch_size):
            if task_relevant[b]:
                task_states[b, :, 0] = 1.0
            else:
                task_states[b, :, 1] = 1.0

        # Cue input: zeros by default
        cues = torch.zeros(batch_size, seq_length, self.n_orientations)

        return SequenceMetadata(
            orientations=orientations,
            states=states,
            contrasts=contrasts,
            is_ambiguous=is_ambiguous,
            task_states=task_states,
            cues=cues,
        )


# Convenience function for simple batch generation (backward compatible)
def generate_batch_sequences(
    batch_size: int,
    seq_length: int,
    p_self: float = 0.95,
    p_transition_cw: float = 0.80,
    p_transition_ccw: float = 0.80,
    n_anchors: int = 12,
    jitter_range: float = 7.5,
    transition_step: float = 15.0,
    period: float = 180.0,
    contrast_range: tuple[float, float] = (0.15, 1.0),
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Generate a batch of orientation sequences with random contrasts.

    Simpler interface that returns (orientations, states, contrasts).
    For full metadata including task_state and ambiguity, use HMMSequenceGenerator.
    """
    all_oris = []
    all_states = []

    for _ in range(batch_size):
        oris, sts = generate_orientation_sequence(
            n_steps=seq_length,
            p_self=p_self,
            p_transition_cw=p_transition_cw,
            p_transition_ccw=p_transition_ccw,
            n_anchors=n_anchors,
            jitter_range=jitter_range,
            transition_step=transition_step,
            period=period,
            generator=generator,
        )
        all_oris.append(oris)
        all_states.append(sts)

    orientations = torch.stack(all_oris)
    states = torch.stack(all_states)

    lo, hi = contrast_range
    contrasts = lo + (hi - lo) * torch.rand(batch_size, seq_length, generator=generator)

    return orientations, states, contrasts
