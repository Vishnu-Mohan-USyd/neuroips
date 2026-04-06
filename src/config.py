"""Configuration dataclasses and mechanism enum for the laminar V1-V2 model."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Tuple

import yaml


class MechanismType(str, Enum):
    """Feedback mechanism variants."""
    DAMPENING = "dampening"             # Model A: SOM inhibits AT expected
    SHARPENING = "sharpening"           # Model B: SOM inhibits broadly, sparing expected
    CENTER_SURROUND = "center_surround" # Model C: narrow excitation + broad inhibition
    ADAPTATION_ONLY = "adaptation_only" # Model D: no feedback
    PREDICTIVE_ERROR = "predictive_error"  # Model E: error = l4 - prediction


@dataclass
class ModelConfig:
    """Neural population and connectivity parameters."""
    # Orientation space
    n_orientations: int = 36
    orientation_range: float = 180.0

    # V1 L4
    sigma_ff: float = 12.0
    tau_l4: int = 5
    tau_adaptation: int = 200
    alpha_adaptation: float = 0.3
    adaptation_clamp: float = 10.0
    naka_rushton_n: float = 2.0
    naka_rushton_c50: float = 0.3

    # PV pool
    tau_pv: int = 5
    sigma_norm: float = 1.0

    # V1 L2/3
    tau_l23: int = 10
    sigma_rec: float = 15.0
    gain_rec: float = 0.3

    # SOM
    tau_som: int = 10

    # VIP interneurons (disinhibitory: VIP→SOM→L2/3)
    tau_vip: int = 10

    # V2 context
    v2_hidden_dim: int = 16
    pi_max: float = 5.0

    # Deep template
    template_gain: float = 1.0

    # Feedback mechanism
    mechanism: MechanismType = MechanismType.CENTER_SURROUND

    # Feedback mode: 'emergent' (learned basis) or 'fixed' (hardcoded mechanism)
    feedback_mode: str = 'emergent'

    # Transition step for analytical q_pred construction (degrees)
    transition_step: float = 15.0

    # Number of basis functions for emergent feedback operator
    n_basis: int = 7

    # V2 input mode: 'l23' (default), 'l4', or 'l4_l23'
    v2_input_mode: str = 'l23'

    # Numerical stability
    dt: int = 1

    @property
    def orientation_step(self) -> float:
        return self.orientation_range / self.n_orientations

    @property
    def preferred_orientations(self) -> List[float]:
        step = self.orientation_step
        return [i * step for i in range(self.n_orientations)]


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Stage 1
    stage1_n_steps: int = 2000
    stage1_lr: float = 1e-3
    stage1_contrast_range: Tuple[float, float] = (0.1, 1.0)

    # Stage 2
    stage2_n_steps: int = 80000
    stage2_lr_v2: float = 3e-4
    stage2_lr_feedback: float = 1e-4
    stage2_weight_decay: float = 1e-4
    stage2_warmup_steps: int = 1000
    stage2_burnin_steps: int = 5000
    stage2_ramp_steps: int = 5000
    gradient_clip: float = 1.0
    stage2_contrast_range: Tuple[float, float] = (0.15, 1.0)
    ambiguous_fraction: float = 0.15

    # Loss weights
    lambda_sensory: float = 1.0
    lambda_pred: float = 0.5
    lambda_energy: float = 0.01
    lambda_homeo: float = 1.0
    lambda_state: float = 0.25
    lambda_fb: float = 0.01          # L1 sparsity on emergent feedback weights
    lambda_surprise: float = 0.0     # Surprise detection loss (0 = disabled)
    lambda_error: float = 0.0        # Prediction error readout loss (0 = disabled)
    lambda_detection: float = 0.0    # Detection confirmation loss (0 = disabled)
    lambda_l4_sensory: float = 0.0  # L4 sensory readout weight (0 = disabled)
    lambda_mismatch: float = 0.0    # L2/3 mismatch detection weight (0 = disabled)
    lambda_sharp: float = 0.0       # Tuning sharpness: penalize L2/3 activity at flanks (0 = disabled)
    lambda_local_disc: float = 0.0  # Phase 4: local 5-way discrimination (expected vs ±1, ±2 neighbors). 0 = disabled.

    # Delta-SOM: bias-corrected softplus in EmergentFeedbackOperator
    delta_som: bool = False

    # Freeze V2 / use oracle predictor
    freeze_v2: bool = False
    oracle_pi: float = 1.0          # pi value when using oracle mode
    # Oracle template type (only used when freeze_v2=True):
    #   oracle_true    — q_pred bumped at next orientation given TRUE state (normal oracle)
    #   oracle_wrong   — q_pred bumped at next orientation given OPPOSITE state (CW<->CCW swap)
    #   oracle_random  — q_pred bumped at a random orientation, independent of stimulus
    #   oracle_uniform — q_pred is flat (uniform over orientations)
    oracle_template: str = "oracle_true"

    # Phase 5: width of the oracle prediction bump (degrees). Defaults to
    # sigma_ff (12.0) so existing configs are bitwise-identical. Lowering
    # this to e.g. 5.0 gives the feedback operator a NARROWER prior than
    # the feedforward tuning, which is the engineering handle for
    # expectation-driven sharpening.
    oracle_sigma: float = 12.0

    # Freeze orientation decoder in Stage 2 (for clean representational claims)
    freeze_decoder: bool = False

    # Phase 3: shift oracle q_pred by +1 presentation so that the template
    # acts as a prior about the CURRENT stimulus rather than a forecast of the
    # next one. Only used when freeze_v2=True. First presentation receives a
    # uniform prior (no valid history).
    oracle_shift_timing: bool = False

    # Stimulus noise (std of Gaussian noise added to population-coded stimulus in Stage 2)
    stimulus_noise: float = 0.0

    # Batching
    batch_size: int = 32
    seq_length: int = 50
    steps_on: int = 8
    steps_isi: int = 4

    # Seeds
    n_seeds: int = 5


@dataclass
class StimulusConfig:
    """Stimulus generation parameters."""
    n_states: int = 3
    p_transition_cw: float = 0.80
    p_transition_ccw: float = 0.80
    p_self: float = 0.95
    n_anchors: int = 12
    jitter_range: float = 7.5
    transition_step: float = 15.0
    cue_dim: int = 2
    task_state_dim: int = 2
    # Angular offset (degrees) between the two orientations that make up an
    # ambiguous mixture stimulus. Used by `build_stimulus_sequence` in
    # `src/training/trainer.py` to construct the second orientation of each
    # ambiguous presentation as `ori + ambiguous_offset`.
    ambiguous_offset: float = 15.0


def load_config(path: str | Path = "config/defaults.yaml") -> tuple[ModelConfig, TrainingConfig, StimulusConfig]:
    """Load configuration from a YAML file and return typed config objects."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    model_raw = raw.get("model", {})
    # Convert mechanism string to enum
    mech_str = model_raw.pop("mechanism", "center_surround")
    # Extract feedback_mode with default
    feedback_mode = model_raw.pop("feedback_mode", "emergent")
    model_cfg = ModelConfig(**model_raw, mechanism=MechanismType(mech_str), feedback_mode=feedback_mode)

    train_raw = raw.get("training", {})
    # Flatten nested stage configs
    stage1 = train_raw.pop("stage1", {})
    stage2 = train_raw.pop("stage2", {})
    training_cfg = TrainingConfig(
        stage1_n_steps=stage1.get("n_steps", 2000),
        stage1_lr=stage1.get("lr", 1e-3),
        stage1_contrast_range=tuple(stage1.get("contrast_range", [0.1, 1.0])),
        stage2_n_steps=stage2.get("n_steps", 80000),
        stage2_lr_v2=stage2.get("lr_v2", 3e-4),
        stage2_lr_feedback=stage2.get("lr_feedback", 1e-4),
        stage2_weight_decay=stage2.get("weight_decay", 1e-4),
        stage2_warmup_steps=stage2.get("warmup_steps", 1000),
        stage2_burnin_steps=stage2.get("burnin_steps", 5000),
        stage2_ramp_steps=stage2.get("ramp_steps", 5000),
        gradient_clip=stage2.get("gradient_clip", 1.0),
        stage2_contrast_range=tuple(stage2.get("contrast_range", [0.15, 1.0])),
        ambiguous_fraction=stage2.get("ambiguous_fraction", 0.15),
        lambda_sensory=train_raw.get("lambda_sensory", 1.0),
        lambda_pred=train_raw.get("lambda_pred", 0.5),
        lambda_energy=train_raw.get("lambda_energy", 0.01),
        lambda_homeo=train_raw.get("lambda_homeo", 1.0),
        lambda_state=train_raw.get("lambda_state", 0.25),
        lambda_fb=train_raw.get("lambda_fb", 0.01),
        lambda_surprise=train_raw.get("lambda_surprise", 0.0),
        lambda_error=train_raw.get("lambda_error", 0.0),
        lambda_detection=train_raw.get("lambda_detection", 0.0),
        lambda_l4_sensory=train_raw.get("lambda_l4_sensory", 0.0),
        lambda_mismatch=train_raw.get("lambda_mismatch", 0.0),
        lambda_sharp=train_raw.get("lambda_sharp", 0.0),
        lambda_local_disc=train_raw.get("lambda_local_disc", 0.0),
        delta_som=train_raw.get("delta_som", False),
        freeze_v2=train_raw.get("freeze_v2", False),
        freeze_decoder=train_raw.get("freeze_decoder", False),
        oracle_shift_timing=train_raw.get("oracle_shift_timing", False),
        oracle_pi=train_raw.get("oracle_pi", 1.0),
        oracle_template=train_raw.get("oracle_template", "oracle_true"),
        oracle_sigma=train_raw.get("oracle_sigma", 12.0),
        stimulus_noise=train_raw.get("stimulus_noise", 0.0),
        batch_size=train_raw.get("batch_size", 32),
        seq_length=train_raw.get("seq_length", 50),
        steps_on=train_raw.get("steps_on", 8),
        steps_isi=train_raw.get("steps_isi", 4),
        n_seeds=train_raw.get("n_seeds", 5),
    )

    stim_raw = raw.get("stimulus", {})
    stim_cfg = StimulusConfig(**stim_raw)

    return model_cfg, training_cfg, stim_cfg
