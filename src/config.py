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

    # V2 context
    v2_hidden_dim: int = 16
    pi_max: float = 5.0

    # Deep template
    template_gain: float = 1.0

    # Feedback mechanism
    mechanism: MechanismType = MechanismType.CENTER_SURROUND

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
    gradient_clip: float = 1.0
    stage2_contrast_range: Tuple[float, float] = (0.15, 1.0)
    ambiguous_fraction: float = 0.15

    # Loss weights
    lambda_sensory: float = 1.0
    lambda_pred: float = 0.5
    lambda_energy: float = 0.01
    lambda_homeo: float = 1.0

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


def load_config(path: str | Path = "config/defaults.yaml") -> tuple[ModelConfig, TrainingConfig, StimulusConfig]:
    """Load configuration from a YAML file and return typed config objects."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    model_raw = raw.get("model", {})
    # Convert mechanism string to enum
    mech_str = model_raw.pop("mechanism", "center_surround")
    model_cfg = ModelConfig(**model_raw, mechanism=MechanismType(mech_str))

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
        gradient_clip=stage2.get("gradient_clip", 1.0),
        stage2_contrast_range=tuple(stage2.get("contrast_range", [0.15, 1.0])),
        ambiguous_fraction=stage2.get("ambiguous_fraction", 0.15),
        lambda_sensory=train_raw.get("lambda_sensory", 1.0),
        lambda_pred=train_raw.get("lambda_pred", 0.5),
        lambda_energy=train_raw.get("lambda_energy", 0.01),
        lambda_homeo=train_raw.get("lambda_homeo", 1.0),
        batch_size=train_raw.get("batch_size", 32),
        seq_length=train_raw.get("seq_length", 50),
        steps_on=train_raw.get("steps_on", 8),
        steps_isi=train_raw.get("steps_isi", 4),
        n_seeds=train_raw.get("n_seeds", 5),
    )

    stim_raw = raw.get("stimulus", {})
    stim_cfg = StimulusConfig(**stim_raw)

    return model_cfg, training_cfg, stim_cfg
