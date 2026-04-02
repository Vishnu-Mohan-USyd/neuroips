"""Typed configuration models for the phase-1 expectation codebase."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class GeometryConfig:
    n_orientations: int = 12
    periodicity_deg: float = 180.0

    def __post_init__(self) -> None:
        if self.n_orientations <= 0:
            raise ValueError("n_orientations must be positive")
        if self.periodicity_deg != 180.0:
            raise ValueError("phase-1 orientation geometry must remain 180-degree periodic")

    @property
    def periodicity_degrees(self) -> float:
        return self.periodicity_deg

    @property
    def bin_width_deg(self) -> float:
        return self.periodicity_deg / self.n_orientations


@dataclass
class SequenceConfig:
    n_steps: int = 12
    prestim_steps: int = 2
    context_count: int = 3
    expected_transition_probability: float = 0.8
    orthogonal_event_probability: float = 0.25
    seed: int = 1234
    counterbalance_physical_transitions: bool = True

    def __post_init__(self) -> None:
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive")
        if self.prestim_steps <= 0 or self.prestim_steps >= self.n_steps:
            raise ValueError("prestim_steps must be positive and smaller than n_steps")
        if not 0.5 < self.expected_transition_probability < 1.0:
            raise ValueError("expected_transition_probability must lie in (0.5, 1.0)")

    @property
    def prestimulus_steps(self) -> int:
        return self.prestim_steps


@dataclass
class WindowConfig:
    early: tuple[int, int] = (1, 3)
    middle: tuple[int, int] = (4, 7)
    late: tuple[int, int] = (8, 12)

    def __post_init__(self) -> None:
        for start, end in (self.early, self.middle, self.late):
            if start < 1 or end < start:
                raise ValueError("windows use one-based inclusive indexing")
        if not (self.early[1] < self.middle[0] <= self.middle[1] < self.late[0] <= self.late[1]):
            raise ValueError("windows must be ordered and non-overlapping")

    def as_dict(self) -> dict[str, tuple[int, int]]:
        return {
            "early": self.early,
            "middle": self.middle,
            "late": self.late,
        }


@dataclass
class ScaffoldConfig:
    ff_gain: float = 1.0
    recurrent_gain: float = 0.15
    template_gain: float = 0.4
    pv_norm_strength: float = 0.2
    som_inhibition_strength: float = 0.35
    adapt_tau: float = 0.8
    adapt_gain: float = 0.25
    process_noise_std: float = 0.01
    observation_noise_std: float = 0.0
    context_insertion_point: str = "l23_additive"
    context_global_gain_max: float = 0.25

    def __post_init__(self) -> None:
        if self.adapt_tau <= 0.0:
            raise ValueError("adapt_tau must be positive")
        if self.process_noise_std < 0.0 or self.observation_noise_std < 0.0:
            raise ValueError("noise terms must be non-negative")


@dataclass
class ContextConfig:
    hidden_dim: int = 16
    use_precision: bool = True
    causal_mask: bool = True
    context_dim: int = 4
    predictive_target: str = "next_orientation_distribution"

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0 or self.context_dim <= 0:
            raise ValueError("context dimensions must be positive")
        if self.predictive_target != "next_orientation_distribution":
            raise ValueError("phase-1 context target must remain next_orientation_distribution")


@dataclass
class TrainingConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    n_epochs: int = 10
    energy_weight: float = 0.0
    homeostasis_weight: float = 0.0
    predictive_objective: str = "next_orientation_distribution"
    heldout_assays_only: bool = True

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.n_epochs <= 0:
            raise ValueError("training batch size and n_epochs must be positive")
        if self.predictive_objective != "next_orientation_distribution":
            raise ValueError("phase-1 training objective must remain predictive-only")


@dataclass
class ObservationConfig:
    schemes: tuple[str, ...] = ("identity", "gaussian_orientation_bank")
    bank_width_deg: float = 30.0

    def __post_init__(self) -> None:
        allowed = {"identity", "gaussian_orientation_bank"}
        if not self.schemes or any(scheme not in allowed for scheme in self.schemes):
            raise ValueError("observation schemes must be drawn from the fixed phase-1 set")
        if len(set(self.schemes)) != len(self.schemes):
            raise ValueError("observation schemes must be unique")
        if self.bank_width_deg <= 0.0:
            raise ValueError("bank_width_deg must be positive")


@dataclass
class ArtifactConfig:
    root_dir: Path = Path("artifacts")
    save_full_trajectories: bool = True
    save_manifest: bool = True

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)


@dataclass
class AssayConfig:
    heldout_evaluation: bool = True
    primary_metrics: tuple[str, ...] = (
        "mean_suppression",
        "tuning_slope",
        "tuning_peak_change",
        "tuning_width_change",
        "decoder_accuracy",
        "rsa_distance",
        "prestimulus_template_specificity",
        "omission_template_specificity",
        "energy_telemetry",
    )

    def __post_init__(self) -> None:
        if not self.heldout_evaluation:
            raise ValueError("phase-1 assays must remain held out from training")
        if len(set(self.primary_metrics)) != len(self.primary_metrics):
            raise ValueError("primary_metrics must not contain duplicates")


@dataclass
class ExperimentConfig:
    name: str = "phase1_core"
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    windows: WindowConfig = field(default_factory=WindowConfig)
    scaffold: ScaffoldConfig = field(default_factory=ScaffoldConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)
    assays: AssayConfig = field(default_factory=AssayConfig)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "geometry": {
                "n_orientations": self.geometry.n_orientations,
                "periodicity_deg": self.geometry.periodicity_deg,
            },
            "sequence": {
                "n_steps": self.sequence.n_steps,
                "prestim_steps": self.sequence.prestim_steps,
                "context_count": self.sequence.context_count,
                "expected_transition_probability": self.sequence.expected_transition_probability,
                "orthogonal_event_probability": self.sequence.orthogonal_event_probability,
                "seed": self.sequence.seed,
                "counterbalance_physical_transitions": self.sequence.counterbalance_physical_transitions,
            },
            "windows": self.windows.as_dict(),
            "scaffold": {
                "ff_gain": self.scaffold.ff_gain,
                "recurrent_gain": self.scaffold.recurrent_gain,
                "template_gain": self.scaffold.template_gain,
                "pv_norm_strength": self.scaffold.pv_norm_strength,
                "som_inhibition_strength": self.scaffold.som_inhibition_strength,
                "adapt_tau": self.scaffold.adapt_tau,
                "adapt_gain": self.scaffold.adapt_gain,
                "process_noise_std": self.scaffold.process_noise_std,
                "observation_noise_std": self.scaffold.observation_noise_std,
                "context_insertion_point": self.scaffold.context_insertion_point,
                "context_global_gain_max": self.scaffold.context_global_gain_max,
            },
            "context": {
                "hidden_dim": self.context.hidden_dim,
                "use_precision": self.context.use_precision,
                "causal_mask": self.context.causal_mask,
                "context_dim": self.context.context_dim,
                "predictive_target": self.context.predictive_target,
            },
            "training": {
                "learning_rate": self.training.learning_rate,
                "weight_decay": self.training.weight_decay,
                "batch_size": self.training.batch_size,
                "n_epochs": self.training.n_epochs,
                "energy_weight": self.training.energy_weight,
                "homeostasis_weight": self.training.homeostasis_weight,
                "predictive_objective": self.training.predictive_objective,
                "heldout_assays_only": self.training.heldout_assays_only,
            },
            "observation": {
                "schemes": list(self.observation.schemes),
                "bank_width_deg": self.observation.bank_width_deg,
            },
            "artifacts": {
                "root_dir": str(self.artifacts.root_dir),
                "save_full_trajectories": self.artifacts.save_full_trajectories,
                "save_manifest": self.artifacts.save_manifest,
            },
            "assays": {
                "heldout_evaluation": self.assays.heldout_evaluation,
                "primary_metrics": list(self.assays.primary_metrics),
            },
        }


def _build_dataclass(cls: type[Any], payload: dict[str, Any] | None) -> Any:
    payload = payload or {}
    return cls(**payload)


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return ExperimentConfig(
        name=payload.get("name", "phase1_core"),
        geometry=_build_dataclass(GeometryConfig, payload.get("geometry")),
        sequence=_build_dataclass(SequenceConfig, payload.get("sequence")),
        windows=_build_dataclass(WindowConfig, payload.get("windows")),
        scaffold=_build_dataclass(ScaffoldConfig, payload.get("scaffold")),
        context=_build_dataclass(ContextConfig, payload.get("context")),
        training=_build_dataclass(TrainingConfig, payload.get("training")),
        observation=_build_dataclass(ObservationConfig, payload.get("observation")),
        artifacts=_build_dataclass(ArtifactConfig, payload.get("artifacts")),
        assays=_build_dataclass(AssayConfig, payload.get("assays")),
    )


def make_phase1_core_config() -> ExperimentConfig:
    return ExperimentConfig()
