"""Held-out evaluation utilities built on full retained trajectories."""

from __future__ import annotations

from .config import ExperimentConfig
from .observation import ObservationPool
from .types import SimulationOutput, WindowSummary


class TrajectorySummarizer:
    def __init__(self, config: ExperimentConfig) -> None:
        self.windows = {
            "early": config.windows.early,
            "middle": config.windows.middle,
            "late": config.windows.late,
        }

    def summarize(self, simulation: SimulationOutput) -> WindowSummary:
        summaries: dict[str, dict[str, object]] = {}
        for window_name, (start, end) in self.windows.items():
            window_slice = slice(start - 1, end)
            summaries[window_name] = {state_name: state[:, window_slice].mean(dim=1) for state_name, state in simulation.states.items()}
        return WindowSummary(summaries=summaries)


def evaluate_with_observations(simulation: SimulationOutput, config: ExperimentConfig) -> tuple[SimulationOutput, WindowSummary]:
    pool = ObservationPool(config)
    simulation.observations = pool.apply_all(simulation.states["l23_readout"])
    summary = TrajectorySummarizer(config).summarize(simulation)
    return simulation, summary
