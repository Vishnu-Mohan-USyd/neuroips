"""P1: Hidden-State Transitions — core paradigm.

8 context presentations establishing CW/CCW rule, then a probe at
various deviations from expected orientation.  Dense probe offsets
(0, 15, 30, 45, 60, 90 deg) for fine-grained suppression profiling.

Neutral baseline: same last context stimulus and probe orientation,
but preceding context lacks predictive structure (only the last
stimulus is matched, destroying the rule while preserving recent
adaptation history).

200 trials per condition, counterbalanced across starting orientations.
"""

from __future__ import annotations

import torch

from src.config import ModelConfig
from src.model.network import LaminarV1V2Network
from src.experiments.paradigm_base import (
    ParadigmBase, TrialConfig, TrialSet,
)


class HiddenStateParadigm(ParadigmBase):
    """P1: CW/CCW/neutral with dense probe offsets."""

    paradigm_name = "hidden_state"

    def __init__(
        self,
        net: LaminarV1V2Network,
        model_cfg: ModelConfig,
        trial_cfg: TrialConfig | None = None,
        probe_deviations: list[float] | None = None,
        transition_step: float = 15.0,
    ):
        super().__init__(net, model_cfg, trial_cfg)
        self.probe_deviations = probe_deviations or [0.0, 15.0, 30.0, 45.0, 60.0, 90.0]
        self.transition_step = transition_step
        self._trial_metadata: dict = {}

    def generate_trials(self, n_trials: int, seed: int) -> dict[str, TrialSet]:
        gen = torch.Generator()
        gen.manual_seed(seed)
        period = self.cfg.orientation_range
        n_ctx = self.trial_cfg.n_context
        step = self.transition_step

        trial_sets: dict[str, TrialSet] = {}
        metadata: dict = {}

        # CW and CCW conditions
        for rule in ["cw", "ccw"]:
            sign = 1.0 if rule == "cw" else -1.0
            for dev in self.probe_deviations:
                cond = f"{rule}_dev{dev:.0f}"
                seqs = []
                info: dict[str, list] = {"start_oris": [], "expected_oris": [], "probe_oris": []}

                for _ in range(n_trials):
                    start = torch.rand(1, generator=gen).item() * period
                    context = [(start + i * sign * step) % period for i in range(n_ctx)]
                    expected = (context[-1] + sign * step) % period
                    probe = (expected + dev) % period

                    seqs.append(self.build_stimulus_sequence(context, probe))
                    info["start_oris"].append(start)
                    info["expected_oris"].append(expected)
                    info["probe_oris"].append(probe)

                trial_sets[cond] = TrialSet(stimulus=torch.cat(seqs))
                metadata[cond] = info

        # Neutral: matched last context + probe, random earlier context
        for dev in self.probe_deviations:
            cond = f"neutral_dev{dev:.0f}"
            seqs = []
            info = {"start_oris": [], "expected_oris": [], "probe_oris": []}

            for _ in range(n_trials):
                start = torch.rand(1, generator=gen).item() * period
                last = (start + (n_ctx - 1) * step) % period
                expected = (last + step) % period
                probe = (expected + dev) % period

                # Random preceding context, matched last stimulus
                context = [torch.rand(1, generator=gen).item() * period
                           for _ in range(n_ctx - 1)]
                context.append(last)

                seqs.append(self.build_stimulus_sequence(context, probe))
                info["start_oris"].append(start)
                info["expected_oris"].append(expected)
                info["probe_oris"].append(probe)

            trial_sets[cond] = TrialSet(stimulus=torch.cat(seqs))
            metadata[cond] = info

        self._trial_metadata = metadata
        return trial_sets

    def _get_trial_info(self) -> dict:
        return {
            "probe_deviations": self.probe_deviations,
            "transition_step": self.transition_step,
            "trial_metadata": self._trial_metadata,
        }
