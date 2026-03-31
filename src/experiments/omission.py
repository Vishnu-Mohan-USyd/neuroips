"""P2: Omission Trials.

10 stable context presentations -> blank (contrast=0) at position 11.
Tests whether the deep template maintains the expected orientation
representation during stimulus omission.

150 trials per condition (CW/CCW x omission/present).
"""

from __future__ import annotations

import torch

from src.config import ModelConfig
from src.model.network import LaminarV1V2Network
from src.experiments.paradigm_base import (
    ParadigmBase, TrialConfig, TrialSet,
)


class OmissionParadigm(ParadigmBase):
    """P2: omission trials after stable context."""

    paradigm_name = "omission"

    def __init__(
        self,
        net: LaminarV1V2Network,
        model_cfg: ModelConfig,
        trial_cfg: TrialConfig | None = None,
        transition_step: float = 15.0,
    ):
        tc = trial_cfg or TrialConfig(n_context=10, steps_post=12)
        super().__init__(net, model_cfg, tc)
        self.transition_step = transition_step

    @property
    def temporal_windows(self) -> dict[str, tuple[int, int]]:
        windows = super().temporal_windows
        po = self.probe_onset
        tc = self.trial_cfg
        windows["omission"] = (po, po + tc.steps_on + tc.steps_post)
        return windows

    def generate_trials(self, n_trials: int, seed: int) -> dict[str, TrialSet]:
        gen = torch.Generator()
        gen.manual_seed(seed)
        period = self.cfg.orientation_range
        n_ctx = self.trial_cfg.n_context
        step = self.transition_step

        trial_sets: dict[str, TrialSet] = {}

        for rule in ["cw", "ccw"]:
            sign = 1.0 if rule == "cw" else -1.0
            seqs_omit = []
            seqs_present = []

            for _ in range(n_trials):
                start = torch.rand(1, generator=gen).item() * period
                context = [(start + i * sign * step) % period for i in range(n_ctx)]
                expected = (context[-1] + sign * step) % period

                seqs_omit.append(self.build_stimulus_sequence(
                    context, expected, probe_contrast=0.0))
                seqs_present.append(self.build_stimulus_sequence(
                    context, expected))

            trial_sets[f"{rule}_omission"] = TrialSet(stimulus=torch.cat(seqs_omit))
            trial_sets[f"{rule}_present"] = TrialSet(stimulus=torch.cat(seqs_present))

        return trial_sets
