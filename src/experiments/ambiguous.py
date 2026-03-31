"""P3: Ambiguous Stimuli.

CW/CCW context -> ambiguous probe:
    - Mixture: 50/50 blend of two orientations at expected +/- 15 deg
    - Low contrast: single grating at expected but c=0.15

Tests perceptual bias: does expectation attract or repel the percept?
Compared against clear (full contrast, expected orientation) control.
"""

from __future__ import annotations

import torch
from torch import Tensor

from src.config import ModelConfig
from src.model.network import LaminarV1V2Network
from src.stimulus.gratings import make_ambiguous_stimulus
from src.experiments.paradigm_base import (
    ParadigmBase, TrialConfig, TrialSet,
)


class AmbiguousParadigm(ParadigmBase):
    """P3: ambiguous stimuli (mixture / low-contrast) after predictable context."""

    paradigm_name = "ambiguous"

    def __init__(
        self,
        net: LaminarV1V2Network,
        model_cfg: ModelConfig,
        trial_cfg: TrialConfig | None = None,
        transition_step: float = 15.0,
        ambiguous_offset: float = 15.0,
        low_contrast: float = 0.15,
    ):
        super().__init__(net, model_cfg, trial_cfg)
        self.transition_step = transition_step
        self.ambiguous_offset = ambiguous_offset
        self.low_contrast = low_contrast

    def generate_trials(self, n_trials: int, seed: int) -> dict[str, TrialSet]:
        gen = torch.Generator()
        gen.manual_seed(seed)
        period = self.cfg.orientation_range
        n_ctx = self.trial_cfg.n_context
        step = self.transition_step

        trial_sets: dict[str, TrialSet] = {}

        for rule in ["cw", "ccw"]:
            sign = 1.0 if rule == "cw" else -1.0
            seqs_mix, seqs_low, seqs_clear = [], [], []

            for _ in range(n_trials):
                start = torch.rand(1, generator=gen).item() * period
                context = [(start + i * sign * step) % period for i in range(n_ctx)]
                expected = (context[-1] + sign * step) % period

                # Mixture probe
                seqs_mix.append(self._build_mixture_sequence(context, expected))
                # Low contrast probe
                seqs_low.append(self.build_stimulus_sequence(
                    context, expected, probe_contrast=self.low_contrast))
                # Clear control
                seqs_clear.append(self.build_stimulus_sequence(context, expected))

            trial_sets[f"{rule}_mixture"] = TrialSet(stimulus=torch.cat(seqs_mix))
            trial_sets[f"{rule}_low_contrast"] = TrialSet(stimulus=torch.cat(seqs_low))
            trial_sets[f"{rule}_clear"] = TrialSet(stimulus=torch.cat(seqs_clear))

        return trial_sets

    def _build_mixture_sequence(
        self, context_oris: list[float], expected_ori: float,
    ) -> Tensor:
        """Build sequence with 50/50 mixture probe at expected +/- offset."""
        tc = self.trial_cfg
        T = self.n_timesteps
        N = self.cfg.n_orientations
        period = self.cfg.orientation_range
        stim = torch.zeros(1, T, N)

        for i, ori in enumerate(context_oris):
            onset = i * (tc.steps_on + tc.steps_isi)
            s = self.make_grating(ori)
            stim[0, onset:onset + tc.steps_on] = s

        theta1 = (expected_ori + self.ambiguous_offset) % period
        theta2 = (expected_ori - self.ambiguous_offset) % period
        probe = make_ambiguous_stimulus(
            torch.tensor([theta1]), torch.tensor([theta2]),
            torch.tensor([tc.contrast]),
            n_orientations=N, sigma=self.cfg.sigma_ff,
            n=self.cfg.naka_rushton_n, c50=self.cfg.naka_rushton_c50,
            period=period,
        )
        po = self.probe_onset
        stim[0, po:po + tc.steps_on] = probe

        return stim

    def _get_trial_info(self) -> dict:
        return {
            "ambiguous_offset": self.ambiguous_offset,
            "low_contrast": self.low_contrast,
        }
