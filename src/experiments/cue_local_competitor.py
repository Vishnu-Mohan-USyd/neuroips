"""Cue-first local-competitor paradigm with prestimulus cueing.

Predictable context establishes an expected probe orientation. During the final
ISI before probe onset, an orientation-channel cue is presented that is valid,
neutral, or invalid with respect to the upcoming probe. The probe is either the
expected orientation or a local competitor offset by a small angle.
"""

from __future__ import annotations

import torch

from src.config import ModelConfig
from src.model.network import LaminarV1V2Network
from src.experiments.paradigm_base import (
    ParadigmBase, TrialConfig, TrialSet,
)


class CueLocalCompetitorParadigm(ParadigmBase):
    """Prestimulus cueing over local expected-vs-competitor probes."""

    paradigm_name = "cue_local_competitor"

    def __init__(
        self,
        net: LaminarV1V2Network,
        model_cfg: ModelConfig,
        trial_cfg: TrialConfig | None = None,
        transition_step: float = 15.0,
        competitor_offset: float = 10.0,
        cue_contrast: float = 1.0,
    ):
        super().__init__(net, model_cfg, trial_cfg)
        self.transition_step = transition_step
        self.competitor_offset = competitor_offset
        self.cue_contrast = cue_contrast

    def _build_prestimulus_cue(self, cue_ori: float | None) -> torch.Tensor:
        """Build a cue tensor active only during the prestimulus ISI."""
        T = self.n_timesteps
        N = self.cfg.n_orientations
        cue = torch.zeros(1, T, N)
        if cue_ori is None:
            return cue

        cue_start, cue_end = self.temporal_windows["prestimulus"]
        cue_frame = self.make_grating(cue_ori, contrast=self.cue_contrast)
        cue[0, cue_start:cue_end] = cue_frame
        return cue

    def generate_trials(self, n_trials: int, seed: int) -> dict[str, TrialSet]:
        gen = torch.Generator()
        gen.manual_seed(seed)
        period = self.cfg.orientation_range
        n_ctx = self.trial_cfg.n_context
        step = self.transition_step

        trial_sets: dict[str, TrialSet] = {}

        for rule in ["cw", "ccw"]:
            sign = 1.0 if rule == "cw" else -1.0
            for cue_kind in ["valid", "neutral", "invalid"]:
                for probe_kind in ["match", "competitor"]:
                    seqs = []
                    cues = []

                    for _ in range(n_trials):
                        start = torch.rand(1, generator=gen).item() * period
                        context = [(start + i * sign * step) % period for i in range(n_ctx)]
                        expected = (context[-1] + sign * step) % period
                        competitor = (expected + sign * self.competitor_offset) % period
                        probe = expected if probe_kind == "match" else competitor

                        if cue_kind == "valid":
                            cue_ori = probe
                        elif cue_kind == "invalid":
                            cue_ori = competitor if probe_kind == "match" else expected
                        else:
                            cue_ori = None

                        seqs.append(self.build_stimulus_sequence(context, probe))
                        cues.append(self._build_prestimulus_cue(cue_ori))

                    trial_sets[f"{rule}_{cue_kind}_{probe_kind}"] = TrialSet(
                        stimulus=torch.cat(seqs),
                        cue=torch.cat(cues),
                    )

        return trial_sets

    def _get_trial_info(self) -> dict:
        cue_start, cue_end = self.temporal_windows["prestimulus"]
        return {
            "transition_step": self.transition_step,
            "competitor_offset": self.competitor_offset,
            "cue_contrast": self.cue_contrast,
            "cue_window": (cue_start, cue_end),
        }
