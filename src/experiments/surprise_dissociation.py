"""P5: Local vs High-Level Surprise Dissociation.

Matched trial pairs that dissociate sensory distance from V2 surprise:

Both trials share:
    - Same last context orientation (theta_L)
    - Same probe orientation (theta_L - transition_step)
    - Same local distance (|probe - last_context| = transition_step)

High V2 surprise (Trial A):
    - CW context ending at theta_L
    - V2 expects theta_L + step (CW rule)
    - Probe = theta_L - step  =>  2*step deviation from prediction

Low V2 surprise (Trial B):
    - CCW context ending at theta_L
    - V2 expects theta_L - step (CCW rule) = probe
    - Probe matches prediction exactly  =>  0 deviation

V2 entropy should be logged during analysis to confirm both conditions
have low entropy (ruling out "general uncertainty" as a confound).
"""

from __future__ import annotations

import torch

from src.config import ModelConfig
from src.model.network import LaminarV1V2Network
from src.experiments.paradigm_base import (
    ParadigmBase, TrialConfig, TrialSet,
)


class SurpriseDissociationParadigm(ParadigmBase):
    """P5: local vs high-level surprise (CW-unexpected vs CCW-expected)."""

    paradigm_name = "surprise_dissociation"

    def __init__(
        self,
        net: LaminarV1V2Network,
        model_cfg: ModelConfig,
        trial_cfg: TrialConfig | None = None,
        transition_step: float = 15.0,
    ):
        super().__init__(net, model_cfg, trial_cfg)
        self.transition_step = transition_step

    def generate_trials(self, n_trials: int, seed: int) -> dict[str, TrialSet]:
        gen = torch.Generator()
        gen.manual_seed(seed)
        period = self.cfg.orientation_range
        n_ctx = self.trial_cfg.n_context
        step = self.transition_step

        seqs_high: list = []
        seqs_low: list = []

        for _ in range(n_trials):
            theta_L = torch.rand(1, generator=gen).item() * period
            probe = (theta_L - step) % period

            # CW context ending at theta_L (V2 expects theta_L + step)
            cw_ctx = [(theta_L - (n_ctx - 1 - i) * step) % period for i in range(n_ctx)]
            seqs_high.append(self.build_stimulus_sequence(cw_ctx, probe))

            # CCW context ending at theta_L (V2 expects theta_L - step = probe)
            ccw_ctx = [(theta_L + (n_ctx - 1 - i) * step) % period for i in range(n_ctx)]
            seqs_low.append(self.build_stimulus_sequence(ccw_ctx, probe))

        return {
            "high_v2_surprise": TrialSet(stimulus=torch.cat(seqs_high)),
            "low_v2_surprise": TrialSet(stimulus=torch.cat(seqs_low)),
        }

    def _get_trial_info(self) -> dict:
        return {"transition_step": self.transition_step}
