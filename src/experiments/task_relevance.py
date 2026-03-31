"""P4: Task Relevance State Manipulation.

Same trial structure as P1 (CW context -> probe), but with a binary
task-state input to V2 that varies across conditions:
    [1, 0] = orientation-relevant
    [0, 1] = orientation-irrelevant

One shared network, no re-training.  Tests whether expectation effects
depend on task relevance via the task_state input to V2.
"""

from __future__ import annotations

import torch

from src.config import ModelConfig
from src.model.network import LaminarV1V2Network
from src.experiments.paradigm_base import (
    ParadigmBase, TrialConfig, TrialSet,
)


class TaskRelevanceParadigm(ParadigmBase):
    """P4: task-state manipulation (NOT fine-tuning)."""

    paradigm_name = "task_relevance"

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
        T = self.n_timesteps

        trial_sets: dict[str, TrialSet] = {}

        for task in ["relevant", "irrelevant"]:
            task_vec = [1.0, 0.0] if task == "relevant" else [0.0, 1.0]
            for cond in ["expected", "unexpected"]:
                seqs = []
                ts_list = []

                for _ in range(n_trials):
                    start = torch.rand(1, generator=gen).item() * period
                    context = [(start + i * step) % period for i in range(n_ctx)]
                    expected_ori = (context[-1] + step) % period
                    probe = expected_ori if cond == "expected" else (expected_ori + 45.0) % period

                    seqs.append(self.build_stimulus_sequence(context, probe))

                    ts = torch.zeros(1, T, 2)
                    ts[0, :, 0] = task_vec[0]
                    ts[0, :, 1] = task_vec[1]
                    ts_list.append(ts)

                trial_sets[f"{task}_{cond}"] = TrialSet(
                    stimulus=torch.cat(seqs),
                    task_state=torch.cat(ts_list),
                )

        return trial_sets
