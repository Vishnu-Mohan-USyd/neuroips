"""Base class and data structures for experimental paradigms.

All paradigms share:
    - Trial structure: context presentations -> probe -> post-probe recording
    - Full temporal trajectory recording across all layers
    - Batched execution for efficiency
    - Temporal window definitions for downstream analysis
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from src.config import ModelConfig
from src.model.network import LaminarV1V2Network
from src.stimulus.gratings import generate_grating


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TrialConfig:
    """Trial timing parameters."""
    n_context: int = 8       # context presentations before probe
    steps_on: int = 8        # timesteps per stimulus presentation
    steps_isi: int = 4       # inter-stimulus interval (blank)
    steps_post: int = 8      # timesteps after probe offset
    contrast: float = 0.8    # default stimulus contrast


@dataclass
class TrialSet:
    """Stimulus sequences for trials in one condition."""
    stimulus: Tensor                  # [n_trials, T, N]
    cue: Tensor | None = None         # [n_trials, T, N] or None -> zeros
    task_state: Tensor | None = None  # [n_trials, T, 2] or None -> zeros


@dataclass
class ConditionData:
    """Stacked temporal trajectories for all trials in one condition.

    All tensors: [n_trials, T, ...].
    """
    r_l4: Tensor           # [n_trials, T, N]
    r_l23: Tensor          # [n_trials, T, N]
    r_pv: Tensor           # [n_trials, T, 1]
    r_som: Tensor          # [n_trials, T, N]
    q_pred: Tensor         # [n_trials, T, N]
    pi_pred: Tensor        # [n_trials, T, 1]
    state_logits: Tensor   # [n_trials, T, 3]
    deep_template: Tensor  # [n_trials, T, N]
    r_vip: Tensor | None = None  # [n_trials, T, N] when VIP scaffold is present


@dataclass
class ExperimentResult:
    """Full results from running a paradigm."""
    paradigm_name: str
    conditions: dict[str, ConditionData]
    trial_info: dict
    temporal_windows: dict[str, tuple[int, int]]


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class ParadigmBase:
    """Base class for experimental paradigms.

    Provides common trial building and batched execution infrastructure.
    Subclasses implement ``generate_trials`` to define conditions.
    """

    paradigm_name: str = "base"

    def __init__(
        self,
        net: LaminarV1V2Network,
        model_cfg: ModelConfig,
        trial_cfg: TrialConfig | None = None,
    ):
        self.net = net
        self.cfg = model_cfg
        self.trial_cfg = trial_cfg or TrialConfig()

    # -- Timing properties --------------------------------------------------

    @property
    def n_timesteps(self) -> int:
        """Total timesteps per trial."""
        tc = self.trial_cfg
        return tc.n_context * (tc.steps_on + tc.steps_isi) + tc.steps_on + tc.steps_post

    @property
    def probe_onset(self) -> int:
        """Timestep index where the probe stimulus starts."""
        tc = self.trial_cfg
        return tc.n_context * (tc.steps_on + tc.steps_isi)

    @property
    def temporal_windows(self) -> dict[str, tuple[int, int]]:
        """Named (start, end) timestep ranges for analysis."""
        po = self.probe_onset
        tc = self.trial_cfg
        return {
            "prestimulus": (po - tc.steps_isi, po),
            "early": (po, po + min(3, tc.steps_on)),
            "sustained": (po + 3, po + tc.steps_on),
            "late": (po + tc.steps_on, po + tc.steps_on + tc.steps_post),
        }

    # -- Stimulus helpers ---------------------------------------------------

    def make_grating(self, ori: float, contrast: float | None = None) -> Tensor:
        """Generate a population-coded grating. Returns [1, N]."""
        c = contrast if contrast is not None else self.trial_cfg.contrast
        return generate_grating(
            torch.tensor([ori]), torch.tensor([c]),
            n_orientations=self.cfg.n_orientations,
            sigma=self.cfg.sigma_ff,
            n=self.cfg.naka_rushton_n,
            c50=self.cfg.naka_rushton_c50,
            period=self.cfg.orientation_range,
        )

    def build_stimulus_sequence(
        self,
        context_oris: list[float],
        probe_ori: float,
        probe_contrast: float | None = None,
    ) -> Tensor:
        """Build a single-trial stimulus sequence [1, T, N].

        Context orientations presented during ON periods, blanks during ISI.
        Probe presented after context. Post-probe period is blank.
        """
        tc = self.trial_cfg
        T = self.n_timesteps
        N = self.cfg.n_orientations
        stim = torch.zeros(1, T, N)

        for i, ori in enumerate(context_oris):
            onset = i * (tc.steps_on + tc.steps_isi)
            s = self.make_grating(ori)
            stim[0, onset:onset + tc.steps_on] = s

        po = self.probe_onset
        s = self.make_grating(probe_ori, probe_contrast)
        stim[0, po:po + tc.steps_on] = s

        return stim

    # -- Trial generation (override in subclasses) --------------------------

    def generate_trials(self, n_trials: int, seed: int) -> dict[str, TrialSet]:
        """Generate trial sets per condition. Subclasses must implement."""
        raise NotImplementedError

    def _get_trial_info(self) -> dict:
        """Return paradigm-specific metadata. Override in subclasses."""
        return {}

    # -- Execution ----------------------------------------------------------

    def run(
        self,
        n_trials: int = 200,
        seed: int = 42,
        batch_size: int = 32,
    ) -> ExperimentResult:
        """Run the paradigm: generate trials, execute in batches, return results."""
        self.net.eval()
        trial_sets = self.generate_trials(n_trials, seed)

        conditions = {}
        for cond_name, ts in trial_sets.items():
            conditions[cond_name] = self._run_trial_set(ts, batch_size)

        return ExperimentResult(
            paradigm_name=self.paradigm_name,
            conditions=conditions,
            trial_info=self._get_trial_info(),
            temporal_windows=self.temporal_windows,
        )

    def _run_trial_set(self, ts: TrialSet, batch_size: int) -> ConditionData:
        """Run a TrialSet through the network in batches.

        Inputs are moved onto the network's device before packing/forward so
        checkpoint-backed CUDA evaluation works when experiment tensors were
        generated on CPU. Outputs are moved back to CPU before accumulation to
        preserve the historical ExperimentResult storage behavior and avoid
        retaining large CUDA buffers across conditions.
        """
        n = ts.stimulus.shape[0]
        model_device = next(self.net.parameters()).device
        chunks: dict[str, list[Tensor]] = {
            k: [] for k in ["r_l4", "r_l23", "r_pv", "r_som",
                            "q_pred", "pi_pred", "state_logits", "deep_template", "r_vip"]
        }

        for i in range(0, n, batch_size):
            stim = ts.stimulus[i:i + batch_size].to(model_device)
            cue = (
                ts.cue[i:i + batch_size].to(model_device)
                if ts.cue is not None else None
            )
            task = (
                ts.task_state[i:i + batch_size].to(model_device)
                if ts.task_state is not None else None
            )

            with torch.no_grad():
                from src.model.network import LaminarV1V2Network
                packed = LaminarV1V2Network.pack_inputs(stim, cue, task)
                r_l23_all, _, aux = self.net.forward(packed)

            chunks["r_l4"].append(aux["r_l4_all"].cpu())
            chunks["r_l23"].append(r_l23_all.cpu())
            chunks["r_pv"].append(aux["r_pv_all"].cpu())
            chunks["r_som"].append(aux["r_som_all"].cpu())
            chunks["q_pred"].append(aux["q_pred_all"].cpu())
            chunks["pi_pred"].append(aux["pi_pred_all"].cpu())
            chunks["state_logits"].append(aux["state_logits_all"].cpu())
            chunks["deep_template"].append(aux["deep_template_all"].cpu())
            chunks["r_vip"].append(aux["r_vip_all"].cpu())

        return ConditionData(**{k: torch.cat(v) for k, v in chunks.items()})
