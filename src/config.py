"""Configuration dataclasses and mechanism enum for the laminar V1-V2 model."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import yaml


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

    # Feedback mode: always 'emergent' (V2 GRU with head_feedback)
    feedback_mode: str = 'emergent'

    # Transition step for orientation prediction (degrees)
    transition_step: float = 15.0

    # V2 input mode: 'l23' (default), 'l4', or 'l4_l23'
    v2_input_mode: str = 'l23'

    # Numerical stability
    dt: int = 1

    # Phase 2: causal E/I gate on the feedback split.
    # When True, LaminarV1V2Network instantiates a small alpha_net:
    #     nn.Linear(2 + 1, 2)  taking (task_state, pi_pred_raw) → (g_E, g_I)
    # whose outputs (via 2*sigmoid) multiply center_exc and som_drive_fb
    # respectively. Init is near-identity (bias=0, tiny weight std),
    # so legacy behavior is preserved at step 0. Default False keeps all
    # pre-Phase-2 configs bit-identical.
    use_ei_gate: bool = False

    # Task #9 / Fix 2 / Network_both: per-regime head_feedback in V2ContextModule.
    # When True, V2 instantiates two independent feedback heads
    # (`head_feedback_focused`, `head_feedback_routine`) and gates them by
    # `task_state[:, 0:1]` / `task_state[:, 1:2]`. When False (default),
    # V2 uses the legacy single shared `head_feedback`. Default False keeps
    # the Network_mm configuration bit-identical.
    use_per_regime_feedback: bool = False

    # Fix 1: dual V2 architecture. When True, LaminarV1V2Network constructs
    # two independent V2ContextModules (v2_focused and v2_routine), selected
    # by task_state at each timestep. Each has its own GRU, head_mu, head_pi,
    # head_feedback. Everything else (L4, PV, L2/3, SOM) remains shared.
    # use_per_regime_feedback is ignored when use_dual_v2=True (each V2
    # already has its own feedback head). Default False = legacy single V2.
    use_dual_v2: bool = False

    # Rescue 2: precision-gated feedback. When True, V2's learned precision
    # multiplicatively gates the feedback signal in the forward dynamics:
    #     precision_gate = pi_pred_raw / pi_max   # [0, 1]
    #     scaled_fb = feedback_signal * feedback_scale * precision_gate
    # At max precision (pi_pred_raw = pi_max), feedback is unchanged.
    # At zero precision, feedback is silenced. The [0, 1] range means
    # precision can only ATTENUATE feedback, never amplify it.
    # Requires lambda_state > 0 so the prior KL loss trains V2 to produce
    # meaningful precision values. Default False = legacy (feedback_signal
    # scaled only by feedback_scale, precision unused in dynamics).
    use_precision_gating: bool = False

    # R1+2-only surround option: spread the inhibitory feedback drive
    # (`som_drive_fb`) with a fixed circular Gaussian kernel before SOM
    # integration. This leaves `center_exc` unchanged and does not instantiate
    # any Rescue-3 VIP machinery.
    use_fb_surround: bool = False
    sigma_fb_surround: float = 20.0

    # Rescue 3: VIP-SOM disinhibition circuit. When True, adds:
    #   - VIPRing population driven by V2 head_vip (Linear H→N)
    #   - Structured SOM surround kernel (fixed circular Gaussian)
    #   - VIP→SOM subtractive connection: relu(som_drive - w_vip_som * r_vip)
    # Two modes from one circuit:
    #   Routine (low VIP): SOM uninhibited → surround active → dampening
    #   Focused (high VIP at center): SOM suppressed at center → sharpening
    # Default False = legacy (SOM receives raw som_drive_fb, no VIP).
    use_vip: bool = False
    tau_vip: int = 10
    sigma_som_surround: float = 20.0

    # Rescue 4: learnable deep V1 template + error-based mismatch readout.
    # When use_deep_template=True, LaminarV1V2Network constructs a DeepTemplate
    # leaky integrator driven by (q_pred * pi_pred_eff) with a single learnable
    # scalar gain (softplus-positive). State lives in NetworkState.deep_template
    # (already present as a placeholder in state.py). Time constant tau_template
    # defaults to 10 steps for parity with tau_l23/tau_som/tau_vip.
    # When use_error_mismatch=True, the mismatch_head in CompositeLoss receives
    # r_error_windows = relu(r_l23 - r_template) instead of r_l23_windows, i.e.
    # mismatch is detected from the positive prediction error rather than the
    # raw representation. Default False on both = legacy bit-identical (the
    # placeholder `deep_tmpl = q_pred * pi_pred_eff` path is preserved).
    use_deep_template: bool = False
    template_gain: float = 1.0
    tau_template: int = 10
    use_error_mismatch: bool = False

    # Rescue 5: shape-matched predictive suppression. When True, the Stage-2
    # calibration builds T_stage1 ∈ R^{N x N}, where T_stage1[j, :] is the mean
    # Stage-1 L2/3 response profile at orientation j with FB=0 (i.e. the
    # sensory-basis tuning curve). At loss time, q_match = q_pred @ T_stage1
    # projects the softmax V2 prediction into the sensory basis before it is
    # used in expected_suppress_loss. This addresses the central-clipping
    # pattern in R1-R4 where the narrow softmax q_pred over-subtracted the
    # peak of r_l23 (broad tuning curve), leaving expected < unexpected on
    # both activity and decoding. Default False = legacy bit-identical.
    use_shape_matched_prediction: bool = False

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
    # Phase 2.4: LR multiplier applied to the alpha_net param group on top of
    # stage2_lr_v2. 1.0 = legacy (alpha_net uses the same LR as V2). >1.0
    # accelerates alpha_net learning to break out of identity init when the
    # burn-in starves its gradient path. No-op when use_ei_gate=False.
    lr_mult_alpha: float = 1.0
    stage2_warmup_steps: int = 1000
    stage2_burnin_steps: int = 5000
    stage2_ramp_steps: int = 5000
    gradient_clip: float = 1.0
    stage2_contrast_range: Tuple[float, float] = (0.15, 1.0)
    ambiguous_fraction: float = 0.15

    # Loss weights
    lambda_sensory: float = 1.0
    lambda_pred: float = 0.5
    lambda_energy: float = 0.01
    lambda_homeo: float = 1.0
    lambda_state: float = 0.25
    lambda_surprise: float = 0.0     # Surprise detection loss (0 = disabled)
    lambda_error: float = 0.0        # Prediction error readout loss (0 = disabled)
    lambda_detection: float = 0.0    # Detection confirmation loss (0 = disabled)
    lambda_l4_sensory: float = 0.0  # L4 sensory readout weight (0 = disabled)
    lambda_mismatch: float = 0.0    # L2/3 mismatch detection weight (0 = disabled)
    lambda_sharp: float = 0.0       # Tuning sharpness: penalize L2/3 activity at flanks (0 = disabled)
    lambda_local_disc: float = 0.0  # Phase 4: local 5-way discrimination (expected vs ±1, ±2 neighbors). 0 = disabled.
    lambda_pred_suppress: float = 0.0  # Prediction suppression: penalize L2/3 activity matching V2 prediction. 0 = disabled.
    lambda_fb_energy: float = 0.0      # Feedback energy: penalize magnitude of excitatory feedback (center_exc). 0 = disabled.
    # Fix 3: expected-suppress loss. Penalizes mean |r_l23| ONLY on routine
    # presentations where mismatch_label=0 (stimulus matched prediction).
    # Provides direct gradient toward Kok-style expectation suppression.
    # 0.0 = disabled → legacy bit-identical.
    lambda_expected_suppress: float = 0.0
    # Expected-only width loss: penalize shoulder activity above a per-trial
    # half-max reference. The center dead zone remains protected; the shoulder
    # mask can now be controlled explicitly via lower/upper bounds. When those
    # bounds are omitted, the legacy shoulder geometry is preserved:
    #   lower = expected_width_deadzone_deg
    #   upper = expected_width_deadzone_deg + 10.0
    # so the default 10 deg dead zone still yields the original 10-20 deg band.
    # Unlike expected_suppress, this term is applied on all expected
    # presentations regardless of task state. 0.0 = disabled → clean no-op.
    lambda_expected_width: float = 0.0
    expected_width_deadzone_deg: float = 10.0
    expected_width_shoulder_lower_deg: float | None = None
    expected_width_shoulder_upper_deg: float | None = None
    # Phase 2.4: routine E/I symmetry-break loss.
    #   shape_per_sample = |center_exc|.mean(T,N) - 0.5 * |som_drive_fb|.mean(T,N)
    # Weighted per-sample by task_routing[*]['routine_shape'] (0 for focused,
    # 2.0 for routine in sweep_dual_2_4.yaml) so only routine samples are
    # rewarded for routing feedback through the inhibitory (SOM) branch.
    # 0.0 = disabled → legacy bit-identical.
    lambda_routine_shape: float = 0.0
    l2_energy: bool = False             # Use L2 (quadratic) penalty on r_l23 in energy cost instead of L1.
    l23_energy_weight: float = 1.0      # Multiplier on L2/3 term in energy cost. >1 penalizes L2/3 output more.

    # Phase 1A (dual-regime): per-sample loss routing by task_state. When set,
    # CompositeLoss.forward() weights sensory / energy (r_l23 component) / fb_energy
    # per-sample according to whether the sequence's task_state is focused [1,0]
    # or routine [0,1]. None → legacy non-routed path (bit-identical to pre-1A).
    # Expected shape:
    #   {
    #     "focused": {"sensory": float, "energy": float, "fb_energy": float},
    #     "routine": {"sensory": float, "energy": float, "fb_energy": float},
    #   }
    task_routing: dict | None = None

    # Freeze V2 / use oracle predictor
    freeze_v2: bool = False
    oracle_pi: float = 1.0          # pi value when using oracle mode
    # Oracle template type (only used when freeze_v2=True):
    #   oracle_true    — q_pred bumped at next orientation given TRUE state (normal oracle)
    #   oracle_wrong   — q_pred bumped at next orientation given OPPOSITE state (CW<->CCW swap)
    #   oracle_random  — q_pred bumped at a random orientation, independent of stimulus
    #   oracle_uniform — q_pred is flat (uniform over orientations)
    oracle_template: str = "oracle_true"

    # Phase 5: width of the oracle prediction bump (degrees). Defaults to
    # sigma_ff (12.0) so existing configs are bitwise-identical. Lowering
    # this to e.g. 5.0 gives the feedback operator a NARROWER prior than
    # the feedforward tuning, which is the engineering handle for
    # expectation-driven sharpening.
    oracle_sigma: float = 12.0

    # Freeze orientation decoder in Stage 2 (for clean representational claims)
    freeze_decoder: bool = False

    # Phase 3: shift oracle q_pred by +1 presentation so that the template
    # acts as a prior about the CURRENT stimulus rather than a forecast of the
    # next one. Only used when freeze_v2=True. First presentation receives a
    # uniform prior (no valid history).
    oracle_shift_timing: bool = False

    # Stimulus noise (std of Gaussian noise added to population-coded stimulus in Stage 2)
    stimulus_noise: float = 0.0

    # Fix 2: gradient isolation. When True, alternate between focused-only and
    # routine-only task_state overrides every `isolation_period` steps.
    # Stimulus statistics (HMM) still run normally; only task_state is overridden.
    # False = legacy (Markov per-presentation task_state, no override).
    gradient_isolation: bool = False
    isolation_period: int = 100

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
    # Angular offset (degrees) between the two orientations that make up an
    # ambiguous mixture stimulus. Used by `build_stimulus_sequence` in
    # `src/training/trainer.py` to construct the second orientation of each
    # ambiguous presentation as `ori + ambiguous_offset`.
    ambiguous_offset: float = 15.0
    cue_valid_fraction: float = 0.75
    # Simple-dual-regime: per-presentation Markov task_state switch probability.
    # P(regime flip at next presentation | current regime). 0.0 reproduces the
    # pre-simple-dual "per-sequence Bernoulli" behavior (task_state constant
    # across a sequence). 0.2 = user target spec.
    task_p_switch: float = 0.0


def load_config(path: str | Path = "config/defaults.yaml") -> tuple[ModelConfig, TrainingConfig, StimulusConfig]:
    """Load configuration from a YAML file and return typed config objects."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    model_raw = raw.get("model", {})
    # Remove legacy keys that may exist in old YAML configs
    for legacy_key in ('mechanism', 'n_basis', 'max_apical_gain', 'tau_vip',
                       'simple_feedback'):
        model_raw.pop(legacy_key, None)
    # Extract feedback_mode with default
    feedback_mode = model_raw.pop("feedback_mode", "emergent")
    model_cfg = ModelConfig(**model_raw, feedback_mode=feedback_mode)

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
        lr_mult_alpha=stage2.get("lr_mult_alpha", 1.0),
        stage2_warmup_steps=stage2.get("warmup_steps", 1000),
        stage2_burnin_steps=stage2.get("burnin_steps", 5000),
        stage2_ramp_steps=stage2.get("ramp_steps", 5000),
        gradient_clip=stage2.get("gradient_clip", 1.0),
        stage2_contrast_range=tuple(stage2.get("contrast_range", [0.15, 1.0])),
        ambiguous_fraction=stage2.get("ambiguous_fraction", 0.15),
        lambda_sensory=train_raw.get("lambda_sensory", 1.0),
        lambda_pred=train_raw.get("lambda_pred", 0.5),
        lambda_energy=train_raw.get("lambda_energy", 0.01),
        lambda_homeo=train_raw.get("lambda_homeo", 1.0),
        lambda_state=train_raw.get("lambda_state", 0.25),
        lambda_surprise=train_raw.get("lambda_surprise", 0.0),
        lambda_error=train_raw.get("lambda_error", 0.0),
        lambda_detection=train_raw.get("lambda_detection", 0.0),
        lambda_l4_sensory=train_raw.get("lambda_l4_sensory", 0.0),
        lambda_mismatch=train_raw.get("lambda_mismatch", 0.0),
        lambda_sharp=train_raw.get("lambda_sharp", 0.0),
        lambda_local_disc=train_raw.get("lambda_local_disc", 0.0),
        lambda_pred_suppress=train_raw.get("lambda_pred_suppress", 0.0),
        lambda_fb_energy=train_raw.get("lambda_fb_energy", 0.0),
        lambda_expected_suppress=train_raw.get("lambda_expected_suppress", 0.0),
        lambda_expected_width=train_raw.get("lambda_expected_width", 0.0),
        expected_width_deadzone_deg=train_raw.get("expected_width_deadzone_deg", 10.0),
        expected_width_shoulder_lower_deg=train_raw.get("expected_width_shoulder_lower_deg", None),
        expected_width_shoulder_upper_deg=train_raw.get("expected_width_shoulder_upper_deg", None),
        lambda_routine_shape=train_raw.get("lambda_routine_shape", 0.0),
        l2_energy=train_raw.get("l2_energy", False),
        l23_energy_weight=train_raw.get("l23_energy_weight", 1.0),
        task_routing=train_raw.get("task_routing", None),
        freeze_v2=train_raw.get("freeze_v2", False),
        freeze_decoder=train_raw.get("freeze_decoder", False),
        oracle_shift_timing=train_raw.get("oracle_shift_timing", False),
        oracle_pi=train_raw.get("oracle_pi", 1.0),
        oracle_template=train_raw.get("oracle_template", "oracle_true"),
        oracle_sigma=train_raw.get("oracle_sigma", 12.0),
        stimulus_noise=train_raw.get("stimulus_noise", 0.0),
        gradient_isolation=train_raw.get("gradient_isolation", False),
        isolation_period=train_raw.get("isolation_period", 100),
        batch_size=train_raw.get("batch_size", 32),
        seq_length=train_raw.get("seq_length", 50),
        steps_on=train_raw.get("steps_on", 8),
        steps_isi=train_raw.get("steps_isi", 4),
        n_seeds=train_raw.get("n_seeds", 5),
    )

    # ModelConfig carries the optional R1+2-only inhibitory surround fields
    # directly, so they load automatically from model_raw when present.

    stim_raw = raw.get("stimulus", {})
    stim_cfg = StimulusConfig(**stim_raw)

    return model_cfg, training_cfg, stim_cfg
