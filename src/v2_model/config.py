"""V2 model configuration dataclasses (scaffold — placeholder values, no training logic).

Mirrors the v4 spec in plans/come-to-me-with-streamed-grove.md §Architecture. Sizes
and learning-rate placeholders are populated so `state.initial_state()` and the
tests/v2 smoke suite can run; numerical values are provisional and will be tuned
during Phase 2 training (Task #9+).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ArchitectureConfig:
    """Population sizes per architecture table.

    Counts from plans/come-to-me-with-streamed-grove.md §Architecture table.
    """
    # V1 L4 (fixed)
    n_l4_e: int = 128
    n_l4_pv: int = 16

    # V1 L2/3 (plastic in Phase 2)
    n_l23_e: int = 256
    n_l23_pv: int = 16
    n_l23_som: int = 32

    # H (higher area; plastic in Phase 2)
    n_h_e: int = 64
    n_h_pv: int = 8

    # C (context memory)
    n_c: int = 48

    # Retinotopic grid for LGN + L4 receptive field tiling
    grid_h: int = 32
    grid_w: int = 32
    n_orientations: int = 8  # Gabor bank channels (placeholder)


@dataclass
class TimeConstantsConfig:
    """Population time constants (milliseconds).

    From architecture table. dt (integration step) is the simulation step size.
    Ratios τ/dt are what show up in the discrete-time update, not absolute values.
    """
    dt_ms: float = 5.0

    tau_lgn_ms: float = 20.0
    tau_l4_e_ms: float = 10.0
    tau_l4_pv_ms: float = 5.0
    tau_l23_e_ms: float = 20.0
    tau_l23_pv_ms: float = 5.0
    tau_l23_som_ms: float = 20.0
    tau_h_ms: float = 50.0
    tau_c_ms: float = 500.0  # C dynamics in 300–800 ms range


@dataclass
class ConnectivityConfig:
    """Sparse like-to-like recurrent mask generator parameters.

    P(i→j) ∝ exp(−d²/2σ_r²) · exp(−Δθ²/2σ_θ²); target sparsity 12%.
    """
    sparsity: float = 0.12
    sigma_r_px: float = 4.0
    sigma_theta_deg: float = 25.0

    # Task #74 Fix K — feedforward L4→L23E orientation-like-to-like mask.
    # Sparse (top-k per row) orientation-biased retinotopic mask is applied
    # multiplicatively to ``W_l4_l23_eff`` so L23E inherits L4's orientation
    # tuning at init. Without this, dense uniform W_l4_l23 saturates every
    # L23E unit at ~30 Hz and eliminates orientation selectivity in the
    # Level-2 isolation probe (coder2 #74, 2026-04-22).
    l4_l23_mask_sparsity: float = 0.12            # top-k = round(0.12·n_l4_e)
    l4_l23_sigma_theta_deg: float = 30.0           # orient-bias Gaussian σ
    l4_l23_retino_radius_cells: int = 1            # Chebyshev in pool cells
    # Drop from +4.0 (dense uniform) to +1.5 so per-unit drive stays in
    # biological range once sparsified: softplus(1.5)≈1.70; 15 selected
    # L4 units × 1.70 × mean-matched-r_l4≈0.2 ≈ 5 Hz.
    w_l4_l23_init_mean: float = 1.5


@dataclass
class EnergyConfig:
    """Circuit-wide metabolic cost (global, not feedback-specific).

    L = ... + α · ‖r_E‖₁ + β · ‖I_syn‖₂².
    """
    alpha_rate: float = 1e-3
    beta_syn: float = 1e-4


@dataclass
class PlasticityConfig:
    """Learning rates + plasticity-rule family knobs.

    Placeholder values — Phase 2 tuning will set the real LRs.
    """
    lr_urbanczik_senn: float = 1e-4       # L2/3 recurrent E, L2/3 ↔ H, H recurrent E
    lr_vogels_istdp: float = 1e-4          # All inhibitory synapses
    lr_homeostasis: float = 1e-5           # Per-unit threshold drift
    lr_three_factor_kok: float = 1e-3      # Phase 3 Kok task-specific
    lr_three_factor_richter: float = 1e-3  # Phase 3 Richter task-specific

    weight_decay: float = 1e-5

    # Vogels iSTDP / FastInhibitoryPopulation threshold (log-normal target).
    # Reverted to 1.0 after Fix-H-global (Task #74, 2026-04-21) failed: a
    # single scalar was overloaded for (a) Vogels target on L23 I→E, (b)
    # Vogels target on H I→E, and (c) FastInhibitoryPopulation ReLU
    # threshold for L23PV/L23SOM/HPV. Raising to 3.0 broke (b) by driving
    # a Vogels vs. homeostasis conflict on HE (homeostasis target=0.1 per
    # Debugger #37). Surgical fix: this field now governs only the I-pop
    # ReLU threshold and PV-self-regulation Vogels target; L23E- and HE-
    # targeting Vogels targets live in their own fields below.
    target_rate_hz: float = 1.0

    # Vogels iSTDP targets for synapses that push *E-population* rates —
    # separated from the I-pop ReLU threshold above so the two can be
    # tuned independently. Fix H (Task #74, 2026-04-21): L23E target
    # raised to 3.0 Hz to match the init equilibrium r_l23 ≈ 3.5 Hz; at
    # the legacy 1.0 Hz Vogels was driving L23E down to ~0.8 Hz through
    # Phase-2, starving Hebbian rules on tiny pre·post products and
    # collapsing orientation-preference bins 10/12 → 1/12. HE target
    # stays at 0.1 Hz to match the HE homeostasis target (network.py
    # L23E constructor, Debugger #37: "target 1.0 is unreachable and
    # drives θ downward without bound").
    vogels_target_l23e_hz: float = 3.0
    vogels_target_h_hz: float = 0.1


@dataclass
class RegimeConfig:
    """Hidden-regime g_t parameters for the procedural world.

    Per §Synthetic training world. The hidden regime forces H + C to
    jointly infer latent context across frames.
    """
    n_regimes: int = 4  # CW-drift, CCW-drift, low-hazard, high-hazard
    regime_switch_prob: float = 0.02  # P(g_{t+1} ≠ g_t) = 1 − 0.98
    jump_prob_low_hazard: float = 0.05
    jump_prob_high_hazard: float = 0.30
    drift_step_deg: float = 5.0


@dataclass
class ModelConfig:
    """Top-level V2 model config.

    Composed of sub-configs to keep the field list scannable. Loader tolerance
    (yaml → flattened/nested) is out of scope for this scaffold; will be added
    when we port `src/config.py::load_config`.
    """
    arch: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    tau: TimeConstantsConfig = field(default_factory=TimeConstantsConfig)
    conn: ConnectivityConfig = field(default_factory=ConnectivityConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    plasticity: PlasticityConfig = field(default_factory=PlasticityConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)

    seed: int = 42
    device: str = "cpu"
    dtype: str = "float32"
