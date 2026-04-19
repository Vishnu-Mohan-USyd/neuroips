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

    # Vogels iSTDP target rate (log-normal target)
    target_rate_hz: float = 1.0


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
