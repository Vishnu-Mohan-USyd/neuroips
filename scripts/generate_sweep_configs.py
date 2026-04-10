#!/usr/bin/env python3
"""Generate all 19 sweep config YAMLs from the base template.

Usage:
    python scripts/generate_sweep_configs.py

Writes to config/sweep/sweep_*.yaml
"""

from pathlib import Path

SWEEP_DIR = Path("config/sweep")
SWEEP_DIR.mkdir(parents=True, exist_ok=True)

# Sweep grid definition
# (name, description, lambda_sensory, lambda_energy, lambda_mismatch, l23_energy_weight)
SWEEP_GRID = [
    # Group A: Phase boundary (λ_energy=2.0, λ_mm=0.0, l23w=1.0)
    ("a1", "Phase boundary: λ_sens=0.0", 0.0, 2.0, 0.0, 1.0),
    ("a2", "Phase boundary: λ_sens=0.1", 0.1, 2.0, 0.0, 1.0),
    ("a3", "Phase boundary: λ_sens=0.2", 0.2, 2.0, 0.0, 1.0),
    ("a4", "Phase boundary: λ_sens=0.3", 0.3, 2.0, 0.0, 1.0),
    ("a5", "Phase boundary: λ_sens=0.5", 0.5, 2.0, 0.0, 1.0),
    ("a6", "Phase boundary: λ_sens=0.7", 0.7, 2.0, 0.0, 1.0),
    ("a7", "Phase boundary: λ_sens=1.0 (sharpening baseline)", 1.0, 2.0, 0.0, 1.0),
    # Group B: Energy interaction (λ_mm=0.0, l23w=1.0)
    ("b1", "Energy interaction: λ_sens=0.0, λ_energy=5.0", 0.0, 5.0, 0.0, 1.0),
    ("b2", "Energy interaction: λ_sens=0.3, λ_energy=5.0", 0.3, 5.0, 0.0, 1.0),
    ("b3", "Energy interaction: λ_sens=1.0, λ_energy=5.0", 1.0, 5.0, 0.0, 1.0),
    ("b4", "Energy interaction: λ_sens=0.3, λ_energy=0.5", 0.3, 0.5, 0.0, 1.0),
    ("b5", "Energy interaction: λ_sens=1.0, λ_energy=0.5", 1.0, 0.5, 0.0, 1.0),
    # Group C: L2/3 weight (λ_energy=2.0, λ_mm=0.0)
    ("c1", "L2/3 weight: λ_sens=0.3, l23w=5.0", 0.3, 2.0, 0.0, 5.0),
    ("c2", "L2/3 weight: λ_sens=1.0, l23w=5.0", 1.0, 2.0, 0.0, 5.0),
    # Group D: Mismatch interaction (λ_energy=2.0, l23w=1.0)
    ("d1", "Mismatch interaction: λ_sens=0.1, λ_mm=1.0", 0.1, 2.0, 1.0, 1.0),
    ("d2", "Mismatch interaction: λ_sens=0.5, λ_mm=1.0", 0.5, 2.0, 1.0, 1.0),
    ("d3", "Mismatch interaction: λ_sens=0.3, λ_mm=0.5", 0.3, 2.0, 0.5, 1.0),
    # Group E: Deconfound (λ_energy=2.0, λ_mm=0.0)
    ("e1", "Deconfound: λ_sens=0.3, l23w=3.0", 0.3, 2.0, 0.0, 3.0),
    ("e2", "Deconfound: λ_sens=0.0, l23w=3.0", 0.0, 2.0, 0.0, 3.0),
]


def generate_config(name: str, desc: str, lam_sens: float, lam_energy: float,
                    lam_mm: float, l23w: float) -> str:
    """Generate a sweep config YAML string."""
    return f"""# Sweep {name}: {desc}
# λ_sensory={lam_sens}, λ_energy={lam_energy}, λ_mismatch={lam_mm}, l23_energy_weight={l23w}

model:
  n_orientations: 36
  orientation_range: 180.0
  sigma_ff: 12.0
  tau_l4: 5
  tau_adaptation: 200
  alpha_adaptation: 0.3
  adaptation_clamp: 10.0
  naka_rushton_n: 2.0
  naka_rushton_c50: 0.3
  tau_pv: 5
  sigma_norm: 1.0
  tau_l23: 10
  sigma_rec: 15.0
  gain_rec: 0.3
  tau_som: 10
  tau_vip: 10
  v2_hidden_dim: 16
  v2_input_mode: l4_l23
  pi_max: 5.0
  template_gain: 1.0
  mechanism: "center_surround"
  feedback_mode: "emergent"
  simple_feedback: true
  transition_step: 5.0
  n_basis: 7
  max_apical_gain: 0.0
  dt: 1

training:
  stage1:
    n_steps: 2000
    lr: 1.0e-3
    contrast_range: [0.1, 1.0]

  stage2:
    n_steps: 5000
    lr_v2: 3.0e-4
    lr_feedback: 1.0e-4
    weight_decay: 1.0e-4
    warmup_steps: 500
    burnin_steps: 1000
    ramp_steps: 1000
    gradient_clip: 1.0
    contrast_range: [0.15, 1.0]
    ambiguous_fraction: 0.3

  lambda_sensory: {lam_sens}
  lambda_l4_sensory: 0.0
  lambda_mismatch: {lam_mm}
  lambda_pred: 0.0
  lambda_energy: {lam_energy}
  lambda_homeo: 1.0
  lambda_state: 1.0
  lambda_fb: 0.0
  lambda_local_disc: 0.0
  lambda_pred_suppress: 0.0
  l23_energy_weight: {l23w}
  delta_som: true
  freeze_v2: false
  oracle_pi: 3.0
  oracle_template: "oracle_true"
  oracle_shift_timing: true
  stimulus_noise: 0.25

  batch_size: 32
  seq_length: 25
  steps_on: 12
  steps_isi: 4
  n_seeds: 5

stimulus:
  n_states: 2
  p_transition_cw: 0.80
  p_transition_ccw: 0.80
  p_self: 0.95
  n_anchors: 12
  jitter_range: 0.0
  transition_step: 5.0
  ambiguous_offset: 15.0
  cue_valid_fraction: 0.75
  cue_dim: 2
  task_state_dim: 2
"""


def main():
    for name, desc, lam_sens, lam_energy, lam_mm, l23w in SWEEP_GRID:
        content = generate_config(name, desc, lam_sens, lam_energy, lam_mm, l23w)
        path = SWEEP_DIR / f"sweep_{name}.yaml"
        path.write_text(content)
        print(f"  {path}")

    # Also generate the config list for run_sweep_batch.sh
    config_list_path = SWEEP_DIR / "sweep_config_list.txt"
    lines = []
    for name, desc, lam_sens, lam_energy, lam_mm, l23w in SWEEP_GRID:
        config_path = f"config/sweep/sweep_{name}.yaml"
        output_dir = f"results/sweep/{name}"
        lines.append(f"{config_path} {output_dir}")
    config_list_path.write_text("\n".join(lines) + "\n")
    print(f"\n  Config list: {config_list_path}")
    print(f"  Total: {len(SWEEP_GRID)} configs")


if __name__ == "__main__":
    main()
