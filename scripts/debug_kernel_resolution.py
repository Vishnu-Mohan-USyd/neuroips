"""Debugger: Characterize kernel resolution limits.

Experiment 1: Measure actual learned kernel profiles
Experiment 2: Minimum feature width from basis set
Experiment 3: Ideal energy-efficient kernel (free circulant)
"""

import sys
sys.path.insert(0, "/mnt/c/Users/User/codingproj/freshstart")

import torch
import torch.nn.functional as F
import numpy as np
from src.config import load_config
from src.model.network import LaminarV1V2Network
from src.state import initial_state
from src.stimulus.gratings import generate_grating

model_cfg, train_cfg, stim_cfg = load_config("config/exp_branch_a.yaml")
N = model_cfg.n_orientations
period = model_cfg.orientation_range
sigma_ff = model_cfg.sigma_ff
step_deg = period / N  # 5° per channel

net = LaminarV1V2Network(model_cfg, delta_som=train_cfg.delta_som)
ckpt = torch.load("results/control_no_gate/s42/center_surround_seed42/checkpoint.pt",
                   map_location="cpu", weights_only=False)
net.load_state_dict(ckpt["model_state"], strict=False)
net.eval()

theta_stim = 90.0
peak_ch = int(theta_stim / step_deg)  # 18


def compute_fwhm(profile, step_deg):
    """Compute FWHM of a 1D profile in degrees."""
    peak = profile.max().item()
    if peak <= 0:
        return float('nan')
    half_max = peak / 2
    above = (profile >= half_max).float()
    # Count contiguous channels above half-max (handle circularity)
    # Shift profile so peak is at center
    peak_idx = profile.argmax().item()
    shifted = torch.roll(profile, N // 2 - peak_idx)
    above_shifted = (shifted >= half_max).float()
    # Find contiguous region around center
    center = N // 2
    left = center
    while left > 0 and above_shifted[left - 1] > 0:
        left -= 1
    right = center
    while right < N - 1 and above_shifted[right + 1] > 0:
        right += 1
    return (right - left + 1) * step_deg


def find_zero_crossings(profile, step_deg):
    """Find positions where profile crosses zero."""
    crossings = []
    for i in range(len(profile)):
        j = (i + 1) % len(profile)
        if profile[i] * profile[j] < 0:
            # Linear interpolation
            frac = abs(profile[i].item()) / (abs(profile[i].item()) + abs(profile[j].item()))
            pos = (i + frac) * step_deg
            crossings.append(pos)
    return crossings


def report_kernel(name, profile, step_deg):
    """Full report on a kernel profile."""
    print(f"\n  {name}:")
    print(f"    Peak value: {profile.max().item():.6f} at ch {profile.argmax().item()} ({profile.argmax().item() * step_deg:.1f}°)")
    print(f"    Min value:  {profile.min().item():.6f} at ch {profile.argmin().item()} ({profile.argmin().item() * step_deg:.1f}°)")
    print(f"    Sum:        {profile.sum().item():.6f}")

    pos_area = profile[profile > 0].sum().item()
    neg_area = profile[profile < 0].sum().item()
    print(f"    Positive area: {pos_area:.6f}")
    print(f"    Negative area: {neg_area:.6f}")
    if abs(neg_area) > 1e-8:
        print(f"    Pos/|Neg| ratio: {pos_area / abs(neg_area):.3f}")
    else:
        print(f"    Pos/|Neg| ratio: inf (no negative region)")

    fwhm = compute_fwhm(profile, step_deg)
    print(f"    FWHM: {fwhm:.1f}°")

    crossings = find_zero_crossings(profile, step_deg)
    if crossings:
        print(f"    Zero crossings at: {[f'{c:.1f}°' for c in crossings]}")
    else:
        print(f"    Zero crossings: none (all same sign)")

    # Full profile
    print(f"    {'Ch':>4s} {'Deg':>6s}  {'Value':>12s}")
    print(f"    " + "-" * 28)
    for ch in range(N):
        deg = ch * step_deg
        v = profile[ch].item()
        print(f"    {ch:>4d} {deg:>5.1f}°  {v:>12.6f}")


# ======================================================================
# EXPERIMENT 1: Learned kernel profiles
# ======================================================================
print("=" * 80)
print("EXPERIMENT 1: Learned kernel profiles from checkpoint")
print("=" * 80)

# Extract alpha weights
alpha_inh = net.feedback.alpha_inh.data  # [K]
alpha_vip = net.feedback.alpha_vip.data  # [K]
alpha_apical = net.feedback.alpha_apical.data  # [K]
basis = net.feedback.basis  # [K, N]
K = basis.shape[0]

print(f"\nBasis set: {K} functions, N={N} channels")
print(f"\nAlpha weights:")
print(f"  {'Basis':>6s}  {'σ/type':>12s}  {'α_inh':>10s}  {'α_apical':>10s}  {'α_vip':>10s}")
print(f"  " + "-" * 55)
basis_labels = ["G(σ=5°)", "G(σ=15°)", "G(σ=30°)", "G(σ=60°)", "MexHat(10-30)", "Constant", "Odd/sin"]
for k in range(K):
    print(f"  {k:>6d}  {basis_labels[k]:>12s}  {alpha_inh[k].item():>10.4f}  "
          f"{alpha_apical[k].item():>10.4f}  {alpha_vip[k].item():>10.4f}")

# Compute full kernel profiles
K_inh = (alpha_inh.unsqueeze(-1) * basis).sum(dim=0)  # [N]
K_apical = (alpha_apical.unsqueeze(-1) * basis).sum(dim=0)  # [N]
K_vip = (alpha_vip.unsqueeze(-1) * basis).sum(dim=0)  # [N]

report_kernel("SOM (inhibitory) kernel", K_inh, step_deg)
report_kernel("Apical (gain) kernel", K_apical, step_deg)
report_kernel("VIP (disinhibitory) kernel", K_vip, step_deg)

# Also report delta_som parameters
if net.feedback.delta_som:
    print(f"\n  Delta-SOM parameters:")
    print(f"    som_baseline: {net.feedback.som_baseline.item():.4f}")
    print(f"    som_tonic: {net.feedback.som_tonic.item():.4f} → softplus = {F.softplus(net.feedback.som_tonic).item():.6f}")
    print(f"    vip_baseline: {net.feedback.vip_baseline.item():.4f}")

# ======================================================================
# EXPERIMENT 2: Minimum feature width from basis set
# ======================================================================
print("\n" + "=" * 80)
print("EXPERIMENT 2: Minimum feature width achievable with basis set")
print("=" * 80)

# Narrowest single basis: G(σ=5°)
print("\n--- Basis 0: G(σ=5°) alone ---")
K_narrow = basis[0]
report_kernel("G(σ=5°) basis", K_narrow, step_deg)

# What happens when convolved with q_pred?
# apical_field = circulant(K) @ q_centered
# q_pred is an unnormalized Gaussian with σ=12°
oracle_sigma = getattr(train_cfg, 'oracle_sigma', 12.0)
q_pred = net._make_bump(torch.tensor([theta_stim]), sigma=oracle_sigma)
q_centered = q_pred - 1.0 / N  # as in feedback.forward()

# Compute convolution: circulant @ q_centered
circulant_narrow = net.feedback._to_circulant(K_narrow)
field_narrow = (circulant_narrow @ q_centered[0].unsqueeze(-1)).squeeze(-1)
print(f"\n  Convolution of G(σ=5°) with q_pred (σ={oracle_sigma}°):")
report_kernel("G(5°) ⊛ q_pred", field_narrow, step_deg)

# Narrowest difference feature: basis 0 - basis 1 (5° - 15°)
print("\n--- Difference: G(σ=5°) - G(σ=15°) ---")
K_diff_5_15 = basis[0] - basis[1]
report_kernel("G(5°) - G(15°)", K_diff_5_15, step_deg)

circulant_diff = net.feedback._to_circulant(K_diff_5_15)
field_diff = (circulant_diff @ q_centered[0].unsqueeze(-1)).squeeze(-1)
print(f"\n  Convolution of [G(5°)-G(15°)] with q_pred (σ={oracle_sigma}°):")
report_kernel("[G(5°)-G(15°)] ⊛ q_pred", field_diff, step_deg)

# Mexican hat basis
print("\n--- Built-in Mexican Hat basis (10° - 30°) ---")
K_mexhat = basis[4]
report_kernel("MexHat(10-30) basis", K_mexhat, step_deg)

circulant_mh = net.feedback._to_circulant(K_mexhat)
field_mh = (circulant_mh @ q_centered[0].unsqueeze(-1)).squeeze(-1)
print(f"\n  Convolution of MexHat with q_pred (σ={oracle_sigma}°):")
report_kernel("MexHat ⊛ q_pred", field_mh, step_deg)

# What's the actual LEARNED apical field?
print("\n--- Actual learned apical field (current kernel ⊛ q_pred) ---")
circulant_apical = net.feedback._to_circulant(K_apical)
field_actual = (circulant_apical @ q_centered[0].unsqueeze(-1)).squeeze(-1)
report_kernel("Learned apical ⊛ q_pred", field_actual, step_deg)

# How does this translate to gain?
oracle_pi = train_cfg.oracle_pi
pi_eff = torch.tensor(oracle_pi)  # scalar
gain_actual = 1.0 + net.feedback.max_apical_gain * torch.tanh(pi_eff * field_actual)
report_kernel("Actual apical gain from learned kernel", gain_actual, step_deg)

# ======================================================================
# EXPERIMENT 3: Ideal energy-efficient kernel
# ======================================================================
print("\n" + "=" * 80)
print("EXPERIMENT 3: Ideal energy-efficient kernel (free 36-weight circulant)")
print("=" * 80)

# Current situation:
# - Base L2/3 activity (FB OFF): mostly at center channels
# - Current gain: enhancing everywhere (mean=1.28)
# - Goal: same peak enhancement, but mean gain = 1.0 (energy neutral)

# First, get the base activity profile (FB OFF)
stim = generate_grating(torch.tensor([theta_stim]), torch.tensor([0.5]),
                        N, sigma_ff, period=period)
cue = torch.zeros(1, 2)
task = torch.zeros(1, 2)

net.oracle_mode = True
net.feedback_scale.fill_(0.0)
q_true = net._make_bump(torch.tensor([theta_stim]), sigma=oracle_sigma)
pi_oracle = torch.full((1, 1), oracle_pi)
net.oracle_q_pred = q_true
net.oracle_pi_pred = pi_oracle

state = initial_state(1, N, model_cfg.v2_hidden_dim)
with torch.no_grad():
    for t in range(12):
        state, _ = net.step(stim, cue, task, state)
base_activity = state.r_l23[0].clone()  # [N]

print(f"\nBase activity (FB OFF) at stim ch: {base_activity[peak_ch].item():.6f}")
print(f"Base activity sum: {base_activity.sum().item():.6f}")

# Current gain (from actual learned kernel, pure top-down)
gain_current = gain_actual  # [N] from above
current_weighted_activity = (base_activity * gain_current).sum().item()
current_peak_activity = (base_activity * gain_current)[peak_ch].item()

print(f"\nCurrent gain at stim: {gain_current[peak_ch].item():.6f}")
print(f"Current weighted activity sum: {current_weighted_activity:.6f}")
print(f"Current peak weighted activity: {current_peak_activity:.6f}")

# Goal: find apical_field such that:
# gain = 1 + 0.7 * tanh(3.0 * apical_field)
# gain[peak_ch] = gain_current[peak_ch]  (preserve peak)
# mean(gain) = 1.0  (energy neutral)

# Target gain profile:
# - At center: same as current (1.70)
# - Elsewhere: shift down so mean = 1.0
# Simple approach: normalize current gain to mean=1.0
gain_target = gain_current / gain_current.mean()  # mean=1.0 by construction

print(f"\nTarget (mean-normalized) gain:")
print(f"  At stim: {gain_target[peak_ch].item():.6f}")
print(f"  Mean: {gain_target.mean().item():.6f}")
print(f"  Min: {gain_target.min().item():.6f}")

# What apical_field would produce this gain?
# gain = 1 + 0.7 * tanh(3 * field)
# tanh(3 * field) = (gain - 1) / 0.7
# field = arctanh((gain - 1) / 0.7) / 3
mag = net.feedback.max_apical_gain
target_tanh = (gain_target - 1.0) / mag
# Clip to avoid arctanh(±1)
target_tanh = torch.clamp(target_tanh, -0.999, 0.999)
target_field = torch.atanh(target_tanh) / pi_eff.item()

print(f"\nRequired apical_field for target gain:")
print(f"  At stim: {target_field[peak_ch].item():.6f}")
print(f"  At far (ch 0): {target_field[0].item():.6f}")

# Now: what kernel K would produce this field?
# field = circulant(K) @ q_centered
# We need K such that circulant(K) @ q_centered = target_field
# For a circulant matrix, field[i] = sum_j K[(j-i)%N] * q_centered[j]
# This is a convolution. K = IFFT(FFT(field) / FFT(q_centered))
# But q_centered might have zero FFT components...

# Use the circulant property: in Fourier space, circulant multiplication = element-wise
q_centered_shifted = q_centered[0]  # [N]
fft_q = torch.fft.fft(q_centered_shifted)
fft_target = torch.fft.fft(target_field)

# Regularized inversion (avoid division by zero)
eps = 1e-6
fft_K_ideal = fft_target / (fft_q + eps * torch.sign(fft_q + eps))
K_ideal = torch.fft.ifft(fft_K_ideal).real

# Verify
circulant_ideal = net.feedback._to_circulant(K_ideal)
field_verify = (circulant_ideal @ q_centered_shifted.unsqueeze(-1)).squeeze(-1)
gain_verify = 1.0 + mag * torch.tanh(pi_eff * field_verify)

print(f"\nIdeal kernel (36 free weights):")
report_kernel("Ideal energy-efficient kernel", K_ideal, step_deg)

print(f"\nVerification (ideal kernel → gain):")
print(f"  Gain at stim: {gain_verify[peak_ch].item():.6f} (target: {gain_target[peak_ch].item():.6f})")
print(f"  Gain mean: {gain_verify.mean().item():.6f} (target: 1.0)")
print(f"  Gain min: {gain_verify.min().item():.6f}")
print(f"  Max field reconstruction error: {(field_verify - target_field).abs().max().item():.2e}")

# Energy comparison
weighted_ideal = (base_activity * gain_verify).sum().item()
weighted_current = (base_activity * gain_current).sum().item()
weighted_base = base_activity.sum().item()

print(f"\nEnergy comparison (activity × gain):")
print(f"  Base (no FB):    {weighted_base:.6f}")
print(f"  Current kernel:  {weighted_current:.6f} ({weighted_current/weighted_base:.3f}×)")
print(f"  Ideal kernel:    {weighted_ideal:.6f} ({weighted_ideal/weighted_base:.3f}×)")
print(f"  Savings:         {(weighted_current - weighted_ideal)/weighted_current*100:.1f}%")

# Can the 7-basis set approximate the ideal kernel?
# Project ideal kernel onto basis set: K_ideal ≈ Σ c_k * basis_k
# Solve least squares: c = (B^T B)^{-1} B^T K_ideal
B = basis.T  # [N, K]
c_opt = torch.linalg.lstsq(B, K_ideal).solution  # [K]
K_projected = (c_opt.unsqueeze(-1) * basis).sum(dim=0)
residual = (K_ideal - K_projected).norm().item()

print(f"\n--- Can the 7-basis set approximate the ideal kernel? ---")
print(f"\nOptimal basis coefficients (least squares):")
for k in range(K):
    print(f"  {basis_labels[k]:>15s}: {c_opt[k].item():>10.4f} (current α_apical: {alpha_apical[k].item():>10.4f})")
print(f"\n  Residual norm: {residual:.6f}")
print(f"  Relative error: {residual / K_ideal.norm().item() * 100:.1f}%")

# What gain does the projected kernel produce?
circulant_proj = net.feedback._to_circulant(K_projected)
field_proj = (circulant_proj @ q_centered_shifted.unsqueeze(-1)).squeeze(-1)
gain_proj = 1.0 + mag * torch.tanh(pi_eff * field_proj)

print(f"\nProjected kernel gain:")
print(f"  Gain at stim: {gain_proj[peak_ch].item():.6f} (target: {gain_target[peak_ch].item():.6f})")
print(f"  Gain mean: {gain_proj.mean().item():.6f} (target: 1.0)")
print(f"  Gain min: {gain_proj.min().item():.6f}")

weighted_proj = (base_activity * gain_proj).sum().item()
print(f"  Weighted activity: {weighted_proj:.6f} ({weighted_proj/weighted_base:.3f}×)")

# Compare all kernel profiles side by side
print(f"\n--- Kernel comparison: current vs ideal vs projected ---")
print(f"{'Ch':>4s} {'Deg':>6s}  {'Current':>10s}  {'Ideal':>10s}  {'Projected':>10s}  {'Basis0(5°)':>10s}")
print("-" * 55)
for ch in range(N):
    deg = ch * step_deg
    print(f"{ch:>4d} {deg:>5.1f}°  {K_apical[ch].item():>10.6f}  {K_ideal[ch].item():>10.6f}  "
          f"{K_projected[ch].item():>10.6f}  {basis[0, ch].item():>10.6f}")

# Compare gain profiles
print(f"\n--- Gain comparison: current vs ideal vs projected ---")
print(f"{'Ch':>4s} {'Deg':>6s}  {'Current':>10s}  {'Ideal':>10s}  {'Projected':>10s}  {'base_act':>10s}")
print("-" * 60)
for ch in range(N):
    deg = ch * step_deg
    marker = " <-- stim" if ch == peak_ch else ""
    print(f"{ch:>4d} {deg:>5.1f}°  {gain_current[ch].item():>10.6f}  {gain_verify[ch].item():>10.6f}  "
          f"{gain_proj[ch].item():>10.6f}  {base_activity[ch].item():>10.6f}{marker}")
