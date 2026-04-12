"""Debugger: Experiment 3 — ideal energy-efficient kernel (fixed FFT)."""

import sys
sys.path.insert(0, "/mnt/c/Users/User/codingproj/freshstart")

import torch
import torch.nn.functional as F
from src.config import load_config
from src.model.network import LaminarV1V2Network
from src.state import initial_state
from src.stimulus.gratings import generate_grating

model_cfg, train_cfg, stim_cfg = load_config("config/exp_branch_a.yaml")
N = model_cfg.n_orientations
period = model_cfg.orientation_range
sigma_ff = model_cfg.sigma_ff
step_deg = period / N

net = LaminarV1V2Network(model_cfg, delta_som=train_cfg.delta_som)
ckpt = torch.load("results/control_no_gate/s42/center_surround_seed42/checkpoint.pt",
                   map_location="cpu", weights_only=False)
net.load_state_dict(ckpt["model_state"], strict=False)
net.eval()

theta_stim = 90.0
peak_ch = int(theta_stim / step_deg)
oracle_sigma = getattr(train_cfg, 'oracle_sigma', 12.0)
oracle_pi = train_cfg.oracle_pi
mag = net.feedback.max_apical_gain

print(f"max_apical_gain: {mag}")
print(f"oracle_pi: {oracle_pi}")

# Get current gain and base activity
q_pred = net._make_bump(torch.tensor([theta_stim]), sigma=oracle_sigma)
q_centered = (q_pred - 1.0 / N)[0]  # [N]

K_apical = net.feedback.get_apical_profile()
circ_apical = net.feedback._to_circulant(K_apical)
field_current = (circ_apical @ q_centered.unsqueeze(-1)).squeeze(-1)  # [N]
gain_current = 1.0 + mag * torch.tanh(oracle_pi * field_current)

# Base activity (FB OFF)
stim = generate_grating(torch.tensor([theta_stim]), torch.tensor([0.5]),
                        N, sigma_ff, period=period)
cue = torch.zeros(1, 2)
task = torch.zeros(1, 2)
net.oracle_mode = True
net.oracle_q_pred = q_pred
net.oracle_pi_pred = torch.full((1, 1), oracle_pi)
net.feedback_scale.fill_(0.0)

state = initial_state(1, N, model_cfg.v2_hidden_dim)
with torch.no_grad():
    for t in range(12):
        state, _ = net.step(stim, cue, task, state)
base_activity = state.r_l23[0].clone()

print(f"\nBase activity sum: {base_activity.sum().item():.6f}")
print(f"Base activity at stim: {base_activity[peak_ch].item():.6f}")
print(f"Current gain at stim: {gain_current[peak_ch].item():.6f}")
print(f"Current gain mean: {gain_current.mean().item():.6f}")

# Target: mean-normalized gain (energy neutral)
gain_target = gain_current / gain_current.mean()

print(f"\nTarget gain at stim: {gain_target[peak_ch].item():.6f}")
print(f"Target gain mean: {gain_target.mean().item():.6f}")
print(f"Target gain min: {gain_target.min().item():.6f}")

# Required apical field for target gain
# gain = 1 + mag * tanh(pi * field)
# field = atanh((gain - 1) / mag) / pi
target_tanh = torch.clamp((gain_target - 1.0) / mag, -0.999, 0.999)
target_field = torch.atanh(target_tanh) / oracle_pi

print(f"\nTarget field at stim: {target_field[peak_ch].item():.6f}")
print(f"Target field at far (ch 0): {target_field[0].item():.6f}")

# Ideal kernel via FFT deconvolution
fft_q = torch.fft.fft(q_centered)
fft_target = torch.fft.fft(target_field)

# Regularized division (Wiener-like)
eps = 1e-4
fft_K_ideal = fft_target * fft_q.conj() / (fft_q.abs()**2 + eps)
K_ideal = torch.fft.ifft(fft_K_ideal).real

# Verify
circ_ideal = net.feedback._to_circulant(K_ideal)
field_verify = (circ_ideal @ q_centered.unsqueeze(-1)).squeeze(-1)
gain_verify = 1.0 + mag * torch.tanh(oracle_pi * field_verify)

print(f"\n{'='*80}")
print("IDEAL KERNEL (36 free weights)")
print(f"{'='*80}")
print(f"\nReconstruction error (field): {(field_verify - target_field).abs().max().item():.2e}")
print(f"Gain at stim: {gain_verify[peak_ch].item():.6f} (target: {gain_target[peak_ch].item():.6f})")
print(f"Gain mean: {gain_verify.mean().item():.6f} (target: 1.0)")
print(f"Gain min: {gain_verify.min().item():.6f}")

# Energy comparison
weighted_base = base_activity.sum().item()
weighted_current = (base_activity * gain_current).sum().item()
weighted_ideal = (base_activity * gain_verify).sum().item()

print(f"\nEnergy comparison (activity × gain sum):")
print(f"  Base (no FB):    {weighted_base:.6f} (1.000×)")
print(f"  Current kernel:  {weighted_current:.6f} ({weighted_current/weighted_base:.3f}×)")
print(f"  Ideal kernel:    {weighted_ideal:.6f} ({weighted_ideal/weighted_base:.3f}×)")
if weighted_current > weighted_base:
    savings = (weighted_current - weighted_ideal) / (weighted_current - weighted_base) * 100
    print(f"  Energy surplus removed: {savings:.1f}%")

# Project ideal kernel onto 7-basis set
basis = net.feedback.basis  # [K, N]
K_basis = basis.shape[0]
B = basis.T  # [N, K]
c_opt = torch.linalg.lstsq(B, K_ideal).solution  # [K]
K_projected = (c_opt.unsqueeze(-1) * basis).sum(dim=0)
residual = (K_ideal - K_projected).norm().item()

alpha_apical = net.feedback.alpha_apical.data

basis_labels = ["G(σ=5°)", "G(σ=15°)", "G(σ=30°)", "G(σ=60°)", "MexHat(10-30)", "Constant", "Odd/sin"]

print(f"\n{'='*80}")
print("CAN THE 7-BASIS SET APPROXIMATE THE IDEAL KERNEL?")
print(f"{'='*80}")
print(f"\nResidual norm: {residual:.6f} ({residual / K_ideal.norm().item() * 100:.1f}% of ideal kernel norm)")

print(f"\n{'Basis':>15s}  {'Ideal coeff':>12s}  {'Current α':>12s}  {'Diff':>12s}")
print("-" * 55)
for k in range(K_basis):
    print(f"{basis_labels[k]:>15s}  {c_opt[k].item():>12.6f}  {alpha_apical[k].item():>12.6f}  "
          f"{c_opt[k].item() - alpha_apical[k].item():>+12.6f}")

# Projected kernel gain
circ_proj = net.feedback._to_circulant(K_projected)
field_proj = (circ_proj @ q_centered.unsqueeze(-1)).squeeze(-1)
gain_proj = 1.0 + mag * torch.tanh(oracle_pi * field_proj)
weighted_proj = (base_activity * gain_proj).sum().item()

print(f"\nProjected kernel performance:")
print(f"  Gain at stim: {gain_proj[peak_ch].item():.6f} (current: {gain_current[peak_ch].item():.6f})")
print(f"  Gain mean: {gain_proj.mean().item():.6f} (target: 1.0)")
print(f"  Gain min: {gain_proj.min().item():.6f}")
print(f"  Weighted activity: {weighted_proj:.6f} ({weighted_proj/weighted_base:.3f}×)")

# Full profile comparison
print(f"\n{'='*80}")
print("GAIN PROFILE COMPARISON")
print(f"{'='*80}")
print(f"{'Ch':>4s} {'Deg':>6s}  {'Current':>10s}  {'Ideal':>10s}  {'Projected':>10s}  {'base_act':>10s}")
print("-" * 55)
for ch in range(N):
    deg = ch * step_deg
    marker = " <-- stim" if ch == peak_ch else ""
    print(f"{ch:>4d} {deg:>5.1f}°  {gain_current[ch].item():>10.6f}  {gain_verify[ch].item():>10.6f}  "
          f"{gain_proj[ch].item():>10.6f}  {base_activity[ch].item():>10.6f}{marker}")

# Kernel profile comparison
print(f"\n{'='*80}")
print("KERNEL PROFILE COMPARISON")
print(f"{'='*80}")
print(f"{'Ch':>4s} {'Deg':>6s}  {'Current':>10s}  {'Ideal':>10s}  {'Projected':>10s}")
print("-" * 40)
for ch in range(N):
    deg = ch * step_deg
    print(f"{ch:>4d} {deg:>5.1f}°  {K_apical[ch].item():>10.6f}  {K_ideal[ch].item():>10.6f}  "
          f"{K_projected[ch].item():>10.6f}")

# What are the FWHM of the ideal kernel?
peak_val = K_ideal.max().item()
if peak_val > 0:
    half_max = peak_val / 2
    above = (K_ideal >= half_max).sum().item()
    print(f"\nIdeal kernel FWHM: ~{above * step_deg:.0f}° ({above} channels above half-max)")

# Does the ideal kernel have negative regions?
pos_area = K_ideal[K_ideal > 0].sum().item()
neg_area = K_ideal[K_ideal < 0].sum().item()
print(f"Ideal kernel positive area: {pos_area:.6f}")
print(f"Ideal kernel negative area: {neg_area:.6f}")
n_neg = (K_ideal < 0).sum().item()
print(f"Ideal kernel negative channels: {n_neg}/{N}")
