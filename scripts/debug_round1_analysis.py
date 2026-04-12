"""Debugger: Round 1 analysis — mag=0.3, λ_energy=0.5, 36 direct weights.

Focus areas:
1. Gain profile — saturation check (is tanh(pi*field) ≈ 1 at center?)
2. Kernel shape — did center-surround emerge?
3. Amplitude decomposition — gain vs SOM vs structural overhead
4. Extrapolation — what mag gives amp ≈ 1.10?
"""

import sys
sys.path.insert(0, "/mnt/c/Users/User/codingproj/freshstart")

import torch
import torch.nn.functional as F
import math
from src.config import load_config
from src.model.network import LaminarV1V2Network
from src.state import initial_state
from src.stimulus.gratings import generate_grating

# ── Setup ────────────────────────────────────────────────────────────
model_cfg, train_cfg, stim_cfg = load_config("config/exp_iter1.yaml")
N = model_cfg.n_orientations
period = model_cfg.orientation_range
sigma_ff = model_cfg.sigma_ff
step_deg = period / N
steps_on = train_cfg.steps_on

theta_stim = 90.0
contrast = 0.5
peak_ch = int(theta_stim / step_deg)  # 18
oracle_sigma = getattr(train_cfg, 'oracle_sigma', 12.0)
oracle_pi = train_cfg.oracle_pi

stim = generate_grating(torch.tensor([theta_stim]), torch.tensor([contrast]),
                        N, sigma_ff, period=period)
cue = torch.zeros(1, 2)
task = torch.zeros(1, 2)

# Load
net = LaminarV1V2Network(model_cfg, delta_som=train_cfg.delta_som)
ckpt = torch.load("results/iter/r1_mag03_e05/center_surround_seed42/checkpoint.pt",
                   map_location="cpu", weights_only=False)
net.load_state_dict(ckpt["model_state"], strict=False)
net.eval()

mag = net.feedback.max_apical_gain
print(f"max_apical_gain: {mag}")
print(f"oracle_pi: {oracle_pi}")

# ── Kernel profiles ──────────────────────────────────────────────────
alpha_inh = net.feedback.alpha_inh.data.clone()
alpha_vip = net.feedback.alpha_vip.data.clone()
alpha_apical = net.feedback.alpha_apical.data.clone()

print(f"\n{'='*80}")
print("KERNEL PROFILES")
print(f"{'='*80}")
print(f"\n{'Ch':>4s} {'Deg':>6s}  {'α_inh':>10s}  {'α_vip':>10s}  {'α_apical':>10s}")
print("-" * 50)
for ch in range(N):
    deg = ch * step_deg
    marker = " <-- center" if ch == 0 else ""
    print(f"{ch:>4d} {deg:>5.1f}°  {alpha_inh[ch].item():>10.6f}  "
          f"{alpha_vip[ch].item():>10.6f}  {alpha_apical[ch].item():>10.6f}{marker}")

print(f"\nApical: max={alpha_apical.max().item():.6f} (ch {alpha_apical.argmax().item()}), "
      f"min={alpha_apical.min().item():.6f} (ch {alpha_apical.argmin().item()}), "
      f"mean={alpha_apical.mean().item():.6f}")
print(f"  positive: {(alpha_apical > 0).sum().item()}/{N}, negative: {(alpha_apical < 0).sum().item()}/{N}")
print(f"SOM: max={alpha_inh.max().item():.6f} (ch {alpha_inh.argmax().item()}), "
      f"min={alpha_inh.min().item():.6f} (ch {alpha_inh.argmin().item()})")
print(f"  positive: {(alpha_inh > 0).sum().item()}/{N}, negative: {(alpha_inh < 0).sum().item()}/{N}")
print(f"VIP: max={alpha_vip.max().item():.6f}, min={alpha_vip.min().item():.6f}")

# ======================================================================
# 1. GAIN PROFILE + SATURATION CHECK
# ======================================================================
print(f"\n{'='*80}")
print("1. GAIN PROFILE + SATURATION CHECK")
print(f"{'='*80}")

q_pred = net._make_bump(torch.tensor([theta_stim]), sigma=oracle_sigma)
q_centered = q_pred - 1.0 / N  # [1, N]

# Get apical field manually to check saturation
K_apical = net.feedback.get_apical_profile()
circ_apical = net.feedback._to_circulant(K_apical)
apical_field = (circ_apical @ q_centered[0].unsqueeze(-1)).squeeze(-1)  # [N]

# tanh argument = pi * field
tanh_arg = oracle_pi * apical_field
tanh_val = torch.tanh(tanh_arg)
gain = 1.0 + mag * tanh_val

print(f"\nq_pred sum: {q_pred.sum().item():.4f}, mean: {q_pred.mean().item():.6f}")

# Gain statistics
print(f"\nGain profile:")
print(f"  mean: {gain.mean().item():.6f}")
print(f"  at stim (ch {peak_ch}): {gain[peak_ch].item():.6f}")
print(f"  max: {gain.max().item():.6f} at ch {gain.argmax().item()}")
print(f"  min: {gain.min().item():.6f} at ch {gain.argmin().item()}")
print(f"  channels < 1.0: {(gain < 1.0).sum().item()}/{N}")
print(f"  sum(gain-1): {(gain - 1.0).sum().item():+.6f}")

print(f"\nSATURATION CHECK (tanh argument and value):")
print(f"{'Ch':>4s} {'Deg':>6s}  {'field':>10s}  {'pi*field':>10s}  {'tanh':>10s}  {'gain':>10s}  {'|tanh|':>8s}")
print("-" * 70)
for ch in range(N):
    deg = ch * step_deg
    f = apical_field[ch].item()
    ta = tanh_arg[ch].item()
    tv = tanh_val[ch].item()
    g = gain[ch].item()
    sat = abs(tv)
    marker = " <-- stim" if ch == peak_ch else ""
    # Flag saturation
    sat_flag = " SAT" if sat > 0.99 else (" ~SAT" if sat > 0.95 else "")
    print(f"{ch:>4d} {deg:>5.1f}°  {f:>10.6f}  {ta:>10.4f}  {tv:>10.6f}  {g:>10.6f}  {sat:>8.4f}{sat_flag}{marker}")

# Count saturated channels
n_sat_pos = (tanh_val > 0.99).sum().item()
n_sat_neg = (tanh_val < -0.99).sum().item()
n_near_sat_pos = ((tanh_val > 0.95) & (tanh_val <= 0.99)).sum().item()
n_near_sat_neg = ((tanh_val < -0.95) & (tanh_val >= -0.99)).sum().item()
print(f"\nSaturation summary:")
print(f"  Fully saturated positive (tanh > 0.99): {n_sat_pos}/{N}")
print(f"  Near-saturated positive (0.95-0.99): {n_near_sat_pos}/{N}")
print(f"  Fully saturated negative (tanh < -0.99): {n_sat_neg}/{N}")
print(f"  Near-saturated negative (-0.99 to -0.95): {n_near_sat_neg}/{N}")
print(f"  Unsaturated (|tanh| < 0.95): {N - n_sat_pos - n_sat_neg - n_near_sat_pos - n_near_sat_neg}/{N}")

# ======================================================================
# 2. KERNEL SHAPE — CENTER-SURROUND?
# ======================================================================
print(f"\n{'='*80}")
print("2. KERNEL SHAPE ANALYSIS")
print(f"{'='*80}")

# Apical kernel structure
# Channels near 0 (offset 0) are "center", channels near N/2 are "anti-preferred"
center_region = list(range(0, 6)) + list(range(31, 36))  # ±25° from center
surround_region = list(range(6, 31))  # 30-150°

apical_center_mean = alpha_apical[center_region].mean().item()
apical_surround_mean = alpha_apical[surround_region].mean().item()

print(f"\nApical kernel:")
print(f"  Center region (±25°): mean = {apical_center_mean:.6f}")
print(f"  Surround region (30-150°): mean = {apical_surround_mean:.6f}")
print(f"  Center-surround contrast: {apical_center_mean - apical_surround_mean:.6f}")
if apical_center_mean > 0 and apical_surround_mean < 0:
    print(f"  Shape: CENTER-SURROUND (positive center, negative surround)")
elif apical_center_mean > apical_surround_mean:
    print(f"  Shape: Peaked (positive bias at center, weaker at surround)")
else:
    print(f"  Shape: Inverted or flat")

# SOM kernel structure
inh_center_mean = alpha_inh[center_region].mean().item()
inh_surround_mean = alpha_inh[surround_region].mean().item()
print(f"\nSOM kernel:")
print(f"  Center region (±25°): mean = {inh_center_mean:.6f}")
print(f"  Surround region (30-150°): mean = {inh_surround_mean:.6f}")
if inh_center_mean < inh_surround_mean:
    print(f"  Shape: CENTER-SPARING (lower at center, higher at surround)")
elif inh_center_mean < 0 and inh_surround_mean > 0:
    print(f"  Shape: INVERTED CENTER-SURROUND (negative center, positive surround)")
else:
    print(f"  Shape: Broad / flat")

# ======================================================================
# 3. AMPLITUDE DECOMPOSITION
# ======================================================================
print(f"\n{'='*80}")
print("3. AMPLITUDE DECOMPOSITION — WHERE DOES amp=1.45 COME FROM?")
print(f"{'='*80}")

# Run with FB ON (oracle mode)
net.oracle_mode = True
net.oracle_q_pred = q_pred
net.oracle_pi_pred = torch.full((1, 1), oracle_pi)
net.feedback_scale.fill_(1.0)

state_on = initial_state(1, N, model_cfg.v2_hidden_dim)
with torch.no_grad():
    net.l23.cache_kernels()
    net.feedback.cache_kernels()
    for t in range(steps_on):
        state_on, aux_on = net.step(stim, cue, task, state_on)
    net.l23.uncache_kernels()
    net.feedback.uncache_kernels()

r_l23_on = state_on.r_l23[0].clone()

# Run with FB OFF
net.feedback_scale.fill_(0.0)
state_off = initial_state(1, N, model_cfg.v2_hidden_dim)
with torch.no_grad():
    net.l23.cache_kernels()
    net.feedback.cache_kernels()
    for t in range(steps_on):
        state_off, _ = net.step(stim, cue, task, state_off)
    net.l23.uncache_kernels()
    net.feedback.uncache_kernels()

r_l23_off = state_off.r_l23[0].clone()
net.feedback_scale.fill_(1.0)

total_on = r_l23_on.sum().item()
total_off = r_l23_off.sum().item()
amp_ratio = total_on / total_off if total_off > 0 else float('inf')

# Get SOM/VIP drives
with torch.no_grad():
    som_drive, vip_drive, apical_gain_full = net.feedback(q_pred, torch.full((1,1), oracle_pi), r_l4=None)

apical_gain_vec = apical_gain_full[0]
activity_weighted_gain = (r_l23_off * apical_gain_vec).sum().item() / total_off if total_off > 0 else 0

print(f"\nL2/3 activity:")
print(f"  FB OFF total: {total_off:.6f}")
print(f"  FB ON total:  {total_on:.6f}")
print(f"  Amplitude ratio (ON/OFF): {amp_ratio:.4f}")
print(f"  Activity-weighted mean gain: {activity_weighted_gain:.4f}")

# Decompose: what contributes to the amplitude increase?
# The ON activity is shaped by: (1) apical gain on excitatory drive, (2) SOM suppression, (3) recurrence
# We can't cleanly separate them in the nonlinear dynamics, but we can measure:

# (a) "Gain-only" effect: multiply OFF activity by gain (ignores SOM interaction)
gain_only_total = (r_l23_off * apical_gain_vec).sum().item()
gain_only_ratio = gain_only_total / total_off if total_off > 0 else 0
print(f"\n  Gain-only effect (OFF × gain): {gain_only_total:.6f} ({gain_only_ratio:.4f}× of OFF)")

# (b) SOM steady state
r_som_on = state_on.r_som[0]
r_som_off = state_off.r_som[0]
print(f"\n  SOM total (ON): {r_som_on.sum().item():.6f}")
print(f"  SOM total (OFF): {r_som_off.sum().item():.6f}")
print(f"  SOM increase: {(r_som_on.sum() - r_som_off.sum()).item():.6f}")

# (c) Effective SOM drive at each channel
r_vip_on = state_on.r_vip[0]
w_vip_som = F.softplus(net.w_vip_som).item()
eff_som = F.relu(som_drive - F.softplus(net.w_vip_som) * r_vip_on.unsqueeze(0))
print(f"\n  Effective SOM (after VIP subtraction):")
print(f"    at stim: {eff_som[0, peak_ch].item():.6f}")
print(f"    mean: {eff_som.mean().item():.6f}")
print(f"    max: {eff_som.max().item():.6f} at ch {eff_som[0].argmax().item()}")

# (d) Energy decomposition
e_l4_on = state_on.r_l4[0].abs().mean().item()
e_l23_on = r_l23_on.abs().mean().item()
e_tmpl_on = state_on.deep_template[0].abs().mean().item()
e_pv_on = state_on.r_pv[0].abs().mean().item()
e_som_on = r_som_on.abs().mean().item()
e_vip_on = r_vip_on.abs().mean().item()
e_total_on = e_l4_on + e_l23_on + e_tmpl_on + e_pv_on + e_som_on + e_vip_on

e_l4_off = state_off.r_l4[0].abs().mean().item()
e_l23_off = r_l23_off.abs().mean().item()
e_tmpl_off = state_off.deep_template[0].abs().mean().item()
e_pv_off = state_off.r_pv[0].abs().mean().item()
e_som_off = r_som_off.abs().mean().item()
e_vip_off = state_off.r_vip[0].abs().mean().item()
e_total_off = e_l4_off + e_l23_off + e_tmpl_off + e_pv_off + e_som_off + e_vip_off

print(f"\nEnergy decomposition:")
print(f"{'Component':>15s}  {'FB OFF':>10s}  {'FB ON':>10s}  {'Δ':>10s}  {'%total_ON':>10s}")
print("-" * 60)
for name, off, on in [
    ("L4", e_l4_off, e_l4_on),
    ("L2/3", e_l23_off, e_l23_on),
    ("Deep template", e_tmpl_off, e_tmpl_on),
    ("PV", e_pv_off, e_pv_on),
    ("SOM", e_som_off, e_som_on),
    ("VIP", e_vip_off, e_vip_on),
    ("TOTAL", e_total_off, e_total_on),
]:
    delta = on - off
    pct_on = on / e_total_on * 100 if e_total_on > 0 else 0
    print(f"{name:>15s}  {off:>10.6f}  {on:>10.6f}  {delta:>+10.6f}  {pct_on:>10.1f}%")

# Per-channel detail
print(f"\n{'Ch':>4s} {'Deg':>6s}  {'OFF':>10s}  {'ON':>10s}  {'gain':>8s}  {'som_eff':>10s}  {'r_som':>10s}")
print("-" * 70)
for ch in range(N):
    deg = ch * step_deg
    marker = " <-- stim" if ch == peak_ch else ""
    print(f"{ch:>4d} {deg:>5.1f}°  {r_l23_off[ch].item():>10.6f}  {r_l23_on[ch].item():>10.6f}  "
          f"{apical_gain_vec[ch].item():>8.4f}  {eff_som[0,ch].item():>10.6f}  "
          f"{r_som_on[ch].item():>10.6f}{marker}")

# ======================================================================
# 4. EXTRAPOLATION — WHAT MAG GIVES AMP ≈ 1.10?
# ======================================================================
print(f"\n{'='*80}")
print("4. EXTRAPOLATION — WHAT MAG GIVES AMP ≈ 1.10?")
print(f"{'='*80}")

# Method: use the learned kernel shape (alpha_apical) to simulate different mag values.
# Keep the kernel fixed, vary only max_apical_gain.
# For each mag: compute gain profile → multiply OFF activity → estimate amp.

# This is an APPROXIMATION because:
# - The actual dynamics are nonlinear (gain affects recurrence which affects SOM etc.)
# - The optimizer would learn a different kernel at different mag
# - But it gives a first-order estimate

print(f"\nMethod: Fix kernel shape from this checkpoint, vary mag, compute gain → multiply OFF activity")
print(f"(Approximation: ignores SOM interaction and recurrence changes)")

mag_values = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70]

print(f"\n{'mag':>6s}  {'gain_peak':>10s}  {'gain_mean':>10s}  {'gain_min':>10s}  {'act_wtd':>10s}  {'est_amp':>10s}")
print("-" * 65)

for m in mag_values:
    g = 1.0 + m * tanh_val  # tanh_val from the learned kernel
    g_peak = g[peak_ch].item()
    g_mean = g.mean().item()
    g_min = g.min().item()

    # Activity-weighted gain = estimated amp ratio
    act_wtd = (r_l23_off * g).sum().item() / total_off if total_off > 0 else 0

    marker = " *** TARGET" if abs(act_wtd - 1.10) < 0.05 else ""
    print(f"{m:>6.2f}  {g_peak:>10.6f}  {g_mean:>10.6f}  {g_min:>10.6f}  {act_wtd:>10.4f}  ~{act_wtd:>8.4f}{marker}")

# Find exact mag for amp=1.10 by interpolation
# act_weighted_gain = sum(r_off * (1 + mag * tanh_val)) / sum(r_off)
#                   = 1 + mag * sum(r_off * tanh_val) / sum(r_off)
# Set to 1.10: mag = 0.10 / (sum(r_off * tanh_val) / sum(r_off))
weighted_tanh = (r_l23_off * tanh_val).sum().item() / total_off
print(f"\nActivity-weighted tanh (from learned kernel): {weighted_tanh:.6f}")
mag_for_1_10 = 0.10 / weighted_tanh if weighted_tanh > 0 else float('inf')
mag_for_1_05 = 0.05 / weighted_tanh if weighted_tanh > 0 else float('inf')
mag_for_1_15 = 0.15 / weighted_tanh if weighted_tanh > 0 else float('inf')

print(f"\nExtrapolated mag for target amplitudes (linear approximation):")
print(f"  amp = 1.05 → mag ≈ {mag_for_1_05:.4f}")
print(f"  amp = 1.10 → mag ≈ {mag_for_1_10:.4f}")
print(f"  amp = 1.15 → mag ≈ {mag_for_1_15:.4f}")

# But this assumes tanh doesn't change saturation. Let's also compute the
# EXACT gain at the extrapolated mag (in case tanh is in the linear regime there)
if mag_for_1_10 < 10:
    g_exact = 1.0 + mag_for_1_10 * tanh_val
    act_wtd_exact = (r_l23_off * g_exact).sum().item() / total_off
    g_at_stim = g_exact[peak_ch].item()
    print(f"\n  At mag={mag_for_1_10:.4f}:")
    print(f"    Gain at stim: {g_at_stim:.6f}")
    print(f"    Gain mean: {g_exact.mean().item():.6f}")
    print(f"    Activity-weighted gain: {act_wtd_exact:.4f}")
    print(f"    Channels < 1.0: {(g_exact < 1.0).sum().item()}/{N}")

# IMPORTANT CAVEAT: This extrapolation assumes the kernel shape stays the same.
# At lower mag, the optimizer might learn LARGER alpha weights to compensate,
# potentially re-saturating tanh. We need the actual training runs to confirm.

# More sophisticated: what if tanh de-saturates at lower mag?
# The field is fixed (from this kernel). So pi*field values don't change.
# If pi*field >> 1 (saturated), then tanh ≈ ±1 regardless of mag.
# Only if pi*field is moderate (transition region) does mag matter.

# The pi*field values from this kernel:
print(f"\nField analysis (determines if tanh saturates):")
print(f"  Field at stim: {apical_field[peak_ch].item():.6f}")
print(f"  pi * field at stim: {tanh_arg[peak_ch].item():.4f}")
print(f"  |tanh| at stim: {abs(tanh_val[peak_ch].item()):.6f}")

# Where is the transition zone?
for ch in range(N):
    tv = abs(tanh_val[ch].item())
    if 0.1 < tv < 0.9:
        deg = ch * step_deg
        print(f"  Transition: ch {ch} ({deg:.1f}°): |tanh| = {tv:.4f}, field = {apical_field[ch].item():.6f}")

# ======================================================================
# 5. COMPARISON ACROSS CHECKPOINTS
# ======================================================================
print(f"\n{'='*80}")
print("5. COMPARISON TABLE")
print(f"{'='*80}")

print(f"\n{'Metric':<40s}  {'mag=0.7(36w)':>14s}  {'mag=0.3(R1)':>14s}")
print("-" * 75)
print(f"{'max_apical_gain':<40s}  {'0.70':>14s}  {'0.30':>14s}")
print(f"{'Gain at stim':<40s}  {'1.699':>14s}  {gain[peak_ch].item():>14.3f}")
print(f"{'Mean gain':<40s}  {'0.780':>14s}  {gain.mean().item():>14.3f}")
print(f"{'Gain min':<40s}  {'0.302':>14s}  {gain.min().item():>14.3f}")
print(f"{'Channels < 1.0':<40s}  {'23/36':>14s}  {f'{(gain<1.0).sum().item():.0f}/36':>14s}")
print(f"{'sum(gain-1)':<40s}  {'-7.92':>14s}  {(gain-1).sum().item():>14.2f}")
print(f"{'L2/3 amp ratio':<40s}  {'2.13':>14s}  {amp_ratio:>14.2f}")
print(f"{'Activity-weighted gain':<40s}  {'1.69':>14s}  {activity_weighted_gain:>14.2f}")
print(f"{'Total energy ratio':<40s}  {'5.16':>14s}  {e_total_on/e_total_off:>14.2f}")
print(f"{'VIP active?':<40s}  {'No':>14s}  {'No' if alpha_vip.abs().max().item() < 0.001 else 'Yes':>14s}")

# Apical kernel center-surround characterization
print(f"\nApical kernel shape (R1):")
print(f"  Center (±25°) mean: {apical_center_mean:.6f}")
print(f"  Surround (30-150°) mean: {apical_surround_mean:.6f}")
if alpha_apical.max().item() > 0:
    peak_val = alpha_apical.max().item()
    half_max = peak_val / 2
    above = (alpha_apical >= half_max).sum().item()
    print(f"  FWHM: ~{above * step_deg:.0f}°")
print(f"  Positive sum: {alpha_apical[alpha_apical > 0].sum().item():.6f}")
neg_sum = alpha_apical[alpha_apical < 0].sum().item() if (alpha_apical < 0).any() else 0
print(f"  Negative sum: {neg_sum:.6f}")
