"""Debugger: Comprehensive analysis of 36-weight checkpoint (exp5_direct36_e05_nofb).

5 investigations:
1. Gain profile at 90° stimulus
2. L2/3 activity ON vs OFF (global amplitude ratio)
3. SOM profile and effective_som_drive
4. Energy decomposition
5. True vs wrong template selectivity
"""

import sys
sys.path.insert(0, "/mnt/c/Users/User/codingproj/freshstart")

import torch
import torch.nn.functional as F
from src.config import load_config
from src.model.network import LaminarV1V2Network
from src.state import initial_state
from src.stimulus.gratings import generate_grating

# ── Setup ────────────────────────────────────────────────────────────
model_cfg, train_cfg, stim_cfg = load_config("config/exp_eff5.yaml")
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

# Load network with checkpoint
net = LaminarV1V2Network(model_cfg, delta_som=train_cfg.delta_som)
ckpt = torch.load("results/eff_tests/exp5_direct36_e05_nofb/center_surround_seed42/checkpoint.pt",
                   map_location="cpu", weights_only=False)

# Check what's in the checkpoint
print("=" * 80)
print("CHECKPOINT CONTENTS")
print("=" * 80)
ckpt_keys = list(ckpt["model_state"].keys())
fb_keys = [k for k in ckpt_keys if "feedback" in k]
print(f"Feedback-related keys: {fb_keys}")
for k in fb_keys:
    v = ckpt["model_state"][k]
    if hasattr(v, 'shape'):
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
    else:
        print(f"  {k}: {v}")

net.load_state_dict(ckpt["model_state"], strict=False)
net.eval()

print(f"\nmax_apical_gain (from config): {net.feedback.max_apical_gain}")
print(f"oracle_pi: {oracle_pi}")
print(f"N={N}, step_deg={step_deg}°, peak_ch={peak_ch}")

# ── Extract raw kernel profiles ──────────────────────────────────────
alpha_inh = net.feedback.alpha_inh.data.clone()
alpha_vip = net.feedback.alpha_vip.data.clone()
alpha_apical = net.feedback.alpha_apical.data.clone()

print(f"\n{'='*80}")
print("RAW KERNEL PROFILES (36 weights each)")
print(f"{'='*80}")
print(f"\n{'Ch':>4s} {'Deg':>6s}  {'α_inh':>10s}  {'α_vip':>10s}  {'α_apical':>10s}")
print("-" * 50)
for ch in range(N):
    deg = ch * step_deg
    marker = " <-- stim center" if ch == peak_ch else ""
    print(f"{ch:>4d} {deg:>5.1f}°  {alpha_inh[ch].item():>10.6f}  "
          f"{alpha_vip[ch].item():>10.6f}  {alpha_apical[ch].item():>10.6f}{marker}")

print(f"\nApical kernel stats:")
print(f"  max: {alpha_apical.max().item():.6f} at ch {alpha_apical.argmax().item()} ({alpha_apical.argmax().item()*step_deg:.1f}°)")
print(f"  min: {alpha_apical.min().item():.6f} at ch {alpha_apical.argmin().item()} ({alpha_apical.argmin().item()*step_deg:.1f}°)")
print(f"  mean: {alpha_apical.mean().item():.6f}")
print(f"  positive: {(alpha_apical > 0).sum().item()}/{N}")
print(f"  negative: {(alpha_apical < 0).sum().item()}/{N}")

print(f"\nSOM kernel stats:")
print(f"  max: {alpha_inh.max().item():.6f} at ch {alpha_inh.argmax().item()}")
print(f"  min: {alpha_inh.min().item():.6f} at ch {alpha_inh.argmin().item()}")
print(f"  positive: {(alpha_inh > 0).sum().item()}/{N}")
print(f"  negative: {(alpha_inh < 0).sum().item()}/{N}")

print(f"\nVIP kernel stats:")
print(f"  max: {alpha_vip.max().item():.6f} at ch {alpha_vip.argmax().item()}")
print(f"  min: {alpha_vip.min().item():.6f} at ch {alpha_vip.argmin().item()}")

# ======================================================================
# INVESTIGATION 1: GAIN PROFILE
# ======================================================================
print(f"\n{'='*80}")
print("INVESTIGATION 1: APICAL GAIN PROFILE")
print(f"{'='*80}")

# Set up oracle mode
q_pred = net._make_bump(torch.tensor([theta_stim]), sigma=oracle_sigma)
print(f"\nq_pred sum: {q_pred.sum().item():.4f}")
print(f"q_pred mean: {q_pred.mean().item():.6f}")
print(f"1/N: {1.0/N:.6f}")

# Run forward pass with feedback ON
net.oracle_mode = True
net.oracle_q_pred = q_pred
net.oracle_pi_pred = torch.full((1, 1), oracle_pi)
net.feedback_scale.fill_(1.0)

state = initial_state(1, N, model_cfg.v2_hidden_dim)
with torch.no_grad():
    net.l23.cache_kernels()
    net.feedback.cache_kernels()
    for t in range(steps_on):
        state, aux = net.step(stim, cue, task, state)
    net.l23.uncache_kernels()
    net.feedback.uncache_kernels()

r_l23_on = state.r_l23[0].clone()
r_l4_on = state.r_l4[0].clone()
r_som_on = state.r_som[0].clone()
r_vip_on = state.r_vip[0].clone()
deep_tmpl_on = state.deep_template[0].clone()

# Reconstruct gain from the feedback operator directly
with torch.no_grad():
    som_drive, vip_drive, apical_gain = net.feedback(q_pred, torch.full((1,1), oracle_pi), r_l4=None)
gain = apical_gain[0]

print(f"\nApical gain profile (pure top-down, r_l4=None):")
print(f"  mean: {gain.mean().item():.6f}")
print(f"  at stim (ch {peak_ch}): {gain[peak_ch].item():.6f}")
print(f"  max: {gain.max().item():.6f} at ch {gain.argmax().item()} ({gain.argmax().item()*step_deg:.1f}°)")
print(f"  min: {gain.min().item():.6f} at ch {gain.argmin().item()} ({gain.argmin().item()*step_deg:.1f}°)")
print(f"  channels < 1.0: {(gain < 1.0).sum().item()}/{N}")
print(f"  channels > 1.0: {(gain > 1.0).sum().item()}/{N}")
print(f"  integral sum(gain-1): {(gain - 1.0).sum().item():+.6f}")

print(f"\n{'Ch':>4s} {'Deg':>6s}  {'gain':>10s}  {'gain-1':>10s}")
print("-" * 40)
for ch in range(N):
    deg = ch * step_deg
    marker = " <-- stim" if ch == peak_ch else ""
    g = gain[ch].item()
    print(f"{ch:>4d} {deg:>5.1f}°  {g:>10.6f}  {g-1:>+10.6f}{marker}")

# ======================================================================
# INVESTIGATION 2: L2/3 ACTIVITY ON vs OFF
# ======================================================================
print(f"\n{'='*80}")
print("INVESTIGATION 2: L2/3 ACTIVITY — FEEDBACK ON vs OFF")
print(f"{'='*80}")

# Run with feedback OFF (feedback_scale=0)
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

# Reset feedback scale
net.feedback_scale.fill_(1.0)

total_on = r_l23_on.sum().item()
total_off = r_l23_off.sum().item()
ratio = total_on / total_off if total_off > 0 else float('inf')

print(f"\nTotal L2/3 activity:")
print(f"  Feedback OFF: {total_off:.6f}")
print(f"  Feedback ON:  {total_on:.6f}")
print(f"  Ratio (ON/OFF): {ratio:.4f}")
print(f"  Change: {(ratio-1)*100:+.1f}%")

# Activity-weighted gain
weighted_gain = (r_l23_off * gain).sum().item() / r_l23_off.sum().item() if r_l23_off.sum().item() > 0 else 0
print(f"\n  Activity-weighted mean gain: {weighted_gain:.4f}")

# Per-channel comparison
print(f"\n{'Ch':>4s} {'Deg':>6s}  {'OFF':>10s}  {'ON':>10s}  {'ratio':>8s}  {'gain':>8s}")
print("-" * 60)
for ch in range(N):
    deg = ch * step_deg
    off = r_l23_off[ch].item()
    on = r_l23_on[ch].item()
    r = on / off if off > 1e-8 else 0
    g = gain[ch].item()
    marker = " <-- stim" if ch == peak_ch else ""
    print(f"{ch:>4d} {deg:>5.1f}°  {off:>10.6f}  {on:>10.6f}  {r:>8.4f}  {g:>8.4f}{marker}")

# ======================================================================
# INVESTIGATION 3: SOM PROFILE
# ======================================================================
print(f"\n{'='*80}")
print("INVESTIGATION 3: SOM PROFILE & effective_som_drive")
print(f"{'='*80}")

# Get SOM and VIP drives directly
with torch.no_grad():
    som_drive_raw, vip_drive_raw, _ = net.feedback(q_pred, torch.full((1,1), oracle_pi), r_l4=None)

r_vip_eq = state.r_vip[0].clone()  # VIP from ON run
w_vip_som = F.softplus(net.w_vip_som).item()

# Reconstruct effective_som_drive as in network.step()
effective_som = F.relu(som_drive_raw - F.softplus(net.w_vip_som) * r_vip_on.unsqueeze(0))

print(f"\nSOM baseline (delta_som={net.feedback.delta_som}):")
print(f"  som_baseline: {net.feedback.som_baseline.item():.6f}")
print(f"  som_tonic: {net.feedback.som_tonic.item():.6f}")
print(f"  softplus(som_tonic): {F.softplus(net.feedback.som_tonic).item():.6f}")
print(f"  w_vip_som: {w_vip_som:.6f}")

print(f"\nSOM drive (raw from feedback):")
print(f"  mean: {som_drive_raw.mean().item():.6f}")
print(f"  at stim: {som_drive_raw[0, peak_ch].item():.6f}")
print(f"  max: {som_drive_raw.max().item():.6f} at ch {som_drive_raw[0].argmax().item()}")
print(f"  min: {som_drive_raw.min().item():.6f} at ch {som_drive_raw[0].argmin().item()}")

print(f"\nVIP drive (raw):")
print(f"  mean: {vip_drive_raw.mean().item():.6f}")
print(f"  at stim: {vip_drive_raw[0, peak_ch].item():.6f}")
print(f"  max: {vip_drive_raw.max().item():.6f}")

print(f"\nVIP activity (steady state from ON run):")
print(f"  mean: {r_vip_on.mean().item():.6f}")
print(f"  at stim: {r_vip_on[peak_ch].item():.6f}")
print(f"  max: {r_vip_on.max().item():.6f}")

print(f"\nEffective SOM drive (after VIP subtraction):")
print(f"  mean: {effective_som.mean().item():.6f}")
print(f"  at stim: {effective_som[0, peak_ch].item():.6f}")
print(f"  max: {effective_som.max().item():.6f} at ch {effective_som[0].argmax().item()}")
print(f"  min: {effective_som.min().item():.6f} at ch {effective_som[0].argmin().item()}")

print(f"\nSOM activity (steady state):")
print(f"  mean: {r_som_on.mean().item():.6f}")
print(f"  at stim: {r_som_on[peak_ch].item():.6f}")
print(f"  max: {r_som_on.max().item():.6f} at ch {r_som_on.argmax().item()}")
print(f"  min: {r_som_on.min().item():.6f} at ch {r_som_on.argmin().item()}")

print(f"\n{'Ch':>4s} {'Deg':>6s}  {'som_raw':>10s}  {'vip_act':>10s}  {'eff_som':>10s}  {'r_som':>10s}")
print("-" * 60)
for ch in range(N):
    deg = ch * step_deg
    marker = " <-- stim" if ch == peak_ch else ""
    print(f"{ch:>4d} {deg:>5.1f}°  {som_drive_raw[0,ch].item():>10.6f}  "
          f"{r_vip_on[ch].item():>10.6f}  {effective_som[0,ch].item():>10.6f}  "
          f"{r_som_on[ch].item():>10.6f}{marker}")

# ======================================================================
# INVESTIGATION 4: ENERGY DECOMPOSITION
# ======================================================================
print(f"\n{'='*80}")
print("INVESTIGATION 4: ENERGY DECOMPOSITION")
print(f"{'='*80}")

# Energy with feedback ON (from the ON run state)
e_l4_on = r_l4_on.abs().mean().item()
e_l23_on = r_l23_on.abs().mean().item()
e_tmpl_on = deep_tmpl_on.abs().mean().item()
e_pv_on = state.r_pv[0].abs().mean().item()
e_som_on = r_som_on.abs().mean().item()
e_vip_on = r_vip_on.abs().mean().item()
e_total_on = e_l4_on + e_l23_on + e_tmpl_on + e_pv_on + e_som_on + e_vip_on

# Energy with feedback OFF
e_l4_off = state_off.r_l4[0].abs().mean().item()
e_l23_off = r_l23_off.abs().mean().item()
e_tmpl_off = state_off.deep_template[0].abs().mean().item()
e_pv_off = state_off.r_pv[0].abs().mean().item()
e_som_off = state_off.r_som[0].abs().mean().item()
e_vip_off = state_off.r_vip[0].abs().mean().item()
e_total_off = e_l4_off + e_l23_off + e_tmpl_off + e_pv_off + e_som_off + e_vip_off

print(f"\n{'Component':>15s}  {'FB OFF':>10s}  {'FB ON':>10s}  {'Δ':>10s}  {'%change':>10s}")
print("-" * 65)
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
    pct = delta / off * 100 if off > 1e-8 else 0
    print(f"{name:>15s}  {off:>10.6f}  {on:>10.6f}  {delta:>+10.6f}  {pct:>+10.1f}%")

print(f"\nGlobal amplitude ratio: {e_total_on/e_total_off:.4f}")

# ======================================================================
# INVESTIGATION 5: TRUE vs WRONG TEMPLATE SELECTIVITY
# ======================================================================
print(f"\n{'='*80}")
print("INVESTIGATION 5: TRUE vs WRONG TEMPLATE SELECTIVITY")
print(f"{'='*80}")

offsets_to_test = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 90.0]

print(f"\nStimulus at {theta_stim}°, varying template orientation:")
print(f"Template uses oracle_sigma={oracle_sigma}, oracle_pi={oracle_pi}")
print(f"\n{'Offset':>8s}  {'θ_tmpl':>8s}  {'gain_peak':>10s}  {'gain_mean':>10s}  {'gain_min':>10s}  "
      f"{'sum(g-1)':>10s}  {'<1.0':>5s}")
print("-" * 75)

for offset in offsets_to_test:
    theta_template = (theta_stim + offset) % period
    q_wrong = net._make_bump(torch.tensor([theta_template]), sigma=oracle_sigma)
    with torch.no_grad():
        _, _, gain_wrong = net.feedback(q_wrong, torch.full((1,1), oracle_pi), r_l4=None)
    g = gain_wrong[0]
    print(f"{offset:>8.1f}°  {theta_template:>8.1f}°  {g[peak_ch].item():>10.6f}  "
          f"{g.mean().item():>10.6f}  {g.min().item():>10.6f}  "
          f"{(g-1).sum().item():>+10.4f}  {(g<1.0).sum().item():>5d}")

# Detailed comparison: true (0°) vs 10° offset
print(f"\n--- Detailed: True template (0° offset) vs Wrong template (10° offset) ---")
theta_wrong = (theta_stim + 10.0) % period
q_true = net._make_bump(torch.tensor([theta_stim]), sigma=oracle_sigma)
q_10off = net._make_bump(torch.tensor([theta_wrong]), sigma=oracle_sigma)

with torch.no_grad():
    _, _, gain_true = net.feedback(q_true, torch.full((1,1), oracle_pi), r_l4=None)
    _, _, gain_10off = net.feedback(q_10off, torch.full((1,1), oracle_pi), r_l4=None)

gt = gain_true[0]
gw = gain_10off[0]

# Run forward with wrong template
net.oracle_q_pred = q_10off
state_wrong = initial_state(1, N, model_cfg.v2_hidden_dim)
with torch.no_grad():
    net.l23.cache_kernels()
    net.feedback.cache_kernels()
    for t in range(steps_on):
        state_wrong, _ = net.step(stim, cue, task, state_wrong)
    net.l23.uncache_kernels()
    net.feedback.uncache_kernels()

r_l23_wrong = state_wrong.r_l23[0].clone()

print(f"\n{'Ch':>4s} {'Deg':>6s}  {'gain_true':>10s}  {'gain_10off':>10s}  "
      f"{'l23_off':>10s}  {'l23_true':>10s}  {'l23_wrong':>10s}")
print("-" * 75)
for ch in range(N):
    deg = ch * step_deg
    marker = " <-- stim" if ch == peak_ch else ""
    print(f"{ch:>4d} {deg:>5.1f}°  {gt[ch].item():>10.6f}  {gw[ch].item():>10.6f}  "
          f"{r_l23_off[ch].item():>10.6f}  {r_l23_on[ch].item():>10.6f}  "
          f"{r_l23_wrong[ch].item():>10.6f}{marker}")

# Selectivity metric: peak L2/3 / total L2/3
sel_off = r_l23_off[peak_ch].item() / r_l23_off.sum().item() if r_l23_off.sum().item() > 0 else 0
sel_true = r_l23_on[peak_ch].item() / r_l23_on.sum().item() if r_l23_on.sum().item() > 0 else 0
sel_wrong = r_l23_wrong[peak_ch].item() / r_l23_wrong.sum().item() if r_l23_wrong.sum().item() > 0 else 0

print(f"\nSelectivity (peak / total):")
print(f"  FB OFF:        {sel_off:.6f}")
print(f"  True template: {sel_true:.6f} ({sel_true/sel_off:.3f}× of OFF)")
print(f"  Wrong (+10°):  {sel_wrong:.6f} ({sel_wrong/sel_off:.3f}× of OFF)")

# Total activity comparison
print(f"\nTotal L2/3 activity:")
print(f"  FB OFF:        {r_l23_off.sum().item():.6f}")
print(f"  True template: {r_l23_on.sum().item():.6f} ({r_l23_on.sum().item()/r_l23_off.sum().item():.4f}×)")
print(f"  Wrong (+10°):  {r_l23_wrong.sum().item():.6f} ({r_l23_wrong.sum().item()/r_l23_off.sum().item():.4f}×)")

# Discrimination metric: difference at peak channel
print(f"\nDiscrimination at peak (ch {peak_ch}):")
print(f"  True template L2/3: {r_l23_on[peak_ch].item():.6f}")
print(f"  Wrong template L2/3: {r_l23_wrong[peak_ch].item():.6f}")
print(f"  Difference: {r_l23_on[peak_ch].item() - r_l23_wrong[peak_ch].item():.6f}")
print(f"  Ratio (true/wrong): {r_l23_on[peak_ch].item() / r_l23_wrong[peak_ch].item():.4f}"
      if r_l23_wrong[peak_ch].item() > 0 else "  Wrong = 0")

# ======================================================================
# SUMMARY: Key metrics comparison with old control
# ======================================================================
print(f"\n{'='*80}")
print("SUMMARY — KEY METRICS")
print(f"{'='*80}")
print(f"\n  max_apical_gain: {net.feedback.max_apical_gain}")
print(f"  lambda_energy: {train_cfg.lambda_energy}")
print(f"  Parameters: 36 per pathway (direct channel weights)")
print(f"  Gain mode: pure top-down (r_l4=None)")
print(f"\n  Global amplitude ratio (ON/OFF): {ratio:.4f}")
print(f"  Mean apical gain: {gain.mean().item():.6f}")
print(f"  Gain at stimulus: {gain[peak_ch].item():.6f}")
print(f"  Gain min: {gain.min().item():.6f}")
print(f"  Channels with gain < 1.0: {(gain < 1.0).sum().item()}/{N}")
print(f"  Sum(gain-1): {(gain - 1.0).sum().item():+.6f}")
print(f"  Selectivity improvement (true template): {sel_true/sel_off:.3f}×")

# Apical kernel FWHM
peak_val = alpha_apical.max().item()
if peak_val > 0:
    half_max = peak_val / 2
    above = (alpha_apical >= half_max).sum().item()
    print(f"  Apical kernel FWHM: ~{above * step_deg:.0f}° ({above} channels above half-max)")

# Check for center-surround structure
pos_area = alpha_apical[alpha_apical > 0].sum().item()
neg_area = alpha_apical[alpha_apical < 0].sum().item() if (alpha_apical < 0).any() else 0
print(f"  Apical kernel positive sum: {pos_area:.6f}")
print(f"  Apical kernel negative sum: {neg_area:.6f}")
