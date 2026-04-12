"""Debugger: Energy cost investigation for control (no gate) checkpoint.

Experiments:
1. Energy decomposition: feedback ON vs OFF, activity by region
2. Loss component analysis from training logs
3. Energy fraction of total loss
4. Drive decomposition at peak gain channel
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
from src.utils import rectified_softplus

# ── Load model ──────────────────────────────────────────────────────
model_cfg, train_cfg, stim_cfg = load_config("config/exp_branch_a.yaml")
N = model_cfg.n_orientations
period = model_cfg.orientation_range
sigma_ff = model_cfg.sigma_ff
steps_on = train_cfg.steps_on  # 12

# Load with delta_som=True (from config)
net = LaminarV1V2Network(model_cfg, delta_som=train_cfg.delta_som)
ckpt = torch.load("results/control_no_gate/s42/center_surround_seed42/checkpoint.pt",
                   map_location="cpu", weights_only=False)
net.load_state_dict(ckpt["model_state"], strict=False)
net.eval()

print(f"delta_som: {net.feedback.delta_som}")
print(f"max_apical_gain: {net.feedback.max_apical_gain}")
print(f"w_template_drive: {net.w_template_drive.item():.4f}")
print(f"w_vip_som: {F.softplus(net.w_vip_som).item():.4f}")

# ── Stimulus setup ──────────────────────────────────────────────────
theta_stim = 90.0
contrast = 0.5
stim = generate_grating(torch.tensor([theta_stim]), torch.tensor([contrast]),
                        N, sigma_ff, period=period)

# Oracle mode
net.oracle_mode = True
oracle_sigma = getattr(train_cfg, 'oracle_sigma', 12.0)
oracle_pi = train_cfg.oracle_pi
q_true = net._make_bump(torch.tensor([theta_stim]), sigma=oracle_sigma)
pi_oracle = torch.full((1, 1), oracle_pi)
net.oracle_q_pred = q_true
net.oracle_pi_pred = pi_oracle

cue = torch.zeros(1, 2)
task = torch.zeros(1, 2)

peak_ch = int(theta_stim / (period / N))  # channel 18

# Helper: channel regions relative to stimulus
def get_regions(N, period, theta_stim):
    """Return index masks for center (|d|<=10), surround (10<|d|<=45), far (|d|>45)."""
    thetas = torch.arange(N, dtype=torch.float32) * (period / N)
    dists = torch.abs(thetas - theta_stim)
    dists = torch.min(dists, period - dists)  # circular distance
    center = dists <= 10.0
    surround = (dists > 10.0) & (dists <= 45.0)
    far = dists > 45.0
    return center, surround, far

center_mask, surround_mask, far_mask = get_regions(N, period, theta_stim)

print(f"\nRegion sizes: center={center_mask.sum().item()}, surround={surround_mask.sum().item()}, far={far_mask.sum().item()}")

# ======================================================================
# EXPERIMENT 1: Energy decomposition — feedback ON vs OFF
# ======================================================================
print("\n" + "=" * 80)
print("EXPERIMENT 1: Energy decomposition — feedback ON vs OFF")
print("=" * 80)

def run_simulation(net, stim, cue, task, steps, feedback_on=True):
    """Run full network simulation, return per-step states."""
    if feedback_on:
        net.feedback_scale.fill_(1.0)
    else:
        net.feedback_scale.fill_(0.0)

    state = initial_state(1, N, model_cfg.v2_hidden_dim)
    history = {
        'r_l4': [], 'r_l23': [], 'r_pv': [], 'r_som': [], 'r_vip': [],
        'deep_template': [], 'apical_gain': [],
    }

    with torch.no_grad():
        net.l23.cache_kernels()
        if hasattr(net.feedback, 'cache_kernels'):
            net.feedback.cache_kernels()

        for t in range(steps):
            new_state, aux = net.step(stim, cue, task, state)

            history['r_l4'].append(new_state.r_l4[0].clone())
            history['r_l23'].append(new_state.r_l23[0].clone())
            history['r_pv'].append(new_state.r_pv[0].clone())
            history['r_som'].append(new_state.r_som[0].clone())
            history['r_vip'].append(new_state.r_vip[0].clone())
            history['deep_template'].append(new_state.deep_template[0].clone())

            # Reconstruct apical gain
            q_pred = aux.q_pred
            pi_eff = aux.pi_pred_eff
            q_centered = q_pred - q_pred.mean(dim=-1, keepdim=True)

            if net.feedback._cached_apical_circulant is not None:
                apical_circulant = net.feedback._cached_apical_circulant
            else:
                K_apical = net.feedback.get_apical_profile()
                apical_circulant = net.feedback._to_circulant(K_apical)

            apical_field = (apical_circulant @ q_centered.unsqueeze(-1)).squeeze(-1)

            # Check if r_l4 was passed (depends on network.py state)
            # Reconstruct based on forward logic: if r_l4 is not None, coincidence gate
            basal_field = new_state.r_l4 - new_state.r_l4.mean(dim=-1, keepdim=True)
            coincidence = F.relu(apical_field) * F.relu(basal_field)
            apical_gain = 1.0 + net.feedback.max_apical_gain * torch.tanh(pi_eff * coincidence)

            # But for this control checkpoint, it was trained with r_l4=None
            # So the actual gain during training was pure top-down:
            apical_gain_topdown = 1.0 + net.feedback.max_apical_gain * torch.tanh(pi_eff * apical_field)

            history['apical_gain'].append(apical_gain_topdown[0].clone())

            state = new_state

        net.l23.uncache_kernels()
        if hasattr(net.feedback, 'uncache_kernels'):
            net.feedback.uncache_kernels()

    # Stack
    for k in history:
        history[k] = torch.stack(history[k])  # [T, N] or [T, 1]

    return history

# Run with feedback ON and OFF
hist_on = run_simulation(net, stim, cue, task, steps_on, feedback_on=True)
hist_off = run_simulation(net, stim, cue, task, steps_on, feedback_on=False)

# Steady-state analysis (last 3 timesteps)
print("\n--- Steady-state activity (last 3 steps averaged) ---")
print(f"{'Metric':<30s}  {'FB ON':>10s}  {'FB OFF':>10s}  {'Ratio':>10s}  {'Diff':>10s}")
print("-" * 75)

metrics = [
    ("L2/3 total (sum)", lambda h: h['r_l23'][-3:].sum(dim=-1).mean().item()),
    ("L2/3 mean", lambda h: h['r_l23'][-3:].mean().item()),
    ("L2/3 peak", lambda h: h['r_l23'][-3:].max(dim=-1).values.mean().item()),
    ("L2/3 @ center (|d|<=10)", lambda h: h['r_l23'][-3:, center_mask].mean().item()),
    ("L2/3 @ surround (10<|d|<=45)", lambda h: h['r_l23'][-3:, surround_mask].mean().item()),
    ("L2/3 @ far (|d|>45)", lambda h: h['r_l23'][-3:, far_mask].mean().item()),
    ("L4 total (sum)", lambda h: h['r_l4'][-3:].sum(dim=-1).mean().item()),
    ("L4 mean", lambda h: h['r_l4'][-3:].mean().item()),
    ("SOM mean", lambda h: h['r_som'][-3:].mean().item()),
    ("SOM @ center", lambda h: h['r_som'][-3:, center_mask].mean().item()),
    ("SOM @ surround", lambda h: h['r_som'][-3:, surround_mask].mean().item()),
    ("VIP mean", lambda h: h['r_vip'][-3:].mean().item()),
    ("VIP @ center", lambda h: h['r_vip'][-3:, center_mask].mean().item()),
    ("PV", lambda h: h['r_pv'][-3:].mean().item()),
    ("deep_template mean", lambda h: h['deep_template'][-3:].mean().item()),
]

for name, fn in metrics:
    v_on = fn(hist_on)
    v_off = fn(hist_off)
    ratio = v_on / (v_off + 1e-12)
    diff = v_on - v_off
    print(f"{name:<30s}  {v_on:>10.6f}  {v_off:>10.6f}  {ratio:>10.3f}x  {diff:>+10.6f}")

# Per-channel profile comparison
print("\n--- Per-channel L2/3 profile (steady state, step 11) ---")
print(f"{'Ch':>4s} {'Deg':>6s}  {'ON':>10s}  {'OFF':>10s}  {'Diff':>10s}  {'Ratio':>8s}  {'Region':>10s}")
print("-" * 65)
for ch in range(N):
    deg = ch * period / N
    on_val = hist_on['r_l23'][-1, ch].item()
    off_val = hist_off['r_l23'][-1, ch].item()
    diff = on_val - off_val
    ratio = on_val / (off_val + 1e-12) if off_val > 1e-8 else float('inf')
    region = "CENTER" if center_mask[ch] else ("SURR" if surround_mask[ch] else "FAR")
    marker = " <-- stim" if ch == peak_ch else ""
    print(f"{ch:>4d} {deg:>5.1f}°  {on_val:>10.6f}  {off_val:>10.6f}  {diff:>+10.6f}  {ratio:>8.3f}  {region:>10s}{marker}")

# ======================================================================
# EXPERIMENT 2: Loss component analysis from training logs
# ======================================================================
print("\n" + "=" * 80)
print("EXPERIMENT 2: Loss component analysis from training logs")
print("=" * 80)

import re

log_file = "logs/control_no_gate_s42.log"
steps_log = []
with open(log_file) as f:
    for line in f:
        m = re.search(
            r'step (\d+)/\d+: loss=([\d.]+), sens=([\d.]+), prior_kl=([\d.]+), '
            r'fb_sparse=([\d.]+), energy=([\d.]+), homeo=([\d.]+)',
            line
        )
        if m:
            steps_log.append({
                'step': int(m.group(1)),
                'loss': float(m.group(2)),
                'sens': float(m.group(3)),
                'prior_kl': float(m.group(4)),
                'fb_sparse': float(m.group(5)),
                'energy': float(m.group(6)),
                'homeo': float(m.group(7)),
            })

# Also extract fb_scale
with open(log_file) as f:
    for i, line in enumerate(f):
        m = re.search(r'fb_scale=([\d.]+)', line)
        if m and i < len(steps_log) + 50:
            # Find matching step entry
            step_m = re.search(r'step (\d+)', line)
            if step_m:
                step_num = int(step_m.group(1))
                for entry in steps_log:
                    if entry['step'] == step_num:
                        entry['fb_scale'] = float(m.group(1))

# Lambda values from config
lambda_sensory = 1.0
lambda_energy = 0.01
lambda_homeo = 1.0
lambda_state = 1.0  # prior_kl
lambda_fb = 0.001

print(f"\nLambda weights: sensory={lambda_sensory}, energy={lambda_energy}, "
      f"homeo={lambda_homeo}, state(prior_kl)={lambda_state}, fb={lambda_fb}")

print(f"\n{'Step':>6s}  {'fb_scale':>8s}  {'Total':>8s}  {'Sens':>8s}  {'PriorKL':>8s}  "
      f"{'Energy':>8s}  {'λ*E':>8s}  {'E%total':>8s}  {'Homeo':>8s}")
print("-" * 85)

# Show key milestones
milestones = [100, 500, 1000, 1100, 1500, 2000, 2100, 3000, 5000, 7000, 9000, 10000]
for entry in steps_log:
    if entry['step'] in milestones:
        fb = entry.get('fb_scale', 'N/A')
        weighted_energy = lambda_energy * entry['energy']
        energy_pct = weighted_energy / (entry['loss'] + 1e-12) * 100
        print(f"{entry['step']:>6d}  {str(fb):>8s}  {entry['loss']:>8.4f}  {entry['sens']:>8.4f}  "
              f"{entry['prior_kl']:>8.4f}  {entry['energy']:>8.4f}  {weighted_energy:>8.4f}  "
              f"{energy_pct:>7.2f}%  {entry['homeo']:>8.4f}")

# Energy evolution: early (no FB) vs late (full FB)
early = [e for e in steps_log if e['step'] <= 1000]
late = [e for e in steps_log if e['step'] >= 8000]

early_energy = np.mean([e['energy'] for e in early])
late_energy = np.mean([e['energy'] for e in late])
early_sens = np.mean([e['sens'] for e in early])
late_sens = np.mean([e['sens'] for e in late])

print(f"\nEnergy evolution:")
print(f"  Early (step<=1000, fb_scale≈0): energy={early_energy:.4f}, sens={early_sens:.4f}")
print(f"  Late  (step>=8000, fb_scale=1): energy={late_energy:.4f}, sens={late_sens:.4f}")
print(f"  Energy increase: {late_energy - early_energy:.4f} ({(late_energy/early_energy - 1)*100:.1f}%)")
print(f"  Sensory decrease: {late_sens - early_sens:.4f} ({(late_sens/early_sens - 1)*100:.1f}%)")

# ======================================================================
# EXPERIMENT 3: Energy fraction analysis
# ======================================================================
print("\n" + "=" * 80)
print("EXPERIMENT 3: Energy fraction of total loss & gradient dominance")
print("=" * 80)

# Use final training values
final = steps_log[-1]
weighted_sens = lambda_sensory * final['sens']
weighted_kl = lambda_state * final['prior_kl']
weighted_energy = lambda_energy * final['energy']
weighted_homeo = lambda_homeo * final['homeo']
weighted_fb = lambda_fb * final['fb_sparse']
total_weighted = weighted_sens + weighted_kl + weighted_energy + weighted_homeo + weighted_fb

print(f"\nFinal step ({final['step']}) loss decomposition (weighted):")
print(f"  λ_sens * sens    = {lambda_sensory} × {final['sens']:.4f} = {weighted_sens:.4f}  ({weighted_sens/total_weighted*100:.1f}%)")
print(f"  λ_state * kl     = {lambda_state} × {final['prior_kl']:.4f} = {weighted_kl:.4f}  ({weighted_kl/total_weighted*100:.1f}%)")
print(f"  λ_energy * E     = {lambda_energy} × {final['energy']:.4f} = {weighted_energy:.4f}  ({weighted_energy/total_weighted*100:.2f}%)")
print(f"  λ_homeo * H      = {lambda_homeo} × {final['homeo']:.4f} = {weighted_homeo:.4f}  ({weighted_homeo/total_weighted*100:.2f}%)")
print(f"  λ_fb * sparse    = {lambda_fb} × {final['fb_sparse']:.4f} = {weighted_fb:.4f}  ({weighted_fb/total_weighted*100:.2f}%)")
print(f"  Reported total   = {final['loss']:.4f}")
print(f"  Reconstructed    = {total_weighted:.4f}")

print(f"\n  Energy gradient contribution: {weighted_energy/total_weighted*100:.2f}% of total loss")
print(f"  Sensory gradient contribution: {weighted_sens/total_weighted*100:.1f}% of total loss")
print(f"  Ratio (sensory/energy): {weighted_sens/weighted_energy:.1f}×")
print(f"\n  → Energy loss is {weighted_sens/weighted_energy:.0f}× weaker than sensory loss.")
print(f"  → If energy cost increases by 30%, the gradient impact is only {0.3*weighted_energy:.4f}")
print(f"     while sensory loss change of 0.01 gives gradient impact of {0.01*lambda_sensory:.4f}")

# ======================================================================
# EXPERIMENT 4: Drive decomposition at peak gain
# ======================================================================
print("\n" + "=" * 80)
print("EXPERIMENT 4: Drive decomposition at peak gain channel")
print("=" * 80)

# Run with feedback ON and manually decompose the drive at each step
net.feedback_scale.fill_(1.0)
state = initial_state(1, N, model_cfg.v2_hidden_dim)

print(f"\n{'Step':>4s}  {'ff':>8s}  {'rec':>8s}  {'tmpl':>8s}  {'exc_raw':>8s}  "
      f"{'a_gain':>8s}  {'exc_gnd':>8s}  {'som_inh':>8s}  {'pv_inh':>8s}  "
      f"{'drive':>8s}  {'r_l23':>8s}")
print("-" * 100)

with torch.no_grad():
    net.l23.cache_kernels()
    if hasattr(net.feedback, 'cache_kernels'):
        net.feedback.cache_kernels()

    for t in range(steps_on):
        # Reproduce step internals
        r_l4, adaptation = net.l4(stim, state.r_l4, state.r_pv, state.adaptation)
        r_pv = net.pv(r_l4, state.r_l23, state.r_pv)

        # V2 / oracle
        q_pred = q_true
        pi_pred_raw = pi_oracle
        pi_eff = pi_pred_raw * net.feedback_scale

        # Deep template
        deep_tmpl = net.deep_template(q_pred, pi_eff)

        # Feedback
        som_drive, vip_drive, apical_gain = net.feedback(q_pred, pi_eff, r_l4=r_l4)
        r_vip = net.vip(vip_drive, state.r_vip)
        effective_som_drive = F.relu(som_drive - F.softplus(net.w_vip_som) * r_vip)
        center_exc = net.w_template_drive * deep_tmpl
        r_som = net.som(effective_som_drive, state.r_som)

        # L2/3 internals
        ff = F.linear(r_l4, net.l23.W_l4_to_l23)
        W_rec = net.l23.W_rec
        rec = F.linear(state.r_l23, W_rec)

        excitatory_raw = ff + rec + center_exc
        exc_gained = apical_gain * excitatory_raw

        som_inh = net.l23.w_som(r_som)
        pv_inh = net.l23.w_pv_l23(r_pv)
        l23_drive = exc_gained - som_inh - pv_inh

        r_l23 = state.r_l23 + (net.l23.dt / net.l23.tau_l23) * (
            -state.r_l23 + rectified_softplus(l23_drive)
        )

        print(f"{t:>4d}  {ff[0, peak_ch].item():>8.4f}  {rec[0, peak_ch].item():>8.5f}  "
              f"{center_exc[0, peak_ch].item():>8.5f}  {excitatory_raw[0, peak_ch].item():>8.4f}  "
              f"{apical_gain[0, peak_ch].item():>8.4f}  {exc_gained[0, peak_ch].item():>8.4f}  "
              f"{som_inh[0, peak_ch].item():>8.4f}  {pv_inh[0, 0].item():>8.4f}  "
              f"{l23_drive[0, peak_ch].item():>8.4f}  {r_l23[0, peak_ch].item():>8.5f}")

        state = state._replace(
            r_l4=r_l4, r_l23=r_l23, r_pv=r_pv, r_som=r_som,
            r_vip=r_vip, adaptation=adaptation,
            h_v2=state.h_v2, deep_template=deep_tmpl,
        )

    net.l23.uncache_kernels()
    if hasattr(net.feedback, 'uncache_kernels'):
        net.feedback.uncache_kernels()

# Final summary of gain decomposition
print(f"\n--- Final step gain decomposition at stimulus channel (ch={peak_ch}) ---")
print(f"  Excitatory drive (ff+rec+tmpl) BEFORE gain: {excitatory_raw[0, peak_ch].item():.6f}")
print(f"  Apical gain at stim channel: {apical_gain[0, peak_ch].item():.6f}")
print(f"  Excitatory drive AFTER gain: {exc_gained[0, peak_ch].item():.6f}")
print(f"  Gain contribution: {(apical_gain[0, peak_ch].item() - 1.0) * 100:.1f}% modulation")
print(f"  SOM inhibition: {som_inh[0, peak_ch].item():.6f}")
print(f"  PV inhibition: {pv_inh[0, 0].item():.6f}")
print(f"  Net drive: {l23_drive[0, peak_ch].item():.6f}")

# Check apical gain profile across channels
print(f"\n--- Apical gain profile (steady state, step 11) ---")
print(f"{'Ch':>4s} {'Deg':>6s}  {'gain':>10s}  {'modulation%':>12s}  {'Region':>8s}")
print("-" * 50)
for ch in range(N):
    deg = ch * period / N
    g = apical_gain[0, ch].item()
    mod_pct = (g - 1.0) * 100
    region = "CENTER" if center_mask[ch] else ("SURR" if surround_mask[ch] else "FAR")
    marker = " <-- stim" if ch == peak_ch else ""
    print(f"{ch:>4d} {deg:>5.1f}°  {g:>10.6f}  {mod_pct:>+11.2f}%  {region:>8s}{marker}")

# Where is the apical gain > 1 (enhancement) vs < 1 (suppression)?
gain_enhanced = (apical_gain[0] > 1.01).sum().item()
gain_suppressed = (apical_gain[0] < 0.99).sum().item()
gain_neutral = N - gain_enhanced - gain_suppressed
print(f"\n  Enhanced (gain>1.01): {gain_enhanced}/{N} channels")
print(f"  Suppressed (gain<0.99): {gain_suppressed}/{N} channels")
print(f"  Neutral: {gain_neutral}/{N} channels")
print(f"  Peak gain: {apical_gain[0].max().item():.4f} at ch {apical_gain[0].argmax().item()}")
print(f"  Min gain: {apical_gain[0].min().item():.4f} at ch {apical_gain[0].argmin().item()}")

# ======================================================================
# EXPERIMENT 4b: What does "no SOM at center" mean for energy?
# ======================================================================
print("\n" + "=" * 80)
print("EXPERIMENT 4b: SOM profile (does SOM suppress at center or surround?)")
print("=" * 80)

print(f"\n--- SOM drive and r_som profile (steady state) ---")
print(f"{'Ch':>4s} {'Deg':>6s}  {'som_drive':>10s}  {'r_som':>10s}  {'som_inh':>10s}  {'Region':>8s}")
print("-" * 55)
for ch in range(N):
    deg = ch * period / N
    sd = effective_som_drive[0, ch].item() if ch < effective_som_drive.shape[1] else 0
    rs = r_som[0, ch].item()
    si = som_inh[0, ch].item()
    region = "CENTER" if center_mask[ch] else ("SURR" if surround_mask[ch] else "FAR")
    marker = " <-- stim" if ch == peak_ch else ""
    print(f"{ch:>4d} {deg:>5.1f}°  {sd:>10.6f}  {rs:>10.6f}  {si:>10.6f}  {region:>8s}{marker}")

# ======================================================================
# FINAL: What fraction of total energy increase goes to center vs surround?
# ======================================================================
print("\n" + "=" * 80)
print("SUMMARY: Where does the energy go?")
print("=" * 80)

l23_on = hist_on['r_l23'][-3:]  # [3, N]
l23_off = hist_off['r_l23'][-3:]

center_on = l23_on[:, center_mask].sum(dim=-1).mean().item()
center_off = l23_off[:, center_mask].sum(dim=-1).mean().item()
surround_on = l23_on[:, surround_mask].sum(dim=-1).mean().item()
surround_off = l23_off[:, surround_mask].sum(dim=-1).mean().item()
far_on = l23_on[:, far_mask].sum(dim=-1).mean().item()
far_off = l23_off[:, far_mask].sum(dim=-1).mean().item()

total_increase = (center_on - center_off) + (surround_on - surround_off) + (far_on - far_off)

print(f"\nL2/3 activity increase from feedback (sum per region):")
print(f"  Center (|d|<=10°):   ON={center_on:.4f}, OFF={center_off:.4f}, Δ={center_on-center_off:+.4f} ({(center_on-center_off)/total_increase*100:.1f}% of total increase)")
print(f"  Surround (10-45°):   ON={surround_on:.4f}, OFF={surround_off:.4f}, Δ={surround_on-surround_off:+.4f} ({(surround_on-surround_off)/total_increase*100:.1f}%)")
print(f"  Far (>45°):          ON={far_on:.4f}, OFF={far_off:.4f}, Δ={far_on-far_off:+.4f} ({(far_on-far_off)/total_increase*100:.1f}%)")
print(f"  Total:               ON={center_on+surround_on+far_on:.4f}, OFF={center_off+surround_off+far_off:.4f}, Δ={total_increase:+.4f}")

# Global amplitude ratio
total_on = l23_on.sum(dim=-1).mean().item()
total_off = l23_off.sum(dim=-1).mean().item()
print(f"\n  Global L2/3 amplitude ratio (ON/OFF): {total_on/total_off:.3f}x")
print(f"  This means feedback {'increases' if total_on > total_off else 'decreases'} total L2/3 activity by {abs(total_on/total_off - 1)*100:.1f}%")
