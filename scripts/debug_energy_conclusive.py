"""Debugger: CONCLUSIVE energy investigation.

Gap identified: Experiment 1 ran net.step() which uses r_l4=r_l4 (coincidence gate),
but checkpoint was TRAINED with r_l4=None (pure top-down). Need to verify under
the CORRECT gain mode.

This script runs 3 modes:
A) Pure top-down gain (r_l4=None) — HOW THE MODEL WAS TRAINED
B) Coincidence-gated gain (r_l4=r_l4) — current code, NOT how it was trained
C) No feedback (fb_scale=0) — baseline

For each mode, we measure:
- Total L2/3 activity
- Activity by region (center/surround/far)
- Activity-weighted gain (to verify asymmetric base explanation)
- The actual energy cost as computed by the loss function
"""

import sys
sys.path.insert(0, "/mnt/c/Users/User/codingproj/freshstart")

import torch
import torch.nn.functional as F
from src.config import load_config
from src.model.network import LaminarV1V2Network
from src.state import initial_state
from src.stimulus.gratings import generate_grating
from src.utils import rectified_softplus

model_cfg, train_cfg, stim_cfg = load_config("config/exp_branch_a.yaml")
N = model_cfg.n_orientations
period = model_cfg.orientation_range
sigma_ff = model_cfg.sigma_ff
steps_on = train_cfg.steps_on

theta_stim = 90.0
contrast = 0.5
stim = generate_grating(torch.tensor([theta_stim]), torch.tensor([contrast]),
                        N, sigma_ff, period=period)
cue = torch.zeros(1, 2)
task = torch.zeros(1, 2)
peak_ch = int(theta_stim / (period / N))

oracle_sigma = getattr(train_cfg, 'oracle_sigma', 12.0)
oracle_pi = train_cfg.oracle_pi

# Region masks
thetas = torch.arange(N, dtype=torch.float32) * (period / N)
dists = torch.abs(thetas - theta_stim)
dists = torch.min(dists, period - dists)
center_mask = dists <= 10.0
surround_mask = (dists > 10.0) & (dists <= 45.0)
far_mask = dists > 45.0


def run_manual_simulation(net, stim, steps, gain_mode="topdown"):
    """Run simulation with explicit control over gain computation.

    gain_mode:
        "topdown": r_l4=None (pure top-down, how checkpoint was trained)
        "gated": r_l4=r_l4 (coincidence gate)
        "off": feedback_scale=0 (no feedback)
    """
    if gain_mode == "off":
        net.feedback_scale.fill_(0.0)
    else:
        net.feedback_scale.fill_(1.0)

    q_true = net._make_bump(torch.tensor([theta_stim]), sigma=oracle_sigma)
    pi_oracle = torch.full((1, 1), oracle_pi)
    net.oracle_q_pred = q_true
    net.oracle_pi_pred = pi_oracle

    state = initial_state(1, N, model_cfg.v2_hidden_dim)

    # Track per-step values
    history = {'r_l23': [], 'r_l4': [], 'r_som': [], 'r_vip': [], 'r_pv': [],
               'apical_gain': [], 'deep_template': [],
               'exc_raw': [], 'exc_gained': [], 'som_inh': [], 'pv_inh': []}

    with torch.no_grad():
        net.l23.cache_kernels()
        if hasattr(net.feedback, 'cache_kernels'):
            net.feedback.cache_kernels()

        for t in range(steps):
            # L4
            r_l4, adaptation = net.l4(stim, state.r_l4, state.r_pv, state.adaptation)
            r_pv = net.pv(r_l4, state.r_l23, state.r_pv)

            # V2/oracle
            q_pred = q_true
            pi_pred_raw = pi_oracle
            pi_eff = pi_pred_raw * net.feedback_scale

            # Deep template
            deep_tmpl = net.deep_template(q_pred, pi_eff)

            # Feedback with explicit gain mode control
            if gain_mode == "topdown":
                som_drive, vip_drive, apical_gain = net.feedback(q_pred, pi_eff, r_l4=None)
            elif gain_mode == "gated":
                som_drive, vip_drive, apical_gain = net.feedback(q_pred, pi_eff, r_l4=r_l4)
            else:  # "off"
                som_drive, vip_drive, apical_gain = net.feedback(q_pred, pi_eff, r_l4=None)

            r_vip = net.vip(vip_drive, state.r_vip)
            effective_som_drive = F.relu(som_drive - F.softplus(net.w_vip_som) * r_vip)
            center_exc = net.w_template_drive * deep_tmpl
            r_som = net.som(effective_som_drive, state.r_som)

            # L2/3
            ff = F.linear(r_l4, net.l23.W_l4_to_l23)
            W_rec = net.l23.W_rec
            rec = F.linear(state.r_l23, W_rec)
            exc_raw = ff + rec + center_exc
            exc_gained = apical_gain * exc_raw
            som_inh = net.l23.w_som(r_som)
            pv_inh = net.l23.w_pv_l23(r_pv)
            l23_drive = exc_gained - som_inh - pv_inh
            r_l23 = state.r_l23 + (net.l23.dt / net.l23.tau_l23) * (
                -state.r_l23 + rectified_softplus(l23_drive)
            )

            history['r_l23'].append(r_l23[0].clone())
            history['r_l4'].append(r_l4[0].clone())
            history['r_som'].append(r_som[0].clone())
            history['r_vip'].append(r_vip[0].clone())
            history['r_pv'].append(r_pv[0].clone())
            history['apical_gain'].append(apical_gain[0].clone())
            history['deep_template'].append(deep_tmpl[0].clone())
            history['exc_raw'].append(exc_raw[0].clone())
            history['exc_gained'].append(exc_gained[0].clone())
            history['som_inh'].append(som_inh[0].clone())
            history['pv_inh'].append(pv_inh[0].clone())

            state = state._replace(
                r_l4=r_l4, r_l23=r_l23, r_pv=r_pv, r_som=r_som,
                r_vip=r_vip, adaptation=adaptation,
                h_v2=state.h_v2, deep_template=deep_tmpl,
            )

        net.l23.uncache_kernels()
        if hasattr(net.feedback, 'uncache_kernels'):
            net.feedback.uncache_kernels()

    for k in history:
        history[k] = torch.stack(history[k])
    return history


# Load checkpoint
net = LaminarV1V2Network(model_cfg, delta_som=train_cfg.delta_som)
ckpt = torch.load("results/control_no_gate/s42/center_surround_seed42/checkpoint.pt",
                   map_location="cpu", weights_only=False)
net.load_state_dict(ckpt["model_state"], strict=False)
net.eval()
net.oracle_mode = True

print(f"delta_som: {net.feedback.delta_som}")
print(f"max_apical_gain: {net.feedback.max_apical_gain}")

# Run all 3 modes
print("\nRunning 3 modes...")
hist_topdown = run_manual_simulation(net, stim, steps_on, gain_mode="topdown")
hist_gated = run_manual_simulation(net, stim, steps_on, gain_mode="gated")
hist_off = run_manual_simulation(net, stim, steps_on, gain_mode="off")

# ======================================================================
# ANALYSIS 1: L2/3 activity comparison across all 3 modes
# ======================================================================
print("\n" + "=" * 80)
print("ANALYSIS 1: L2/3 activity — 3 gain modes (steady state, last 3 steps)")
print("=" * 80)

def ss_mean(hist, key, mask=None):
    """Steady-state mean of last 3 steps."""
    data = hist[key][-3:]  # [3, N] or [3, 1]
    if mask is not None:
        return data[:, mask].mean().item()
    return data.mean().item()

def ss_sum(hist, key, mask=None):
    data = hist[key][-3:]
    if mask is not None:
        return data[:, mask].sum(dim=-1).mean().item()
    return data.sum(dim=-1).mean().item()

modes = [("Top-down (TRAINED)", hist_topdown), ("Gated (current code)", hist_gated), ("FB OFF", hist_off)]

print(f"\n{'Metric':<35s}", end="")
for name, _ in modes:
    print(f"  {name:>22s}", end="")
print()
print("-" * 105)

metrics = [
    ("L2/3 total sum", lambda h: ss_sum(h, 'r_l23')),
    ("L2/3 mean", lambda h: ss_mean(h, 'r_l23')),
    ("L2/3 peak", lambda h: h['r_l23'][-3:].max(dim=-1).values.mean().item()),
    ("L2/3 @ stim ch", lambda h: h['r_l23'][-3:, peak_ch].mean().item()),
    ("L2/3 @ center (≤10°)", lambda h: ss_mean(h, 'r_l23', center_mask)),
    ("L2/3 @ surround (10-45°)", lambda h: ss_mean(h, 'r_l23', surround_mask)),
    ("L2/3 @ far (>45°)", lambda h: ss_mean(h, 'r_l23', far_mask)),
    ("SOM mean", lambda h: ss_mean(h, 'r_som')),
    ("SOM @ center", lambda h: ss_mean(h, 'r_som', center_mask)),
    ("PV", lambda h: h['r_pv'][-3:].mean().item()),
    ("Gain @ stim", lambda h: h['apical_gain'][-3:, peak_ch].mean().item()),
    ("Gain mean", lambda h: ss_mean(h, 'apical_gain')),
    ("Gain integral sum(g-1)", lambda h: (h['apical_gain'][-3:] - 1.0).sum(dim=-1).mean().item()),
]

for name, fn in metrics:
    print(f"{name:<35s}", end="")
    for _, hist in modes:
        print(f"  {fn(hist):>22.6f}", end="")
    print()

# Ratios relative to FB OFF
print(f"\n{'Ratio vs FB OFF':<35s}", end="")
for name, _ in modes:
    print(f"  {name:>22s}", end="")
print()
print("-" * 105)

for name, fn in metrics[:7]:  # L2/3 metrics only
    off_val = fn(hist_off)
    print(f"{name:<35s}", end="")
    for _, hist in modes:
        val = fn(hist)
        ratio = val / (off_val + 1e-12)
        print(f"  {ratio:>21.3f}x", end="")
    print()

# ======================================================================
# ANALYSIS 2: Activity-weighted gain (direct test of asymmetric explanation)
# ======================================================================
print("\n" + "=" * 80)
print("ANALYSIS 2: Activity-weighted gain (WHY net-suppressive gain → net amplification)")
print("=" * 80)

# For top-down mode: compute what L2/3 WOULD be if gain were 1.0 everywhere (pure FF)
# vs what it actually is with gain applied
# This isolates the contribution of gain from other factors

gain_td = hist_topdown['apical_gain'][-1]  # [N]
r_l23_off = hist_off['r_l23'][-1]  # [N], baseline L2/3 without feedback
r_l23_td = hist_topdown['r_l23'][-1]  # [N], actual L2/3 with top-down gain

print(f"\nBase activity (FB OFF) by region:")
print(f"  Center (≤10°): mean={r_l23_off[center_mask].mean().item():.6f}, sum={r_l23_off[center_mask].sum().item():.6f}")
print(f"  Surround:      mean={r_l23_off[surround_mask].mean().item():.6f}, sum={r_l23_off[surround_mask].sum().item():.6f}")
print(f"  Far:           mean={r_l23_off[far_mask].mean().item():.6f}, sum={r_l23_off[far_mask].sum().item():.6f}")

# Activity-weighted gain: sum(r_base * gain) / sum(r_base)
# This tells us what effective multiplicative factor the gain applies to total activity
weighted_gain_sum = (r_l23_off * gain_td).sum().item()
base_sum = r_l23_off.sum().item()
effective_gain = weighted_gain_sum / (base_sum + 1e-12)

print(f"\nUnweighted gain statistics:")
print(f"  Mean gain: {gain_td.mean().item():.6f}")
print(f"  Sum(gain-1): {(gain_td - 1.0).sum().item():.4f} ({'net suppressive' if (gain_td-1.0).sum() < 0 else 'net enhancing'})")

print(f"\nActivity-weighted gain:")
print(f"  sum(r_base * gain) / sum(r_base) = {effective_gain:.6f}")
print(f"  → Despite unweighted gain being net suppressive,")
print(f"    the activity-weighted gain is {'> 1 (net amplification)' if effective_gain > 1 else '< 1 (net suppression)'}!")

# Decompose by region
for region_name, mask in [("Center (≤10°)", center_mask), ("Surround", surround_mask), ("Far", far_mask)]:
    r_base_region = r_l23_off[mask]
    g_region = gain_td[mask]
    w_gain = (r_base_region * g_region).sum().item() / (r_base_region.sum().item() + 1e-12)
    activity_frac = r_base_region.sum().item() / (base_sum + 1e-12) * 100
    print(f"  {region_name:>20s}: weighted_gain={w_gain:.4f}, base_activity={activity_frac:.1f}% of total, unweighted_gain={g_region.mean().item():.4f}")

# Direct calculation: how much does gain ADD or REMOVE from each region?
print(f"\nDirect gain contribution (from steady state):")
print(f"  {'Region':<20s}  {'base_sum':>10s}  {'gained_sum':>10s}  {'Δ':>10s}  {'Δ/base%':>10s}")
print(f"  " + "-" * 65)
for region_name, mask in [("Center (≤10°)", center_mask), ("Surround", surround_mask), ("Far", far_mask), ("ALL", torch.ones(N, dtype=torch.bool))]:
    base = r_l23_off[mask].sum().item()
    gained = r_l23_td[mask].sum().item()
    delta = gained - base
    pct = delta / (base + 1e-12) * 100
    print(f"  {region_name:<20s}  {base:>10.6f}  {gained:>10.6f}  {delta:>+10.6f}  {pct:>+9.1f}%")

# ======================================================================
# ANALYSIS 3: Per-channel comparison of all 3 modes
# ======================================================================
print("\n" + "=" * 80)
print("ANALYSIS 3: Per-channel L2/3 comparison (steady state, step 11)")
print("=" * 80)

print(f"{'Ch':>4s} {'Deg':>6s}  {'TopDown':>10s}  {'Gated':>10s}  {'OFF':>10s}  "
      f"{'TD/OFF':>8s}  {'Gated/OFF':>10s}  {'TD_gain':>8s}  {'G_gain':>8s}  {'Region':>8s}")
print("-" * 95)

for ch in range(N):
    deg = ch * period / N
    td = hist_topdown['r_l23'][-1, ch].item()
    gt = hist_gated['r_l23'][-1, ch].item()
    off = hist_off['r_l23'][-1, ch].item()
    td_ratio = td / (off + 1e-12) if off > 1e-8 else float('inf')
    gt_ratio = gt / (off + 1e-12) if off > 1e-8 else float('inf')
    g_td = hist_topdown['apical_gain'][-1, ch].item()
    g_gt = hist_gated['apical_gain'][-1, ch].item()
    region = "CENTER" if center_mask[ch] else ("SURR" if surround_mask[ch] else "FAR")
    marker = " <-- stim" if ch == peak_ch else ""

    td_r = f"{td_ratio:.3f}" if td_ratio < 100 else "inf"
    gt_r = f"{gt_ratio:.3f}" if gt_ratio < 100 else "inf"
    print(f"{ch:>4d} {deg:>5.1f}°  {td:>10.6f}  {gt:>10.6f}  {off:>10.6f}  "
          f"{td_r:>8s}  {gt_r:>10s}  {g_td:>8.4f}  {g_gt:>8.4f}  {region:>8s}{marker}")

# ======================================================================
# ANALYSIS 4: Energy cost as computed by the loss function
# ======================================================================
print("\n" + "=" * 80)
print("ANALYSIS 4: Energy cost matching the loss function")
print("=" * 80)

for mode_name, hist in modes:
    r_l4_mean = hist['r_l4'][-3:].abs().mean().item()
    r_l23_mean = hist['r_l23'][-3:].abs().mean().item()
    dt_mean = hist['deep_template'][-3:].abs().mean().item()
    r_pv_mean = hist['r_pv'][-3:].abs().mean().item()
    r_som_mean = hist['r_som'][-3:].abs().mean().item()
    r_vip_mean = hist['r_vip'][-3:].abs().mean().item()

    e_exc = r_l4_mean + r_l23_mean + dt_mean
    e_inh = r_pv_mean + r_som_mean + r_vip_mean
    e_total = e_exc + e_inh

    print(f"\n  {mode_name}:")
    print(f"    r_l4.abs().mean():     {r_l4_mean:.6f}")
    print(f"    r_l23.abs().mean():    {r_l23_mean:.6f}")
    print(f"    deep_tmpl.abs().mean():{dt_mean:.6f}")
    print(f"    E_excitatory:          {e_exc:.6f}")
    print(f"    r_pv.abs().mean():     {r_pv_mean:.6f}")
    print(f"    r_som.abs().mean():    {r_som_mean:.6f}")
    print(f"    r_vip.abs().mean():    {r_vip_mean:.6f}")
    print(f"    E_inhibitory:          {e_inh:.6f}")
    print(f"    E_total:               {e_total:.6f}")

# Energy comparison
td_e = sum(hist_topdown[k][-3:].abs().mean().item() for k in ['r_l4', 'r_l23', 'deep_template', 'r_pv', 'r_som', 'r_vip'])
gt_e = sum(hist_gated[k][-3:].abs().mean().item() for k in ['r_l4', 'r_l23', 'deep_template', 'r_pv', 'r_som', 'r_vip'])
off_e = sum(hist_off[k][-3:].abs().mean().item() for k in ['r_l4', 'r_l23', 'deep_template', 'r_pv', 'r_som', 'r_vip'])

print(f"\n  Energy comparison:")
print(f"    Top-down (TRAINED): {td_e:.6f}")
print(f"    Gated:              {gt_e:.6f}")
print(f"    FB OFF:             {off_e:.6f}")
print(f"    Top-down / OFF:     {td_e/off_e:.3f}x")
print(f"    Gated / OFF:        {gt_e/off_e:.3f}x")
print(f"    Top-down / Gated:   {td_e/gt_e:.3f}x")

# ======================================================================
# ANALYSIS 5: Verify the gain is actually being applied correctly
# ======================================================================
print("\n" + "=" * 80)
print("ANALYSIS 5: Sanity check — verify gain computation is correct")
print("=" * 80)

# Manually recompute gain at last step for top-down mode
with torch.no_grad():
    net.feedback.cache_kernels()
    q_true = net._make_bump(torch.tensor([theta_stim]), sigma=oracle_sigma)
    pi_eff = torch.full((1, 1), oracle_pi)
    q_centered = q_true - q_true.mean(dim=-1, keepdim=True)

    apical_circulant = net.feedback._cached_apical_circulant
    apical_field = (apical_circulant @ q_centered.unsqueeze(-1)).squeeze(-1)

    # Pure top-down: gain = 1 + mag * tanh(pi * apical_field)
    gain_manual_td = 1.0 + net.feedback.max_apical_gain * torch.tanh(pi_eff * apical_field)

    # Coincidence-gated: gain = 1 + mag * tanh(pi * relu(apical) * relu(basal))
    r_l4_last = hist_topdown['r_l4'][-1].unsqueeze(0)  # [1, N]
    basal_field = r_l4_last - r_l4_last.mean(dim=-1, keepdim=True)
    coincidence = F.relu(apical_field) * F.relu(basal_field)
    gain_manual_gated = 1.0 + net.feedback.max_apical_gain * torch.tanh(pi_eff * coincidence)

    net.feedback.uncache_kernels()

# Compare manual vs simulation
gain_sim_td = hist_topdown['apical_gain'][-1]
gain_sim_gt = hist_gated['apical_gain'][-1]

print(f"\n  Top-down gain match (manual vs sim):")
print(f"    Max absolute diff: {(gain_manual_td[0] - gain_sim_td).abs().max().item():.2e}")
print(f"    Manual at stim:    {gain_manual_td[0, peak_ch].item():.6f}")
print(f"    Sim at stim:       {gain_sim_td[peak_ch].item():.6f}")

print(f"\n  Gated gain (manual computation for reference):")
print(f"    Manual at stim:    {gain_manual_gated[0, peak_ch].item():.6f}")
print(f"    Sim at stim:       {gain_sim_gt[peak_ch].item():.6f}")
print(f"    (These may differ slightly because L4 evolves differently under gated vs top-down)")

print(f"\n  Key difference: apical_field can be negative!")
print(f"    apical_field at ch 0 (0°, far): {apical_field[0, 0].item():.6f}")
print(f"    apical_field at ch 18 (90°, stim): {apical_field[0, peak_ch].item():.6f}")
print(f"    tanh(pi * field) at ch 0: {torch.tanh(pi_eff[0,0] * apical_field[0, 0]).item():.6f}")
print(f"    tanh(pi * field) at ch 18: {torch.tanh(pi_eff[0,0] * apical_field[0, peak_ch]).item():.6f}")
print(f"    relu(field) at ch 0: {F.relu(apical_field[0, 0]).item():.6f} (clipped to 0 → gain=1.0 in gated mode)")
print(f"    relu(field) at ch 18: {F.relu(apical_field[0, peak_ch]).item():.6f}")

# ======================================================================
# ANALYSIS 6: Would the model benefit from higher energy lambda?
# ======================================================================
print("\n" + "=" * 80)
print("ANALYSIS 6: Loss sensitivity analysis")
print("=" * 80)

# From training log final values
sens_final = 1.4738
kl_final = 0.8467
energy_final_topdown = td_e  # our measured value for training mode
lambda_e = 0.01

print(f"  Current: λ_energy={lambda_e}, energy={energy_final_topdown:.4f}, λ*E={lambda_e*energy_final_topdown:.4f}")
print(f"  Sensory loss: {sens_final:.4f} (contributes {sens_final/(sens_final + kl_final + lambda_e*energy_final_topdown)*100:.1f}% of gradient)")
print(f"  Energy loss:  {lambda_e*energy_final_topdown:.4f} (contributes {lambda_e*energy_final_topdown/(sens_final + kl_final + lambda_e*energy_final_topdown)*100:.2f}% of gradient)")

# What λ_energy would make energy 10% of gradient?
target_frac = 0.10
# λ_e_new * E / (sens + kl + λ_e_new * E) = target_frac
# λ_e_new * E = target_frac * (sens + kl + λ_e_new * E)
# λ_e_new * E * (1 - target_frac) = target_frac * (sens + kl)
# λ_e_new = target_frac * (sens + kl) / (E * (1 - target_frac))
lambda_e_10pct = target_frac * (sens_final + kl_final) / (energy_final_topdown * (1 - target_frac))
print(f"\n  To make energy 10% of gradient: λ_energy = {lambda_e_10pct:.2f}")
print(f"  That's {lambda_e_10pct/lambda_e:.0f}× the current value")
