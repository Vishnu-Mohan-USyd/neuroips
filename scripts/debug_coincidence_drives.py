"""Diagnose WHY L2/3 is near-zero: inspect drive components.

Previous experiment showed:
- L2/3 ≈ 0 for all conditions (wrong, wrong30, uniform) except true (0.00002)
- SOM at stim ch: true=0.47, wrong10=0.58, wrong30=0.96, uniform=1.08
- Gain: true=1.107, wrong10=1.085 (2% diff)

Questions:
1. What are the L2/3 drive components (excitatory, SOM, PV)?
2. Is L2/3 clamped at zero because SOM overwhelms excitatory?
3. Does this change if we pre-warm the network (simulate sequence context)?
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

model_cfg, train_cfg, stim_cfg = load_config("config/apical_template_true.yaml")
N = model_cfg.n_orientations
period = model_cfg.orientation_range
sigma_ff = model_cfg.sigma_ff

net = LaminarV1V2Network(model_cfg)
ckpt = torch.load("results/batch1/true_s42/center_surround_seed42/checkpoint.pt",
                   map_location="cpu", weights_only=False)
net.load_state_dict(ckpt["model_state"], strict=False)
net.eval()

net.oracle_mode = True
net.feedback_scale = torch.tensor(1.0)

theta_stim = 90.0
contrast = 0.5
stim = generate_grating(torch.tensor([theta_stim]), torch.tensor([contrast]),
                        N, sigma_ff, period=period)
cue = torch.zeros(1, 2)
task = torch.zeros(1, 2)

# True oracle
q_true = net._make_bump(torch.tensor([theta_stim]), sigma=train_cfg.oracle_sigma)
pi_oracle = torch.full((1, 1), train_cfg.oracle_pi)
net.oracle_q_pred = q_true
net.oracle_pi_pred = pi_oracle

# ── Inspect L2/3 internal drive components ──────────────────────────
print("=" * 80)
print("L2/3 DRIVE DECOMPOSITION (true template, from zero state)")
print("=" * 80)

state = initial_state(1, N, model_cfg.v2_hidden_dim)
peak_ch = 18

# Inspect learned weights
print(f"\nL2/3 learned parameters:")
print(f"  w_som: {net.l23.w_som.gain.item():.4f}")
print(f"  w_pv_l23: {net.l23.w_pv_l23.gain.item():.4f}")
print(f"  sigma_rec: {F.softplus(net.l23.sigma_rec_raw).item():.2f}°")
print(f"  gain_rec: {F.softplus(net.l23.gain_rec_raw).item():.4f}")

with torch.no_grad():
    net.l23.cache_kernels()
    if hasattr(net.feedback, 'cache_kernels'):
        net.feedback.cache_kernels()

    print(f"\n{'Step':>4s}  {'ff@stim':>10s}  {'rec@stim':>10s}  {'tmpl@stim':>10s}  "
          f"{'exc@stim':>10s}  {'gain@stim':>10s}  {'exc*g@stim':>10s}  "
          f"{'som_inh':>10s}  {'pv_inh':>10s}  {'drive':>10s}  {'r_l23':>10s}")
    print("-" * 120)

    for t in range(12):
        # L4 step
        r_l4, adaptation = net.l4(stim, state.r_l4, state.r_pv, state.adaptation)
        r_pv = net.pv(r_l4, state.r_l23, state.r_pv)

        # Feedback
        q_pred = q_true
        pi_eff = pi_oracle * net.feedback_scale
        som_drive, vip_drive, apical_gain = net.feedback(q_pred, pi_eff, r_l4=r_l4)
        r_vip = net.vip(vip_drive, state.r_vip)
        effective_som_drive = F.relu(som_drive - F.softplus(net.w_vip_som) * r_vip)
        center_exc = net.w_template_drive * net.deep_template(q_pred, pi_eff)
        r_som = net.som(effective_som_drive, state.r_som)

        # L2/3 internal components
        ff = F.linear(r_l4, net.l23.W_l4_to_l23)  # [1, N]
        W_rec = net.l23.W_rec
        rec = F.linear(state.r_l23, W_rec)
        excitatory_drive = ff + rec + center_exc
        exc_gained = apical_gain * excitatory_drive

        som_inh = net.l23.w_som(r_som)
        pv_inh = net.l23.w_pv_l23(r_pv)
        l23_drive = exc_gained - som_inh - pv_inh

        # Actual L2/3 step
        from src.utils import rectified_softplus
        r_l23 = state.r_l23 + (net.l23.dt / net.l23.tau_l23) * (
            -state.r_l23 + rectified_softplus(l23_drive)
        )

        print(f"{t:>4d}  {ff[0, peak_ch].item():>10.6f}  {rec[0, peak_ch].item():>10.6f}  "
              f"{center_exc[0, peak_ch].item():>10.6f}  "
              f"{excitatory_drive[0, peak_ch].item():>10.6f}  "
              f"{apical_gain[0, peak_ch].item():>10.6f}  "
              f"{exc_gained[0, peak_ch].item():>10.6f}  "
              f"{som_inh[0, peak_ch].item():>10.6f}  "
              f"{pv_inh[0, 0].item():>10.6f}  "
              f"{l23_drive[0, peak_ch].item():>10.6f}  "
              f"{r_l23[0, peak_ch].item():>10.6f}")

        state = state._replace(
            r_l4=r_l4, r_l23=r_l23, r_pv=r_pv, r_som=r_som,
            r_vip=r_vip, adaptation=adaptation,
        )

    net.l23.uncache_kernels()
    if hasattr(net.feedback, 'uncache_kernels'):
        net.feedback.uncache_kernels()

# ── Now compare true vs wrong drives at steady state ────────────
print("\n" + "=" * 80)
print("STEADY-STATE DRIVE COMPARISON: true vs wrong10 vs wrong30 vs uniform")
print("=" * 80)

conditions = {
    "true (0°)":  net._make_bump(torch.tensor([theta_stim]), sigma=train_cfg.oracle_sigma),
    "wrong (10°)": net._make_bump(torch.tensor([theta_stim + 10.0]), sigma=train_cfg.oracle_sigma),
    "wrong (30°)": net._make_bump(torch.tensor([theta_stim + 30.0]), sigma=train_cfg.oracle_sigma),
    "uniform": torch.full((1, N), 1.0 / N),
}

for cond_name, q_oracle in conditions.items():
    net.oracle_q_pred = q_oracle
    net.oracle_pi_pred = pi_oracle

    state = initial_state(1, N, model_cfg.v2_hidden_dim)

    with torch.no_grad():
        net.l23.cache_kernels()
        if hasattr(net.feedback, 'cache_kernels'):
            net.feedback.cache_kernels()

        for t in range(12):
            r_l4, adaptation = net.l4(stim, state.r_l4, state.r_pv, state.adaptation)
            r_pv = net.pv(r_l4, state.r_l23, state.r_pv)
            q_pred = q_oracle
            pi_eff = pi_oracle * net.feedback_scale
            som_drive, vip_drive, apical_gain = net.feedback(q_pred, pi_eff, r_l4=r_l4)
            r_vip = net.vip(vip_drive, state.r_vip)
            effective_som_drive = F.relu(som_drive - F.softplus(net.w_vip_som) * r_vip)
            center_exc = net.w_template_drive * net.deep_template(q_pred, pi_eff)
            r_som = net.som(effective_som_drive, state.r_som)

            ff = F.linear(r_l4, net.l23.W_l4_to_l23)
            W_rec = net.l23.W_rec
            rec = F.linear(state.r_l23, W_rec)
            excitatory_drive = ff + rec + center_exc
            exc_gained = apical_gain * excitatory_drive
            som_inh = net.l23.w_som(r_som)
            pv_inh = net.l23.w_pv_l23(r_pv)
            l23_drive = exc_gained - som_inh - pv_inh
            r_l23 = state.r_l23 + (net.l23.dt / net.l23.tau_l23) * (
                -state.r_l23 + rectified_softplus(l23_drive)
            )

            state = state._replace(
                r_l4=r_l4, r_l23=r_l23, r_pv=r_pv, r_som=r_som,
                r_vip=r_vip, adaptation=adaptation,
            )

        net.l23.uncache_kernels()
        if hasattr(net.feedback, 'uncache_kernels'):
            net.feedback.uncache_kernels()

    # Report step 11 values
    print(f"\n{cond_name}:")
    print(f"  ff@stim:      {ff[0, peak_ch].item():.6f}")
    print(f"  rec@stim:     {rec[0, peak_ch].item():.8f}")
    print(f"  template:     {center_exc[0, peak_ch].item():.6f}")
    print(f"  exc(ff+r+t):  {excitatory_drive[0, peak_ch].item():.6f}")
    print(f"  gain@stim:    {apical_gain[0, peak_ch].item():.6f}")
    print(f"  exc*gain:     {exc_gained[0, peak_ch].item():.6f}")
    print(f"  som_inh:      {som_inh[0, peak_ch].item():.6f}")
    print(f"  pv_inh:       {pv_inh[0, 0].item():.6f}")
    print(f"  net_drive:    {l23_drive[0, peak_ch].item():.6f}")
    print(f"  r_l23@stim:   {r_l23[0, peak_ch].item():.8f}")
    print(f"  r_som@stim:   {r_som[0, peak_ch].item():.6f}")

# ── Check what w_template_drive and deep_template contribute ─────
print("\n" + "=" * 80)
print("TEMPLATE EXCITATION ANALYSIS")
print("=" * 80)
with torch.no_grad():
    for cond_name, q in conditions.items():
        pi_eff = pi_oracle * net.feedback_scale
        dt = net.deep_template(q, pi_eff)
        ce = net.w_template_drive * dt
        print(f"{cond_name:>15s}: w_template_drive={net.w_template_drive.item():.4f}, "
              f"deep_template@stim={dt[0, peak_ch].item():.6f}, "
              f"center_exc@stim={ce[0, peak_ch].item():.6f}, "
              f"dt_peak={dt.max().item():.6f}")

# ── Run same test with NO SOM (to see if overlap alone kills signal) ─
print("\n" + "=" * 80)
print("COUNTERFACTUAL: What if SOM were removed? (apical gain only)")
print("=" * 80)
print("Simulating by computing L2/3 drive without SOM subtraction:")

for cond_name, q_oracle in conditions.items():
    net.oracle_q_pred = q_oracle
    net.oracle_pi_pred = pi_oracle

    state = initial_state(1, N, model_cfg.v2_hidden_dim)

    with torch.no_grad():
        net.l23.cache_kernels()
        if hasattr(net.feedback, 'cache_kernels'):
            net.feedback.cache_kernels()

        for t in range(12):
            r_l4, adaptation = net.l4(stim, state.r_l4, state.r_pv, state.adaptation)
            r_pv = net.pv(r_l4, state.r_l23, state.r_pv)
            q_pred = q_oracle
            pi_eff = pi_oracle * net.feedback_scale
            som_drive, vip_drive, apical_gain = net.feedback(q_pred, pi_eff, r_l4=r_l4)
            r_vip = net.vip(vip_drive, state.r_vip)
            center_exc = net.w_template_drive * net.deep_template(q_pred, pi_eff)

            ff = F.linear(r_l4, net.l23.W_l4_to_l23)
            W_rec = net.l23.W_rec
            rec = F.linear(state.r_l23, W_rec)
            excitatory_drive = ff + rec + center_exc
            exc_no_som = apical_gain * excitatory_drive - net.l23.w_pv_l23(r_pv)
            # Note: we still subtract PV but NOT SOM

            from src.utils import rectified_softplus
            r_l23_no_som = state.r_l23 + (net.l23.dt / net.l23.tau_l23) * (
                -state.r_l23 + rectified_softplus(exc_no_som)
            )

            # Use real SOM for consistency but compute "what if no SOM" L2/3
            r_som_real = net.som(F.relu(som_drive - F.softplus(net.w_vip_som) * r_vip), state.r_som)
            state = state._replace(
                r_l4=r_l4, r_l23=r_l23_no_som, r_pv=r_pv, r_som=r_som_real,
                r_vip=r_vip, adaptation=adaptation,
            )

        net.l23.uncache_kernels()
        if hasattr(net.feedback, 'uncache_kernels'):
            net.feedback.uncache_kernels()

    print(f"{cond_name:>15s}: L2/3@stim (no SOM) = {r_l23_no_som[0, peak_ch].item():.6f}, "
          f"peak = {r_l23_no_som.max().item():.6f}")
