"""Full-network dynamic simulation: true vs wrong template over 12 timesteps.

Tests whether:
1. The 2.3% gain difference compounds over recurrent L2/3 dynamics
2. The shifted gain profile for wrong actively hurts population response
3. pi_eff actually reaches 3.0 as assumed
4. SOM/VIP pathways differentially interact with the coincidence

This is the definitive test: run the actual network step() with oracle
templates and measure L2/3 population responses.
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
steps_on = 12

# Load trained checkpoint
net = LaminarV1V2Network(model_cfg)
ckpt = torch.load("results/batch1/true_s42/center_surround_seed42/checkpoint.pt",
                   map_location="cpu", weights_only=False)
net.load_state_dict(ckpt["model_state"], strict=False)
net.eval()

# Enable oracle mode
net.oracle_mode = True
net.feedback_scale = torch.tensor(1.0)  # full feedback (no warmup ramp)

# Create stimulus at 90°
theta_stim = 90.0
contrast = 0.5
stim = generate_grating(torch.tensor([theta_stim]), torch.tensor([contrast]),
                        N, sigma_ff, period=period)
cue = torch.zeros(1, 2)
task = torch.zeros(1, 2)

oracle_pi = train_cfg.oracle_pi
oracle_sigma = train_cfg.oracle_sigma

# ── Build oracle templates ──────────────────────────────────────────
# True: prediction aligned with stimulus (bump at 90°)
q_true = net._make_bump(torch.tensor([theta_stim]), sigma=oracle_sigma)  # [1, N]
# Wrong: prediction at stimulus + 10° (simulating CW state with CCW oracle)
q_wrong = net._make_bump(torch.tensor([theta_stim + 10.0]), sigma=oracle_sigma)
# Wrong-30: larger offset for contrast
q_wrong30 = net._make_bump(torch.tensor([theta_stim + 30.0]), sigma=oracle_sigma)
# No template: uniform
q_uniform = torch.full((1, N), 1.0 / N)

pi_oracle = torch.full((1, 1), oracle_pi)

conditions = {
    "true (0°)":  q_true,
    "wrong (10°)": q_wrong,
    "wrong (30°)": q_wrong30,
    "uniform":    q_uniform,
}

# ── Run full network simulation ─────────────────────────────────────
print("=" * 80)
print("FULL-NETWORK DYNAMIC SIMULATION: 12 timesteps with oracle templates")
print("=" * 80)
print(f"Stimulus: {theta_stim}°, contrast={contrast}, oracle_pi={oracle_pi}")
print(f"oracle_sigma={oracle_sigma}°, steps_on={steps_on}")
print()

results = {}
for cond_name, q_oracle in conditions.items():
    # Set oracle
    net.oracle_q_pred = q_oracle  # [1, N] — per-step mode
    net.oracle_pi_pred = pi_oracle  # [1, 1]

    # Initialize state
    state = initial_state(1, N, model_cfg.v2_hidden_dim)

    # Track per-step values
    r_l23_history = []
    r_l4_history = []
    gain_history = []
    pi_eff_history = []
    som_history = []
    vip_history = []
    coincidence_history = []

    with torch.no_grad():
        net.l23.cache_kernels()
        if hasattr(net.feedback, 'cache_kernels'):
            net.feedback.cache_kernels()

        for t in range(steps_on):
            # Run one network step
            new_state, aux = net.step(stim, cue, task, state)

            # Capture intermediate values by re-running feedback manually
            q_pred = aux.q_pred
            pi_eff = aux.pi_pred_eff

            # Re-compute coincidence for inspection
            q_centered = q_pred - q_pred.mean(dim=-1, keepdim=True)
            apical_circulant = net.feedback._cached_apical_circulant
            apical_field = (apical_circulant @ q_centered.unsqueeze(-1)).squeeze(-1)
            basal_field = new_state.r_l4 - new_state.r_l4.mean(dim=-1, keepdim=True)
            coincidence = F.relu(apical_field) * F.relu(basal_field)
            apical_gain = 1.0 + net.feedback.max_apical_gain * torch.tanh(pi_eff * coincidence)

            r_l23_history.append(new_state.r_l23[0].clone())
            r_l4_history.append(new_state.r_l4[0].clone())
            gain_history.append(apical_gain[0].clone())
            pi_eff_history.append(pi_eff[0, 0].item())
            som_history.append(new_state.r_som[0].clone())
            vip_history.append(new_state.r_vip[0].clone())
            coincidence_history.append(coincidence[0].clone())

            state = new_state

        net.l23.uncache_kernels()
        if hasattr(net.feedback, 'uncache_kernels'):
            net.feedback.uncache_kernels()

    results[cond_name] = {
        'r_l23': torch.stack(r_l23_history),  # [T, N]
        'r_l4': torch.stack(r_l4_history),
        'gain': torch.stack(gain_history),
        'pi_eff': pi_eff_history,
        'r_som': torch.stack(som_history),
        'r_vip': torch.stack(vip_history),
        'coincidence': torch.stack(coincidence_history),
    }

# ── Analysis ────────────────────────────────────────────────────────
peak_ch = 18  # 90° stimulus channel

print("─" * 80)
print("1. pi_eff over time (does it actually reach oracle_pi?)")
print("─" * 80)
for cond_name in conditions:
    pis = results[cond_name]['pi_eff']
    print(f"  {cond_name:>15s}: pi_eff = {pis[0]:.4f} → {pis[-1]:.4f}")

print("\n" + "─" * 80)
print("2. Apical gain at stimulus channel (ch=18) over time")
print("─" * 80)
print(f"{'Step':>4s}", end="")
for cn in conditions:
    print(f"  {cn:>14s}", end="")
print()
print("-" * (4 + 16 * len(conditions)))
for t in range(steps_on):
    print(f"{t:>4d}", end="")
    for cn in conditions:
        g = results[cn]['gain'][t, peak_ch].item()
        print(f"  {g:>14.6f}", end="")
    print()

print("\n" + "─" * 80)
print("3. L2/3 at stimulus channel (ch=18) over time")
print("─" * 80)
print(f"{'Step':>4s}", end="")
for cn in conditions:
    print(f"  {cn:>14s}", end="")
print(f"  {'true-wrong10':>14s}  {'true-unif':>14s}")
print("-" * (4 + 16 * (len(conditions) + 2)))
for t in range(steps_on):
    print(f"{t:>4d}", end="")
    for cn in conditions:
        r = results[cn]['r_l23'][t, peak_ch].item()
        print(f"  {r:>14.6f}", end="")
    diff_tw = results["true (0°)"]['r_l23'][t, peak_ch].item() - results["wrong (10°)"]['r_l23'][t, peak_ch].item()
    diff_tu = results["true (0°)"]['r_l23'][t, peak_ch].item() - results["uniform"]['r_l23'][t, peak_ch].item()
    print(f"  {diff_tw:>14.6f}  {diff_tu:>14.6f}")

print("\n" + "─" * 80)
print("4. L2/3 population peak firing rate over time")
print("─" * 80)
print(f"{'Step':>4s}", end="")
for cn in conditions:
    print(f"  {cn:>14s}", end="")
print()
print("-" * (4 + 16 * len(conditions)))
for t in range(steps_on):
    print(f"{t:>4d}", end="")
    for cn in conditions:
        r = results[cn]['r_l23'][t].max().item()
        print(f"  {r:>14.6f}", end="")
    print()

print("\n" + "─" * 80)
print("5. L2/3 population mean firing rate (sum/N) over time")
print("─" * 80)
print(f"{'Step':>4s}", end="")
for cn in conditions:
    print(f"  {cn:>14s}", end="")
print()
print("-" * (4 + 16 * len(conditions)))
for t in range(steps_on):
    print(f"{t:>4d}", end="")
    for cn in conditions:
        r = results[cn]['r_l23'][t].mean().item()
        print(f"  {r:>14.6f}", end="")
    print()

print("\n" + "─" * 80)
print("6. SOM at stimulus channel (ch=18) over time")
print("─" * 80)
print(f"{'Step':>4s}", end="")
for cn in conditions:
    print(f"  {cn:>14s}", end="")
print()
print("-" * (4 + 16 * len(conditions)))
for t in range(steps_on):
    print(f"{t:>4d}", end="")
    for cn in conditions:
        r = results[cn]['r_som'][t, peak_ch].item()
        print(f"  {r:>14.6f}", end="")
    print()

print("\n" + "─" * 80)
print("7. Coincidence sum over time")
print("─" * 80)
print(f"{'Step':>4s}", end="")
for cn in conditions:
    print(f"  {cn:>14s}", end="")
print()
print("-" * (4 + 16 * len(conditions)))
for t in range(steps_on):
    print(f"{t:>4d}", end="")
    for cn in conditions:
        c = results[cn]['coincidence'][t].sum().item()
        print(f"  {c:>14.6f}", end="")
    print()

# ── Steady-state differences ─────────────────────────────────────
print("\n" + "=" * 80)
print("STEADY-STATE ANALYSIS (last 3 timesteps averaged)")
print("=" * 80)

for metric_name, key, ch in [
    ("L2/3 @ stim ch", 'r_l23', peak_ch),
    ("L2/3 peak", 'r_l23', None),
    ("L2/3 mean", 'r_l23', None),
    ("Gain @ stim ch", 'gain', peak_ch),
    ("SOM @ stim ch", 'r_som', peak_ch),
]:
    print(f"\n{metric_name}:")
    vals = {}
    for cn in conditions:
        if ch is not None:
            v = results[cn][key][-3:, ch].mean().item()
        elif "peak" in metric_name:
            v = results[cn][key][-3:].max(dim=-1).values.mean().item()
        else:
            v = results[cn][key][-3:].mean().item()
        vals[cn] = v
        print(f"  {cn:>15s}: {v:.6f}")

    if "true (0°)" in vals and "wrong (10°)" in vals:
        diff = vals["true (0°)"] - vals["wrong (10°)"]
        rel = diff / (vals["true (0°)"] + 1e-12) * 100
        print(f"  true - wrong10 = {diff:.6f} ({rel:.3f}%)")

# ── Final: What would happen with LARGER offset? ────────────────
print("\n" + "=" * 80)
print("COUNTERFACTUAL: wrong at 30° offset (full-network dynamics)")
print("=" * 80)
r_true_ss = results["true (0°)"]['r_l23'][-3:, peak_ch].mean().item()
r_wrong10_ss = results["wrong (10°)"]['r_l23'][-3:, peak_ch].mean().item()
r_wrong30_ss = results["wrong (30°)"]['r_l23'][-3:, peak_ch].mean().item()
r_uniform_ss = results["uniform"]['r_l23'][-3:, peak_ch].mean().item()

print(f"L2/3 at stim channel (steady state):")
print(f"  true (0°):   {r_true_ss:.6f}")
print(f"  wrong (10°): {r_wrong10_ss:.6f}  (Δ = {r_true_ss - r_wrong10_ss:.6f})")
print(f"  wrong (30°): {r_wrong30_ss:.6f}  (Δ = {r_true_ss - r_wrong30_ss:.6f})")
print(f"  uniform:     {r_uniform_ss:.6f}  (Δ = {r_true_ss - r_uniform_ss:.6f})")
print()
print(f"Discrimination ratio (true-wrong)/(true-uniform):")
denom = r_true_ss - r_uniform_ss + 1e-12
print(f"  wrong10: {(r_true_ss - r_wrong10_ss) / denom:.4f}")
print(f"  wrong30: {(r_true_ss - r_wrong30_ss) / denom:.4f}")

# ── Spatial profile comparison ──────────────────────────────────
print("\n" + "=" * 80)
print("SPATIAL PROFILE: L2/3 steady-state across channels (true vs wrong10)")
print("=" * 80)
print(f"{'Ch':>4s} {'Deg':>6s}  {'true':>10s}  {'wrong10':>10s}  {'diff':>10s}  "
      f"{'wrong30':>10s}  {'uniform':>10s}")
print("-" * 70)
for ch in range(12, 25):
    deg = ch * period / N
    rt = results["true (0°)"]['r_l23'][-1, ch].item()
    rw = results["wrong (10°)"]['r_l23'][-1, ch].item()
    rw30 = results["wrong (30°)"]['r_l23'][-1, ch].item()
    ru = results["uniform"]['r_l23'][-1, ch].item()
    marker = " <-- stim" if ch == peak_ch else ""
    print(f"{ch:>4d} {deg:>5.1f}°  {rt:>10.6f}  {rw:>10.6f}  {rt-rw:>10.6f}  "
          f"{rw30:>10.6f}  {ru:>10.6f}{marker}")
