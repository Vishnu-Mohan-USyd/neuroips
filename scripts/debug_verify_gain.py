"""Debugger: Verify 3 claims about gain profiles.

Verify 1: Does pure top-down gain go below 1.0 at flanks?
Verify 2: Would mean-normalized gain preserve sharpening?
Verify 3: What was the gain profile at mag=0.2 (Branch A)?
"""

import sys
sys.path.insert(0, "/mnt/c/Users/User/codingproj/freshstart")

import torch
import torch.nn.functional as F
from src.config import load_config
from src.model.network import LaminarV1V2Network
from src.state import initial_state
from src.stimulus.gratings import generate_grating

# ── Common setup ────────────────────────────────────────────────────
model_cfg, train_cfg, stim_cfg = load_config("config/exp_branch_a.yaml")
N = model_cfg.n_orientations
period = model_cfg.orientation_range
sigma_ff = model_cfg.sigma_ff
steps_on = train_cfg.steps_on  # 12

theta_stim = 90.0
contrast = 0.5
stim = generate_grating(torch.tensor([theta_stim]), torch.tensor([contrast]),
                        N, sigma_ff, period=period)
cue = torch.zeros(1, 2)
task = torch.zeros(1, 2)
peak_ch = int(theta_stim / (period / N))  # 18

oracle_sigma = getattr(train_cfg, 'oracle_sigma', 12.0)
oracle_pi = train_cfg.oracle_pi


def run_and_get_gain(net, stim, cue, task, steps, peak_ch, N, period, label):
    """Run forward pass, return steady-state apical_gain tensor [N]."""
    net.oracle_mode = True
    net.feedback_scale.fill_(1.0)
    q_true = net._make_bump(torch.tensor([theta_stim]), sigma=oracle_sigma)
    pi_oracle = torch.full((1, 1), oracle_pi)
    net.oracle_q_pred = q_true
    net.oracle_pi_pred = pi_oracle

    state = initial_state(1, N, model_cfg.v2_hidden_dim)

    with torch.no_grad():
        net.l23.cache_kernels()
        if hasattr(net.feedback, 'cache_kernels'):
            net.feedback.cache_kernels()

        for t in range(steps):
            new_state, aux = net.step(stim, cue, task, state)
            state = new_state

        # Reconstruct apical gain from last step
        q_pred = aux.q_pred
        pi_eff = aux.pi_pred_eff

        q_centered = q_pred - q_pred.mean(dim=-1, keepdim=True)

        if net.feedback._cached_apical_circulant is not None:
            apical_circulant = net.feedback._cached_apical_circulant
        else:
            K_apical = net.feedback.get_apical_profile()
            apical_circulant = net.feedback._to_circulant(K_apical)

        apical_field = (apical_circulant @ q_centered.unsqueeze(-1)).squeeze(-1)  # [1, N]

        # This checkpoint was trained with r_l4=None (pure top-down)
        apical_gain_topdown = 1.0 + net.feedback.max_apical_gain * torch.tanh(pi_eff * apical_field)

        # Also compute coincidence-gated version for comparison
        basal_field = state.r_l4 - state.r_l4.mean(dim=-1, keepdim=True)
        coincidence = F.relu(apical_field) * F.relu(basal_field)
        apical_gain_gated = 1.0 + net.feedback.max_apical_gain * torch.tanh(pi_eff * coincidence)

        net.l23.uncache_kernels()
        if hasattr(net.feedback, 'uncache_kernels'):
            net.feedback.uncache_kernels()

    return apical_gain_topdown[0], apical_gain_gated[0], state, pi_eff


def report_gain(gain, label, peak_ch, N, period):
    """Print gain statistics."""
    below_1 = (gain < 1.0).sum().item()
    above_1 = (gain > 1.0).sum().item()
    at_1 = N - below_1 - above_1
    min_gain = gain.min().item()
    max_gain = gain.max().item()
    mean_gain = gain.mean().item()
    integral = (gain - 1.0).sum().item()
    gain_at_stim = gain[peak_ch].item()

    print(f"\n  {label}:")
    print(f"    Channels with gain < 1.0: {below_1}/{N}")
    print(f"    Channels with gain > 1.0: {above_1}/{N}")
    print(f"    Channels with gain ≈ 1.0: {at_1}/{N}")
    print(f"    Min gain: {min_gain:.6f} (at ch {gain.argmin().item()})")
    print(f"    Max gain: {max_gain:.6f} (at ch {gain.argmax().item()})")
    print(f"    Mean gain: {mean_gain:.6f}")
    print(f"    Gain at stimulus (ch={peak_ch}): {gain_at_stim:.6f}")
    print(f"    Integral sum(gain-1): {integral:+.6f} ({'net enhancement' if integral > 0 else 'net suppression'})")

    # Per-channel detail
    print(f"\n    {'Ch':>4s} {'Deg':>6s}  {'gain':>10s}  {'gain-1':>10s}")
    print(f"    " + "-" * 40)
    for ch in range(N):
        deg = ch * period / N
        g = gain[ch].item()
        marker = " <-- stim" if ch == peak_ch else ""
        print(f"    {ch:>4d} {deg:>5.1f}°  {g:>10.6f}  {g-1:>+10.6f}{marker}")


# ======================================================================
# VERIFY 1: Does pure top-down gain go below 1.0 at flanks?
# ======================================================================
print("=" * 80)
print("VERIFY 1: Pure top-down gain profile (control checkpoint, mag=0.7)")
print("=" * 80)

net_ctrl = LaminarV1V2Network(model_cfg, delta_som=train_cfg.delta_som)
ckpt = torch.load("results/control_no_gate/s42/center_surround_seed42/checkpoint.pt",
                   map_location="cpu", weights_only=False)
net_ctrl.load_state_dict(ckpt["model_state"], strict=False)
net_ctrl.eval()

print(f"max_apical_gain: {net_ctrl.feedback.max_apical_gain}")

gain_topdown, gain_gated, state_ctrl, pi_eff_ctrl = run_and_get_gain(
    net_ctrl, stim, cue, task, steps_on, peak_ch, N, period, "control"
)

report_gain(gain_topdown, "Pure top-down gain (how it was TRAINED)", peak_ch, N, period)
report_gain(gain_gated, "Coincidence-gated gain (for comparison)", peak_ch, N, period)

# ======================================================================
# VERIFY 2: Mean-normalized gain
# ======================================================================
print("\n" + "=" * 80)
print("VERIFY 2: Mean-normalized gain (energy-neutral by construction)")
print("=" * 80)

# Normalize the pure top-down gain (which is what this checkpoint actually uses)
gain_raw = gain_topdown
mean_g = gain_raw.mean()
gain_normalized = gain_raw / mean_g

print(f"\n  Raw gain mean: {mean_g.item():.6f}")
print(f"  After normalization, mean: {gain_normalized.mean().item():.6f}")
print(f"\n  Normalized gain at stimulus (ch={peak_ch}): {gain_normalized[peak_ch].item():.6f}")
print(f"  Raw gain at stimulus: {gain_raw[peak_ch].item():.6f}")
print(f"  Ratio (normalized/raw at stim): {gain_normalized[peak_ch].item()/gain_raw[peak_ch].item():.4f}")

# Find the center-surround contrast
# Center: channels within 10° of stimulus
# Flanks: channels 15-25° from stimulus
center_chs = []
flank_chs = []
far_chs = []
for ch in range(N):
    deg = ch * period / N
    dist = min(abs(deg - theta_stim), period - abs(deg - theta_stim))
    if dist <= 10:
        center_chs.append(ch)
    elif dist <= 25:
        flank_chs.append(ch)
    else:
        far_chs.append(ch)

norm_center = gain_normalized[center_chs].mean().item()
norm_flank = gain_normalized[flank_chs].mean().item()
norm_far = gain_normalized[far_chs].mean().item()

raw_center = gain_raw[center_chs].mean().item()
raw_flank = gain_raw[flank_chs].mean().item()
raw_far = gain_raw[far_chs].mean().item()

print(f"\n  Region means:")
print(f"  {'Region':<15s}  {'Raw':>10s}  {'Normalized':>10s}")
print(f"  {'-'*40}")
print(f"  {'Center (≤10°)':<15s}  {raw_center:>10.6f}  {norm_center:>10.6f}")
print(f"  {'Flank (10-25°)':<15s}  {raw_flank:>10.6f}  {norm_flank:>10.6f}")
print(f"  {'Far (>25°)':<15s}  {raw_far:>10.6f}  {norm_far:>10.6f}")

print(f"\n  Center-surround contrast (center/flank):")
print(f"    Raw: {raw_center/raw_flank:.4f}")
print(f"    Normalized: {norm_center/norm_flank:.4f}")
print(f"    (Same contrast means sharpening is preserved)")

print(f"\n  Center-far contrast (center/far):")
print(f"    Raw: {raw_center/raw_far:.4f}")
print(f"    Normalized: {norm_center/norm_far:.4f}")

# Show full normalized profile
print(f"\n  {'Ch':>4s} {'Deg':>6s}  {'raw':>10s}  {'normalized':>10s}  {'norm-1':>10s}")
print(f"  " + "-" * 50)
for ch in range(N):
    deg = ch * period / N
    r = gain_raw[ch].item()
    n = gain_normalized[ch].item()
    marker = " <-- stim" if ch == peak_ch else ""
    print(f"  {ch:>4d} {deg:>5.1f}°  {r:>10.6f}  {n:>10.6f}  {n-1:>+10.6f}{marker}")

# Does normalized gain go below 1.0?
norm_below = (gain_normalized < 1.0).sum().item()
norm_above = (gain_normalized > 1.0).sum().item()
print(f"\n  Normalized: {norm_below} channels below 1.0, {norm_above} above 1.0")
print(f"  Min normalized: {gain_normalized.min().item():.6f}")
print(f"  Max normalized: {gain_normalized.max().item():.6f}")

# ======================================================================
# VERIFY 3: Gain profile at mag=0.2 (Branch A checkpoint)
# ======================================================================
print("\n" + "=" * 80)
print("VERIFY 3: Gain profile at mag=0.2 (Branch A checkpoint)")
print("=" * 80)

net_ba = LaminarV1V2Network(model_cfg, delta_som=train_cfg.delta_som)
ckpt_ba = torch.load("results/branch_a/ba_s42/center_surround_seed42/checkpoint.pt",
                      map_location="cpu", weights_only=False)
net_ba.load_state_dict(ckpt_ba["model_state"], strict=False)
net_ba.eval()

print(f"max_apical_gain (Branch A): {net_ba.feedback.max_apical_gain}")

# Branch A was trained with max_apical_gain=0.5 (default at the time)
# but let's check what the checkpoint actually has

gain_ba_topdown, gain_ba_gated, state_ba, pi_eff_ba = run_and_get_gain(
    net_ba, stim, cue, task, steps_on, peak_ch, N, period, "branch_a"
)

print(f"pi_eff (Branch A): {pi_eff_ba[0,0].item():.4f}")

report_gain(gain_ba_topdown, "Branch A pure top-down gain", peak_ch, N, period)
report_gain(gain_ba_gated, "Branch A coincidence-gated gain (hypothetical)", peak_ch, N, period)

# Compare control vs Branch A
print("\n" + "=" * 80)
print("COMPARISON: Control (mag=0.7) vs Branch A")
print("=" * 80)

print(f"\n  {'Metric':<35s}  {'Control':>10s}  {'Branch A':>10s}")
print(f"  " + "-" * 60)
print(f"  {'max_apical_gain':<35s}  {net_ctrl.feedback.max_apical_gain:>10.2f}  {net_ba.feedback.max_apical_gain:>10.2f}")
print(f"  {'pi_eff':<35s}  {pi_eff_ctrl[0,0].item():>10.4f}  {pi_eff_ba[0,0].item():>10.4f}")
print(f"  {'Gain at stim (top-down)':<35s}  {gain_topdown[peak_ch].item():>10.6f}  {gain_ba_topdown[peak_ch].item():>10.6f}")
print(f"  {'Mean gain (top-down)':<35s}  {gain_topdown.mean().item():>10.6f}  {gain_ba_topdown.mean().item():>10.6f}")
print(f"  {'Integral sum(g-1) (top-down)':<35s}  {(gain_topdown-1).sum().item():>+10.6f}  {(gain_ba_topdown-1).sum().item():>+10.6f}")
print(f"  {'Channels < 1.0 (top-down)':<35s}  {(gain_topdown<1.0).sum().item():>10d}  {(gain_ba_topdown<1.0).sum().item():>10d}")
print(f"  {'Min gain (top-down)':<35s}  {gain_topdown.min().item():>10.6f}  {gain_ba_topdown.min().item():>10.6f}")
