"""Debugger: Decompose M7 contributions — SOM-only vs gain-only vs full.

Loads R1 checkpoint, manipulates alpha weights to isolate each pathway's
contribution to sharpening. Measures L2/3 response profiles under 4 conditions:
  1. Baseline (FB OFF)
  2. SOM-only (alpha_apical → 0, alpha_inh kept)
  3. Gain-only (alpha_inh → 0, alpha_apical kept)
  4. Full (both active)

For each condition, measures across 8 anchor orientations:
  - L2/3 amplitude ratio (ON/OFF)
  - FWHM of L2/3 response
  - Peak/total selectivity
  - Response contrast at ±10° offset (simplified M7 proxy)
"""

import sys
sys.path.insert(0, "/mnt/c/Users/User/codingproj/freshstart")

import torch
import torch.nn.functional as F
import copy
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
oracle_sigma = getattr(train_cfg, 'oracle_sigma', 12.0)
oracle_pi = train_cfg.oracle_pi

# Load R1 checkpoint
net = LaminarV1V2Network(model_cfg, delta_som=train_cfg.delta_som)
ckpt = torch.load("results/iter/r1_mag03_e05/center_surround_seed42/checkpoint.pt",
                   map_location="cpu", weights_only=False)
net.load_state_dict(ckpt["model_state"], strict=False)
net.eval()

# Save learned weights
alpha_inh_learned = net.feedback.alpha_inh.data.clone()
alpha_apical_learned = net.feedback.alpha_apical.data.clone()
alpha_vip_learned = net.feedback.alpha_vip.data.clone()
mag = net.feedback.max_apical_gain

print(f"max_apical_gain: {mag}")
print(f"oracle_pi: {oracle_pi}")
print(f"alpha_inh L1: {alpha_inh_learned.abs().sum().item():.4f}")
print(f"alpha_apical L1: {alpha_apical_learned.abs().sum().item():.4f}")
print(f"alpha_vip L1: {alpha_vip_learned.abs().sum().item():.6f}")


def run_forward(net, theta_stim, contrast=0.5, steps=12):
    """Run forward pass and return steady-state L2/3 [N]."""
    stim = generate_grating(torch.tensor([theta_stim]), torch.tensor([contrast]),
                            N, sigma_ff, period=period)
    cue = torch.zeros(1, 2)
    task = torch.zeros(1, 2)

    q_pred = net._make_bump(torch.tensor([theta_stim]), sigma=oracle_sigma)
    net.oracle_mode = True
    net.oracle_q_pred = q_pred
    net.oracle_pi_pred = torch.full((1, 1), oracle_pi)

    state = initial_state(1, N, model_cfg.v2_hidden_dim)
    with torch.no_grad():
        net.l23.cache_kernels()
        net.feedback.cache_kernels()
        for t in range(steps):
            state, aux = net.step(stim, cue, task, state)
        net.l23.uncache_kernels()
        net.feedback.uncache_kernels()

    return state.r_l23[0].clone(), state


def measure_profile(r_l23, theta_stim):
    """Compute sharpening metrics from L2/3 response profile."""
    peak_ch = int(theta_stim / step_deg) % N
    total = r_l23.sum().item()
    peak = r_l23[peak_ch].item()

    # FWHM: count channels above half-max
    if peak > 0:
        half_max = peak / 2
        above_hm = (r_l23 >= half_max).sum().item()
        fwhm_deg = above_hm * step_deg
    else:
        fwhm_deg = 0.0

    # Selectivity: peak / total
    selectivity = peak / total if total > 1e-10 else 0.0

    # Response at ±10° offset (2 channels away from peak)
    ch_p10 = (peak_ch + 2) % N  # +10°
    ch_m10 = (peak_ch - 2) % N  # -10°
    resp_p10 = r_l23[ch_p10].item()
    resp_m10 = r_l23[ch_m10].item()
    resp_10_avg = (resp_p10 + resp_m10) / 2

    # Contrast between peak and ±10°: (peak - nearby) / (peak + nearby)
    if peak + resp_10_avg > 1e-10:
        contrast_10 = (peak - resp_10_avg) / (peak + resp_10_avg)
    else:
        contrast_10 = 0.0

    # Response at ±5° (1 channel)
    ch_p5 = (peak_ch + 1) % N
    ch_m5 = (peak_ch - 1) % N
    resp_5_avg = (r_l23[ch_p5].item() + r_l23[ch_m5].item()) / 2

    return {
        "total": total,
        "peak": peak,
        "fwhm": fwhm_deg,
        "selectivity": selectivity,
        "resp_10": resp_10_avg,
        "contrast_10": contrast_10,
        "resp_5": resp_5_avg,
    }


def set_condition(net, condition, alpha_inh_orig, alpha_apical_orig):
    """Set network weights for a given condition."""
    if condition == "baseline":
        net.feedback_scale.fill_(0.0)
        # Doesn't matter what alphas are — fb_scale=0 kills everything
    elif condition == "som_only":
        net.feedback_scale.fill_(1.0)
        net.feedback.alpha_inh.data.copy_(alpha_inh_orig)
        net.feedback.alpha_apical.data.zero_()  # Kill apical gain
    elif condition == "gain_only":
        net.feedback_scale.fill_(1.0)
        net.feedback.alpha_inh.data.zero_()  # Kill SOM
        net.feedback.alpha_apical.data.copy_(alpha_apical_orig)
    elif condition == "full":
        net.feedback_scale.fill_(1.0)
        net.feedback.alpha_inh.data.copy_(alpha_inh_orig)
        net.feedback.alpha_apical.data.copy_(alpha_apical_orig)
    else:
        raise ValueError(f"Unknown condition: {condition}")


# ======================================================================
# RUN ALL CONDITIONS ACROSS 8 ANCHORS
# ======================================================================
anchors = [0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5]
conditions = ["baseline", "som_only", "gain_only", "full"]

# Store results: condition -> anchor -> metrics dict
all_results = {c: {} for c in conditions}

for condition in conditions:
    print(f"\n--- Running condition: {condition} ---")
    set_condition(net, condition, alpha_inh_learned, alpha_apical_learned)

    for anchor in anchors:
        r_l23, _ = run_forward(net, anchor)
        metrics = measure_profile(r_l23, anchor)
        all_results[condition][anchor] = metrics

# Restore original weights
net.feedback.alpha_inh.data.copy_(alpha_inh_learned)
net.feedback.alpha_apical.data.copy_(alpha_apical_learned)
net.feedback_scale.fill_(1.0)

# ======================================================================
# AGGREGATE RESULTS
# ======================================================================
print(f"\n{'='*80}")
print("RESULTS BY CONDITION (averaged across 8 anchors)")
print(f"{'='*80}")

print(f"\n{'Condition':>12s}  {'total':>8s}  {'peak':>8s}  {'FWHM':>6s}  {'sel':>8s}  "
      f"{'resp@10':>8s}  {'ctrst@10':>8s}  {'resp@5':>8s}")
print("-" * 85)

for condition in conditions:
    avgs = {}
    for key in ["total", "peak", "fwhm", "selectivity", "resp_10", "contrast_10", "resp_5"]:
        vals = [all_results[condition][a][key] for a in anchors]
        avgs[key] = sum(vals) / len(vals)

    print(f"{condition:>12s}  {avgs['total']:>8.4f}  {avgs['peak']:>8.4f}  "
          f"{avgs['fwhm']:>5.1f}°  {avgs['selectivity']:>8.4f}  "
          f"{avgs['resp_10']:>8.4f}  {avgs['contrast_10']:>8.4f}  {avgs['resp_5']:>8.4f}")

# Derived metrics
bl = {k: sum(all_results["baseline"][a][k] for a in anchors) / len(anchors) for k in ["total", "peak", "selectivity", "fwhm", "contrast_10"]}

print(f"\n{'='*80}")
print("DERIVED METRICS (relative to baseline)")
print(f"{'='*80}")
print(f"\n{'Condition':>12s}  {'amp_ratio':>10s}  {'ΔFWHM':>8s}  {'Δsel':>10s}  {'Δctrst@10':>10s}")
print("-" * 55)

for condition in conditions:
    avgs = {}
    for key in ["total", "peak", "fwhm", "selectivity", "contrast_10"]:
        vals = [all_results[condition][a][key] for a in anchors]
        avgs[key] = sum(vals) / len(vals)

    amp = avgs["total"] / bl["total"] if bl["total"] > 0 else 0
    d_fwhm = avgs["fwhm"] - bl["fwhm"]
    d_sel = avgs["selectivity"] - bl["selectivity"]
    d_ctrst = avgs["contrast_10"] - bl["contrast_10"]

    print(f"{condition:>12s}  {amp:>10.4f}  {d_fwhm:>+7.1f}°  {d_sel:>+10.4f}  {d_ctrst:>+10.4f}")

# ======================================================================
# DETAILED: 90° ANCHOR PROFILES
# ======================================================================
theta_detail = 90.0
peak_ch = int(theta_detail / step_deg)

print(f"\n{'='*80}")
print(f"DETAILED PROFILES AT {theta_detail}° STIMULUS")
print(f"{'='*80}")

# Regenerate profiles for detailed view
profiles = {}
for condition in conditions:
    set_condition(net, condition, alpha_inh_learned, alpha_apical_learned)
    r_l23, state = run_forward(net, theta_detail)
    profiles[condition] = r_l23

# Restore
net.feedback.alpha_inh.data.copy_(alpha_inh_learned)
net.feedback.alpha_apical.data.copy_(alpha_apical_learned)
net.feedback_scale.fill_(1.0)

print(f"\n{'Ch':>4s} {'Deg':>6s}  {'baseline':>10s}  {'som_only':>10s}  {'gain_only':>10s}  {'full':>10s}")
print("-" * 60)
for ch in range(N):
    deg = ch * step_deg
    marker = " <-- stim" if ch == peak_ch else ""
    vals = [profiles[c][ch].item() for c in conditions]
    print(f"{ch:>4d} {deg:>5.1f}°  {vals[0]:>10.6f}  {vals[1]:>10.6f}  {vals[2]:>10.6f}  {vals[3]:>10.6f}{marker}")

# ======================================================================
# SIMPLIFIED M7 PROXY: match vs near-miss discrimination
# ======================================================================
print(f"\n{'='*80}")
print("SIMPLIFIED M7 PROXY: Match vs Near-Miss Response Difference")
print(f"{'='*80}")
print("For each condition, generate L2/3 at θ=90° (match) and θ=100° (near-miss at δ=10°)")
print("Measure: response difference at peak channel (ch 18)")

deltas_to_test = [5.0, 10.0, 15.0]

for delta in deltas_to_test:
    print(f"\n--- δ = {delta}° ---")
    theta_match = 90.0
    theta_miss = (90.0 + delta) % period
    peak_match = int(theta_match / step_deg)

    print(f"{'Condition':>12s}  {'L23_match':>10s}  {'L23_miss':>10s}  {'diff':>10s}  {'ratio':>8s}  "
          f"{'total_m':>8s}  {'total_mm':>8s}  {'amp_diff':>10s}")
    print("-" * 90)

    for condition in conditions:
        set_condition(net, condition, alpha_inh_learned, alpha_apical_learned)

        # Match: stimulus=90°, oracle=90°
        r_match, _ = run_forward(net, theta_match)

        # Near-miss: stimulus=90°+δ, oracle=90° (oracle still thinks 90° is correct)
        # Actually — for M7, the STIMULUS is at θ or θ+δ, and the ORACLE is at θ (the anchor)
        # So for match: stim=θ, oracle=θ
        # For near-miss: stim=θ+δ, oracle=θ
        stim_miss = generate_grating(torch.tensor([theta_miss]), torch.tensor([0.5]),
                                     N, sigma_ff, period=period)
        q_oracle = net._make_bump(torch.tensor([theta_match]), sigma=oracle_sigma)
        net.oracle_mode = True
        net.oracle_q_pred = q_oracle  # Oracle at match orientation
        net.oracle_pi_pred = torch.full((1, 1), oracle_pi)

        state_miss = initial_state(1, N, model_cfg.v2_hidden_dim)
        cue = torch.zeros(1, 2)
        task = torch.zeros(1, 2)
        with torch.no_grad():
            net.l23.cache_kernels()
            net.feedback.cache_kernels()
            for t in range(steps_on):
                state_miss, _ = net.step(stim_miss, cue, task, state_miss)
            net.l23.uncache_kernels()
            net.feedback.uncache_kernels()

        r_miss = state_miss.r_l23[0].clone()

        l23_match = r_match[peak_match].item()
        l23_miss = r_miss[peak_match].item()
        diff = l23_match - l23_miss
        ratio = l23_match / l23_miss if l23_miss > 1e-8 else float('inf')
        total_m = r_match.sum().item()
        total_mm = r_miss.sum().item()
        amp_diff = total_m - total_mm

        print(f"{condition:>12s}  {l23_match:>10.6f}  {l23_miss:>10.6f}  {diff:>+10.6f}  "
              f"{ratio:>8.4f}  {total_m:>8.4f}  {total_mm:>8.4f}  {amp_diff:>+10.4f}")

# Restore
net.feedback.alpha_inh.data.copy_(alpha_inh_learned)
net.feedback.alpha_apical.data.copy_(alpha_apical_learned)
net.feedback_scale.fill_(1.0)

# ======================================================================
# KEY QUESTION: What does SOM-only contribute to sharpening?
# ======================================================================
print(f"\n{'='*80}")
print("KEY ANALYSIS: SOM CONTRIBUTION TO SHARPENING")
print(f"{'='*80}")

# Compute SOM-only effect metrics
bl_metrics = {k: sum(all_results["baseline"][a][k] for a in anchors) / len(anchors)
              for k in ["total", "peak", "fwhm", "selectivity", "contrast_10"]}
som_metrics = {k: sum(all_results["som_only"][a][k] for a in anchors) / len(anchors)
               for k in ["total", "peak", "fwhm", "selectivity", "contrast_10"]}
gain_metrics = {k: sum(all_results["gain_only"][a][k] for a in anchors) / len(anchors)
                for k in ["total", "peak", "fwhm", "selectivity", "contrast_10"]}
full_metrics = {k: sum(all_results["full"][a][k] for a in anchors) / len(anchors)
                for k in ["total", "peak", "fwhm", "selectivity", "contrast_10"]}

som_amp = som_metrics["total"] / bl_metrics["total"] if bl_metrics["total"] > 0 else 0
gain_amp = gain_metrics["total"] / bl_metrics["total"] if bl_metrics["total"] > 0 else 0
full_amp = full_metrics["total"] / bl_metrics["total"] if bl_metrics["total"] > 0 else 0

print(f"\nAmplitude ratios:")
print(f"  SOM-only:  {som_amp:.4f}")
print(f"  Gain-only: {gain_amp:.4f}")
print(f"  Full:      {full_amp:.4f}")

print(f"\nFWHM changes:")
print(f"  Baseline:  {bl_metrics['fwhm']:.1f}°")
print(f"  SOM-only:  {som_metrics['fwhm']:.1f}° (Δ={som_metrics['fwhm'] - bl_metrics['fwhm']:+.1f}°)")
print(f"  Gain-only: {gain_metrics['fwhm']:.1f}° (Δ={gain_metrics['fwhm'] - bl_metrics['fwhm']:+.1f}°)")
print(f"  Full:      {full_metrics['fwhm']:.1f}° (Δ={full_metrics['fwhm'] - bl_metrics['fwhm']:+.1f}°)")

print(f"\nSelectivity (peak/total):")
print(f"  Baseline:  {bl_metrics['selectivity']:.4f}")
print(f"  SOM-only:  {som_metrics['selectivity']:.4f} (Δ={som_metrics['selectivity'] - bl_metrics['selectivity']:+.4f})")
print(f"  Gain-only: {gain_metrics['selectivity']:.4f} (Δ={gain_metrics['selectivity'] - bl_metrics['selectivity']:+.4f})")
print(f"  Full:      {full_metrics['selectivity']:.4f} (Δ={full_metrics['selectivity'] - bl_metrics['selectivity']:+.4f})")

print(f"\nContrast at ±10°:")
print(f"  Baseline:  {bl_metrics['contrast_10']:.4f}")
print(f"  SOM-only:  {som_metrics['contrast_10']:.4f} (Δ={som_metrics['contrast_10'] - bl_metrics['contrast_10']:+.4f})")
print(f"  Gain-only: {gain_metrics['contrast_10']:.4f} (Δ={gain_metrics['contrast_10'] - bl_metrics['contrast_10']:+.4f})")
print(f"  Full:      {full_metrics['contrast_10']:.4f} (Δ={full_metrics['contrast_10'] - bl_metrics['contrast_10']:+.4f})")

# Estimate: if SOM sharpening is independent of gain, total effect ≈ SOM + gain
som_sel_delta = som_metrics["selectivity"] - bl_metrics["selectivity"]
gain_sel_delta = gain_metrics["selectivity"] - bl_metrics["selectivity"]
full_sel_delta = full_metrics["selectivity"] - bl_metrics["selectivity"]
interaction = full_sel_delta - (som_sel_delta + gain_sel_delta)

print(f"\nAdditivity check (selectivity):")
print(f"  SOM Δsel:  {som_sel_delta:+.4f}")
print(f"  Gain Δsel: {gain_sel_delta:+.4f}")
print(f"  Sum:       {som_sel_delta + gain_sel_delta:+.4f}")
print(f"  Full Δsel: {full_sel_delta:+.4f}")
print(f"  Interaction: {interaction:+.4f} ({'synergistic' if interaction > 0 else 'antagonistic'})")

# Predict M7 at lower mag values
# If SOM provides a fixed M7 floor, then at mag=0.10:
# M7(total) ≈ M7(gain, mag=0.10) + M7(SOM)
# M7(gain, mag=0.10) ≈ M7(gain, mag=0.30) * 0.10/0.30
print(f"\n{'='*80}")
print("PREDICTION: CAN mag=0.10 REACH M7 > +0.03?")
print(f"{'='*80}")
print(f"\nAssuming SOM contribution is independent of mag:")
print(f"  SOM Δselectivity: {som_sel_delta:+.4f}")
print(f"  Gain Δselectivity at mag=0.30: {gain_sel_delta:+.4f}")
print(f"  Gain Δselectivity at mag=0.10 (extrapolated): {gain_sel_delta * 0.10/0.30:+.4f}")
print(f"  Total at mag=0.10: {som_sel_delta + gain_sel_delta * 0.10/0.30:+.4f}")
print(f"  Full Δselectivity at mag=0.30: {full_sel_delta:+.4f}")
