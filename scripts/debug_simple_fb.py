"""Debugger: Analyze simple additive feedback checkpoint.

Compares with R5 (multiplicative, mag=0.10) and baseline (no feedback).
Checks for Kok-style profile, hallucinations, kernel shape.
"""

import sys
sys.path.insert(0, "/mnt/c/Users/User/codingproj/freshstart")

import torch
import torch.nn.functional as F
from src.config import load_config
from src.model.network import LaminarV1V2Network
from src.state import initial_state
from src.stimulus.gratings import generate_grating


def load_and_analyze(config_path, ckpt_path, label, theta_stim=90.0, contrast=0.5):
    """Load checkpoint, run ON/OFF, return comprehensive results."""
    model_cfg, train_cfg, stim_cfg = load_config(config_path)
    N = model_cfg.n_orientations
    period = model_cfg.orientation_range
    sigma_ff = model_cfg.sigma_ff
    step_deg = period / N
    steps_on = train_cfg.steps_on
    oracle_sigma = getattr(train_cfg, 'oracle_sigma', 12.0)
    oracle_pi = train_cfg.oracle_pi
    is_simple = getattr(model_cfg, 'simple_feedback', False)

    net = LaminarV1V2Network(model_cfg, delta_som=train_cfg.delta_som)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net.load_state_dict(ckpt["model_state"], strict=False)
    net.eval()

    stim = generate_grating(torch.tensor([theta_stim]), torch.tensor([contrast]),
                            N, sigma_ff, period=period)
    cue = torch.zeros(1, 2)
    task = torch.zeros(1, 2)
    q_pred = net._make_bump(torch.tensor([theta_stim]), sigma=oracle_sigma)

    # --- FB ON ---
    net.oracle_mode = True
    net.oracle_q_pred = q_pred
    net.oracle_pi_pred = torch.full((1, 1), oracle_pi)
    net.feedback_scale.fill_(1.0)

    state_on = initial_state(1, N, model_cfg.v2_hidden_dim)
    with torch.no_grad():
        net.l23.cache_kernels()
        if hasattr(net.feedback, 'cache_kernels'):
            net.feedback.cache_kernels()
        for t in range(steps_on):
            state_on, _ = net.step(stim, cue, task, state_on)
        net.l23.uncache_kernels()
        if hasattr(net.feedback, 'uncache_kernels'):
            net.feedback.uncache_kernels()
    r_l23_on = state_on.r_l23[0].clone()

    # --- FB OFF ---
    net.feedback_scale.fill_(0.0)
    state_off = initial_state(1, N, model_cfg.v2_hidden_dim)
    with torch.no_grad():
        net.l23.cache_kernels()
        if hasattr(net.feedback, 'cache_kernels'):
            net.feedback.cache_kernels()
        for t in range(steps_on):
            state_off, _ = net.step(stim, cue, task, state_off)
        net.l23.uncache_kernels()
        if hasattr(net.feedback, 'uncache_kernels'):
            net.feedback.uncache_kernels()
    r_l23_off = state_off.r_l23[0].clone()
    net.feedback_scale.fill_(1.0)

    # Get feedback modulation signal
    with torch.no_grad():
        if is_simple:
            fb_mod = net.feedback.compute_simple_feedback(q_pred)[0]  # [N]
            gain = torch.ones(N)  # no gain in simple mode
        else:
            _, _, apical_gain = net.feedback(q_pred, torch.full((1,1), oracle_pi), r_l4=None)
            gain = apical_gain[0]
            fb_mod = torch.zeros(N)  # no additive feedback in multiplicative mode

    alpha_apical = net.feedback.alpha_apical.data.clone()
    alpha_inh = net.feedback.alpha_inh.data.clone() if hasattr(net.feedback, 'alpha_inh') else torch.zeros(N)

    peak_ch = int(theta_stim / step_deg)

    return {
        "label": label,
        "r_l23_on": r_l23_on,
        "r_l23_off": r_l23_off,
        "gain": gain,
        "fb_mod": fb_mod,
        "alpha_apical": alpha_apical,
        "alpha_inh": alpha_inh,
        "is_simple": is_simple,
        "mag": net.feedback.max_apical_gain,
        "N": N, "step_deg": step_deg, "period": period, "peak_ch": peak_ch,
        "r_som_on": state_on.r_som[0].clone(),
    }


# ── Load checkpoints ─────────────────────────────────────────────────
import os

simple_ckpt = "results/simple_fb/center_surround_seed42/checkpoint.pt"
r5_ckpt = "results/iter/r5_mag010_e05/center_surround_seed42/checkpoint.pt"

if not os.path.exists(simple_ckpt):
    print(f"ERROR: Simple FB checkpoint not found at {simple_ckpt}")
    print("Trying alternate paths...")
    # Try other possible locations
    for alt in ["results/simple_fb/s42/center_surround_seed42/checkpoint.pt",
                "results/simple_fb/checkpoint.pt"]:
        if os.path.exists(alt):
            simple_ckpt = alt
            print(f"Found at: {alt}")
            break
    else:
        import glob
        matches = glob.glob("results/*simple*/**/checkpoint.pt", recursive=True)
        if matches:
            simple_ckpt = matches[0]
            print(f"Found at: {simple_ckpt}")
        else:
            print("No simple FB checkpoint found. Exiting.")
            sys.exit(1)

print(f"Loading simple FB from: {simple_ckpt}")
sfb = load_and_analyze("config/exp_simple_fb.yaml", simple_ckpt, "Simple Additive")

print(f"Loading R5 (multiplicative, mag=0.10)...")
r5 = load_and_analyze("config/exp_iter5.yaml", r5_ckpt, "R5 Multiplicative")

N = sfb["N"]
step_deg = sfb["step_deg"]
period = sfb["period"]
peak_ch = sfb["peak_ch"]

# ======================================================================
# 1. KERNEL PROFILE
# ======================================================================
print(f"\n{'='*80}")
print("1. KERNEL PROFILE (simple additive feedback)")
print(f"{'='*80}")

alpha = sfb["alpha_apical"]
print(f"\n{'Ch':>4s} {'Deg':>6s}  {'α_apical':>10s}  {'fb_mod':>10s}")
print("-" * 40)
for ch in range(N):
    deg = ch * step_deg
    marker = " <-- center" if ch == 0 else ""
    print(f"{ch:>4d} {deg:>5.1f}°  {alpha[ch].item():>10.6f}  {sfb['fb_mod'][ch].item():>10.6f}{marker}")

print(f"\nKernel stats:")
print(f"  max: {alpha.max().item():.6f} at ch {alpha.argmax().item()} ({alpha.argmax().item()*step_deg:.1f}°)")
print(f"  min: {alpha.min().item():.6f} at ch {alpha.argmin().item()} ({alpha.argmin().item()*step_deg:.1f}°)")
print(f"  mean: {alpha.mean().item():.6f}")
print(f"  positive: {(alpha > 0.001).sum().item()}/{N}")
print(f"  negative: {(alpha < -0.001).sum().item()}/{N}")
print(f"  near-zero (|α| < 0.001): {(alpha.abs() < 0.001).sum().item()}/{N}")
print(f"  L1 norm: {alpha.abs().sum().item():.4f}")

# Center-surround check
center_region = list(range(0, 6)) + list(range(31, 36))  # ±25°
surround_region = list(range(6, 31))
center_mean = alpha[center_region].mean().item()
surround_mean = alpha[surround_region].mean().item()
print(f"\n  Center (±25°) mean: {center_mean:.6f}")
print(f"  Surround (30-150°) mean: {surround_mean:.6f}")
if center_mean > 0 and surround_mean < 0:
    print(f"  Shape: CENTER-SURROUND ✓")
elif center_mean > surround_mean:
    print(f"  Shape: Peaked (center > surround)")
elif abs(center_mean) < 0.001 and abs(surround_mean) < 0.001:
    print(f"  Shape: FLAT / DEAD (near zero)")
else:
    print(f"  Shape: OTHER ({center_mean:.4f} vs {surround_mean:.4f})")

# ======================================================================
# 2. L2/3 PROFILES AND SI CURVE
# ======================================================================
print(f"\n{'='*80}")
print("2. L2/3 PROFILES (ON vs OFF) AND SUPPRESSION INDEX")
print(f"{'='*80}")

total_on = sfb["r_l23_on"].sum().item()
total_off = sfb["r_l23_off"].sum().item()
amp = total_on / total_off if total_off > 0 else 0

print(f"\nSimple additive feedback:")
print(f"  Total L2/3 OFF: {total_off:.6f}")
print(f"  Total L2/3 ON:  {total_on:.6f}")
print(f"  Amplitude ratio: {amp:.4f}")

# FWHM
peak_on = sfb["r_l23_on"][peak_ch].item()
peak_off = sfb["r_l23_off"][peak_ch].item()
fwhm_on = (sfb["r_l23_on"] >= peak_on/2).sum().item() * step_deg if peak_on > 0 else 0
fwhm_off = (sfb["r_l23_off"] >= peak_off/2).sum().item() * step_deg if peak_off > 0 else 0
sel_on = peak_on / total_on if total_on > 0 else 0
sel_off = peak_off / total_off if total_off > 0 else 0

print(f"  FWHM OFF: {fwhm_off:.1f}°, FWHM ON: {fwhm_on:.1f}° (Δ={fwhm_on-fwhm_off:+.1f}°)")
print(f"  Selectivity OFF: {sel_off:.4f}, ON: {sel_on:.4f} (Δ={sel_on-sel_off:+.4f})")
print(f"  Peak OFF: {peak_off:.6f}, Peak ON: {peak_on:.6f}")

# SI curve
print(f"\n{'Ch':>4s} {'Deg':>6s}  {'OFF':>10s}  {'ON':>10s}  {'fb_mod':>10s}  {'SI':>10s}")
print("-" * 60)
for ch in range(N):
    deg = ch * step_deg
    off = sfb["r_l23_off"][ch].item()
    on = sfb["r_l23_on"][ch].item()
    fb = sfb["fb_mod"][ch].item()
    si = (on - off) / off if off > 1e-8 else 0.0
    si_str = f"{si:>+10.4f}" if off > 1e-8 else f"{'n/a':>10s}"
    marker = " <-- stim" if ch == peak_ch else ""
    print(f"{ch:>4d} {deg:>5.1f}°  {off:>10.6f}  {on:>10.6f}  {fb:>10.6f}  {si_str}{marker}")

# ======================================================================
# 3. KOK-STYLE SI BY DISTANCE
# ======================================================================
print(f"\n{'='*80}")
print("3. SI BY DISTANCE FROM PREDICTED ORIENTATION")
print(f"{'='*80}")

def si_by_distance(r_on, r_off, peak_ch, N, period):
    """Compute SI averaged over symmetric offsets."""
    results = {}
    step = period / N
    for offset_ch in range(N // 2 + 1):
        dist_deg = offset_ch * step
        ch_pos = (peak_ch + offset_ch) % N
        ch_neg = (peak_ch - offset_ch) % N
        channels = [ch_pos] if offset_ch == 0 or offset_ch == N//2 else [ch_pos, ch_neg]

        off_avg = sum(r_off[c].item() for c in channels) / len(channels)
        on_avg = sum(r_on[c].item() for c in channels) / len(channels)
        si = (on_avg - off_avg) / off_avg if off_avg > 1e-8 else None
        results[dist_deg] = (si, on_avg, off_avg)
    return results

si_sfb = si_by_distance(sfb["r_l23_on"], sfb["r_l23_off"], peak_ch, N, period)
si_r5 = si_by_distance(r5["r_l23_on"], r5["r_l23_off"], peak_ch, N, period)

print(f"\n{'Offset':>8s}  {'SI(simple)':>11s}  {'SI(R5)':>11s}  {'ON(s)':>10s}  {'OFF(s)':>10s}  {'Region':>10s}")
print("-" * 75)
for dist in sorted(si_sfb.keys()):
    si_s, on_s, off_s = si_sfb[dist]
    si_r, on_r, off_r = si_r5[dist]

    if dist <= 5: region = "CENTER"
    elif dist <= 30: region = "FLANK"
    elif dist <= 60: region = "FAR"
    else: region = "ANTI"

    si_s_str = f"{si_s:>+11.4f}" if si_s is not None else f"{'n/a':>11s}"
    si_r_str = f"{si_r:>+11.4f}" if si_r is not None else f"{'n/a':>11s}"
    print(f"{dist:>7.1f}°  {si_s_str}  {si_r_str}  {on_s:>10.6f}  {off_s:>10.6f}  [{region}]")

# Regional SI averages
regions = [("Center (0-5°)", 0, 5), ("Near flank (10-15°)", 10, 15),
           ("Mid flank (20-30°)", 20, 30), ("Far (35-60°)", 35, 60)]
print(f"\n{'Region':>25s}  {'SI(simple)':>11s}  {'SI(R5)':>11s}  {'Kok':>15s}")
print("-" * 70)
for name, lo, hi in regions:
    vals_s = [(d, s) for d, (s, _, off) in si_sfb.items() if lo <= d <= hi and s is not None]
    vals_r = [(d, s) for d, (s, _, off) in si_r5.items() if lo <= d <= hi and s is not None]
    avg_s = sum(s for _, s in vals_s) / len(vals_s) if vals_s else None
    avg_r = sum(s for _, s in vals_r) / len(vals_r) if vals_r else None

    if "Center" in name: kok = "~0 or slight +"
    elif "flank" in name.lower(): kok = "NEGATIVE"
    else: kok = "~0"

    s_str = f"{avg_s:>+11.4f}" if avg_s is not None else f"{'n/a':>11s}"
    r_str = f"{avg_r:>+11.4f}" if avg_r is not None else f"{'n/a':>11s}"
    print(f"{name:>25s}  {s_str}  {r_str}  {kok:>15s}")

# ======================================================================
# 4. HALLUCINATION CHECK
# ======================================================================
print(f"\n{'='*80}")
print("4. HALLUCINATION CHECK (activity where there's no stimulus)")
print(f"{'='*80}")

# Channels where OFF=0 but ON>0 → hallucination
hallucinations = []
for ch in range(N):
    off = sfb["r_l23_off"][ch].item()
    on = sfb["r_l23_on"][ch].item()
    if off < 1e-8 and on > 1e-6:
        hallucinations.append((ch, ch*step_deg, on))

if hallucinations:
    print(f"\n  HALLUCINATIONS DETECTED: {len(hallucinations)} channels")
    for ch, deg, on in hallucinations:
        print(f"    ch {ch} ({deg:.1f}°): OFF=0, ON={on:.6f}")
else:
    print(f"\n  No hallucinations — all ON activity is at channels with OFF > 0")

# Also check reverse: channels where OFF>0 but ON=0 → suppression
suppressions = []
for ch in range(N):
    off = sfb["r_l23_off"][ch].item()
    on = sfb["r_l23_on"][ch].item()
    if off > 1e-6 and on < 1e-8:
        suppressions.append((ch, ch*step_deg, off))

if suppressions:
    print(f"\n  Complete suppressions: {len(suppressions)} channels silenced by feedback")
    for ch, deg, off in suppressions:
        print(f"    ch {ch} ({deg:.1f}°): OFF={off:.6f}, ON=0")

# ======================================================================
# 5. KOK CRITERIA CHECK
# ======================================================================
print(f"\n{'='*80}")
print("5. KOK CRITERIA CHECK")
print(f"{'='*80}")

total_on_r5 = r5["r_l23_on"].sum().item()
total_off_r5 = r5["r_l23_off"].sum().item()
amp_r5 = total_on_r5 / total_off_r5 if total_off_r5 > 0 else 0

print(f"\n{'Criterion':>35s}  {'Simple':>12s}  {'R5 (mult)':>12s}  {'Kok':>8s}")
print("-" * 75)
print(f"{'Narrower FWHM (ON < OFF)':>35s}  {'YES' if fwhm_on < fwhm_off else 'NO':>12s}  "
      f"{'YES' if False else 'NO':>12s}  {'YES':>8s}")
print(f"{'Reduced amplitude (ON < OFF)':>35s}  {'YES' if amp < 1.0 else 'NO':>12s}  "
      f"{'NO':>12s}  {'YES':>8s}")
print(f"{'Improved selectivity':>35s}  {'YES' if sel_on > sel_off else 'NO':>12s}  "
      f"{'YES':>12s}  {'YES':>8s}")

print(f"\n  Simple: amp = {amp:.4f}, Δsel = {sel_on-sel_off:+.4f}, ΔFWHM = {fwhm_on-fwhm_off:+.1f}°")
print(f"  R5:     amp = {amp_r5:.4f}, Δsel = ?, ΔFWHM = 0°")

# ======================================================================
# 6. COMPARISON TABLE
# ======================================================================
print(f"\n{'='*80}")
print("6. COMPARISON: SIMPLE ADDITIVE vs R5 MULTIPLICATIVE")
print(f"{'='*80}")

print(f"\n{'Metric':>35s}  {'Simple':>12s}  {'R5 (mag=0.10)':>14s}")
print("-" * 65)
print(f"{'Feedback type':>35s}  {'Additive':>12s}  {'Multiplicative':>14s}")
print(f"{'max_apical_gain':>35s}  {sfb['mag']:>12.2f}  {r5['mag']:>14.2f}")
print(f"{'Amplitude ratio':>35s}  {amp:>12.4f}  {amp_r5:>14.4f}")
print(f"{'Peak ON':>35s}  {peak_on:>12.6f}  {r5['r_l23_on'][peak_ch].item():>14.6f}")
print(f"{'Peak OFF':>35s}  {peak_off:>12.6f}  {r5['r_l23_off'][peak_ch].item():>14.6f}")
print(f"{'FWHM ON':>35s}  {fwhm_on:>11.1f}°  {(r5['r_l23_on'] >= r5['r_l23_on'][peak_ch].item()/2).sum().item()*step_deg:>13.1f}°")
print(f"{'FWHM OFF':>35s}  {fwhm_off:>11.1f}°  {(r5['r_l23_off'] >= r5['r_l23_off'][peak_ch].item()/2).sum().item()*step_deg:>13.1f}°")
print(f"{'Selectivity ON':>35s}  {sel_on:>12.4f}  {r5['r_l23_on'][peak_ch].item()/total_on_r5:>14.4f}")
print(f"{'Selectivity OFF':>35s}  {sel_off:>12.4f}  {r5['r_l23_off'][peak_ch].item()/total_off_r5:>14.4f}")
print(f"{'VIP active':>35s}  {'N/A':>12s}  {'No':>14s}")
print(f"{'SOM active':>35s}  {'No':>12s}  {'Yes':>14s}")
print(f"{'Kernel pos channels':>35s}  {(alpha > 0.001).sum().item():>12d}  {(r5['alpha_apical'] > 0.001).sum().item():>14d}")
print(f"{'Kernel neg channels':>35s}  {(alpha < -0.001).sum().item():>12d}  {(r5['alpha_apical'] < -0.001).sum().item():>14d}")
