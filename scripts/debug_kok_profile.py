"""Debugger: Kok-style sharpening analysis for R5 (mag=0.10) vs R1 (mag=0.30).

Kok et al. (2012) found for expected stimuli:
- Narrower orientation tuning (sharper population response)
- Reduced overall BOLD (lower amplitude)
- Improved orientation decoding despite lower amplitude

Measures:
1. Full L2/3 population response profile (ON vs OFF)
2. Suppression index SI = (ON - OFF) / OFF at each channel offset
3. Comparison R5 vs R1
"""

import sys
sys.path.insert(0, "/mnt/c/Users/User/codingproj/freshstart")

import torch
import torch.nn.functional as F
from src.config import load_config
from src.model.network import LaminarV1V2Network
from src.state import initial_state
from src.stimulus.gratings import generate_grating


def load_and_run(config_path, ckpt_path, theta_stim=90.0, contrast=0.5):
    """Load a checkpoint, run ON and OFF, return profiles and metadata."""
    model_cfg, train_cfg, stim_cfg = load_config(config_path)
    N = model_cfg.n_orientations
    period = model_cfg.orientation_range
    sigma_ff = model_cfg.sigma_ff
    step_deg = period / N
    steps_on = train_cfg.steps_on
    oracle_sigma = getattr(train_cfg, 'oracle_sigma', 12.0)
    oracle_pi = train_cfg.oracle_pi

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
        net.feedback.cache_kernels()
        for t in range(steps_on):
            state_on, aux_on = net.step(stim, cue, task, state_on)
        net.l23.uncache_kernels()
        net.feedback.uncache_kernels()
    r_l23_on = state_on.r_l23[0].clone()

    # --- FB OFF ---
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

    # Get gain profile
    with torch.no_grad():
        _, _, apical_gain = net.feedback(q_pred, torch.full((1,1), oracle_pi), r_l4=None)
    gain = apical_gain[0].clone()

    # Kernel profiles
    alpha_inh = net.feedback.alpha_inh.data.clone()
    alpha_apical = net.feedback.alpha_apical.data.clone()
    alpha_vip = net.feedback.alpha_vip.data.clone()

    # SOM state
    r_som_on = state_on.r_som[0].clone()

    return {
        "r_l23_on": r_l23_on,
        "r_l23_off": r_l23_off,
        "gain": gain,
        "alpha_inh": alpha_inh,
        "alpha_apical": alpha_apical,
        "alpha_vip": alpha_vip,
        "r_som_on": r_som_on,
        "mag": net.feedback.max_apical_gain,
        "N": N,
        "step_deg": step_deg,
        "period": period,
        "peak_ch": int(theta_stim / step_deg),
    }


# ── Load both checkpoints ───────────────────────────────────────────
print("Loading R5 (mag=0.10)...")
r5 = load_and_run("config/exp_iter5.yaml",
                   "results/iter/r5_mag010_e05/center_surround_seed42/checkpoint.pt")

print("Loading R1 (mag=0.30)...")
r1 = load_and_run("config/exp_iter1.yaml",
                   "results/iter/r1_mag03_e05/center_surround_seed42/checkpoint.pt")

N = r5["N"]
step_deg = r5["step_deg"]
period = r5["period"]
peak_ch = r5["peak_ch"]

# ======================================================================
# 1. POPULATION RESPONSE PROFILES
# ======================================================================
print(f"\n{'='*80}")
print("1. POPULATION RESPONSE PROFILES (L2/3 at 90° stimulus)")
print(f"{'='*80}")

print(f"\n--- R5 (mag={r5['mag']}) ---")
print(f"{'Ch':>4s} {'Deg':>6s}  {'OFF':>10s}  {'ON':>10s}  {'gain':>8s}")
print("-" * 50)
for ch in range(N):
    deg = ch * step_deg
    off = r5["r_l23_off"][ch].item()
    on = r5["r_l23_on"][ch].item()
    g = r5["gain"][ch].item()
    marker = " <-- stim" if ch == peak_ch else ""
    print(f"{ch:>4d} {deg:>5.1f}°  {off:>10.6f}  {on:>10.6f}  {g:>8.4f}{marker}")

total_off_r5 = r5["r_l23_off"].sum().item()
total_on_r5 = r5["r_l23_on"].sum().item()
amp_r5 = total_on_r5 / total_off_r5 if total_off_r5 > 0 else 0
print(f"\nTotal OFF: {total_off_r5:.6f}")
print(f"Total ON:  {total_on_r5:.6f}")
print(f"Amplitude ratio: {amp_r5:.4f}")
print(f"Mean gain: {r5['gain'].mean().item():.6f}")
print(f"Gain at stim: {r5['gain'][peak_ch].item():.6f}")
print(f"Gain min: {r5['gain'].min().item():.6f}")
print(f"Channels < 1.0: {(r5['gain'] < 1.0).sum().item()}/{N}")

# ======================================================================
# 2. SUPPRESSION INDEX BY DISTANCE
# ======================================================================
print(f"\n{'='*80}")
print("2. SUPPRESSION INDEX: SI = (ON - OFF) / OFF at each offset from predicted")
print(f"{'='*80}")


def compute_si_curve(r_on, r_off, peak_ch, N, period):
    """Compute SI as a function of distance from peak."""
    offsets = []  # in degrees
    si_vals = []
    on_vals = []
    off_vals = []

    for offset_ch in range(N):
        # Circular distance from peak
        dist_ch = min(offset_ch, N - offset_ch)
        dist_deg = dist_ch * (period / N)

        # Average the two symmetric channels (if not 0 or N/2)
        ch_pos = (peak_ch + offset_ch) % N
        ch_neg = (peak_ch - offset_ch) % N

        if offset_ch == 0 or offset_ch == N // 2:
            channels = [ch_pos]
        else:
            channels = [ch_pos, ch_neg]

        off_avg = sum(r_off[c].item() for c in channels) / len(channels)
        on_avg = sum(r_on[c].item() for c in channels) / len(channels)

        if off_avg > 1e-8:
            si = (on_avg - off_avg) / off_avg
        else:
            si = 0.0  # Both are essentially zero

        offsets.append(dist_deg)
        si_vals.append(si)
        on_vals.append(on_avg)
        off_vals.append(off_avg)

    # Deduplicate (since we iterate 0..N-1 but distances repeat)
    seen = {}
    for d, s, on, off in zip(offsets, si_vals, on_vals, off_vals):
        if d not in seen:
            seen[d] = (s, on, off)
    return seen


si_r5 = compute_si_curve(r5["r_l23_on"], r5["r_l23_off"], peak_ch, N, period)
si_r1 = compute_si_curve(r1["r_l23_on"], r1["r_l23_off"], peak_ch, N, period)

print(f"\n{'Offset':>8s}  {'SI (R5)':>10s}  {'ON (R5)':>10s}  {'OFF (R5)':>10s}  "
      f"{'SI (R1)':>10s}  {'ON (R1)':>10s}  {'OFF (R1)':>10s}")
print("-" * 80)

for dist_deg in sorted(si_r5.keys()):
    si5, on5, off5 = si_r5[dist_deg]
    si1, on1, off1 = si_r1[dist_deg]

    # Label Kok regions
    if dist_deg <= 5:
        region = "CENTER"
    elif dist_deg <= 30:
        region = "FLANK"
    elif dist_deg <= 60:
        region = "FAR"
    else:
        region = "ANTI"

    si5_str = f"{si5:>+10.4f}" if off5 > 1e-8 else f"{'n/a':>10s}"
    si1_str = f"{si1:>+10.4f}" if off1 > 1e-8 else f"{'n/a':>10s}"

    print(f"{dist_deg:>7.1f}°  {si5_str}  {on5:>10.6f}  {off5:>10.6f}  "
          f"{si1_str}  {on1:>10.6f}  {off1:>10.6f}  [{region}]")

# ======================================================================
# 3. KOK-STYLE SUMMARY
# ======================================================================
print(f"\n{'='*80}")
print("3. KOK-STYLE SUMMARY")
print(f"{'='*80}")

# Kok profile: center enhancement, surround suppression, far neutral
# Compute regional averages
def regional_si(si_dict, region_bounds):
    """Average SI within distance bounds [min_deg, max_deg]."""
    vals = [(d, s, on, off) for d, (s, on, off) in si_dict.items()
            if region_bounds[0] <= d <= region_bounds[1] and off > 1e-8]
    if not vals:
        return None, None, None
    avg_si = sum(s for _, s, _, _ in vals) / len(vals)
    avg_on = sum(on for _, _, on, _ in vals) / len(vals)
    avg_off = sum(off for _, _, _, off in vals) / len(vals)
    return avg_si, avg_on, avg_off


regions = [
    ("Center (0-5°)", (0, 5)),
    ("Near flank (10-15°)", (10, 15)),
    ("Mid flank (20-30°)", (20, 30)),
    ("Far (35-60°)", (35, 60)),
    ("Anti-pref (65-90°)", (65, 90)),
]

print(f"\n{'Region':>25s}  {'SI (R5)':>10s}  {'SI (R1)':>10s}  {'Kok prediction':>20s}")
print("-" * 75)
for name, bounds in regions:
    si5, _, _ = regional_si(si_r5, bounds)
    si1, _, _ = regional_si(si_r1, bounds)

    if "Center" in name:
        kok = "~0 or slight +"
    elif "Near" in name or "Mid" in name:
        kok = "NEGATIVE (suppressed)"
    else:
        kok = "~0 (neutral)"

    si5_str = f"{si5:>+10.4f}" if si5 is not None else f"{'n/a':>10s}"
    si1_str = f"{si1:>+10.4f}" if si1 is not None else f"{'n/a':>10s}"
    print(f"{name:>25s}  {si5_str}  {si1_str}  {kok:>20s}")

# ======================================================================
# 4. AMPLITUDE AND SELECTIVITY COMPARISON
# ======================================================================
print(f"\n{'='*80}")
print("4. R5 vs R1 COMPARISON")
print(f"{'='*80}")

total_off_r1 = r1["r_l23_off"].sum().item()
total_on_r1 = r1["r_l23_on"].sum().item()
amp_r1 = total_on_r1 / total_off_r1 if total_off_r1 > 0 else 0

peak_off_r5 = r5["r_l23_off"][peak_ch].item()
peak_on_r5 = r5["r_l23_on"][peak_ch].item()
peak_off_r1 = r1["r_l23_off"][peak_ch].item()
peak_on_r1 = r1["r_l23_on"][peak_ch].item()

sel_off_r5 = peak_off_r5 / total_off_r5 if total_off_r5 > 0 else 0
sel_on_r5 = peak_on_r5 / total_on_r5 if total_on_r5 > 0 else 0
sel_off_r1 = peak_off_r1 / total_off_r1 if total_off_r1 > 0 else 0
sel_on_r1 = peak_on_r1 / total_on_r1 if total_on_r1 > 0 else 0

# FWHM
def compute_fwhm(r_l23, peak_ch):
    peak = r_l23[peak_ch].item()
    if peak <= 0:
        return 0.0
    hm = peak / 2
    return (r_l23 >= hm).sum().item() * step_deg

fwhm_off_r5 = compute_fwhm(r5["r_l23_off"], peak_ch)
fwhm_on_r5 = compute_fwhm(r5["r_l23_on"], peak_ch)
fwhm_off_r1 = compute_fwhm(r1["r_l23_off"], peak_ch)
fwhm_on_r1 = compute_fwhm(r1["r_l23_on"], peak_ch)

print(f"\n{'Metric':>35s}  {'R5 (mag=0.10)':>14s}  {'R1 (mag=0.30)':>14s}")
print("-" * 70)
print(f"{'max_apical_gain':>35s}  {r5['mag']:>14.2f}  {r1['mag']:>14.2f}")
print(f"{'Total L2/3 OFF':>35s}  {total_off_r5:>14.4f}  {total_off_r1:>14.4f}")
print(f"{'Total L2/3 ON':>35s}  {total_on_r5:>14.4f}  {total_on_r1:>14.4f}")
print(f"{'Amplitude ratio (ON/OFF)':>35s}  {amp_r5:>14.4f}  {amp_r1:>14.4f}")
print(f"{'Peak ON':>35s}  {peak_on_r5:>14.6f}  {peak_on_r1:>14.6f}")
print(f"{'Peak OFF':>35s}  {peak_off_r5:>14.6f}  {peak_off_r1:>14.6f}")
print(f"{'Selectivity OFF (peak/total)':>35s}  {sel_off_r5:>14.4f}  {sel_off_r1:>14.4f}")
print(f"{'Selectivity ON (peak/total)':>35s}  {sel_on_r5:>14.4f}  {sel_on_r1:>14.4f}")
print(f"{'Δselectivity (ON-OFF)':>35s}  {sel_on_r5 - sel_off_r5:>+14.4f}  {sel_on_r1 - sel_off_r1:>+14.4f}")
print(f"{'FWHM OFF':>35s}  {fwhm_off_r5:>13.1f}°  {fwhm_off_r1:>13.1f}°")
print(f"{'FWHM ON':>35s}  {fwhm_on_r5:>13.1f}°  {fwhm_on_r1:>13.1f}°")
print(f"{'ΔFWHM (ON-OFF)':>35s}  {fwhm_on_r5 - fwhm_off_r5:>+13.1f}°  {fwhm_on_r1 - fwhm_off_r1:>+13.1f}°")
print(f"{'Gain at stim':>35s}  {r5['gain'][peak_ch].item():>14.4f}  {r1['gain'][peak_ch].item():>14.4f}")
print(f"{'Mean gain':>35s}  {r5['gain'].mean().item():>14.4f}  {r1['gain'].mean().item():>14.4f}")
print(f"{'Gain min':>35s}  {r5['gain'].min().item():>14.4f}  {r1['gain'].min().item():>14.4f}")
print(f"{'Channels gain < 1.0':>35s}  {(r5['gain']<1.0).sum().item():>14d}  {(r1['gain']<1.0).sum().item():>14d}")

# Kok criteria check
print(f"\n{'='*80}")
print("KOK CRITERIA CHECK (R5, mag=0.10)")
print(f"{'='*80}")

kok_narrower = fwhm_on_r5 < fwhm_off_r5
kok_lower_amp = total_on_r5 < total_off_r5
kok_better_sel = sel_on_r5 > sel_off_r5

print(f"\n  1. Narrower tuning (FWHM ON < OFF)?  {'YES' if kok_narrower else 'NO'} "
      f"({fwhm_on_r5:.1f}° vs {fwhm_off_r5:.1f}°)")
print(f"  2. Reduced amplitude (ON < OFF)?      {'YES' if kok_lower_amp else 'NO'} "
      f"({total_on_r5:.4f} vs {total_off_r5:.4f}, ratio={amp_r5:.4f})")
print(f"  3. Improved selectivity?              {'YES' if kok_better_sel else 'NO'} "
      f"({sel_on_r5:.4f} vs {sel_off_r5:.4f}, Δ={sel_on_r5-sel_off_r5:+.4f})")

# ======================================================================
# 5. KERNEL SHAPE COMPARISON
# ======================================================================
print(f"\n{'='*80}")
print("5. KERNEL PROFILES COMPARISON")
print(f"{'='*80}")

print(f"\n{'Ch':>4s} {'Deg':>6s}  {'α_ap(R5)':>10s}  {'α_ap(R1)':>10s}  {'α_inh(R5)':>10s}  {'α_inh(R1)':>10s}")
print("-" * 55)
for ch in range(N):
    deg = ch * step_deg
    marker = " <-- center" if ch == 0 else ""
    print(f"{ch:>4d} {deg:>5.1f}°  {r5['alpha_apical'][ch].item():>10.6f}  "
          f"{r1['alpha_apical'][ch].item():>10.6f}  "
          f"{r5['alpha_inh'][ch].item():>10.6f}  "
          f"{r1['alpha_inh'][ch].item():>10.6f}{marker}")

print(f"\nApical kernel comparison:")
print(f"  R5: pos={r5['alpha_apical'][r5['alpha_apical']>0].sum().item():.4f}, "
      f"neg={r5['alpha_apical'][r5['alpha_apical']<0].sum().item():.4f}, "
      f"n_neg={(r5['alpha_apical']<0).sum().item()}")
print(f"  R1: pos={r1['alpha_apical'][r1['alpha_apical']>0].sum().item():.4f}, "
      f"neg={r1['alpha_apical'][r1['alpha_apical']<0].sum().item():.4f}, "
      f"n_neg={(r1['alpha_apical']<0).sum().item()}")

print(f"\nSOM kernel comparison:")
print(f"  R5: pos={r5['alpha_inh'][r5['alpha_inh']>0].sum().item():.4f}, "
      f"neg={r5['alpha_inh'][r5['alpha_inh']<0].sum().item():.4f}, "
      f"n_neg={(r5['alpha_inh']<0).sum().item()}")
print(f"  R1: pos={r1['alpha_inh'][r1['alpha_inh']>0].sum().item():.4f}, "
      f"neg={r1['alpha_inh'][r1['alpha_inh']<0].sum().item():.4f}, "
      f"n_neg={(r1['alpha_inh']<0).sum().item()}")

print(f"\nVIP active?")
print(f"  R5: max |α_vip| = {r5['alpha_vip'].abs().max().item():.6f}")
print(f"  R1: max |α_vip| = {r1['alpha_vip'].abs().max().item():.6f}")

# ======================================================================
# 6. ASCII VISUALIZATION OF SI CURVE
# ======================================================================
print(f"\n{'='*80}")
print("6. SI CURVE VISUALIZATION (R5, mag=0.10)")
print(f"{'='*80}")
print("\nSI = (ON - OFF) / OFF")
print("Kok prediction: center ~0, flanks negative, far neutral")
print()

# Scale: -1.0 to +1.0 → 40-char bar
bar_width = 40
for dist_deg in sorted(si_r5.keys()):
    si, on, off = si_r5[dist_deg]
    if off < 1e-8:
        bar = " " * bar_width + "| (zero activity)"
    else:
        # Clamp SI to [-1, 1] for display
        si_clamped = max(-1.0, min(1.0, si))
        # Map to position: -1 → 0, 0 → 20, +1 → 40
        pos = int((si_clamped + 1.0) * bar_width / 2)
        bar = " " * 20 + "|"  # center line at position 20
        if si_clamped >= 0:
            bar = " " * 20 + "|" + "█" * (pos - 20) + " " * (bar_width - pos)
        else:
            bar = " " * pos + "█" * (20 - pos) + "|" + " " * (bar_width - 20)

    print(f"  {dist_deg:>5.1f}°  {bar}  SI={si:+.4f}" if off > 1e-8 else
          f"  {dist_deg:>5.1f}°  {'':>{bar_width+1}s}  (zero)")
