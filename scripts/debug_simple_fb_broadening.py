"""Debugger: Investigate why simple additive feedback broadens FWHM.

5 questions:
1. Where does additive feedback add activity relative to FF peak?
2. Why is the positive center ±20° (not narrower)?
3. Compare kernel width to FF tuning width
4. Can additive feedback produce narrowing at all?
5. Manual narrowing test: zero out channels 3-4 (±15-20°)
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


def compute_fwhm(profile, step_deg=5.0):
    """Compute FWHM of a 1D profile in degrees."""
    peak = profile.max().item()
    half_max = peak / 2.0
    above = (profile >= half_max).float()
    return above.sum().item() * step_deg


def load_model(config_path, ckpt_path):
    model_cfg, train_cfg, stim_cfg = load_config(config_path)
    net = LaminarV1V2Network(model_cfg, delta_som=train_cfg.delta_som)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net.load_state_dict(ckpt["model_state"], strict=False)
    net.eval()
    return net, model_cfg, train_cfg


def run_model(net, model_cfg, train_cfg, theta_stim=90.0, contrast=0.5, fb_on=True):
    """Run model and return final L2/3 rates."""
    N = model_cfg.n_orientations
    sigma_ff = model_cfg.sigma_ff
    period = model_cfg.orientation_range
    steps_on = train_cfg.steps_on
    oracle_pi = train_cfg.oracle_pi

    stim = generate_grating(torch.tensor([theta_stim]), torch.tensor([contrast]),
                            N, sigma_ff, period=period)
    cue = torch.zeros(1, 2)
    task = torch.zeros(1, 2)
    q_pred = net._make_bump(torch.tensor([theta_stim]), sigma=12.0)

    net.oracle_mode = True
    net.oracle_q_pred = q_pred
    net.oracle_pi_pred = torch.full((1, 1), oracle_pi)
    net.feedback_scale.fill_(1.0 if fb_on else 0.0)

    state = initial_state(1, N, model_cfg.v2_hidden_dim)
    with torch.no_grad():
        net.l23.cache_kernels()
        if hasattr(net.feedback, 'cache_kernels'):
            net.feedback.cache_kernels()
        for t in range(steps_on):
            state, _ = net.step(stim, cue, task, state)
        net.l23.uncache_kernels()
        if hasattr(net.feedback, 'uncache_kernels'):
            net.feedback.uncache_kernels()

    net.feedback_scale.fill_(1.0)
    return state


def main():
    config_path = "config/exp_simple_fb.yaml"
    ckpt_path = "results/iter/simple_fb/center_surround_seed42/checkpoint.pt"

    print("=" * 80)
    print("INVESTIGATION: Simple Additive Feedback Broadening")
    print("=" * 80)

    net, model_cfg, train_cfg = load_model(config_path, ckpt_path)
    N = model_cfg.n_orientations  # 36
    period = model_cfg.orientation_range  # 180
    step_deg = period / N  # 5.0

    # =====================================================================
    # Q0: Extract and display the learned kernel
    # =====================================================================
    print("\n[Q0] Learned alpha_apical kernel")
    alpha = net.feedback.alpha_apical.detach().clone()
    print(f"  Shape: {alpha.shape}")
    print(f"  Sum: {alpha.sum().item():.4f}")
    print(f"  Min: {alpha.min().item():.4f}, Max: {alpha.max().item():.4f}")
    print(f"  Channel-by-channel (index → offset_deg → value):")
    for i in range(N):
        # Kernel index i corresponds to offset i*step_deg from center
        # But circulant wraps, so index > N/2 means negative offset
        offset = i * step_deg
        if offset > period / 2:
            offset -= period
        print(f"    ch={i:2d}  offset={offset:+7.1f}°  alpha={alpha[i].item():+.6f}")

    # =====================================================================
    # Q1: Where does additive feedback add activity relative to FF peak?
    # =====================================================================
    print("\n" + "=" * 80)
    print("[Q1] L2/3 profile ON vs OFF — where feedback adds activity")
    print("=" * 80)

    theta_stim = 90.0
    state_on = run_model(net, model_cfg, train_cfg, theta_stim, fb_on=True)
    state_off = run_model(net, model_cfg, train_cfg, theta_stim, fb_on=False)

    r_on = state_on.r_l23[0].detach()
    r_off = state_off.r_l23[0].detach()
    r_l4 = state_off.r_l4[0].detach()  # L4 is same with/without feedback

    # Also get the raw feedback modulation signal
    q_pred = net._make_bump(torch.tensor([theta_stim]), sigma=12.0)
    with torch.no_grad():
        fb_mod = net.feedback.compute_simple_feedback(q_pred)[0]  # [N]
    fb_scaled = fb_mod * net.feedback_scale.item()

    # Also get pre-rectification drives
    # Run one more step capturing the drive
    stim = generate_grating(torch.tensor([theta_stim]), torch.tensor([0.5]),
                            N, model_cfg.sigma_ff, period=period)
    net.oracle_mode = True
    net.oracle_q_pred = q_pred
    net.oracle_pi_pred = torch.full((1, 1), train_cfg.oracle_pi)

    # Get FF drive (L4 → L2/3 weights)
    ff_drive = F.linear(r_l4.unsqueeze(0), net.l23.W_l4_to_l23)[0]

    # Get recurrent drive (from converged OFF state)
    W_rec = net.l23.W_rec
    rec_drive_off = F.linear(r_off.unsqueeze(0), W_rec)[0]
    rec_drive_on = F.linear(r_on.unsqueeze(0), W_rec)[0]

    # Total excitatory drive
    exc_off = ff_drive + rec_drive_off  # no template_modulation when fb=off means center_exc=0
    exc_on = ff_drive + rec_drive_on + fb_scaled  # center_exc = fb_mod * fb_scale

    stim_ch = int(theta_stim / step_deg)  # channel 18

    print(f"\n  Stimulus: {theta_stim}° (channel {stim_ch})")
    print(f"  feedback_scale: {net.feedback_scale.item():.4f}")
    print(f"\n  Channel-by-channel (centered on stimulus):")
    print(f"  {'dist':>6s} {'r_OFF':>8s} {'r_ON':>8s} {'delta':>8s} {'ratio':>8s} {'r_L4':>8s} {'ff_drv':>8s} {'fb_mod':>8s} {'fb_scl':>8s}")
    for i in range(N):
        dist = (i - stim_ch) * step_deg
        if dist > period / 2:
            dist -= period
        if dist < -period / 2:
            dist += period
        if abs(dist) <= 45:
            r_off_i = r_off[i].item()
            r_on_i = r_on[i].item()
            delta = r_on_i - r_off_i
            ratio = r_on_i / r_off_i if r_off_i > 1e-6 else float('nan')
            print(f"  {dist:+6.1f}° {r_off_i:8.4f} {r_on_i:8.4f} {delta:+8.4f} {ratio:8.4f} "
                  f"{r_l4[i].item():8.4f} {ff_drive[i].item():8.4f} {fb_mod[i].item():+8.4f} {fb_scaled[i].item():+8.4f}")

    fwhm_on = compute_fwhm(r_on, step_deg)
    fwhm_off = compute_fwhm(r_off, step_deg)
    print(f"\n  FWHM OFF: {fwhm_off:.2f}°")
    print(f"  FWHM ON:  {fwhm_on:.2f}°")
    print(f"  Delta FWHM: {fwhm_on - fwhm_off:+.2f}°")

    # Peak analysis
    peak_off = r_off.max().item()
    peak_on = r_on.max().item()
    print(f"\n  Peak OFF: {peak_off:.4f}")
    print(f"  Peak ON:  {peak_on:.4f}")
    print(f"  Peak ratio: {peak_on/peak_off:.4f}")

    # Global amplitude
    mean_on = r_on.mean().item()
    mean_off = r_off.mean().item()
    print(f"\n  Mean activity OFF: {mean_off:.6f}")
    print(f"  Mean activity ON:  {mean_on:.6f}")
    print(f"  Global amp ratio:  {mean_on/mean_off:.4f}")

    # =====================================================================
    # Q2 & Q3: Compare kernel width to FF tuning width
    # =====================================================================
    print("\n" + "=" * 80)
    print("[Q2/Q3] Kernel width vs FF tuning width")
    print("=" * 80)

    # FF tuning = L4 response to a single grating
    # L4 is a von Mises tuning curve with sigma_ff
    print(f"\n  FF tuning (L4 response at stim={theta_stim}°):")
    fwhm_l4 = compute_fwhm(r_l4, step_deg)
    print(f"  L4 FWHM: {fwhm_l4:.2f}°")

    # FF drive after W_l4_to_l23
    fwhm_ff_drive = compute_fwhm(F.relu(ff_drive), step_deg)
    print(f"  FF drive (post W_l4_l23) FWHM: {fwhm_ff_drive:.2f}°")

    # L2/3 OFF FWHM (the actual tuning width that feedback modifies)
    print(f"  L2/3 OFF FWHM: {fwhm_off:.2f}°")

    # Feedback modulation kernel FWHM (positive part only)
    fb_pos = F.relu(fb_mod)
    if fb_pos.max() > 0:
        fwhm_fb_pos = compute_fwhm(fb_pos, step_deg)
        print(f"\n  Feedback modulation (positive part) FWHM: {fwhm_fb_pos:.2f}°")
    else:
        print(f"\n  Feedback modulation has no positive values!")

    # Feedback kernel (alpha_apical) positive part FWHM
    alpha_centered = alpha.clone()
    # Roll to center at channel 0
    alpha_pos = F.relu(alpha_centered)
    fwhm_kernel_pos = compute_fwhm(alpha_pos, step_deg)
    print(f"  Kernel (alpha_apical) positive part FWHM: {fwhm_kernel_pos:.2f}°")

    # Convolved feedback signal FWHM
    fwhm_fb_signal = compute_fwhm(F.relu(fb_mod), step_deg)
    print(f"  Convolved fb signal (positive part) FWHM: {fwhm_fb_signal:.2f}°")

    # The critical comparison
    print(f"\n  CRITICAL COMPARISON:")
    print(f"    L2/3 OFF FWHM:      {fwhm_off:.2f}°")
    print(f"    FB signal pos FWHM: {fwhm_fb_signal:.2f}°")
    if fwhm_fb_signal > fwhm_off:
        print(f"    → Feedback positive region is WIDER than L2/3 tuning → BROADENS")
    elif fwhm_fb_signal < fwhm_off:
        print(f"    → Feedback positive region is NARROWER than L2/3 tuning → SHARPENS")
    else:
        print(f"    → Same width → neutral")

    # Q2: Why didn't optimizer learn narrower center?
    print(f"\n  Why ±20° center?")
    print(f"    The q_pred bump (sigma=12°) has most mass within ±15-20°.")
    print(f"    After centering (q_pred - 1/N), the positive region extends ~±25°.")
    # Compute q_pred shape
    q_pred_vals = q_pred[0].detach()
    q_centered = q_pred_vals - 1.0 / N
    print(f"    q_pred peak: {q_pred_vals.max().item():.4f}, min: {q_pred_vals.min().item():.4f}")
    print(f"    q_centered peak: {q_centered.max().item():.4f}, min: {q_centered.min().item():.4f}")
    fwhm_qpred = compute_fwhm(q_pred_vals, step_deg)
    fwhm_qcentered_pos = compute_fwhm(F.relu(q_centered), step_deg)
    print(f"    q_pred FWHM: {fwhm_qpred:.2f}°")
    print(f"    q_centered positive FWHM: {fwhm_qcentered_pos:.2f}°")
    print(f"    The convolution of kernel × q_centered produces a signal")
    print(f"    whose positive FWHM depends on BOTH kernel and q_centered widths.")

    # =====================================================================
    # Q4: Can additive feedback produce FWHM narrowing at all?
    # =====================================================================
    print("\n" + "=" * 80)
    print("[Q4] Can additive feedback produce narrowing? — Analytical test")
    print("=" * 80)

    # For narrowing, the additive signal must REDUCE flanks relative to center.
    # This requires the convolution result to be NARROWER than the L2/3 OFF bump.
    # Test: what if kernel were a delta function (only channel 0 positive)?
    # The convolution of delta × q_centered = q_centered itself.
    # q_centered has FWHM ≈ fwhm_qpred. If fwhm_qpred < fwhm_off, narrowing is possible.
    print(f"  If kernel = delta(0): fb_signal = q_centered (FWHM={fwhm_qpred:.2f}°)")
    print(f"  L2/3 OFF FWHM: {fwhm_off:.2f}°")
    if fwhm_qpred < fwhm_off:
        print(f"  → Delta kernel COULD narrow (fb signal narrower than L2/3)")
    else:
        print(f"  → Even delta kernel cannot narrow (q_pred wider than L2/3)")

    # But there's a catch: delta kernel means only one channel positive.
    # The sensory loss improvement depends on boosting channels near the stimulus.
    # A delta kernel only boosts the peak channel → small improvement.

    # Test: what FWHM of additive signal is needed?
    # For narrowing: we need fb_signal to have MOST of its positive mass
    # WITHIN the FWHM of the OFF bump, and NEGATIVE outside.
    # This naturally happens with center-surround, BUT the CONVOLUTION
    # of a center-surround kernel with q_centered broadens the center.

    # Let's compute: the effective additive signal profile
    print(f"\n  Effective additive signal at each distance from stimulus:")
    for i in range(N):
        dist = (i - stim_ch) * step_deg
        if dist > period / 2:
            dist -= period
        if dist < -period / 2:
            dist += period
        if abs(dist) <= 45:
            r_off_i = r_off[i].item()
            fb_i = fb_scaled[i].item()
            # Relative boost: fb / r_off
            rel = fb_i / r_off_i if r_off_i > 1e-6 else float('nan')
            print(f"    d={dist:+6.1f}°  r_off={r_off_i:.4f}  fb_add={fb_i:+.4f}  "
                  f"relative_boost={rel:+.4f}  equiv_gain={1+rel:.4f}")

    # =====================================================================
    # Q5: Manual narrowing test
    # =====================================================================
    print("\n" + "=" * 80)
    print("[Q5] Manual kernel narrowing tests")
    print("=" * 80)

    # Save original kernel
    original_alpha = net.feedback.alpha_apical.data.clone()

    # Test configurations: progressively narrow the positive center
    tests = [
        ("original", None),
        ("zero_ch3-4 (kill ±15-20°)", [3, 4, 32, 33]),  # channels at ±15° and ±20°
        ("zero_ch2-4 (kill ±10-20°)", [2, 3, 4, 32, 33, 34]),  # ±10° to ±20°
        ("only_ch0 (delta kernel)", list(range(1, 36))),  # only center channel
        ("narrower: ch0-1 only (±5°)", list(range(2, 35))),  # only ±5°
        ("invert: neg center, pos surround", "invert"),  # flip sign
    ]

    for label, channels_to_zero in tests:
        # Reset kernel
        net.feedback.alpha_apical.data.copy_(original_alpha)

        if channels_to_zero == "invert":
            net.feedback.alpha_apical.data.mul_(-1.0)
        elif channels_to_zero is not None:
            for ch in channels_to_zero:
                if ch < N:
                    net.feedback.alpha_apical.data[ch] = 0.0

        # Clear cache so new kernel takes effect
        net.feedback.uncache_kernels()

        # Run ON
        state_test = run_model(net, model_cfg, train_cfg, theta_stim, fb_on=True)
        r_test = state_test.r_l23[0].detach()

        fwhm_test = compute_fwhm(r_test, step_deg)
        peak_test = r_test.max().item()
        mean_test = r_test.mean().item()

        # Get fb modulation for this kernel
        net.feedback.uncache_kernels()
        with torch.no_grad():
            fb_test = net.feedback.compute_simple_feedback(q_pred)[0]
        fb_test_scaled = fb_test * net.feedback_scale.item()
        fwhm_fb_test = compute_fwhm(F.relu(fb_test_scaled), step_deg)

        # M7-like discrimination test (quick version)
        # Match: stim=90, oracle=90; Near-miss: stim=100, oracle=90
        state_match = run_model(net, model_cfg, train_cfg, 90.0, fb_on=True)
        state_miss = run_model(net, model_cfg, train_cfg, 100.0, fb_on=True)
        r_match = state_match.r_l23[0].detach()
        r_miss = state_miss.r_l23[0].detach()
        diff_on = (r_match - r_miss).abs().sum().item()

        # Same without feedback
        net.feedback_scale.fill_(0.0)
        state_match_off = run_model(net, model_cfg, train_cfg, 90.0, fb_on=False)
        state_miss_off = run_model(net, model_cfg, train_cfg, 100.0, fb_on=False)
        r_match_off = state_match_off.r_l23[0].detach()
        r_miss_off = state_miss_off.r_l23[0].detach()
        diff_off = (r_match_off - r_miss_off).abs().sum().item()
        net.feedback_scale.fill_(1.0)

        print(f"\n  [{label}]")
        print(f"    Kernel sum: {net.feedback.alpha_apical.data.sum().item():+.4f}")
        print(f"    FB signal pos FWHM: {fwhm_fb_test:.2f}°")
        print(f"    L2/3 FWHM ON: {fwhm_test:.2f}° (OFF: {fwhm_off:.2f}°, delta: {fwhm_test-fwhm_off:+.2f}°)")
        print(f"    Peak ON: {peak_test:.4f} (OFF: {peak_off:.4f}, ratio: {peak_test/peak_off:.4f})")
        print(f"    Global amp: {mean_test/mean_off:.4f}")
        print(f"    Discrimination (Σ|match-miss|): ON={diff_on:.4f} OFF={diff_off:.4f} delta={diff_on-diff_off:+.4f}")

    # Restore original
    net.feedback.alpha_apical.data.copy_(original_alpha)
    net.feedback.uncache_kernels()

    # =====================================================================
    # BONUS: What happens with a pure sharpening kernel?
    # =====================================================================
    print("\n" + "=" * 80)
    print("[BONUS] Synthetic sharpening kernel: narrow positive, wide negative")
    print("=" * 80)

    # Design a kernel that is positive only at ch0 (center) and negative everywhere else
    # This should produce narrowing IF the magnitude is right
    synth_kernels = [
        ("narrow_cs_weak", lambda: _make_cs_kernel(N, pos_width=1, pos_val=0.1, neg_val=-0.01)),
        ("narrow_cs_medium", lambda: _make_cs_kernel(N, pos_width=1, pos_val=0.5, neg_val=-0.05)),
        ("narrow_cs_strong", lambda: _make_cs_kernel(N, pos_width=1, pos_val=1.0, neg_val=-0.1)),
        ("2ch_cs_medium", lambda: _make_cs_kernel(N, pos_width=2, pos_val=0.3, neg_val=-0.05)),
        ("3ch_cs_medium", lambda: _make_cs_kernel(N, pos_width=3, pos_val=0.2, neg_val=-0.05)),
    ]

    for label, kernel_fn in synth_kernels:
        kernel = kernel_fn()
        net.feedback.alpha_apical.data.copy_(kernel)
        net.feedback.uncache_kernels()

        state_test = run_model(net, model_cfg, train_cfg, theta_stim, fb_on=True)
        r_test = state_test.r_l23[0].detach()
        fwhm_test = compute_fwhm(r_test, step_deg)
        peak_test = r_test.max().item()
        mean_test = r_test.mean().item()

        with torch.no_grad():
            fb_test = net.feedback.compute_simple_feedback(q_pred)[0]
        fb_test_scaled = fb_test * net.feedback_scale.item()

        print(f"\n  [{label}] kernel sum={kernel.sum().item():+.4f}")
        print(f"    FWHM ON: {fwhm_test:.2f}° (OFF: {fwhm_off:.2f}°, delta: {fwhm_test-fwhm_off:+.2f}°)")
        print(f"    Peak ON: {peak_test:.4f} (ratio: {peak_test/peak_off:.4f})")
        print(f"    Global amp: {mean_test/mean_off:.4f}")

        # Show profile at key distances
        for dist_deg in [0, 5, 10, 15, 20, 25]:
            ch_pos = stim_ch + int(dist_deg / step_deg)
            ch_neg = stim_ch - int(dist_deg / step_deg)
            if 0 <= ch_pos < N:
                r_t = r_test[ch_pos].item()
                r_o = r_off[ch_pos].item()
                fb_v = fb_test_scaled[ch_pos].item()
                print(f"      d={dist_deg:+3d}°  r_off={r_o:.4f}  r_on={r_t:.4f}  fb={fb_v:+.4f}")

    # Restore original
    net.feedback.alpha_apical.data.copy_(original_alpha)
    net.feedback.uncache_kernels()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  L2/3 OFF FWHM: {fwhm_off:.2f}°")
    print(f"  q_pred FWHM: {fwhm_qpred:.2f}°")
    print(f"  Learned kernel positive FWHM: {fwhm_kernel_pos:.2f}°")
    print(f"  Learned fb signal positive FWHM: {fwhm_fb_signal:.2f}°")
    print(f"  For narrowing: fb signal must be NARROWER than L2/3 OFF FWHM")
    print(f"  Minimum achievable fb signal FWHM = q_pred FWHM = {fwhm_qpred:.2f}°")
    print(f"  → Narrowing possible only if q_pred FWHM < L2/3 OFF FWHM")


def _make_cs_kernel(N, pos_width=1, pos_val=0.1, neg_val=-0.01):
    """Make a center-surround kernel with specified positive width."""
    kernel = torch.full((N,), neg_val)
    kernel[0] = pos_val
    for i in range(1, pos_width):
        kernel[i] = pos_val
        kernel[N - i] = pos_val
    return kernel


if __name__ == "__main__":
    main()
