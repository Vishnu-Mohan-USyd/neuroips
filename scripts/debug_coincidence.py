"""Diagnostic: investigate why coincidence gate produces no selectivity
between true and wrong oracle templates.

Tests 4 hypotheses:
1. Overlap: wrong pred only 10° from stimulus; with sigma=12° Gaussians, massive overlap
2. Kernel compensation: alpha_apical learned broad profiles
3. Gate kills all: relu products uniformly small for both conditions
4. Centering kills signal: basal_field too narrow after centering

Loads a trained Batch 1 true checkpoint and runs controlled forward passes.
"""

import sys
sys.path.insert(0, "/mnt/c/Users/User/codingproj/freshstart")

import torch
import torch.nn.functional as F
import numpy as np
from src.config import load_config
from src.model.network import LaminarV1V2Network as LaminarV1V2
from src.state import initial_state
from src.stimulus.gratings import generate_grating

# ── Load model ──────────────────────────────────────────────────────
config_path = "config/apical_template_true.yaml"
ckpt_path = "results/batch1/true_s42/center_surround_seed42/checkpoint.pt"

model_cfg, train_cfg, stim_cfg = load_config(config_path)

net = LaminarV1V2(model_cfg)
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
net.load_state_dict(ckpt["model_state"], strict=False)
net.eval()

N = model_cfg.n_orientations  # 36
period = model_cfg.orientation_range  # 180.0
sigma_ff = model_cfg.sigma_ff  # 12.0
step_deg = model_cfg.transition_step  # 5.0

print(f"N={N}, period={period}°, sigma_ff={sigma_ff}°, transition_step={step_deg}°")
print(f"oracle_sigma={train_cfg.oracle_sigma}°, oracle_pi={train_cfg.oracle_pi}")
print()

# ── Hypothesis 2: Inspect learned alpha_apical ──────────────────────
print("=" * 60)
print("HYPOTHESIS 2: Learned alpha_apical weights")
print("=" * 60)
fb = net.feedback
alpha_apical = fb.alpha_apical.detach().numpy()
basis_sigmas = ["σ=5°", "σ=15°", "σ=30°", "σ=60°", "MexHat", "Constant", "Odd/sin"]
print("Basis functions and their learned weights:")
for i, (name, w) in enumerate(zip(basis_sigmas, alpha_apical)):
    print(f"  [{i}] {name:12s}: alpha = {w:.6f}")
print(f"  Sum of |alpha|: {np.abs(alpha_apical).sum():.6f}")
print()

# Show the apical kernel profile
K_apical = fb.get_apical_profile().detach().numpy()
print(f"Apical kernel: peak={K_apical.max():.6f}, min={K_apical.min():.6f}, "
      f"sum={K_apical.sum():.6f}")
# Effective width: FWHM of the kernel
half_max = K_apical.max() / 2
above_half = np.where(K_apical >= half_max)[0]
if len(above_half) > 0:
    fwhm_channels = len(above_half)
    fwhm_deg = fwhm_channels * (period / N)
    print(f"Apical kernel FWHM: ~{fwhm_channels} channels = ~{fwhm_deg:.1f}°")
print()

# ── Create controlled stimulus ──────────────────────────────────────
# Stimulus at 90° (middle of range), contrast=0.5
theta_stim = torch.tensor([90.0])
contrast = torch.tensor([0.5])
stim = generate_grating(theta_stim, contrast, N, sigma_ff, period=period)
print("=" * 60)
print(f"STIMULUS at {theta_stim.item()}°, contrast={contrast.item()}")
print("=" * 60)

# ── Get steady-state L4 by running several steps ────────────────────
with torch.no_grad():
    state = initial_state(batch_size=1, n_orientations=N, v2_hidden_dim=model_cfg.v2_hidden_dim)
    cue = torch.zeros(1, 2)
    task = torch.zeros(1, 2)

    # Run 20 steps to reach L4 steady state
    for _ in range(20):
        r_l4, adaptation = net.l4(stim, state.r_l4, state.r_pv, state.adaptation)
        r_pv = net.pv(r_l4, state.r_l23, state.r_pv)
        state = state._replace(r_l4=r_l4, r_pv=r_pv, adaptation=adaptation)

    r_l4 = state.r_l4  # [1, 36]

print(f"\nL4 profile: peak={r_l4.max().item():.4f}, min={r_l4.min().item():.6f}")
peak_ch = r_l4.argmax(dim=-1).item()
print(f"L4 peak channel: {peak_ch} = {peak_ch * period / N:.1f}°")

# FWHM of L4
l4_np = r_l4[0].numpy()
l4_half = l4_np.max() / 2
l4_above = np.where(l4_np >= l4_half)[0]
print(f"L4 FWHM: {len(l4_above)} channels = {len(l4_above) * period / N:.1f}°")

# ── Hypothesis 1 + 4: Compute basal_field ──────────────────────────
print("\n" + "=" * 60)
print("HYPOTHESIS 4: Basal field after centering")
print("=" * 60)
basal_field = r_l4 - r_l4.mean(dim=-1, keepdim=True)
basal_np = basal_field[0].detach().numpy()
n_positive = (basal_np > 0).sum()
print(f"basal_field: peak={basal_np.max():.6f}, min={basal_np.min():.6f}")
print(f"Positive channels: {n_positive}/{N} = {n_positive/N*100:.1f}%")
relu_basal = np.maximum(basal_np, 0)
print(f"relu(basal_field): nonzero channels: {(relu_basal > 0).sum()}")
print(f"relu(basal_field) peak at channel {np.argmax(relu_basal)} "
      f"= {relu_basal.max():.6f}")
print()

# Show profile around stimulus channel
print("basal_field profile around stimulus (channels 15-21, i.e. 75°-105°):")
for ch in range(15, 22):
    deg = ch * period / N
    print(f"  ch={ch:2d} ({deg:5.1f}°): basal={basal_np[ch]:.6f}  "
          f"relu(basal)={max(0, basal_np[ch]):.6f}")

# ── Create true and wrong predictions ──────────────────────────────
# True: prediction aligned with stimulus (bump at 90°)
# Wrong: prediction at 90° + 2*step = 90°+10° = 100° (CW state predicts CCW = -5°,
#         so wrong is at 85°; or if CCW state, wrong is at 95°)
# Actually: if stim is at 90° during CW, previous was at 85°, and oracle_wrong
# predicted CCW = 85°-5°=80° while true was 85°+5°=90°. Offset = 10°.
# Let's just directly test at several offsets to see the gradient.

offsets_deg = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0]
print("\n" + "=" * 60)
print("HYPOTHESIS 1 + 3: Coincidence vs prediction offset")
print("=" * 60)

oracle_pi = train_cfg.oracle_pi  # 3.0
oracle_sigma = train_cfg.oracle_sigma  # 12.0

with torch.no_grad():
    fb.cache_kernels()

    print(f"\noracle_pi={oracle_pi}, oracle_sigma={oracle_sigma}°")
    print(f"max_apical_gain={fb.max_apical_gain}")
    print()

    print(f"{'Offset':>8s}  {'coincidence_sum':>15s}  {'coincidence_peak':>16s}  "
          f"{'gain_peak':>10s}  {'gain_mean':>10s}  {'gain_at_stim':>12s}")
    print("-" * 85)

    results = {}
    for offset in offsets_deg:
        # Prediction bump at (stimulus + offset)
        theta_pred = torch.tensor([theta_stim.item() + offset])
        q_pred = net._make_bump(theta_pred, sigma=oracle_sigma)  # [1, N]

        # Center q_pred (as done in feedback.forward)
        q_centered = q_pred - q_pred.mean(dim=-1, keepdim=True)

        # Apical field via circulant convolution
        apical_circulant = fb._cached_apical_circulant
        apical_field = (apical_circulant @ q_centered.unsqueeze(-1)).squeeze(-1)

        # Coincidence
        coincidence = F.relu(apical_field) * F.relu(basal_field)

        # Gain
        pi_eff = torch.tensor([[oracle_pi]])
        apical_gain = 1.0 + fb.max_apical_gain * torch.tanh(pi_eff * coincidence)

        # Metrics
        coin_sum = coincidence.sum().item()
        coin_peak = coincidence.max().item()
        gain_peak = apical_gain.max().item()
        gain_mean = apical_gain.mean().item()
        gain_at_stim = apical_gain[0, peak_ch].item()

        results[offset] = {
            'coin_sum': coin_sum, 'coin_peak': coin_peak,
            'gain_peak': gain_peak, 'gain_mean': gain_mean,
            'gain_at_stim': gain_at_stim,
            'apical_field': apical_field[0].numpy().copy(),
            'coincidence': coincidence[0].numpy().copy(),
            'gain': apical_gain[0].numpy().copy(),
        }

        print(f"{offset:>7.1f}°  {coin_sum:>15.8f}  {coin_peak:>16.8f}  "
              f"{gain_peak:>10.6f}  {gain_mean:>10.6f}  {gain_at_stim:>12.6f}")

    fb.uncache_kernels()

# ── Key comparison: true (0°) vs wrong (10°) ───────────────────────
print("\n" + "=" * 60)
print("KEY COMPARISON: true (0° offset) vs wrong (10° offset)")
print("=" * 60)
r0 = results[0.0]
r10 = results[10.0]
print(f"  coincidence sum:  true={r0['coin_sum']:.8f}  wrong={r10['coin_sum']:.8f}  "
      f"ratio={r10['coin_sum']/(r0['coin_sum']+1e-12):.4f}")
print(f"  coincidence peak: true={r0['coin_peak']:.8f}  wrong={r10['coin_peak']:.8f}  "
      f"ratio={r10['coin_peak']/(r0['coin_peak']+1e-12):.4f}")
print(f"  gain at stim ch:  true={r0['gain_at_stim']:.6f}  wrong={r10['gain_at_stim']:.6f}")
print(f"  gain peak:        true={r0['gain_peak']:.6f}  wrong={r10['gain_peak']:.6f}")
print(f"  gain mean:        true={r0['gain_mean']:.6f}  wrong={r10['gain_mean']:.6f}")

# ── Detailed channel-level comparison ──────────────────────────────
print("\nChannel-level comparison around stimulus (true vs wrong 10°):")
print(f"{'Ch':>4s} {'Deg':>6s}  {'apical_0':>10s} {'apical_10':>10s}  "
      f"{'relu_bas':>10s}  {'coin_0':>10s} {'coin_10':>10s}  "
      f"{'gain_0':>8s} {'gain_10':>8s}")
print("-" * 95)
for ch in range(14, 24):
    deg = ch * period / N
    a0 = r0['apical_field'][ch]
    a10 = r10['apical_field'][ch]
    rb = relu_basal[ch]
    c0 = r0['coincidence'][ch]
    c10 = r10['coincidence'][ch]
    g0 = r0['gain'][ch]
    g10 = r10['gain'][ch]
    marker = " <-- stim" if ch == peak_ch else ""
    print(f"{ch:>4d} {deg:>5.1f}°  {a0:>10.6f} {a10:>10.6f}  "
          f"{rb:>10.6f}  {c0:>10.8f} {c10:>10.8f}  "
          f"{g0:>8.6f} {g10:>8.6f}{marker}")

# ── Summary diagnosis ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY OF EVIDENCE")
print("=" * 60)

# 1. Is alpha_apical effectively zero?
alpha_mag = np.abs(alpha_apical).sum()
if alpha_mag < 0.05:
    print(f"[H3] alpha_apical magnitude is very small ({alpha_mag:.6f}) — "
          f"apical field may be near-zero regardless of prediction alignment.")

# 2. Is the coincidence difference meaningful?
if r0['coin_sum'] > 0:
    ratio = r10['coin_sum'] / r0['coin_sum']
    print(f"[H1] Coincidence wrong/true ratio: {ratio:.4f}")
    if ratio > 0.7:
        print(f"  → MASSIVE OVERLAP: wrong template retains {ratio*100:.1f}% of true's "
              f"coincidence. 10° offset with σ=12° gives negligible discrimination.")
    elif ratio > 0.4:
        print(f"  → Moderate overlap. Some discrimination possible but may be insufficient.")
    else:
        print(f"  → Good discrimination. Overlap is not the primary problem.")
else:
    print("[H3] CONFIRMED: coincidence is effectively zero for BOTH conditions. "
          "Gate kills all gain.")

# 3. How broad is the basal field?
print(f"[H4] Basal field: {n_positive} positive channels out of {N}.")
if n_positive < 5:
    print("  → Very narrow. Centering heavily restricts the basal signal.")
