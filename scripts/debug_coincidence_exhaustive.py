"""Exhaustive coincidence gate diagnostic — cross-seed, cross-condition analysis.

Tests:
1. Cross-seed consistency (true_s42, true_s123, true_s456)
2. Wrong-trained checkpoint alpha_apical (wrong_s42, wrong_s123, wrong_s456)
3. Theoretical overlap analysis: what offset WOULD produce meaningful discrimination?
4. The gain contrast required for ~1% M7 effect
5. Does the apical kernel compound the overlap problem?
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

# ── EXPERIMENT 1: Cross-seed alpha_apical for TRUE checkpoints ──────
print("=" * 70)
print("EXPERIMENT 1: Cross-seed alpha_apical consistency (TRUE checkpoints)")
print("=" * 70)

true_ckpts = [
    ("true_s42",  "results/batch1/true_s42/center_surround_seed42/checkpoint.pt"),
    ("true_s123", "results/batch1/true_s123/center_surround_seed123/checkpoint.pt"),
    ("true_s456", "results/batch1/true_s456/center_surround_seed456/checkpoint.pt"),
]

basis_names = ["σ=5°", "σ=15°", "σ=30°", "σ=60°", "MexHat", "Const", "Odd"]

print(f"\n{'Seed':>10s}  ", end="")
for bn in basis_names:
    print(f"{bn:>8s}", end="")
print(f"  {'|sum|':>8s}  {'FWHM':>6s}  {'peak':>8s}  {'min':>8s}")
print("-" * 100)

true_alpha_all = []
for name, path in true_ckpts:
    net = LaminarV1V2Network(model_cfg)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net.load_state_dict(ckpt["model_state"], strict=False)
    net.eval()

    alpha = net.feedback.alpha_apical.detach().numpy()
    true_alpha_all.append(alpha)
    K = net.feedback.get_apical_profile().detach().numpy()
    half_max = K.max() / 2
    fwhm_ch = (K >= half_max).sum()

    print(f"{name:>10s}  ", end="")
    for a in alpha:
        print(f"{a:>8.4f}", end="")
    print(f"  {np.abs(alpha).sum():>8.4f}  {fwhm_ch*period/N:>5.1f}°  "
          f"{K.max():>8.6f}  {K.min():>8.6f}")

true_alpha_all = np.array(true_alpha_all)
print(f"\n  Cross-seed std: ", end="")
for i in range(len(basis_names)):
    print(f"{true_alpha_all[:, i].std():>8.4f}", end="")
print()

# ── EXPERIMENT 2: WRONG checkpoint alpha_apical ─────────────────────
print("\n" + "=" * 70)
print("EXPERIMENT 2: WRONG checkpoint alpha_apical")
print("=" * 70)

# Load wrong config
model_cfg_w, train_cfg_w, _ = load_config("config/apical_template_true.yaml")

wrong_ckpts = [
    ("wrong_s42",  "results/batch1/wrong_s42/center_surround_seed42/checkpoint.pt"),
    ("wrong_s123", "results/batch1/wrong_s123/center_surround_seed123/checkpoint.pt"),
    ("wrong_s456", "results/batch1/wrong_s456/center_surround_seed456/checkpoint.pt"),
]

print(f"\n{'Seed':>10s}  ", end="")
for bn in basis_names:
    print(f"{bn:>8s}", end="")
print(f"  {'|sum|':>8s}  {'FWHM':>6s}  {'peak':>8s}")
print("-" * 100)

wrong_alpha_all = []
for name, path in wrong_ckpts:
    net_w = LaminarV1V2Network(model_cfg_w)
    ckpt_w = torch.load(path, map_location="cpu", weights_only=False)
    net_w.load_state_dict(ckpt_w["model_state"], strict=False)
    net_w.eval()

    alpha = net_w.feedback.alpha_apical.detach().numpy()
    wrong_alpha_all.append(alpha)
    K = net_w.feedback.get_apical_profile().detach().numpy()
    half_max = K.max() / 2
    fwhm_ch = (K >= half_max).sum()

    print(f"{name:>10s}  ", end="")
    for a in alpha:
        print(f"{a:>8.4f}", end="")
    print(f"  {np.abs(alpha).sum():>8.4f}  {fwhm_ch*period/N:>5.1f}°  "
          f"{K.max():>8.6f}")

# Compare true vs wrong
print("\nTrue vs Wrong alpha_apical comparison:")
true_mean = np.array(true_alpha_all).mean(axis=0)
wrong_mean = np.array(wrong_alpha_all).mean(axis=0)
print(f"  True mean:  ", end="")
for a in true_mean:
    print(f"{a:>8.4f}", end="")
print()
print(f"  Wrong mean: ", end="")
for a in wrong_mean:
    print(f"{a:>8.4f}", end="")
print()
print(f"  Difference: ", end="")
for a, b in zip(true_mean, wrong_mean):
    print(f"{a-b:>8.4f}", end="")
print()

# ── EXPERIMENT 3: Coincidence sweep across all TRUE seeds ────────────
print("\n" + "=" * 70)
print("EXPERIMENT 3: Coincidence sweep across all TRUE seeds")
print("=" * 70)

offsets = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0]
theta_stim = torch.tensor([90.0])
contrast = torch.tensor([0.5])
stim = generate_grating(theta_stim, contrast, N, sigma_ff, period=period)

for name, path in true_ckpts:
    net = LaminarV1V2Network(model_cfg)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net.load_state_dict(ckpt["model_state"], strict=False)
    net.eval()

    with torch.no_grad():
        state = initial_state(1, N, model_cfg.v2_hidden_dim)
        for _ in range(20):
            r_l4, adaptation = net.l4(stim, state.r_l4, state.r_pv, state.adaptation)
            r_pv = net.pv(r_l4, state.r_l23, state.r_pv)
            state = state._replace(r_l4=r_l4, r_pv=r_pv, adaptation=adaptation)

        basal_field = r_l4 - r_l4.mean(dim=-1, keepdim=True)
        peak_ch = r_l4.argmax(dim=-1).item()

        net.feedback.cache_kernels()
        pi = torch.tensor([[train_cfg.oracle_pi]])

        print(f"\n{name}: L4 peak ch={peak_ch}, L4 peak={r_l4.max().item():.4f}")
        print(f"  {'Offset':>8s}  {'coin_sum':>12s}  {'gain@stim':>10s}  {'gain_peak':>10s}  {'wrong/true':>10s}")
        print("  " + "-" * 60)

        coin_at_0 = None
        for offset in offsets:
            theta_pred = torch.tensor([theta_stim.item() + offset])
            q_pred = net._make_bump(theta_pred, sigma=train_cfg.oracle_sigma)
            q_centered = q_pred - q_pred.mean(dim=-1, keepdim=True)
            apical_circulant = net.feedback._cached_apical_circulant
            apical_field = (apical_circulant @ q_centered.unsqueeze(-1)).squeeze(-1)
            coincidence = F.relu(apical_field) * F.relu(basal_field)
            gain = 1.0 + net.feedback.max_apical_gain * torch.tanh(pi * coincidence)

            cs = coincidence.sum().item()
            gs = gain[0, peak_ch].item()
            gp = gain.max().item()

            if offset == 0.0:
                coin_at_0 = cs
            ratio = cs / (coin_at_0 + 1e-12) if coin_at_0 else 0

            print(f"  {offset:>7.1f}°  {cs:>12.6f}  {gs:>10.6f}  {gp:>10.6f}  {ratio:>10.4f}")

        net.feedback.uncache_kernels()

# ── EXPERIMENT 4: Theoretical overlap analysis ──────────────────────
print("\n" + "=" * 70)
print("EXPERIMENT 4: Theoretical overlap analysis")
print("=" * 70)

# Pure Gaussian overlap: if two Gaussians have sigma_a and sigma_b,
# their product is a Gaussian with sigma_p = 1/sqrt(1/sigma_a^2 + 1/sigma_b^2)
# and peak at (sigma_b^2*mu_a + sigma_a^2*mu_b)/(sigma_a^2 + sigma_b^2),
# with amplitude reduction exp(-d^2/(2*(sigma_a^2 + sigma_b^2)))

sigma_q = train_cfg.oracle_sigma  # 12.0° — q_pred bump width
sigma_l4 = sigma_ff  # 12.0° — L4 tuning width

# The apical kernel FWHM is ~15° (from data above), which further broadens
# the apical_field. Effective apical field width is convolution of q (sigma=12)
# with kernel. For a Gaussian kernel with sigma_k, the result has
# sigma_eff = sqrt(sigma_q^2 + sigma_k^2).
# Kernel FWHM = 15° → sigma_k ≈ 15°/(2*sqrt(2*ln(2))) ≈ 15/2.355 ≈ 6.4°
# But the kernel is not a pure Gaussian, so let's measure directly.

# Measure effective apical_field width from the 0° offset experiment
net = LaminarV1V2Network(model_cfg)
ckpt = torch.load(true_ckpts[0][1], map_location="cpu", weights_only=False)
net.load_state_dict(ckpt["model_state"], strict=False)
net.eval()

with torch.no_grad():
    q_0 = net._make_bump(torch.tensor([90.0]), sigma=sigma_q)
    q_c = q_0 - q_0.mean(dim=-1, keepdim=True)
    net.feedback.cache_kernels()
    af = (net.feedback._cached_apical_circulant @ q_c.unsqueeze(-1)).squeeze(-1)
    net.feedback.uncache_kernels()

af_np = af[0].numpy()
af_half = af_np.max() / 2
af_fwhm_ch = (af_np >= af_half).sum()
af_fwhm_deg = af_fwhm_ch * period / N
print(f"q_pred bump FWHM: {2.355*sigma_q:.1f}° (sigma={sigma_q}°)")
print(f"Apical kernel FWHM: ~15.0° (from data)")
print(f"Resulting apical_field FWHM: {af_fwhm_deg:.1f}° ({af_fwhm_ch} channels)")

# For L4 (basal) side:
print(f"L4 (basal) FWHM: ~25.0° (from data, sigma_ff={sigma_ff}°)")

# Overlap ratio for the product of two centered Gaussians offset by d:
# overlap = integral of max(af(x-d),0) * max(bf(x),0) / integral of max(af(x),0) * max(bf(x),0)
# For Gaussians: ratio ≈ exp(-d^2 / (2 * (sigma_eff_af^2 + sigma_eff_bf^2)))

# Estimate effective sigmas from FWHM
sigma_eff_af = af_fwhm_deg / 2.355
sigma_eff_bf = 25.0 / 2.355  # From L4 FWHM data
sigma_combined = np.sqrt(sigma_eff_af**2 + sigma_eff_bf**2)

print(f"\nEffective sigma (apical field): {sigma_eff_af:.1f}°")
print(f"Effective sigma (basal field): {sigma_eff_bf:.1f}°")
print(f"Combined sigma for overlap decay: {sigma_combined:.1f}°")

print(f"\nTheoretical overlap ratio at various offsets:")
print(f"  {'Offset':>8s}  {'Ratio (theory)':>14s}  {'Ratio (measured)':>16s}")
print("  " + "-" * 45)

# Recompute measured ratios from first seed
net2 = LaminarV1V2Network(model_cfg)
ckpt2 = torch.load(true_ckpts[0][1], map_location="cpu", weights_only=False)
net2.load_state_dict(ckpt2["model_state"], strict=False)
net2.eval()

with torch.no_grad():
    state2 = initial_state(1, N, model_cfg.v2_hidden_dim)
    for _ in range(20):
        r_l4_2, adapt_2 = net2.l4(stim, state2.r_l4, state2.r_pv, state2.adaptation)
        r_pv_2 = net2.pv(r_l4_2, state2.r_l23, state2.r_pv)
        state2 = state2._replace(r_l4=r_l4_2, r_pv=r_pv_2, adaptation=adapt_2)

    basal_2 = r_l4_2 - r_l4_2.mean(dim=-1, keepdim=True)
    net2.feedback.cache_kernels()

    coin_0_ref = None
    for offset in offsets:
        q = net2._make_bump(torch.tensor([90.0 + offset]), sigma=sigma_q)
        qc = q - q.mean(dim=-1, keepdim=True)
        af2 = (net2.feedback._cached_apical_circulant @ qc.unsqueeze(-1)).squeeze(-1)
        coin2 = F.relu(af2) * F.relu(basal_2)
        cs2 = coin2.sum().item()
        if offset == 0.0:
            coin_0_ref = cs2

        theory_ratio = np.exp(-offset**2 / (2 * sigma_combined**2))
        measured_ratio = cs2 / (coin_0_ref + 1e-12)
        print(f"  {offset:>7.1f}°  {theory_ratio:>14.4f}  {measured_ratio:>16.4f}")

    net2.feedback.uncache_kernels()

# ── EXPERIMENT 5: What offset IS needed for 50% discrimination? ─────
print("\n" + "=" * 70)
print("EXPERIMENT 5: Required offset for meaningful discrimination")
print("=" * 70)

# From theory: ratio = exp(-d^2 / (2*sigma_combined^2))
# For ratio = 0.5: d = sigma_combined * sqrt(2 * ln(2)) ≈ sigma_combined * 1.177
d_50 = sigma_combined * np.sqrt(2 * np.log(2))
d_25 = sigma_combined * np.sqrt(2 * np.log(4))
d_10 = sigma_combined * np.sqrt(2 * np.log(10))

print(f"For 50% discrimination (ratio=0.5): offset ≈ {d_50:.1f}°")
print(f"For 75% discrimination (ratio=0.25): offset ≈ {d_25:.1f}°")
print(f"For 90% discrimination (ratio=0.10): offset ≈ {d_10:.1f}°")
print(f"\nCurrent true/wrong offset: 10.0° (2 × transition_step)")
print(f"  → Expected discrimination at 10°: {(1-np.exp(-10**2/(2*sigma_combined**2)))*100:.1f}%")

# ── EXPERIMENT 6: Gain difference impact estimate ───────────────────
print("\n" + "=" * 70)
print("EXPERIMENT 6: Gain difference impact on L2/3")
print("=" * 70)
print("At stimulus channel (0° vs 10° offset):")
print(f"  gain_true = 1.109, gain_wrong = 1.086")
print(f"  Absolute difference: {1.109 - 1.086:.3f}")
print(f"  Relative difference: {(1.109 - 1.086)/1.109 * 100:.2f}%")
print(f"  Max possible gain: 1 + max_apical_gain = 1.20")
print(f"  Using {(0.109/0.2)*100:.1f}% of available gain range (true)")
print(f"  Using {(0.086/0.2)*100:.1f}% of available gain range (wrong)")
print()
print("The gain difference is 2.3 percentage points.")
print("With L2/3 baseline firing rates, this produces negligible")
print("differential suppression — both conditions are nearly identical.")

# ── EXPERIMENT 7: Check random and uniform for comparison ───────────
print("\n" + "=" * 70)
print("EXPERIMENT 7: Random & uniform checkpoint alpha_apical")
print("=" * 70)

for label, ckpts_list in [
    ("random", [
        ("random_s42",  "results/batch1/random_s42/center_surround_seed42/checkpoint.pt"),
        ("random_s123", "results/batch1/random_s123/center_surround_seed123/checkpoint.pt"),
        ("random_s456", "results/batch1/random_s456/center_surround_seed456/checkpoint.pt"),
    ]),
    ("uniform", [
        ("uniform_s42",  "results/batch1/uniform_s42/center_surround_seed42/checkpoint.pt"),
        ("uniform_s123", "results/batch1/uniform_s123/center_surround_seed123/checkpoint.pt"),
        ("uniform_s456", "results/batch1/uniform_s456/center_surround_seed456/checkpoint.pt"),
    ]),
]:
    print(f"\n{label.upper()} checkpoints:")
    print(f"{'Seed':>12s}  ", end="")
    for bn in basis_names:
        print(f"{bn:>8s}", end="")
    print(f"  {'|sum|':>8s}  {'FWHM':>6s}")
    print("-" * 95)

    for name, path in ckpts_list:
        net_r = LaminarV1V2Network(model_cfg)
        ckpt_r = torch.load(path, map_location="cpu", weights_only=False)
        net_r.load_state_dict(ckpt_r["model_state"], strict=False)
        net_r.eval()

        alpha = net_r.feedback.alpha_apical.detach().numpy()
        K = net_r.feedback.get_apical_profile().detach().numpy()
        half_max = K.max() / 2
        fwhm_ch = (K >= half_max).sum()

        print(f"{name:>12s}  ", end="")
        for a in alpha:
            print(f"{a:>8.4f}", end="")
        print(f"  {np.abs(alpha).sum():>8.4f}  {fwhm_ch*period/N:>5.1f}°")

print("\n" + "=" * 70)
print("FINAL DIAGNOSIS")
print("=" * 70)
