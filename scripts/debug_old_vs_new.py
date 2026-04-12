"""Compare old (pre-gate) vs new (coincidence-gated) checkpoints.

Old: results/apical/ap_s42 — trained WITHOUT coincidence gate (r_l4 not passed to feedback)
New: results/batch1/true_s42 — trained WITH coincidence gate

Questions:
1. SOM inhibition magnitude (old vs new)
2. Apical gain at stimulus channel (old vs new)
3. Net L2/3 drive (old vs new)
4. alpha_apical and alpha_inh learned values (old vs new)
5. Was the old model's M7=+0.014 from a completely different drive balance?
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
from src.utils import rectified_softplus

model_cfg, train_cfg, stim_cfg = load_config("config/apical_template_true.yaml")
N = model_cfg.n_orientations
period = model_cfg.orientation_range
sigma_ff = model_cfg.sigma_ff

theta_stim = 90.0
contrast = 0.5
stim = generate_grating(torch.tensor([theta_stim]), torch.tensor([contrast]),
                        N, sigma_ff, period=period)
cue = torch.zeros(1, 2)
task = torch.zeros(1, 2)
peak_ch = 18

oracle_sigma = 12.0
oracle_pi = 3.0
q_true = None  # will be set per-model

# ── Load checkpoints ────────────────────────────────────────────────
checkpoints = {
    "OLD (apical/ap_s42)": "results/apical/ap_s42/center_surround_seed42/checkpoint.pt",
    "OLD (apical/ap_s123)": "results/apical/ap_s123/center_surround_seed123/checkpoint.pt",
    "OLD (apical/ap_s456)": "results/apical/ap_s456/center_surround_seed456/checkpoint.pt",
    "NEW (batch1/true_s42)": "results/batch1/true_s42/center_surround_seed42/checkpoint.pt",
    "NEW (batch1/true_s123)": "results/batch1/true_s123/center_surround_seed123/checkpoint.pt",
    "NEW (batch1/true_s456)": "results/batch1/true_s456/center_surround_seed456/checkpoint.pt",
}

# ── EXPERIMENT 1: Compare learned parameters ────────────────────────
print("=" * 90)
print("EXPERIMENT 1: Learned parameter comparison (OLD pre-gate vs NEW with-gate)")
print("=" * 90)

basis_names = ["σ=5°", "σ=15°", "σ=30°", "σ=60°", "MexHat", "Const", "Odd"]

print(f"\n{'Checkpoint':>28s}  ", end="")
for bn in basis_names:
    print(f"{'α_a_'+bn:>10s}", end="")
print(f"  {'|α_a|sum':>8s}  {'K_a FWHM':>8s}")
print("-" * 120)

for name, path in checkpoints.items():
    net = LaminarV1V2Network(model_cfg)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net.load_state_dict(ckpt["model_state"], strict=False)
    net.eval()

    aa = net.feedback.alpha_apical.detach().numpy()
    K_a = net.feedback.get_apical_profile().detach().numpy()
    hm = K_a.max() / 2
    fwhm = (K_a >= hm).sum() * period / N if K_a.max() > 0 else 0

    print(f"{name:>28s}  ", end="")
    for a in aa:
        print(f"{a:>10.5f}", end="")
    print(f"  {np.abs(aa).sum():>8.4f}  {fwhm:>7.1f}°")

# alpha_inh comparison
print(f"\n{'Checkpoint':>28s}  ", end="")
for bn in basis_names:
    print(f"{'α_i_'+bn:>10s}", end="")
print(f"  {'|α_i|sum':>8s}  {'K_i FWHM':>8s}")
print("-" * 120)

for name, path in checkpoints.items():
    net = LaminarV1V2Network(model_cfg)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net.load_state_dict(ckpt["model_state"], strict=False)
    net.eval()

    ai = net.feedback.alpha_inh.detach().numpy()
    K_i = net.feedback.get_profiles().detach().numpy()
    hm = K_i.max() / 2
    fwhm = (K_i >= hm).sum() * period / N if K_i.max() > 0 else 0

    print(f"{name:>28s}  ", end="")
    for a in ai:
        print(f"{a:>10.5f}", end="")
    print(f"  {np.abs(ai).sum():>8.4f}  {fwhm:>7.1f}°")

# Other key parameters
print(f"\n{'Checkpoint':>28s}  {'w_som':>8s}  {'w_pv_l23':>8s}  {'σ_rec':>8s}  "
      f"{'g_rec':>8s}  {'w_tmpl':>8s}  {'vip_base':>8s}  {'w_vip_som':>10s}")
print("-" * 110)

for name, path in checkpoints.items():
    net = LaminarV1V2Network(model_cfg)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net.load_state_dict(ckpt["model_state"], strict=False)
    net.eval()

    w_som = net.l23.w_som.gain.item()
    w_pv = net.l23.w_pv_l23.gain.item()
    sig_rec = F.softplus(net.l23.sigma_rec_raw).item()
    g_rec = F.softplus(net.l23.gain_rec_raw).item()
    w_tmpl = net.w_template_drive.item() if hasattr(net, 'w_template_drive') else float('nan')
    vip_b = net.feedback.vip_baseline.item()
    w_vs = F.softplus(net.w_vip_som).item()

    print(f"{name:>28s}  {w_som:>8.4f}  {w_pv:>8.4f}  {sig_rec:>7.2f}°  "
          f"{g_rec:>8.4f}  {w_tmpl:>8.4f}  {vip_b:>8.4f}  {w_vs:>10.4f}")

# ── EXPERIMENT 2: Drive decomposition for OLD models ────────────────
print("\n" + "=" * 90)
print("EXPERIMENT 2: Drive decomposition — OLD models (pure top-down, r_l4=None path)")
print("=" * 90)

for name, path in list(checkpoints.items())[:3]:  # OLD models only
    net = LaminarV1V2Network(model_cfg)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net.load_state_dict(ckpt["model_state"], strict=False)
    net.eval()

    net.oracle_mode = True
    net.feedback_scale = torch.tensor(1.0)

    q_true = net._make_bump(torch.tensor([theta_stim]), sigma=oracle_sigma)
    pi_tensor = torch.full((1, 1), oracle_pi)
    net.oracle_q_pred = q_true
    net.oracle_pi_pred = pi_tensor

    state = initial_state(1, N, model_cfg.v2_hidden_dim)

    with torch.no_grad():
        net.l23.cache_kernels()
        if hasattr(net.feedback, 'cache_kernels'):
            net.feedback.cache_kernels()

        for t in range(12):
            r_l4, adaptation = net.l4(stim, state.r_l4, state.r_pv, state.adaptation)
            r_pv = net.pv(r_l4, state.r_l23, state.r_pv)
            pi_eff = pi_tensor * net.feedback_scale

            # OLD code path: r_l4=None → pure top-down apical
            som_drive, vip_drive, apical_gain = net.feedback(q_true, pi_eff, r_l4=None)
            r_vip = net.vip(vip_drive, state.r_vip)
            effective_som_drive = F.relu(som_drive - F.softplus(net.w_vip_som) * r_vip)
            center_exc = net.w_template_drive * net.deep_template(q_true, pi_eff) if hasattr(net, 'w_template_drive') else torch.zeros(1, N)
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

    print(f"\n{name} (step 11, r_l4=None / pure top-down):")
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
    print(f"  r_l23 peak:   {r_l23.max().item():.8f}")
    print(f"  r_som@stim:   {r_som[0, peak_ch].item():.6f}")
    print(f"  gain peak:    {apical_gain.max().item():.6f}")
    print(f"  gain profile: {apical_gain[0, peak_ch-2:peak_ch+3].tolist()}")

# ── EXPERIMENT 2b: Same but with coincidence gate (for direct comparison) ─
print("\n" + "=" * 90)
print("EXPERIMENT 2b: Drive decomposition — OLD models WITH coincidence gate")
print("=" * 90)

for name, path in list(checkpoints.items())[:3]:  # OLD models
    net = LaminarV1V2Network(model_cfg)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net.load_state_dict(ckpt["model_state"], strict=False)
    net.eval()

    net.oracle_mode = True
    net.feedback_scale = torch.tensor(1.0)

    q_true = net._make_bump(torch.tensor([theta_stim]), sigma=oracle_sigma)
    pi_tensor = torch.full((1, 1), oracle_pi)
    net.oracle_q_pred = q_true
    net.oracle_pi_pred = pi_tensor

    state = initial_state(1, N, model_cfg.v2_hidden_dim)

    with torch.no_grad():
        net.l23.cache_kernels()
        if hasattr(net.feedback, 'cache_kernels'):
            net.feedback.cache_kernels()

        for t in range(12):
            r_l4, adaptation = net.l4(stim, state.r_l4, state.r_pv, state.adaptation)
            r_pv = net.pv(r_l4, state.r_l23, state.r_pv)
            pi_eff = pi_tensor * net.feedback_scale

            # NEW code path: r_l4 passed → coincidence gate
            som_drive, vip_drive, apical_gain = net.feedback(q_true, pi_eff, r_l4=r_l4)
            r_vip = net.vip(vip_drive, state.r_vip)
            effective_som_drive = F.relu(som_drive - F.softplus(net.w_vip_som) * r_vip)
            center_exc = net.w_template_drive * net.deep_template(q_true, pi_eff) if hasattr(net, 'w_template_drive') else torch.zeros(1, N)
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

    print(f"\n{name} (step 11, r_l4 passed / coincidence gate):")
    print(f"  ff@stim:      {ff[0, peak_ch].item():.6f}")
    print(f"  gain@stim:    {apical_gain[0, peak_ch].item():.6f}")
    print(f"  exc*gain:     {exc_gained[0, peak_ch].item():.6f}")
    print(f"  som_inh:      {som_inh[0, peak_ch].item():.6f}")
    print(f"  net_drive:    {l23_drive[0, peak_ch].item():.6f}")
    print(f"  r_l23@stim:   {r_l23[0, peak_ch].item():.8f}")
    print(f"  r_som@stim:   {r_som[0, peak_ch].item():.6f}")

# ── EXPERIMENT 3: NEW model drive decomposition (recap) ─────────────
print("\n" + "=" * 90)
print("EXPERIMENT 3: Drive decomposition — NEW models WITH coincidence gate")
print("=" * 90)

for name, path in list(checkpoints.items())[3:]:  # NEW models
    net = LaminarV1V2Network(model_cfg)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net.load_state_dict(ckpt["model_state"], strict=False)
    net.eval()

    net.oracle_mode = True
    net.feedback_scale = torch.tensor(1.0)

    q_true = net._make_bump(torch.tensor([theta_stim]), sigma=oracle_sigma)
    pi_tensor = torch.full((1, 1), oracle_pi)
    net.oracle_q_pred = q_true
    net.oracle_pi_pred = pi_tensor

    state = initial_state(1, N, model_cfg.v2_hidden_dim)

    with torch.no_grad():
        net.l23.cache_kernels()
        if hasattr(net.feedback, 'cache_kernels'):
            net.feedback.cache_kernels()

        for t in range(12):
            r_l4, adaptation = net.l4(stim, state.r_l4, state.r_pv, state.adaptation)
            r_pv = net.pv(r_l4, state.r_l23, state.r_pv)
            pi_eff = pi_tensor * net.feedback_scale

            som_drive, vip_drive, apical_gain = net.feedback(q_true, pi_eff, r_l4=r_l4)
            r_vip = net.vip(vip_drive, state.r_vip)
            effective_som_drive = F.relu(som_drive - F.softplus(net.w_vip_som) * r_vip)
            center_exc = net.w_template_drive * net.deep_template(q_true, pi_eff) if hasattr(net, 'w_template_drive') else torch.zeros(1, N)
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

    print(f"\n{name} (step 11, coincidence gate):")
    print(f"  ff@stim:      {ff[0, peak_ch].item():.6f}")
    print(f"  gain@stim:    {apical_gain[0, peak_ch].item():.6f}")
    print(f"  exc*gain:     {exc_gained[0, peak_ch].item():.6f}")
    print(f"  som_inh:      {som_inh[0, peak_ch].item():.6f}")
    print(f"  net_drive:    {l23_drive[0, peak_ch].item():.6f}")
    print(f"  r_l23@stim:   {r_l23[0, peak_ch].item():.8f}")
    print(f"  r_som@stim:   {r_som[0, peak_ch].item():.6f}")

# ── EXPERIMENT 4: Apical gain OLD pure-top-down vs NEW coincidence ─
print("\n" + "=" * 90)
print("EXPERIMENT 4: Apical gain comparison (pure top-down vs coincidence)")
print("=" * 90)
print("Using OLD ap_s42 checkpoint parameters:")

net = LaminarV1V2Network(model_cfg)
ckpt = torch.load("results/apical/ap_s42/center_surround_seed42/checkpoint.pt",
                   map_location="cpu", weights_only=False)
net.load_state_dict(ckpt["model_state"], strict=False)
net.eval()

q_true = net._make_bump(torch.tensor([theta_stim]), sigma=oracle_sigma)
pi_tensor = torch.full((1, 1), oracle_pi)

with torch.no_grad():
    state = initial_state(1, N, model_cfg.v2_hidden_dim)
    for _ in range(12):
        r_l4, adaptation = net.l4(stim, state.r_l4, state.r_pv, state.adaptation)
        r_pv = net.pv(r_l4, state.r_l23, state.r_pv)
        state = state._replace(r_l4=r_l4, r_pv=r_pv, adaptation=adaptation)

    # Get both gain computations
    net.feedback.cache_kernels()

    q_centered = q_true - q_true.mean(dim=-1, keepdim=True)
    apical_circulant = net.feedback._cached_apical_circulant
    apical_field = (apical_circulant @ q_centered.unsqueeze(-1)).squeeze(-1)
    pi_eff = pi_tensor

    # Pure top-down (old path)
    gain_pure_td = 1.0 + net.feedback.max_apical_gain * torch.tanh(pi_eff * apical_field)

    # Coincidence (new path)
    basal_field = r_l4 - r_l4.mean(dim=-1, keepdim=True)
    coincidence = F.relu(apical_field) * F.relu(basal_field)
    gain_coinc = 1.0 + net.feedback.max_apical_gain * torch.tanh(pi_eff * coincidence)

    net.feedback.uncache_kernels()

print(f"\nAt stimulus channel (ch=18):")
print(f"  apical_field:    {apical_field[0, peak_ch].item():.6f}")
print(f"  basal_field:     {basal_field[0, peak_ch].item():.6f}")
print(f"  coincidence:     {coincidence[0, peak_ch].item():.6f}")
print(f"  gain (pure TD):  {gain_pure_td[0, peak_ch].item():.6f}")
print(f"  gain (coincid):  {gain_coinc[0, peak_ch].item():.6f}")
print(f"\nFull profile comparison (channels 14-22):")
print(f"{'Ch':>4s} {'Deg':>6s}  {'apical_f':>10s} {'basal_f':>10s}  {'gain_TD':>10s} {'gain_CO':>10s}")
for ch in range(14, 23):
    deg = ch * period / N
    af = apical_field[0, ch].item()
    bf = basal_field[0, ch].item()
    gt = gain_pure_td[0, ch].item()
    gc = gain_coinc[0, ch].item()
    marker = " <-- stim" if ch == peak_ch else ""
    print(f"{ch:>4d} {deg:>5.1f}°  {af:>10.6f} {bf:>10.6f}  {gt:>10.6f} {gc:>10.6f}{marker}")

print(f"\nPure top-down gain: peak={gain_pure_td.max().item():.6f}, mean={gain_pure_td.mean().item():.6f}")
print(f"Coincidence gain:   peak={gain_coinc.max().item():.6f}, mean={gain_coinc.mean().item():.6f}")
print(f"\nKey difference: pure TD uses apical_field DIRECTLY in tanh")
print(f"  apical_field ranges [{apical_field.min().item():.6f}, {apical_field.max().item():.6f}]")
print(f"  Coincidence ranges [{coincidence.min().item():.6f}, {coincidence.max().item():.6f}]")
print(f"  pi_eff * apical_field: [{(pi_eff * apical_field).min().item():.4f}, {(pi_eff * apical_field).max().item():.4f}]")
print(f"  pi_eff * coincidence:  [{(pi_eff * coincidence).min().item():.4f}, {(pi_eff * coincidence).max().item():.4f}]")
