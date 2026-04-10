"""Profile analysis: suppression-by-tuning for each feedback mechanism.

Part 1: Direct kernel analysis (feedback mechanism at init params, no network dynamics)
Part 2: Full network simulation with oracle predictor (using Stage 1 checkpoints)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
import math
from pathlib import Path

from src.config import ModelConfig
from src.model.network import LaminarV1V2Network
from src.stimulus.gratings import generate_grating, population_code
from src.utils import circular_distance_abs, circular_gaussian

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

N = 36
PERIOD = 180.0
STEP = PERIOD / N  # 5 deg

# Oracle prediction: peaked at 90° (channel 18)
EXPECTED_CH = 18
EXPECTED_THETA = 90.0

results = {}

# ============================================================================
# PART 1: DIRECT KERNEL ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("PART 1: DIRECT KERNEL ANALYSIS (feedback mechanism at init params)")
print("="*70)

for mech_name in ['dampening', 'sharpening', 'center_surround']:
    cfg = ModelConfig(mechanism=MechanismType(mech_name))
    fb = FeedbackMechanism(cfg).to(device)
    fb.eval()

    # Print raw parameter values
    print(f"\n--- {mech_name.upper()} ---")
    print(f"  surround_gain = {fb.surround_gain.item():.4f}")
    print(f"  surround_width = {fb.surround_width.item():.4f} deg")
    if hasattr(fb, 'center_gain_raw'):
        print(f"  center_gain = {fb.center_gain.item():.4f}")
        print(f"  center_width = {fb.center_width.item():.4f} deg")

    # Create oracle q_pred: Gaussian peaked at 90° (channel 18)
    prefs = torch.arange(N, dtype=torch.float32, device=device) * STEP
    dists = circular_distance_abs(prefs, torch.tensor(EXPECTED_THETA, device=device), PERIOD)
    q_pred = circular_gaussian(dists, sigma=10.0).unsqueeze(0)  # [1, N]
    q_pred = q_pred / q_pred.sum(dim=-1, keepdim=True)  # normalize to distribution

    # Use moderate precision
    pi_pred = torch.tensor([[3.0]], device=device)  # [1, 1]

    # Compute SOM drive
    fb.cache_kernels()
    som_drive = fb.compute_som_drive(q_pred, pi_pred)  # [1, N]
    center_exc = fb.compute_center_excitation(q_pred, pi_pred)  # [1, N]
    fb.uncache_kernels()

    som_drive_np = som_drive[0].detach().cpu()
    center_exc_np = center_exc[0].detach().cpu()

    # Compute angular offsets from expected orientation for each channel
    offsets = []
    for ch in range(N):
        d = abs(ch - EXPECTED_CH) * STEP
        if d > 90:
            d = 180 - d
        offsets.append(d)

    print(f"\n  SOM Drive Profile (Δθ from expected):")
    print(f"  {'Offset':>8} {'SOM_drive':>12} {'Ctr_exc':>12} {'Net_inhib':>12}")
    for offset_deg in [0, 5, 10, 15, 20, 25, 30, 45, 60, 90]:
        matching = [i for i, o in enumerate(offsets) if abs(o - offset_deg) < 2.5]
        if matching:
            avg_som = sum(som_drive_np[i].item() for i in matching) / len(matching)
            avg_ctr = sum(center_exc_np[i].item() for i in matching) / len(matching)
            net = avg_som - avg_ctr  # net inhibition = SOM - center excitation
            print(f"  {offset_deg:>6}° {avg_som:>12.6f} {avg_ctr:>12.6f} {net:>12.6f}")

    # Full channel-by-channel printout
    print(f"\n  Channel-by-channel SOM drive:")
    for ch in range(N):
        marker = " <-- expected" if ch == EXPECTED_CH else ""
        print(f"    ch {ch:>2} ({ch*STEP:>5.1f}°, offset {offsets[ch]:>5.1f}°): "
              f"SOM={som_drive_np[ch].item():>10.6f}  CtrExc={center_exc_np[ch].item():>10.6f}{marker}")

    results[mech_name] = {
        'som_drive': som_drive_np,
        'center_exc': center_exc_np,
        'offsets': offsets,
    }


# ============================================================================
# PART 2: FULL NETWORK SIMULATION WITH ORACLE
# ============================================================================
print("\n\n" + "="*70)
print("PART 2: FULL NETWORK SIMULATION WITH ORACLE PREDICTOR")
print("="*70)

# Check available checkpoints
ckpt_dir = Path("results/ablation")
ckpt_map = {}
for sub in ckpt_dir.iterdir():
    if sub.is_dir():
        for ckpt in sub.rglob("stage1_checkpoint.pt"):
            ckpt_map[sub.name] = str(ckpt)

print(f"\nAvailable checkpoints: {list(ckpt_map.keys())}")

# Map mechanism to a checkpoint (prefer l4l23 variant, then l4, then l23)
def find_ckpt(mech_prefix):
    for suffix in ['l4l23', 'l4', 'l23']:
        key = f"{mech_prefix}_{suffix}"
        if key in ckpt_map:
            return ckpt_map[key], key
    return None, None

# Create oracle q_pred for the full network
prefs = torch.arange(N, dtype=torch.float32, device=device) * STEP
dists = circular_distance_abs(prefs, torch.tensor(EXPECTED_THETA, device=device), PERIOD)
oracle_q = circular_gaussian(dists, sigma=10.0).unsqueeze(0)  # [1, N]
oracle_q = oracle_q / oracle_q.sum(dim=-1, keepdim=True)
oracle_pi = torch.tensor([[3.0]], device=device)

T_SIM = 30  # timesteps per stimulus
CONTRAST = 0.5

for mech_name, mech_prefix in [('dampening', 'damp'), ('center_surround', 'cs')]:
    ckpt_path, ckpt_key = find_ckpt(mech_prefix)
    if ckpt_path is None:
        print(f"\n--- {mech_name.upper()} --- (no checkpoint found, skipping)")
        continue

    # Determine v2_input_mode from ckpt_key
    if ckpt_key.endswith('l4l23'):
        v2_input_mode = 'l4_l23'
    elif ckpt_key.endswith('l4'):
        v2_input_mode = 'l4'
    else:
        v2_input_mode = 'l23'

    cfg = ModelConfig(mechanism=MechanismType(mech_name), v2_input_mode=v2_input_mode)
    net = LaminarV1V2Network(cfg).to(device)

    # Load Stage 1 checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if 'model_state' in ckpt:
        state_dict = ckpt['model_state']
    elif 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt

    # Load with strict=False to handle any missing/extra keys
    missing, unexpected = net.load_state_dict(state_dict, strict=False)
    print(f"\n--- {mech_name.upper()} (checkpoint: {ckpt_key}) ---")
    print(f"  Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    if missing:
        print(f"  Missing: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    # Print feedback params after loading
    fb = net.feedback
    print(f"  surround_gain = {fb.surround_gain.item():.4f}")
    print(f"  surround_width = {fb.surround_width.item():.4f} deg")
    if hasattr(fb, 'center_gain_raw'):
        print(f"  center_gain = {fb.center_gain.item():.4f}")
        print(f"  center_width = {fb.center_width.item():.4f} deg")

    net.eval()

    responses_oracle = []  # [N_stim, N_channels] — L2/3 steady-state with oracle
    responses_neutral = [] # [N_stim, N_channels] — L2/3 steady-state without oracle

    for ori_idx in range(N):
        theta = torch.tensor([ori_idx * STEP], device=device)
        contrast = torch.tensor([CONTRAST], device=device)
        stim = generate_grating(theta, contrast)  # [1, N]
        stim_seq = stim.unsqueeze(1).expand(1, T_SIM, N)  # [1, T, N]

        # --- With oracle prediction ---
        net.oracle_mode = True
        net.oracle_q_pred = oracle_q  # [1, N] — constant per step
        net.oracle_pi_pred = oracle_pi  # [1, 1]

        with torch.no_grad():
            r_l23_all, final_state, aux = net(stim_seq)
        # Steady-state: average last 10 steps
        resp_oracle = r_l23_all[0, -10:, :].mean(dim=0)  # [N]
        responses_oracle.append(resp_oracle.cpu())

        # --- Without oracle (neutral) ---
        net.oracle_mode = False
        net.oracle_q_pred = None
        net.oracle_pi_pred = None

        with torch.no_grad():
            r_l23_neu, _, _ = net(stim_seq)
        resp_neu = r_l23_neu[0, -10:, :].mean(dim=0)
        responses_neutral.append(resp_neu.cpu())

    responses_oracle = torch.stack(responses_oracle)   # [36, 36]
    responses_neutral = torch.stack(responses_neutral)  # [36, 36]

    # -------------------------------------------------------------------
    # Profile 1: STIMULUS-ALIGNED analysis
    # For each stimulus orientation, look at the L2/3 response of the unit
    # tuned to that orientation (the "preferred" unit).
    # Then compute suppression = (neutral - oracle) / neutral
    # as a function of angular distance of stimulus from expected.
    # -------------------------------------------------------------------
    print(f"\n  STIMULUS-ALIGNED PROFILE (response of preferred unit per stimulus):")
    print(f"  {'Δθ_stim':>8} {'R_oracle':>10} {'R_neutral':>10} {'Delta':>10} {'Suppression%':>12}")

    offsets = []
    for s_idx in range(N):
        d = abs(s_idx - EXPECTED_CH) * STEP
        if d > 90:
            d = 180 - d
        offsets.append(d)

    for offset_deg in [0, 5, 10, 15, 20, 25, 30, 45, 60, 90]:
        matching = [i for i, o in enumerate(offsets) if abs(o - offset_deg) < 2.5]
        if matching:
            # For each matched stimulus, take the response of the unit tuned to that stimulus
            avg_oracle = sum(responses_oracle[i, i].item() for i in matching) / len(matching)
            avg_neutral = sum(responses_neutral[i, i].item() for i in matching) / len(matching)
            delta = avg_oracle - avg_neutral
            supp_pct = -delta / max(abs(avg_neutral), 1e-8) * 100
            print(f"  {offset_deg:>6}° {avg_oracle:>10.4f} {avg_neutral:>10.4f} {delta:>10.4f} {supp_pct:>10.1f}%")

    # -------------------------------------------------------------------
    # Profile 2: UNIT-ALIGNED analysis (fixed stimulus = expected orientation)
    # Present the EXPECTED stimulus (90°), look at all units.
    # This shows how the oracle affects L2/3 across the population
    # when the stimulus matches the expectation.
    # -------------------------------------------------------------------
    stim_at_expected = EXPECTED_CH  # stimulus index for 90°
    r_oracle_at_exp = responses_oracle[stim_at_expected]  # [N]
    r_neutral_at_exp = responses_neutral[stim_at_expected]  # [N]
    delta_at_exp = r_oracle_at_exp - r_neutral_at_exp  # [N]

    print(f"\n  UNIT-ALIGNED PROFILE (stimulus=90°, response across all units):")
    print(f"  {'Δθ_unit':>8} {'R_oracle':>10} {'R_neutral':>10} {'Delta':>10} {'Suppression%':>12}")
    for offset_deg in [0, 5, 10, 15, 20, 25, 30, 45, 60, 90]:
        matching = [i for i, o in enumerate(offsets) if abs(o - offset_deg) < 2.5]
        if matching:
            avg_o = sum(r_oracle_at_exp[i].item() for i in matching) / len(matching)
            avg_n = sum(r_neutral_at_exp[i].item() for i in matching) / len(matching)
            d = avg_o - avg_n
            sp = -d / max(abs(avg_n), 1e-8) * 100
            print(f"  {offset_deg:>6}° {avg_o:>10.4f} {avg_n:>10.4f} {d:>10.4f} {sp:>10.1f}%")

    # -------------------------------------------------------------------
    # Profile 3: SUPPRESSION-BY-TUNING (Keller et al. style)
    # For each stimulus, compute suppression index = (neutral - oracle) / neutral
    # for the unit tuned TO that stimulus.
    # Plot as a function of Δθ_stim from expected.
    # This is the key experimental observable.
    # -------------------------------------------------------------------
    print(f"\n  SUPPRESSION-BY-TUNING PROFILE (Keller et al. style):")
    print(f"  Suppression index = (neutral - oracle) / neutral for preferred unit")
    print(f"  {'Δθ_stim':>8} {'SuppIdx':>10}")

    supp_indices = []
    for s_idx in range(N):
        r_o = responses_oracle[s_idx, s_idx].item()
        r_n = responses_neutral[s_idx, s_idx].item()
        si = (r_n - r_o) / max(abs(r_n), 1e-8)
        supp_indices.append((offsets[s_idx], si))

    # Sort by offset and print
    supp_indices.sort(key=lambda x: x[0])
    for offset_deg in sorted(set(offsets)):
        matching = [(o, s) for o, s in supp_indices if abs(o - offset_deg) < 0.1]
        if matching:
            avg_si = sum(s for _, s in matching) / len(matching)
            print(f"  {offset_deg:>6}° {avg_si:>10.4f}")

    results[f'{mech_name}_network'] = {
        'responses_oracle': responses_oracle,
        'responses_neutral': responses_neutral,
        'offsets': offsets,
    }


# ============================================================================
# PART 3: SHARPENING (no checkpoint, use init params with oracle)
# ============================================================================
print("\n\n" + "="*70)
print("PART 3: SHARPENING (init params, no Stage 1 checkpoint)")
print("="*70)

cfg = ModelConfig(mechanism=MechanismType.SHARPENING)
net = LaminarV1V2Network(cfg).to(device)
net.eval()

fb = net.feedback
print(f"  surround_gain = {fb.surround_gain.item():.4f}")
print(f"  surround_width = {fb.surround_width.item():.4f} deg")
print(f"  center_gain = {fb.center_gain.item():.4f}")
print(f"  center_width = {fb.center_width.item():.4f} deg")

responses_oracle_sharp = []
responses_neutral_sharp = []

for ori_idx in range(N):
    theta = torch.tensor([ori_idx * STEP], device=device)
    contrast = torch.tensor([CONTRAST], device=device)
    stim = generate_grating(theta, contrast)
    stim_seq = stim.unsqueeze(1).expand(1, T_SIM, N)

    net.oracle_mode = True
    net.oracle_q_pred = oracle_q
    net.oracle_pi_pred = oracle_pi
    with torch.no_grad():
        r_l23_all, _, _ = net(stim_seq)
    resp_o = r_l23_all[0, -10:, :].mean(dim=0)
    responses_oracle_sharp.append(resp_o.cpu())

    net.oracle_mode = False
    net.oracle_q_pred = None
    net.oracle_pi_pred = None
    with torch.no_grad():
        r_l23_n, _, _ = net(stim_seq)
    resp_n = r_l23_n[0, -10:, :].mean(dim=0)
    responses_neutral_sharp.append(resp_n.cpu())

responses_oracle_sharp = torch.stack(responses_oracle_sharp)
responses_neutral_sharp = torch.stack(responses_neutral_sharp)

print(f"\n  SUPPRESSION-BY-TUNING PROFILE (Keller et al. style):")
print(f"  {'Δθ_stim':>8} {'R_oracle':>10} {'R_neutral':>10} {'SuppIdx':>10}")
for offset_deg in [0, 5, 10, 15, 20, 25, 30, 45, 60, 90]:
    matching = [i for i, o in enumerate(offsets) if abs(o - offset_deg) < 2.5]
    if matching:
        avg_o = sum(responses_oracle_sharp[i, i].item() for i in matching) / len(matching)
        avg_n = sum(responses_neutral_sharp[i, i].item() for i in matching) / len(matching)
        si = (avg_n - avg_o) / max(abs(avg_n), 1e-8)
        print(f"  {offset_deg:>6}° {avg_o:>10.4f} {avg_n:>10.4f} {si:>10.4f}")

# Print unit-aligned for sharpening too
stim_at_expected = EXPECTED_CH
r_o_e = responses_oracle_sharp[stim_at_expected]
r_n_e = responses_neutral_sharp[stim_at_expected]
d_e = r_o_e - r_n_e

print(f"\n  UNIT-ALIGNED PROFILE (stimulus=90°, response across all units):")
print(f"  {'Δθ_unit':>8} {'R_oracle':>10} {'R_neutral':>10} {'Delta':>10}")
for offset_deg in [0, 5, 10, 15, 20, 25, 30, 45, 60, 90]:
    matching = [i for i, o in enumerate(offsets) if abs(o - offset_deg) < 2.5]
    if matching:
        avg_o = sum(r_o_e[i].item() for i in matching) / len(matching)
        avg_n = sum(r_n_e[i].item() for i in matching) / len(matching)
        d = avg_o - avg_n
        print(f"  {offset_deg:>6}° {avg_o:>10.4f} {avg_n:>10.4f} {d:>10.4f}")


# ============================================================================
# SUMMARY
# ============================================================================
print("\n\n" + "="*70)
print("SUMMARY: Expected vs Observed Profile Shapes")
print("="*70)
print("""
Expected profiles (from mechanism definitions):

DAMPENING (Model A):
  - SOM drive peaks AT expected orientation (narrow Gaussian kernel)
  - Should produce: strongest suppression at Δθ=0°, recovering at large offsets
  - Key signature: monotonically decreasing suppression with Δθ

SHARPENING (Model B):
  - SOM drive = broad - narrow (DoG): minimum at expected, maximum at flanks
  - Should produce: weak/no suppression at Δθ=0°, strongest suppression at ~20-30°
  - Key signature: suppression peaks at INTERMEDIATE Δθ, not at 0°

CENTER-SURROUND (Model C):
  - SOM = broad inhibition, L2/3 excitation = narrow
  - Should produce: facilitation at Δθ=0° (center excitation > inhibition),
    suppression at intermediate Δθ, recovery at large Δθ
  - Key signature: FACILITATION at 0°, suppression at intermediate Δθ
""")

print("Done.")
