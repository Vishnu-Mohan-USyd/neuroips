"""Debugger: Investigate why head_feedback learns nothing.

6 questions:
1. Does head_feedback output anything nonzero?
2. Does gradient reach head_feedback.weight?
3. ReLU hypothesis: fraction of positive l23_drive channels
4. feedback_scale hypothesis: enough gradient steps?
5. pred_suppress gradient chain for PS experiment
6. Architectural dead end or just too small gradient?
"""

import sys
sys.path.insert(0, "/mnt/c/Users/User/codingproj/freshstart")

import torch
import torch.nn.functional as F
from src.config import load_config
from src.model.network import LaminarV1V2Network
from src.state import initial_state
from src.stimulus.gratings import generate_grating
from src.training.losses import CompositeLoss


def investigate(config_path, ckpt_path, label):
    """Full gradient flow investigation for one checkpoint."""
    print(f"\n{'='*80}")
    print(f"INVESTIGATION: {label}")
    print(f"  config: {config_path}")
    print(f"  checkpoint: {ckpt_path}")
    print(f"{'='*80}")

    model_cfg, train_cfg, stim_cfg = load_config(config_path)
    N = model_cfg.n_orientations
    period = model_cfg.orientation_range
    step_deg = period / N
    H = model_cfg.v2_hidden_dim

    net = LaminarV1V2Network(model_cfg, delta_som=train_cfg.delta_som)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net.load_state_dict(ckpt["model_state"], strict=False)

    # ================================================================
    # Q1: Does head_feedback output anything nonzero?
    # ================================================================
    print(f"\n[Q1] head_feedback weights and output magnitude")

    hf_w = net.v2.head_feedback.weight.data
    hf_b = net.v2.head_feedback.bias.data
    init_scale = 1.0 / (H ** 0.5)
    print(f"  head_feedback.weight: norm={hf_w.norm().item():.6f}, "
          f"abs_mean={hf_w.abs().mean().item():.6f}, max={hf_w.abs().max().item():.6f}")
    print(f"  head_feedback.bias: norm={hf_b.norm().item():.6f}, "
          f"abs_mean={hf_b.abs().mean().item():.6f}")
    print(f"  Init scale (1/sqrt(H={H})): {init_scale:.6f}")
    print(f"  Weight abs_mean / init_scale = {hf_w.abs().mean().item() / init_scale:.4f}")

    # Compare heads
    print(f"\n  All V2 heads (norm / abs_mean):")
    for name, mod in [("head_mu", net.v2.head_mu),
                      ("head_pi", net.v2.head_pi),
                      ("head_feedback", net.v2.head_feedback)]:
        w = mod.weight.data
        print(f"    {name:20s}  norm={w.norm().item():.4f}  abs_mean={w.abs().mean().item():.6f}")

    # Run forward pass (NOT oracle mode) to see actual output
    theta_stim = 90.0
    stim = generate_grating(torch.tensor([theta_stim]), torch.tensor([0.5]),
                            N, model_cfg.sigma_ff, period=period)
    cue = torch.zeros(1, N)  # cue is N-dimensional (orientation-coded)
    task = torch.zeros(1, 2)

    net.eval()
    net.oracle_mode = False
    net.feedback_scale.fill_(1.0)

    state = initial_state(1, N, H)
    fb_outputs = []
    h_v2_norms = []

    with torch.no_grad():
        for t in range(train_cfg.steps_on):
            state, aux = net.step(stim, cue, task, state)
            # Capture feedback_signal by running V2 manually
            mu, pi_raw, fb_sig, h = net.v2(
                state.r_l4, state.r_l23, cue, task, state.h_v2
            )
            fb_outputs.append(fb_sig[0].clone())
            h_v2_norms.append(state.h_v2[0].norm().item())

    fb_stack = torch.stack(fb_outputs)
    print(f"\n  Feedback signal over {train_cfg.steps_on} timesteps:")
    for t in [0, 3, 6, 9, 11]:
        if t < len(fb_outputs):
            fb_t = fb_outputs[t]
            print(f"    t={t:2d}: abs_mean={fb_t.abs().mean().item():.6f}, "
                  f"abs_max={fb_t.abs().max().item():.6f}, "
                  f"h_v2_norm={h_v2_norms[t]:.4f}")

    # Check: is the feedback signal comparable to the L2/3 drive?
    r_l23_final = state.r_l23[0]
    print(f"\n  Final r_l23 peak: {r_l23_final.max().item():.6f}")
    print(f"  Final fb_signal abs_mean: {fb_outputs[-1].abs().mean().item():.6f}")
    print(f"  Ratio fb/r_l23_peak: {fb_outputs[-1].abs().mean().item() / max(r_l23_final.max().item(), 1e-8):.6f}")

    # ================================================================
    # Q6 (moved up): Weight change from stage1 to final
    # ================================================================
    print(f"\n[Q6] Weight change from stage1 → final checkpoint")
    stage1_path = ckpt_path.replace("checkpoint.pt", "stage1_checkpoint.pt")
    try:
        stage1_ckpt = torch.load(stage1_path, map_location="cpu", weights_only=False)
        s1 = stage1_ckpt["model_state"]
        s2 = ckpt["model_state"]
        for name in ["v2.head_feedback.weight", "v2.head_feedback.bias",
                      "v2.head_mu.weight", "v2.head_mu.bias",
                      "v2.head_pi.weight", "v2.head_pi.bias",
                      "v2.gru.weight_ih", "v2.gru.weight_hh",
                      "v2.gru.bias_ih", "v2.gru.bias_hh"]:
            if name in s1 and name in s2:
                w1, w2 = s1[name], s2[name]
                diff = (w2 - w1).norm().item()
                rel = diff / max(w1.norm().item(), 1e-8) * 100
                print(f"    {name:35s}  diff_norm={diff:.6f}  rel={rel:.2f}%")
    except FileNotFoundError:
        print(f"    Stage1 checkpoint not found")

    # ================================================================
    # Q2: Gradient flow check — test each loss independently
    # ================================================================
    print(f"\n[Q2] Gradient flow to head_feedback.weight (per loss)")

    loss_fn = CompositeLoss(train_cfg, model_cfg)
    loss_fn.train()

    for loss_name in ["sensory", "energy", "pred_suppress", "combined"]:
        net.train()
        net.oracle_mode = False
        net.feedback_scale.fill_(1.0)
        net.zero_grad()

        state = initial_state(1, N, H)
        # Run forward
        for t in range(train_cfg.steps_on):
            state, aux = net.step(stim, cue, task, state)

        # Compute target loss
        r_l23_final = state.r_l23  # [1, N]
        q_pred_final = aux.q_pred  # [1, N]
        true_ch = int(theta_stim / step_deg)

        if loss_name == "sensory":
            logits = loss_fn.orientation_decoder(r_l23_final)
            target = torch.full((1,), true_ch, dtype=torch.long)
            loss = F.cross_entropy(logits, target)
        elif loss_name == "energy":
            loss = r_l23_final.abs().mean()
        elif loss_name == "pred_suppress":
            loss = (r_l23_final * q_pred_final).sum(dim=-1).mean()
        elif loss_name == "combined":
            logits = loss_fn.orientation_decoder(r_l23_final)
            target = torch.full((1,), true_ch, dtype=torch.long)
            l_sens = F.cross_entropy(logits, target)
            l_energy = r_l23_final.abs().mean()
            l_pred = (r_l23_final * q_pred_final).sum(dim=-1).mean()
            loss = (train_cfg.lambda_sensory * l_sens +
                    train_cfg.lambda_energy * l_energy +
                    getattr(train_cfg, 'lambda_pred_suppress', 0.0) * l_pred)

        loss.backward()

        # Collect gradient info
        grad_info = {}
        for param_name, param in [
            ("head_feedback.weight", net.v2.head_feedback.weight),
            ("head_feedback.bias", net.v2.head_feedback.bias),
            ("head_mu.weight", net.v2.head_mu.weight),
            ("head_pi.weight", net.v2.head_pi.weight),
            ("gru.weight_ih", net.v2.gru.weight_ih),
            ("gru.weight_hh", net.v2.gru.weight_hh),
        ]:
            g = param.grad
            if g is not None:
                grad_info[param_name] = {
                    "norm": g.norm().item(),
                    "abs_mean": g.abs().mean().item(),
                    "max": g.abs().max().item(),
                }
            else:
                grad_info[param_name] = None

        print(f"\n  Loss '{loss_name}' = {loss.item():.6f}")
        for pname, ginfo in grad_info.items():
            if ginfo is not None:
                print(f"    {pname:25s} grad_norm={ginfo['norm']:.8f}  "
                      f"abs_mean={ginfo['abs_mean']:.8f}  max={ginfo['max']:.8f}")
            else:
                print(f"    {pname:25s} grad = NONE")

        # Key diagnostic: compare gradient magnitude to weight magnitude
        hf_g = net.v2.head_feedback.weight.grad
        if hf_g is not None:
            lr = train_cfg.stage2_lr_v2
            update_size = lr * hf_g.abs().mean().item()
            weight_size = hf_w.abs().mean().item()
            print(f"    → Per-step update size: lr×|grad| = {lr}×{hf_g.abs().mean().item():.8f} = {update_size:.10f}")
            print(f"    → Weight abs_mean: {weight_size:.6f}")
            print(f"    → Update/weight ratio: {update_size/max(weight_size, 1e-10):.8f}")
            steps_with_grad = train_cfg.stage2_n_steps - train_cfg.stage2_burnin_steps
            total_update = update_size * steps_with_grad
            print(f"    → Total cumulative update ({steps_with_grad} steps): {total_update:.6f}")
            print(f"    → Total/weight ratio: {total_update/max(weight_size, 1e-10):.4f}")

    # ================================================================
    # Q3: ReLU hypothesis
    # ================================================================
    print(f"\n[Q3] ReLU/rectification analysis")

    net.eval()
    net.oracle_mode = False
    net.feedback_scale.fill_(1.0)
    state = initial_state(1, N, H)

    # Run forward, manually check L2/3 drive at each step
    with torch.no_grad():
        for t in range(train_cfg.steps_on):
            # Get L4 and PV
            r_l4, adaptation = net.l4(stim, state.r_l4, state.r_pv, state.adaptation)
            r_pv = net.pv(r_l4, state.r_l23, state.r_pv)

            # Get V2 output
            mu_pred, pi_raw, fb_sig, h_v2 = net.v2(
                r_l4, state.r_l23, cue, task, state.h_v2
            )
            center_exc = fb_sig * net.feedback_scale

            # Compute L2/3 drive manually
            ff = F.linear(r_l4, net.l23.W_l4_to_l23)
            rec = F.linear(state.r_l23, net.l23.W_rec)
            exc_drive = ff + rec + center_exc
            pv_inh = net.l23.w_pv_l23(r_pv)
            # SOM = 0 in simple_feedback mode
            l23_drive = exc_drive - pv_inh

            n_pos = (l23_drive[0] > 0).sum().item()

            if t in [0, 3, 6, 9, 11]:
                stim_ch = int(theta_stim / step_deg)
                print(f"\n  t={t}:")
                print(f"    Positive drive channels: {n_pos}/{N} ({100*n_pos/N:.1f}%)")
                print(f"    Drive range: [{l23_drive[0].min().item():.4f}, {l23_drive[0].max().item():.4f}]")
                print(f"    center_exc range: [{center_exc[0].min().item():.6f}, {center_exc[0].max().item():.6f}]")
                print(f"    ff range: [{ff[0].min().item():.4f}, {ff[0].max().item():.4f}]")

                # Channel-by-channel at stimulus neighborhood
                for dist_deg in [-15, -10, -5, 0, 5, 10, 15, 20, 25]:
                    ch = stim_ch + int(dist_deg / step_deg)
                    if 0 <= ch < N:
                        print(f"      d={dist_deg:+3d}°  drive={l23_drive[0,ch].item():+.4f}  "
                              f"ff={ff[0,ch].item():.4f}  rec={rec[0,ch].item():.4f}  "
                              f"ce={center_exc[0,ch].item():+.6f}  "
                              f"{'ALIVE' if l23_drive[0,ch].item() > 0 else 'DEAD'}")

            # Step forward
            r_som = torch.zeros_like(state.r_som)
            r_l23 = net.l23(r_l4, state.r_l23, center_exc, r_som, r_pv)
            deep_tmpl = net.deep_template(mu_pred, pi_raw * net.feedback_scale)

            state = state._replace(
                r_l4=r_l4, r_l23=r_l23, r_pv=r_pv,
                h_v2=h_v2, r_som=r_som, adaptation=adaptation,
                deep_template=deep_tmpl, r_vip=torch.zeros_like(state.r_vip)
            )

    # ================================================================
    # Q4: feedback_scale timing
    # ================================================================
    print(f"\n[Q4] feedback_scale timing")
    burnin = train_cfg.stage2_burnin_steps
    ramp = train_cfg.stage2_ramp_steps
    total = train_cfg.stage2_n_steps
    print(f"  burnin: {burnin} steps (fb_scale=0, NO gradient to head_feedback)")
    print(f"  ramp: {ramp} steps (fb_scale 0→1, WEAK gradient)")
    print(f"  full: {total - burnin - ramp} steps (fb_scale=1, FULL gradient)")
    print(f"  Total steps: {total}")
    print(f"  Fraction with any gradient: {(total - burnin)/total:.1%}")

    # ================================================================
    # Q5: Is the gradient direction correct?
    # ================================================================
    print(f"\n[Q5] Gradient direction analysis")

    # For energy loss: gradient should push head_feedback to output NEGATIVE
    # values (to reduce r_l23)
    net.train()
    net.oracle_mode = False
    net.feedback_scale.fill_(1.0)
    net.zero_grad()
    state = initial_state(1, N, H)
    for t in range(train_cfg.steps_on):
        state, aux = net.step(stim, cue, task, state)
    loss_energy = state.r_l23.abs().mean()
    loss_energy.backward()

    hf_grad_energy = net.v2.head_feedback.bias.grad
    if hf_grad_energy is not None:
        # The bias gradient tells us: which output channels want to be more negative?
        # Positive grad → decrease bias → more negative feedback at that channel
        # (because optimizer does: bias -= lr * grad)
        print(f"  Energy loss bias gradient (should be positive → push bias negative):")
        n_pos_grad = (hf_grad_energy > 0).sum().item()
        print(f"    {n_pos_grad}/{N} channels have positive grad ({100*n_pos_grad/N:.1f}%)")
        print(f"    Mean grad: {hf_grad_energy.mean().item():+.8f}")
    else:
        print(f"  Energy loss: NO gradient on head_feedback.bias")

    print(f"\n{'='*80}")
    print(f"END INVESTIGATION: {label}")
    print(f"{'='*80}")


def main():
    configs = [
        ("config/exp_v2_direct_highenergy.yaml",
         "results/iter/v2_highenergy/center_surround_seed42/checkpoint.pt",
         "V2 Direct + High Energy (λ_energy=2.0)"),
        ("config/exp_v2_direct_predsup.yaml",
         "results/iter/v2_predsup/center_surround_seed42/checkpoint.pt",
         "V2 Direct + Pred Suppress (λ_pred_suppress=1.0)"),
    ]

    import os
    for config_path, ckpt_path, label in configs:
        if os.path.exists(ckpt_path):
            investigate(config_path, ckpt_path, label)
        else:
            print(f"\n[WAITING] {label}: checkpoint not at {ckpt_path}")


if __name__ == "__main__":
    main()
