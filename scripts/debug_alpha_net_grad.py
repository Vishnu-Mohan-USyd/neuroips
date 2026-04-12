"""Debugger experiments for Phase 2 alpha_net gate stuck at identity.

Runs four experiments to test hypotheses H1-H6 from the debugger brief:

  E1 direct gradient measurement on Phase 2 checkpoint (tests H5, H6)
  E2 per-regime gradient decomposition                (tests H3)
  E3 training dynamics probe from stage1 ckpt          (tests H1, H2)
  E4 identity vs "perfect" gate loss                  (tests H4)

No fixes, no opinions — pure measurement. Run on the remote machine where
the Phase 2 checkpoint lives. All outputs printed verbatim.
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import load_config
from src.model.network import LaminarV1V2Network
from src.training.losses import CompositeLoss
from src.stimulus.sequences import HMMSequenceGenerator
from src.training.trainer import (
    build_stimulus_sequence,
    compute_readout_indices,
    extract_readout_data,
    freeze_stage1,
    unfreeze_stage2,
    create_stage2_optimizer,
    make_warmup_cosine_scheduler,
)
from src.training.stage2_feedback import compute_mismatch_labels


SWEEP = os.path.join(ROOT, "config", "sweep", "sweep_dual_2.yaml")


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------

def build_model_and_loss(device):
    model_cfg, train_cfg, stim_cfg = load_config(SWEEP)
    net = LaminarV1V2Network(model_cfg).to(device)
    loss_fn = CompositeLoss(train_cfg, model_cfg).to(device)
    return model_cfg, train_cfg, stim_cfg, net, loss_fn


def load_checkpoint(net, loss_fn, ckpt_path, device, strict=True):
    ckpt = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(ckpt["model_state"], strict=strict)
    if "decoder_state" in ckpt:
        loss_fn.orientation_decoder.load_state_dict(ckpt["decoder_state"])
    return ckpt


# ----------------------------------------------------------------------
# Batch builder — forces exactly half focused / half routine
# ----------------------------------------------------------------------

def make_balanced_batch(model_cfg, train_cfg, stim_cfg, dev, B=16, seed=7):
    """Generate an HMM batch, then overwrite task_states to half focused, half routine.

    Returns dict of tensors ready for forward + loss, plus the task_state_batch
    [B, 2] for loss routing and a boolean mask identifying focused samples.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    hmm_gen = HMMSequenceGenerator(
        n_orientations=model_cfg.n_orientations,
        p_self=stim_cfg.p_self,
        p_transition_cw=stim_cfg.p_transition_cw,
        p_transition_ccw=stim_cfg.p_transition_ccw,
        n_anchors=stim_cfg.n_anchors,
        jitter_range=stim_cfg.jitter_range,
        transition_step=stim_cfg.transition_step,
        period=model_cfg.orientation_range,
        contrast_range=train_cfg.stage2_contrast_range,
        ambiguous_fraction=train_cfg.ambiguous_fraction,
        ambiguous_offset=stim_cfg.ambiguous_offset,
        n_states=stim_cfg.n_states,
        cue_valid_fraction=stim_cfg.cue_valid_fraction,
    )
    metadata = hmm_gen.generate(B, train_cfg.seq_length, gen)

    # Force balanced task_state: half focused [1,0], half routine [0,1]
    ts = torch.zeros(B, train_cfg.seq_length, 2)
    ts[: B // 2, :, 0] = 1.0    # focused
    ts[B // 2 :, :, 1] = 1.0    # routine
    metadata.task_states = ts

    stim_seq, cue_seq, task_seq, true_thetas, true_next_thetas, true_states = (
        build_stimulus_sequence(metadata, model_cfg, train_cfg, stim_cfg)
    )
    if train_cfg.stimulus_noise > 0:
        stim_seq = stim_seq + train_cfg.stimulus_noise * torch.randn_like(stim_seq)
        stim_seq = stim_seq.clamp(min=0.0)

    stim_seq = stim_seq.to(dev)
    cue_seq = cue_seq.to(dev)
    task_seq = task_seq.to(dev)
    true_thetas = true_thetas.to(dev)
    true_next_thetas = true_next_thetas.to(dev)
    true_states = true_states.to(dev)

    focused_mask = task_seq[:, 0, 0] > 0.5   # [B]

    return dict(
        stim_seq=stim_seq, cue_seq=cue_seq, task_seq=task_seq,
        true_thetas=true_thetas, true_next_thetas=true_next_thetas,
        true_states=true_states, metadata=metadata, focused_mask=focused_mask,
    )


# ----------------------------------------------------------------------
# Loss wrapper matching stage2_feedback.run_stage2 exactly
# ----------------------------------------------------------------------

def stage2_forward_loss(net, loss_fn, batch, model_cfg, train_cfg, stim_cfg, fb_scale_override=None):
    """One stage2 forward + loss. Returns (total_loss, loss_dict, aux, packed_outputs)."""
    if fb_scale_override is not None:
        net.feedback_scale.fill_(fb_scale_override)

    packed = net.pack_inputs(batch["stim_seq"], batch["cue_seq"], batch["task_seq"])
    r_l23_all, _, aux = net(packed)

    outputs = {
        "r_l23": r_l23_all,
        "q_pred": aux["q_pred_all"],
        "r_l4": aux["r_l4_all"],
        "r_pv": aux["r_pv_all"],
        "r_som": aux["r_som_all"],
        "deep_template": aux["deep_template_all"],
        "state_logits": aux["state_logits_all"],
        "p_cw": aux["p_cw_all"],
        "center_exc": aux["center_exc_all"],
    }

    readout_indices = compute_readout_indices(
        train_cfg.seq_length, train_cfg.steps_on, train_cfg.steps_isi,
        window_start=max(0, train_cfg.steps_on - 3),
        window_end=train_cfg.steps_on - 1,
    )
    r_l23_windows, q_pred_windows, state_logits_windows = extract_readout_data(
        outputs, readout_indices,
        steps_on=train_cfg.steps_on, steps_isi=train_cfg.steps_isi,
    )

    task_state_batch = batch["task_seq"][:, 0, :]  # [B, 2]

    total_loss, loss_dict = loss_fn(
        outputs, batch["true_thetas"], batch["true_next_thetas"],
        r_l23_windows, q_pred_windows,
        state_logits_windows=None,  # emergent mode
        true_states_windows=batch["true_states"],
        p_cw_windows=None,
        model=net,
        fb_scale=net.feedback_scale.item(),
        task_state=task_state_batch,
        task_routing=train_cfg.task_routing,
    )
    return total_loss, loss_dict, aux


# ----------------------------------------------------------------------
# Experiment 1 — gradient magnitudes (H5, H6)
# ----------------------------------------------------------------------

def experiment_1(net, loss_fn, model_cfg, train_cfg, stim_cfg, dev):
    print("\n" + "=" * 72)
    print("EXPERIMENT 1 — direct gradient measurement (tests H5 path, H6 magnitude)")
    print("=" * 72)

    # Critical: ensure alpha_net has requires_grad
    for p in net.alpha_net.parameters():
        p.requires_grad_(True)
    net.train()
    net.feedback_scale.fill_(1.0)

    batch = make_balanced_batch(model_cfg, train_cfg, stim_cfg, dev, B=16, seed=7)

    # Hook to grab intermediate activations
    captured = {}

    def hook_fb(mod, args, out):
        # v2 returns (mu_pred, pi_pred_raw, feedback_signal, h_v2)
        captured["fb_raw"] = out[2].detach()
        return out
    h1 = net.v2.register_forward_hook(hook_fb)

    try:
        # Zero grads
        for p in net.parameters():
            if p.grad is not None:
                p.grad.zero_()

        total_loss, loss_dict, aux = stage2_forward_loss(
            net, loss_fn, batch, model_cfg, train_cfg, stim_cfg, fb_scale_override=1.0
        )
        total_loss.backward()
    finally:
        h1.remove()

    # Inspect
    w_grad = net.alpha_net.weight.grad
    b_grad = net.alpha_net.bias.grad
    w = net.alpha_net.weight.data
    b = net.alpha_net.bias.data

    fb_raw = captured.get("fb_raw", None)
    center_exc = aux["center_exc_all"]  # [B, T, N]
    gains_all = aux["gains_all"]        # [B, T, 2]

    print(f"\nLoss total: {total_loss.item():.6f}")
    print(f"Loss breakdown: {loss_dict}")
    print("\n-- alpha_net state (post-training) --")
    print(f"  weight.data:\n{w.cpu().numpy()}")
    print(f"  weight.abs().max() = {w.abs().max().item():.6f}  (init std was 0.01)")
    print(f"  bias.data: {b.cpu().numpy()}  ({'nonzero' if b.abs().max().item() > 1e-6 else 'zero'})")

    print("\n-- alpha_net gradients on Stage 2 loss (focused+routine balanced batch) --")
    if w_grad is None:
        print("  !!! weight.grad is None — alpha_net NOT in autograd graph (H5 CONFIRMED)")
    else:
        print(f"  weight.grad:\n{w_grad.cpu().numpy()}")
        print(f"  weight.grad.abs().max()  = {w_grad.abs().max().item():.3e}")
        print(f"  weight.grad.norm()       = {w_grad.norm().item():.3e}")
        print(f"  bias.grad = {b_grad.cpu().numpy()}")
        print(f"  bias.grad.norm() = {b_grad.norm().item():.3e}")

    print("\n-- feedback signal and downstream magnitudes (tests H6) --")
    if fb_raw is not None:
        print(f"  |feedback_signal| mean = {fb_raw.abs().mean().item():.6f}")
        print(f"  |feedback_signal| max  = {fb_raw.abs().max().item():.6f}")
        print(f"  feedback_signal nonzero frac = "
              f"{(fb_raw.abs() > 1e-6).float().mean().item():.4f}")
    print(f"  |center_exc| mean = {center_exc.abs().mean().item():.6f}")
    print(f"  |center_exc| max  = {center_exc.abs().max().item():.6f}")
    print(f"  center_exc nonzero frac = "
          f"{(center_exc.abs() > 1e-6).float().mean().item():.4f}")
    print(f"  gains[..., 0] (g_E): mean={gains_all[..., 0].mean().item():.4f} "
          f"std={gains_all[..., 0].std().item():.4f}")
    print(f"  gains[..., 1] (g_I): mean={gains_all[..., 1].mean().item():.4f} "
          f"std={gains_all[..., 1].std().item():.4f}")

    # Compare focused vs routine gains
    fm = batch["focused_mask"]
    gE_focused = gains_all[fm, :, 0].mean().item()
    gE_routine = gains_all[~fm, :, 0].mean().item()
    gI_focused = gains_all[fm, :, 1].mean().item()
    gI_routine = gains_all[~fm, :, 1].mean().item()
    print(f"\n  g_E focused={gE_focused:.6f}  routine={gE_routine:.6f}  "
          f"Δ={gE_focused-gE_routine:+.6f}")
    print(f"  g_I focused={gI_focused:.6f}  routine={gI_routine:.6f}  "
          f"Δ={gI_focused-gI_routine:+.6f}")

    return dict(
        w_grad=w_grad.detach().clone() if w_grad is not None else None,
        b_grad=b_grad.detach().clone() if b_grad is not None else None,
        fb_raw_mean=fb_raw.abs().mean().item() if fb_raw is not None else None,
        center_exc_mean=center_exc.abs().mean().item(),
    )


# ----------------------------------------------------------------------
# Experiment 2 — gradient cancellation across regimes (H3)
# ----------------------------------------------------------------------

def experiment_2(net, loss_fn, model_cfg, train_cfg, stim_cfg, dev):
    print("\n" + "=" * 72)
    print("EXPERIMENT 2 — per-regime gradient decomposition (tests H3 cancellation)")
    print("=" * 72)

    def run_single_regime(regime, seed):
        # Batch where ALL samples are one regime
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        hmm_gen = HMMSequenceGenerator(
            n_orientations=model_cfg.n_orientations,
            p_self=stim_cfg.p_self,
            p_transition_cw=stim_cfg.p_transition_cw,
            p_transition_ccw=stim_cfg.p_transition_ccw,
            n_anchors=stim_cfg.n_anchors,
            jitter_range=stim_cfg.jitter_range,
            transition_step=stim_cfg.transition_step,
            period=model_cfg.orientation_range,
            contrast_range=train_cfg.stage2_contrast_range,
            ambiguous_fraction=train_cfg.ambiguous_fraction,
            ambiguous_offset=stim_cfg.ambiguous_offset,
            n_states=stim_cfg.n_states,
            cue_valid_fraction=stim_cfg.cue_valid_fraction,
        )
        metadata = hmm_gen.generate(16, train_cfg.seq_length, gen)
        ts = torch.zeros(16, train_cfg.seq_length, 2)
        if regime == "focused":
            ts[:, :, 0] = 1.0
        else:
            ts[:, :, 1] = 1.0
        metadata.task_states = ts

        stim_seq, cue_seq, task_seq, true_t, true_nt, true_s = build_stimulus_sequence(
            metadata, model_cfg, train_cfg, stim_cfg
        )
        if train_cfg.stimulus_noise > 0:
            stim_seq = stim_seq + train_cfg.stimulus_noise * torch.randn_like(stim_seq)
            stim_seq = stim_seq.clamp(min=0.0)

        batch = dict(
            stim_seq=stim_seq.to(dev), cue_seq=cue_seq.to(dev), task_seq=task_seq.to(dev),
            true_thetas=true_t.to(dev), true_next_thetas=true_nt.to(dev),
            true_states=true_s.to(dev), metadata=metadata, focused_mask=None,
        )

        for p in net.parameters():
            if p.grad is not None:
                p.grad.zero_()

        total_loss, loss_dict, aux = stage2_forward_loss(
            net, loss_fn, batch, model_cfg, train_cfg, stim_cfg, fb_scale_override=1.0
        )
        total_loss.backward()
        return dict(
            w_grad=net.alpha_net.weight.grad.detach().clone(),
            b_grad=net.alpha_net.bias.grad.detach().clone(),
            loss=total_loss.item(),
            loss_dict=dict(loss_dict),
            gains_mean=aux["gains_all"].detach().mean(dim=(0, 1)).cpu().numpy().tolist(),
        )

    net.feedback_scale.fill_(1.0)
    r_foc = run_single_regime("focused", seed=11)
    r_rout = run_single_regime("routine", seed=13)

    print(f"\n-- focused-only batch --")
    print(f"  loss = {r_foc['loss']:.6f}")
    print(f"  loss_dict = {r_foc['loss_dict']}")
    print(f"  gains mean [g_E, g_I] = {r_foc['gains_mean']}")
    print(f"  alpha_net.weight.grad:\n{r_foc['w_grad'].cpu().numpy()}")
    print(f"  weight.grad.norm = {r_foc['w_grad'].norm().item():.6e}")
    print(f"  bias.grad = {r_foc['b_grad'].cpu().numpy()}")

    print(f"\n-- routine-only batch --")
    print(f"  loss = {r_rout['loss']:.6f}")
    print(f"  loss_dict = {r_rout['loss_dict']}")
    print(f"  gains mean [g_E, g_I] = {r_rout['gains_mean']}")
    print(f"  alpha_net.weight.grad:\n{r_rout['w_grad'].cpu().numpy()}")
    print(f"  weight.grad.norm = {r_rout['w_grad'].norm().item():.6e}")
    print(f"  bias.grad = {r_rout['b_grad'].cpu().numpy()}")

    # Cosine similarity (flattened across all weight entries + bias)
    gf = torch.cat([r_foc['w_grad'].flatten(), r_foc['b_grad'].flatten()])
    gr = torch.cat([r_rout['w_grad'].flatten(), r_rout['b_grad'].flatten()])
    cos = F.cosine_similarity(gf.unsqueeze(0), gr.unsqueeze(0)).item()

    # Per-row (g_E row vs g_I row)
    w_f = r_foc['w_grad']  # [2, 3]
    w_r = r_rout['w_grad']
    cos_gE = F.cosine_similarity(w_f[0:1], w_r[0:1]).item()
    cos_gI = F.cosine_similarity(w_f[1:2], w_r[1:2]).item()

    # Average gradient (what actually gets applied in mixed-batch)
    w_avg = 0.5 * (w_f + w_r)
    b_avg = 0.5 * (r_foc['b_grad'] + r_rout['b_grad'])

    print("\n-- gradient cancellation analysis (H3) --")
    print(f"  cosine(focused_grad, routine_grad) [all params]  = {cos:+.6f}")
    print(f"  cosine(focused, routine) g_E row [weight[0]]     = {cos_gE:+.6f}")
    print(f"  cosine(focused, routine) g_I row [weight[1]]     = {cos_gI:+.6f}")
    print(f"  |focused weight.grad|  = {w_f.norm().item():.3e}")
    print(f"  |routine weight.grad|  = {w_r.norm().item():.3e}")
    print(f"  |0.5*(gF+gR)|          = {w_avg.norm().item():.3e}  "
          f"(averaged/mixed batch-equivalent)")
    print(f"  ratio |avg|/|focused|  = {w_avg.norm().item() / (w_f.norm().item() + 1e-12):.4f}")
    print(f"  ratio |avg|/|routine|  = {w_avg.norm().item() / (w_r.norm().item() + 1e-12):.4f}")
    return dict(cos_all=cos, cos_gE=cos_gE, cos_gI=cos_gI,
                wf_norm=w_f.norm().item(), wr_norm=w_r.norm().item(),
                wavg_norm=w_avg.norm().item())


# ----------------------------------------------------------------------
# Experiment 3 — training dynamics probe (H1 burn-in, H2 LR)
# ----------------------------------------------------------------------

def experiment_3(stage1_ckpt_path, model_cfg_in, train_cfg_in, stim_cfg_in, dev,
                 n_steps=50):
    print("\n" + "=" * 72)
    print("EXPERIMENT 3 — training dynamics from stage1 ckpt (H1 burn-in, H2 LR)")
    print("=" * 72)

    def run_variant(name, fb_scale_schedule, alpha_lr_mult=1.0):
        # Fresh net + loss from the stage1 checkpoint
        model_cfg, train_cfg, stim_cfg = load_config(SWEEP)
        net = LaminarV1V2Network(model_cfg).to(dev)
        loss_fn = CompositeLoss(train_cfg, model_cfg).to(dev)
        ckpt = torch.load(stage1_ckpt_path, map_location=dev)
        net.load_state_dict(ckpt["model_state"], strict=False)
        if "decoder_state" in ckpt:
            loss_fn.orientation_decoder.load_state_dict(ckpt["decoder_state"])

        freeze_stage1(net)
        unfreeze_stage2(net)

        # Record init weight
        w_init = net.alpha_net.weight.data.clone()
        b_init = net.alpha_net.bias.data.clone()

        optimizer = create_stage2_optimizer(net, loss_fn, train_cfg)
        # Boost alpha_net LR if requested
        if alpha_lr_mult != 1.0:
            # alpha_net is the LAST group appended (before readout heads)
            for g in optimizer.param_groups:
                if len(g["params"]) == 2:
                    # heuristic: alpha_net group has weight [2,3] + bias [2]
                    shapes = [tuple(p.shape) for p in g["params"]]
                    if (2, 3) in shapes and (2,) in shapes:
                        g["lr"] = g["lr"] * alpha_lr_mult
                        print(f"  [{name}] Alpha LR boosted to {g['lr']:.3e}")
                        break

        gen = torch.Generator(device="cpu")
        gen.manual_seed(42)
        hmm_gen = HMMSequenceGenerator(
            n_orientations=model_cfg.n_orientations,
            p_self=stim_cfg.p_self,
            p_transition_cw=stim_cfg.p_transition_cw,
            p_transition_ccw=stim_cfg.p_transition_ccw,
            n_anchors=stim_cfg.n_anchors,
            jitter_range=stim_cfg.jitter_range,
            transition_step=stim_cfg.transition_step,
            period=model_cfg.orientation_range,
            contrast_range=train_cfg.stage2_contrast_range,
            ambiguous_fraction=train_cfg.ambiguous_fraction,
            ambiguous_offset=stim_cfg.ambiguous_offset,
            n_states=stim_cfg.n_states,
            cue_valid_fraction=stim_cfg.cue_valid_fraction,
        )

        # Disable burn-in for gain_rec freezing logic — we'll do it manually
        net.l23.gain_rec_raw.requires_grad_(True)
        net.train()

        step_drift = []
        step_diff = []
        last_loss = None

        for step in range(n_steps):
            net.feedback_scale.fill_(fb_scale_schedule(step))

            metadata = hmm_gen.generate(train_cfg.batch_size, train_cfg.seq_length, gen)
            stim_seq, cue_seq, task_seq, true_t, true_nt, true_s = (
                build_stimulus_sequence(metadata, model_cfg, train_cfg, stim_cfg)
            )
            if train_cfg.stimulus_noise > 0:
                stim_seq = stim_seq + train_cfg.stimulus_noise * torch.randn_like(stim_seq)
                stim_seq = stim_seq.clamp(min=0.0)

            batch = dict(
                stim_seq=stim_seq.to(dev), cue_seq=cue_seq.to(dev),
                task_seq=task_seq.to(dev),
                true_thetas=true_t.to(dev), true_next_thetas=true_nt.to(dev),
                true_states=true_s.to(dev),
            )
            optimizer.zero_grad()

            packed = net.pack_inputs(batch["stim_seq"], batch["cue_seq"], batch["task_seq"])
            r_l23_all, _, aux = net(packed)
            outputs = {
                "r_l23": r_l23_all,
                "q_pred": aux["q_pred_all"],
                "r_l4": aux["r_l4_all"],
                "r_pv": aux["r_pv_all"],
                "r_som": aux["r_som_all"],
                "deep_template": aux["deep_template_all"],
                "state_logits": aux["state_logits_all"],
                "p_cw": aux["p_cw_all"],
                "center_exc": aux["center_exc_all"],
            }
            readout_indices = compute_readout_indices(
                train_cfg.seq_length, train_cfg.steps_on, train_cfg.steps_isi,
                window_start=max(0, train_cfg.steps_on - 3),
                window_end=train_cfg.steps_on - 1,
            )
            r_l23_w, q_pred_w, _ = extract_readout_data(
                outputs, readout_indices,
                steps_on=train_cfg.steps_on, steps_isi=train_cfg.steps_isi,
            )
            task_state_batch = batch["task_seq"][:, 0, :]

            total_loss, _ = loss_fn(
                outputs, batch["true_thetas"], batch["true_next_thetas"],
                r_l23_w, q_pred_w,
                state_logits_windows=None,
                true_states_windows=batch["true_states"],
                p_cw_windows=None,
                model=net,
                fb_scale=net.feedback_scale.item(),
                task_state=task_state_batch,
                task_routing=train_cfg.task_routing,
            )
            total_loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), train_cfg.gradient_clip)
            optimizer.step()
            last_loss = total_loss.item()

            # Drift from init
            drift = (net.alpha_net.weight.data - w_init).abs().max().item()
            bdrift = (net.alpha_net.bias.data - b_init).abs().max().item()

            # g_E differentiation
            fm = task_state_batch[:, 0] > 0.5
            gE = aux["gains_all"][..., 0]  # [B, T]
            if fm.any() and (~fm).any():
                diff = gE[fm].mean().item() - gE[~fm].mean().item()
            else:
                diff = float("nan")
            step_drift.append(max(drift, bdrift))
            step_diff.append(diff)

        # Final
        w_final = net.alpha_net.weight.data.clone()
        b_final = net.alpha_net.bias.data.clone()
        final_drift_w = (w_final - w_init).abs().max().item()
        final_drift_b = (b_final - b_init).abs().max().item()

        print(f"\n[{name}] final after {n_steps} steps:")
        print(f"  last loss = {last_loss:.4f}")
        print(f"  alpha_net.weight drift from init (abs max) = {final_drift_w:.6e}")
        print(f"  alpha_net.bias   drift from init (abs max) = {final_drift_b:.6e}")
        print(f"  step-by-step weight drift (max): first 5 = "
              f"{[round(d, 6) for d in step_drift[:5]]}")
        print(f"  step-by-step weight drift (max): last 5  = "
              f"{[round(d, 6) for d in step_drift[-5:]]}")
        print(f"  step-by-step g_E(focused) - g_E(routine): first 5 = "
              f"{[round(d, 5) for d in step_diff[:5]]}")
        print(f"  step-by-step g_E(focused) - g_E(routine): last 5  = "
              f"{[round(d, 5) for d in step_diff[-5:]]}")
        print(f"  final weight:\n{w_final.cpu().numpy()}")
        print(f"  final bias: {b_final.cpu().numpy()}")
        return dict(
            drift_w=final_drift_w, drift_b=final_drift_b,
            last_diff=step_diff[-1], last_loss=last_loss,
        )

    burnin = train_cfg_in.stage2_burnin_steps
    ramp = train_cfg_in.stage2_ramp_steps

    def sched_A(step):
        # Replicate sweep: burn-in 1000, ramp 1000. For 50-step test, compress:
        # burn-in first 10 steps, ramp next 10, then 1.0
        if step < 10:
            return 0.0
        elif step < 20:
            return (step - 10) / 10.0
        else:
            return 1.0

    def sched_B(step):
        return 1.0   # no burn-in at all

    def sched_C(step):
        return sched_A(step)  # same as A, but LR boost applied separately

    print("\nNote: compressing 1000-step burn-in into 10-step burn-in for the probe.")
    print("Intent is DIRECTIONAL — does removing burn-in / boosting LR move the gate?")

    rA = run_variant("Variant A (current schedule, compressed)", sched_A, alpha_lr_mult=1.0)
    rB = run_variant("Variant B (no burn-in, fb_scale=1 from step 0)", sched_B, alpha_lr_mult=1.0)
    rC = run_variant("Variant C (current schedule + alpha LR x100)", sched_C, alpha_lr_mult=100.0)

    print("\n-- Experiment 3 summary --")
    print(f"  A drift_w={rA['drift_w']:.3e}  last g_E diff = {rA['last_diff']:+.6f}")
    print(f"  B drift_w={rB['drift_w']:.3e}  last g_E diff = {rB['last_diff']:+.6f}")
    print(f"  C drift_w={rC['drift_w']:.3e}  last g_E diff = {rC['last_diff']:+.6f}")
    return dict(A=rA, B=rB, C=rC)


# ----------------------------------------------------------------------
# Experiment 4 — identity vs perfect gate loss (H4)
# ----------------------------------------------------------------------

def experiment_4(net, loss_fn, model_cfg, train_cfg, stim_cfg, dev):
    print("\n" + "=" * 72)
    print("EXPERIMENT 4 — identity vs perfect gate loss (tests H4 loss-signal)")
    print("=" * 72)

    # We'll monkey-patch alpha_net to output fixed targets and measure loss.
    # Strategy: replace net.alpha_net with a custom module that takes the
    # same gate_input but returns the desired gains value directly (as
    # logits so 2*sigmoid(...) recovers the intended [g_E, g_I]).
    #
    # To set gain = v via 2*sigmoid(x), we need sigmoid(x) = v/2
    # → x = logit(v/2) = ln((v/2) / (1 - v/2)).
    def logit(v):
        v = max(min(v, 1.999), 1e-3)
        p = v / 2.0
        p = max(min(p, 1 - 1e-6), 1e-6)
        return math.log(p / (1 - p))

    class FixedGate(nn.Module):
        """Returns constant pre-sigmoid logits per task-state regime.

        forward(x) where x = [task[:, 0], task[:, 1], pi_pred_raw[:, 0]]
        returns [B, 2] such that 2*sigmoid(...) = [g_E, g_I] per sample.
        focused (x[:,0]=1) → (gE_f, gI_f); routine (x[:,1]=1) → (gE_r, gI_r).
        """
        def __init__(self, gE_f, gI_f, gE_r, gI_r):
            super().__init__()
            self.f = torch.tensor([logit(gE_f), logit(gI_f)])
            self.r = torch.tensor([logit(gE_r), logit(gI_r)])

        def forward(self, x):
            # x: [B, 3]
            is_focused = (x[:, 0:1] > 0.5).float()  # [B, 1]
            f = self.f.to(x.device)
            r = self.r.to(x.device)
            return is_focused * f + (1 - is_focused) * r

    # Save original alpha_net
    original = net.alpha_net

    def measure(name, alpha_module):
        net.alpha_net = alpha_module.to(dev)
        net.feedback_scale.fill_(1.0)
        batch = make_balanced_batch(model_cfg, train_cfg, stim_cfg, dev, B=32, seed=99)
        with torch.no_grad():
            total_loss, loss_dict, aux = stage2_forward_loss(
                net, loss_fn, batch, model_cfg, train_cfg, stim_cfg, fb_scale_override=1.0
            )
        gains = aux["gains_all"]
        fm = batch["focused_mask"]
        print(f"\n-- {name} --")
        print(f"  gains focused: g_E={gains[fm,:,0].mean().item():.4f} "
              f"g_I={gains[fm,:,1].mean().item():.4f}")
        print(f"  gains routine: g_E={gains[~fm,:,0].mean().item():.4f} "
              f"g_I={gains[~fm,:,1].mean().item():.4f}")
        print(f"  total loss = {total_loss.item():.6f}")
        print(f"  breakdown: {loss_dict}")
        return total_loss.item(), dict(loss_dict)

    try:
        # 1. Trained alpha_net (real checkpoint)
        loss_trained, d_trained = measure("TRAINED (checkpoint alpha_net)", original)

        # 2. True identity (g_E=1, g_I=1)
        loss_identity, d_ident = measure(
            "IDENTITY (g_E=1, g_I=1)", FixedGate(1.0, 1.0, 1.0, 1.0)
        )

        # 3. "Perfect" gate: focused amplify excitation, routine dampen
        loss_perf, d_perf = measure(
            "PERFECT (focused g_E=1.8, g_I=0.2 | routine g_E=0.2, g_I=1.8)",
            FixedGate(1.8, 0.2, 0.2, 1.8),
        )

        # 4. Mildly differentiated gate (more realistic — what the gate should learn)
        loss_mild, d_mild = measure(
            "MILD (focused g_E=1.3, g_I=0.7 | routine g_E=0.7, g_I=1.3)",
            FixedGate(1.3, 0.7, 0.7, 1.3),
        )

    finally:
        # Always restore
        net.alpha_net = original

    print("\n-- Experiment 4 summary --")
    print(f"  trained checkpoint loss = {loss_trained:.6f}")
    print(f"  identity         loss   = {loss_identity:.6f}  (Δ vs trained = {loss_identity-loss_trained:+.6f})")
    print(f"  perfect          loss   = {loss_perf:.6f}  (Δ vs identity = {loss_perf-loss_identity:+.6f})")
    print(f"  mild             loss   = {loss_mild:.6f}  (Δ vs identity = {loss_mild-loss_identity:+.6f})")
    return dict(trained=loss_trained, identity=loss_identity, perfect=loss_perf, mild=loss_mild,
                trained_dict=d_trained, identity_dict=d_ident,
                perfect_dict=d_perf, mild_dict=d_mild)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",
                    default=os.path.join(ROOT, "results/dual_2/emergent_seed42/checkpoint.pt"))
    ap.add_argument("--stage1-ckpt",
                    default=os.path.join(ROOT, "results/dual_2/emergent_seed42/stage1_checkpoint.pt"))
    ap.add_argument("--skip-e3", action="store_true", help="skip experiment 3 (training probe)")
    ap.add_argument("--e3-steps", type=int, default=50)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dev = torch.device(args.device)
    print(f"Device: {dev}")
    print(f"Phase 2 checkpoint:  {args.ckpt}")
    print(f"Stage 1 checkpoint:  {args.stage1_ckpt}")
    print(f"Sweep config:        {SWEEP}")

    torch.manual_seed(12345)

    model_cfg, train_cfg, stim_cfg, net, loss_fn = build_model_and_loss(dev)
    ckpt = load_checkpoint(net, loss_fn, args.ckpt, dev, strict=True)

    print(f"\nCheckpoint contents: {list(ckpt.keys())}")
    print(f"use_ei_gate: {net.use_ei_gate}")
    print(f"task_routing: {train_cfg.task_routing}")

    results = {}

    # EXPERIMENT 1
    try:
        results["E1"] = experiment_1(net, loss_fn, model_cfg, train_cfg, stim_cfg, dev)
    except Exception as e:
        print(f"\n!!! Experiment 1 failed: {e}")
        import traceback; traceback.print_exc()

    # EXPERIMENT 2
    try:
        # Need to reload checkpoint because E1 left grads + modified state
        load_checkpoint(net, loss_fn, args.ckpt, dev, strict=True)
        results["E2"] = experiment_2(net, loss_fn, model_cfg, train_cfg, stim_cfg, dev)
    except Exception as e:
        print(f"\n!!! Experiment 2 failed: {e}")
        import traceback; traceback.print_exc()

    # EXPERIMENT 4 (before E3, which modifies weights via training)
    try:
        load_checkpoint(net, loss_fn, args.ckpt, dev, strict=True)
        results["E4"] = experiment_4(net, loss_fn, model_cfg, train_cfg, stim_cfg, dev)
    except Exception as e:
        print(f"\n!!! Experiment 4 failed: {e}")
        import traceback; traceback.print_exc()

    # EXPERIMENT 3 (slowest — runs training)
    if not args.skip_e3:
        try:
            results["E3"] = experiment_3(
                args.stage1_ckpt, model_cfg, train_cfg, stim_cfg, dev,
                n_steps=args.e3_steps,
            )
        except Exception as e:
            print(f"\n!!! Experiment 3 failed: {e}")
            import traceback; traceback.print_exc()

    print("\n" + "=" * 72)
    print("DONE — see per-experiment sections above for verdicts.")
    print("=" * 72)


if __name__ == "__main__":
    main()
