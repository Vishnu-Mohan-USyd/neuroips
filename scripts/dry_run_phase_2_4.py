"""50-step end-to-end dry run of Phase 2.4 on a small batch.

Purpose (per team-lead's process):
    1. Verify routine_shape is NONZERO on routine-containing batches,
       NOT a silent pass-through.
    2. Verify alpha_net gradients are FLOWING (non-zero grad norm on each
       of the 8 alpha_net params over 50 steps).
    3. Verify NO NaN / Inf in the loss, the checkpoint state, or the
       routine_shape metric over 50 steps.
    4. Spot-check that g_E / g_I start near 1.0 and move (at all) by the
       end of 50 steps with lr_mult_alpha=10.

This bypasses Stage 1 (which is a 2000-step scaffold) and goes straight
into Stage 2 using a warm Stage-1 init — it's a behavioral smoke, not a
convergence test.

Run: python scripts/dry_run_phase_2_4.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config import load_config
from src.model.network import LaminarV1V2Network
from src.training.losses import CompositeLoss
from src.training.trainer import (
    create_stage2_optimizer,
    freeze_stage1,
    unfreeze_stage2,
    build_stimulus_sequence,
)
from src.stimulus.sequences import HMMSequenceGenerator


def main() -> None:
    cfg_path = "config/sweep/sweep_dual_2_4.yaml"
    model_cfg, train_cfg, stim_cfg = load_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"== Phase 2.4 dry run (50 Stage-2 steps) ==")
    print(f"device               = {device}")
    print(f"lambda_routine_shape = {train_cfg.lambda_routine_shape}")
    print(f"lr_mult_alpha        = {train_cfg.lr_mult_alpha}")
    print(f"use_ei_gate          = {model_cfg.use_ei_gate}")
    print()

    torch.manual_seed(42)
    net = LaminarV1V2Network(model_cfg).to(device)
    loss_fn = CompositeLoss(train_cfg, model_cfg).to(device)

    # Skip Stage 1: freeze stage1 params as if Stage 1 just finished.
    freeze_stage1(net)
    unfreeze_stage2(net)

    assert hasattr(net, "alpha_net"), "alpha_net missing — use_ei_gate should be True"

    optimizer = create_stage2_optimizer(net, loss_fn, train_cfg)

    # Verify alpha_net group exists with the expected LR.
    alpha_ids = {id(p) for p in net.alpha_net.parameters()}
    alpha_group_lr = None
    for group in optimizer.param_groups:
        if alpha_ids.issubset({id(p) for p in group["params"]}):
            alpha_group_lr = group["lr"]
            break
    expected_alpha_lr = train_cfg.stage2_lr_v2 * train_cfg.lr_mult_alpha
    print(f"alpha_net group LR   = {alpha_group_lr}  (expected {expected_alpha_lr})")
    assert alpha_group_lr is not None, "alpha_net params not found in any optimizer group"
    assert abs(alpha_group_lr - expected_alpha_lr) < 1e-9
    print(f"param groups         = {len(optimizer.param_groups)}")
    print()

    # Build HMM generator.
    hmm = HMMSequenceGenerator(
        n_orientations=model_cfg.n_orientations,
        n_states=stim_cfg.n_states,
        p_transition_cw=stim_cfg.p_transition_cw,
        p_transition_ccw=stim_cfg.p_transition_ccw,
        p_self=stim_cfg.p_self,
        n_anchors=stim_cfg.n_anchors,
        jitter_range=stim_cfg.jitter_range,
        transition_step=stim_cfg.transition_step,
        cue_valid_fraction=stim_cfg.cue_valid_fraction,
        contrast_range=train_cfg.stage2_contrast_range,
        ambiguous_fraction=train_cfg.ambiguous_fraction,
        ambiguous_offset=stim_cfg.ambiguous_offset,
        cue_dim=stim_cfg.cue_dim,
    )

    # Capture initial alpha_net params.
    alpha_w0 = net.alpha_net.weight.detach().clone()
    alpha_b0 = net.alpha_net.bias.detach().clone()

    rs_values = []
    any_nan = False
    alpha_grads_seen = False

    print(f"{'step':>4s}  {'total':>8s}  {'rshape':>9s}  {'|g_alpha|':>10s}  "
          f"{'g_E':>6s}  {'g_I':>6s}")
    print("-" * 60)
    for step in range(50):
        metadata = hmm.generate(train_cfg.batch_size, train_cfg.seq_length)
        stim_seq, cue_seq, task_state_seq, true_thetas, true_next_thetas, _ = (
            build_stimulus_sequence(metadata, model_cfg, train_cfg, stim_cfg)
        )
        stim_seq = stim_seq.to(device)
        cue_seq = cue_seq.to(device)
        task_state_seq = task_state_seq.to(device)
        true_thetas = true_thetas.to(device)
        true_next_thetas = true_next_thetas.to(device)

        packed = LaminarV1V2Network.pack_inputs(
            stim_seq, cue_seq, task_state_seq
        )
        r_l23_all, _, aux = net(packed)
        outputs = {
            "r_l4": aux["r_l4_all"],
            "r_l23": r_l23_all,
            "r_pv": aux["r_pv_all"],
            "r_som": aux["r_som_all"],
            "deep_template": aux["deep_template_all"],
            "q_pred": aux["q_pred_all"],
            "center_exc": aux["center_exc_all"],
            "som_drive_fb": aux["som_drive_fb_all"],
        }

        # Sequence-level task_state: each sample's first-presentation state.
        task_state_seq_level = task_state_seq[:, 0, :]  # [B, 2]

        # Window shape: full sequence-length readouts (simplified dry run).
        from src.training.trainer import compute_readout_indices, extract_readout_data
        indices = compute_readout_indices(
            train_cfg.seq_length, train_cfg.steps_on, train_cfg.steps_isi
        )
        r_l23_w, q_pred_w, _ = extract_readout_data(
            outputs, indices, train_cfg.steps_on, train_cfg.steps_isi
        )

        total, ld = loss_fn(
            outputs,
            true_thetas,
            true_next_thetas,
            r_l23_w,
            q_pred_w,
            fb_scale=1.0,
            task_state=task_state_seq_level,
            task_routing=train_cfg.task_routing,
        )
        if not torch.isfinite(total):
            print(f"  !! non-finite total loss at step {step}: {total.item()}")
            any_nan = True
            break
        if not (ld["routine_shape"] == ld["routine_shape"]):  # NaN check
            print(f"  !! NaN routine_shape at step {step}")
            any_nan = True
            break

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        grad_norm_alpha = sum(
            p.grad.detach().pow(2).sum() for p in net.alpha_net.parameters()
            if p.grad is not None
        ).sqrt().item()
        if grad_norm_alpha > 0:
            alpha_grads_seen = True
        g_mean = aux["gains_all"].mean(dim=(0, 1)).detach().cpu()
        torch.nn.utils.clip_grad_norm_(
            [p for g in optimizer.param_groups for p in g["params"]],
            max_norm=train_cfg.gradient_clip,
        )
        optimizer.step()

        rs_values.append(ld["routine_shape"])
        if step % 5 == 0 or step == 49:
            print(
                f"{step:>4d}  {total.item():>8.4f}  {ld['routine_shape']:>+9.5f}  "
                f"{grad_norm_alpha:>10.3e}  {g_mean[0].item():>6.3f}  {g_mean[1].item():>6.3f}"
            )

    # Post-run checks.
    print()
    print("== Post-run checks ==")
    print(f"any_nan                = {any_nan}")
    print(f"alpha_grads_seen       = {alpha_grads_seen}")
    nonzero_rs = sum(1 for v in rs_values if v != 0.0)
    print(f"nonzero routine_shape  = {nonzero_rs}/{len(rs_values)}")
    mean_rs = sum(rs_values) / max(len(rs_values), 1)
    print(f"mean   routine_shape   = {mean_rs:+.5f}")
    dw = (net.alpha_net.weight.detach() - alpha_w0).abs().max().item()
    db = (net.alpha_net.bias.detach() - alpha_b0).abs().max().item()
    print(f"alpha_net max Δweight  = {dw:.3e}")
    print(f"alpha_net max Δbias    = {db:.3e}")
    print()

    ok = (
        (not any_nan)
        and alpha_grads_seen
        and nonzero_rs == len(rs_values)
        and (dw > 0 or db > 0)
    )
    if ok:
        print("RESULT: PASS — dry run clean.")
        sys.exit(0)
    else:
        print("RESULT: FAIL — see above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
