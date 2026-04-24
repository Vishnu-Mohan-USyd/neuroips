"""Task #3 — Retrain Dec A using the ORIGINAL co-training data schedule on a
frozen net. Tests H-A (co-training context) vs H-B (moving-target L2/3) as the
load-bearing factor in the Dec-A-vs-Dec-A'/Dec-E dissociation on dampening-legacy
networks.

Design (Option 1b, approved by Lead):
  - Self-contained script, no edits to `src/training/stage{1,2}_*.py`.
  - Full net frozen: every param `requires_grad_(False)`; asserted before training.
  - Fresh Linear(36, 36)+bias decoder, seed 42.
  - Adam lr = `train_cfg.stage1_lr` (same as original Dec A stage-1/stage-2 LR).

Stage 1 (n = `train_cfg.stage1_n_steps`, default 2000):
  - Random grating stimuli (thetas ∈ [0, 180), contrast ∈ `stage1_contrast_range`).
  - Forward: `_run_v1_only` (V1-only, 20 timesteps, no V2/feedback) under
    `torch.no_grad()` since the net is frozen.
  - Readout: final `r_l23` [B, N] (no time-window).
  - Loss: CE(decoder(r_l23), true_channel).

Stage 2 (n = `train_cfg.stage2_n_steps`, default 5000):
  - HMM stream via `HMMSequenceGenerator` with each net's own config (task_p_switch,
    ambiguous_fraction, cue_valid_fraction, etc).
  - Feedback curriculum REPLICATED from `stage2_feedback.py`:
      step ∈ [0, burnin):       feedback_scale = 0        (V1-only L2/3)
      step ∈ [burnin, burnin+ramp): feedback_scale = linear(0 → 1)
      step ∈ [burnin+ramp, end): feedback_scale = 1        (full feedback)
    This matches the data distribution Dec A was exposed to during ORIGINAL training.
  - Forward: `net.pack_inputs` + `net.forward` (full V1+V2+feedback) under
    `torch.no_grad()` since net is frozen.
  - Readout: `r_l23[:, onset+9:onset+11].mean(dim=1)` per presentation (Task #25 /
    Stage 2 `[steps_on-3:steps_on-1]` window, matches Dec A'/Dec E/Dec C convention).
  - Loss: CE(decoder(r_probe), true_channel) per presentation, averaged across
    batch × seq_length.

Val pool (Task #25 convention, comparable to Dec A'/Dec E val numbers):
  - HMM batches seed 1234, 10 × 32 × 25 = 8000 readouts, 50/50 focused/routine
    task_state (deterministic), feedback_scale = 1.0.
  - Evaluated every `log_every` steps (500 by default) + at step 0 and final.

Post-training 10k HMM eval for Dec A comparison:
  - Same 10k HMM protocol as Task #25 / Task #2 per-net (bs=16, 625 batches,
    50/50 task_state, seed 42, fresh rng independent of training stream).
  - Reports Dec A (original, loaded from ckpt) vs Dec A_cotrained on SAME 10k pool.

Outputs:
  - checkpoints/decoder_a_cotrained_{net}.pt
  - results/decoder_a_cotrained_training_{net}.json

Seed 42 throughout. FP drift on CPU may be ≤ 0.02 on val acc under re-runs.

Per-net usage:
    python3 scripts/retrain_decoder_a_cotrained.py --net-name r1r2
    python3 scripts/retrain_decoder_a_cotrained.py --net-name a1
    ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS_DIR = os.path.dirname(__file__)
_REPO = os.path.abspath(os.path.join(_THIS_DIR, ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, _THIS_DIR)

from src.config import load_config, MechanismType  # noqa: F401
from src.model.network import LaminarV1V2Network
from src.stimulus.gratings import generate_grating
from src.stimulus.sequences import HMMSequenceGenerator
from src.training.trainer import build_stimulus_sequence

# Local import: _run_v1_only from stage1_sensory (pure read, no config touch).
from src.training.stage1_sensory import _run_v1_only

SEED = 42
VAL_SEED = 1234
N_VAL_BATCHES = 10
READOUT_WIN = (9, 11)
EVAL_SEED = 42
EVAL_N_TRIALS = 10_000
EVAL_BATCH_SIZE = 16
LOG_EVERY = 500

R1R2_CKPT_DEFAULT = os.path.join(_REPO, "results/simple_dual/emergent_seed42/checkpoint.pt")
DEFAULT_CONFIG = os.path.join(_REPO, "config/sweep/sweep_rescue_1_2.yaml")

PER_NET_CKPT = {
    "r1r2": R1R2_CKPT_DEFAULT,
    "a1":   "/tmp/remote_ckpts/a1/checkpoint.pt",
    "b1":   "/tmp/remote_ckpts/b1/checkpoint.pt",
    "c1":   "/tmp/remote_ckpts/c1/checkpoint.pt",
    "e1":   "/tmp/remote_ckpts/e1/checkpoint.pt",
}
PER_NET_CFG = {
    "r1r2": DEFAULT_CONFIG,
    **{k: os.path.join(_REPO, f"config/sweep/sweep_{k}.yaml") for k in ("a1", "b1", "c1", "e1")},
}


def build_generator(model_cfg, train_cfg, stim_cfg) -> HMMSequenceGenerator:
    """Build a HMM generator that respects the yaml's task_p_switch.
    Matches the Dec E / Stage 2 generator config (task_p_switch, cue_valid_fraction
    all respected from stim_cfg)."""
    return HMMSequenceGenerator(
        n_orientations=model_cfg.n_orientations,
        p_self=stim_cfg.p_self,
        p_transition_cw=stim_cfg.p_transition_cw,
        p_transition_ccw=stim_cfg.p_transition_ccw,
        n_anchors=stim_cfg.n_anchors,
        jitter_range=stim_cfg.jitter_range,
        transition_step=stim_cfg.transition_step,
        period=model_cfg.orientation_range,
        contrast_range=tuple(train_cfg.stage2_contrast_range),
        ambiguous_fraction=train_cfg.ambiguous_fraction,
        ambiguous_offset=stim_cfg.ambiguous_offset,
        cue_dim=stim_cfg.cue_dim,
        n_states=stim_cfg.n_states,
        cue_valid_fraction=stim_cfg.cue_valid_fraction,
        task_p_switch=getattr(stim_cfg, "task_p_switch", 0.0),
    )


def set_50_50_task_state(md, batch_size: int) -> None:
    """Mutate metadata.task_states to 50/50 focused/routine per trial (Task #25)."""
    ts_mode = (torch.arange(batch_size) < batch_size // 2).long()
    new_ts = torch.zeros_like(md.task_states)
    new_ts[..., 0] = ts_mode.float().unsqueeze(-1)
    new_ts[..., 1] = (1 - ts_mode).float().unsqueeze(-1)
    md.task_states = new_ts


@torch.no_grad()
def collect_hmm_readouts(
    net, generator, *,
    batch_size: int, seq_length: int,
    steps_per: int, win_lo: int, win_hi: int,
    step_deg: float, n_ori: int,
    rng: torch.Generator, device,
    model_cfg, train_cfg, stim_cfg,
    stim_noise_seed: int | None = None,
    force_5050: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one HMM batch through the frozen net; return (readouts[B*S,N], labels[B*S]).
    If force_5050 is True, override task_state to 50/50 per Task #25."""
    md = generator.generate(batch_size, seq_length, generator=rng)
    if force_5050:
        set_50_50_task_state(md, batch_size)

    if stim_noise_seed is not None:
        torch.manual_seed(stim_noise_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(stim_noise_seed)

    stim_seq, cue_seq, ts_seq, _, _, _ = build_stimulus_sequence(
        md, model_cfg, train_cfg, stim_cfg)
    stim_seq = stim_seq.to(device)
    cue_seq = cue_seq.to(device)
    ts_seq = ts_seq.to(device)

    packed = net.pack_inputs(stim_seq, cue_seq, ts_seq)
    r_l23_all, _, _ = net.forward(packed)  # [B, T_total, N]

    B = batch_size
    S = seq_length
    N = n_ori
    readouts = torch.empty(B * S, N, device=device)
    for s in range(S):
        onset = s * steps_per
        readouts[s * B:(s + 1) * B] = r_l23_all[:, onset + win_lo:onset + win_hi, :].mean(dim=1)
    true_ori = md.orientations.to(device)
    true_ch = (true_ori / step_deg).round().long() % n_ori
    labels = true_ch.transpose(0, 1).reshape(-1)  # [S*B]
    return readouts, labels


def build_val_pool(
    net, gen_val, *,
    batch_size: int, seq_length: int,
    steps_per: int, win_lo: int, win_hi: int,
    step_deg: float, n_ori: int,
    device, model_cfg, train_cfg, stim_cfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed val pool: 10 batches × 32 × 25 = 8000 readouts, 50/50 task_state,
    feedback_scale=1.0. Mirrors Dec A' / Dec E val pool convention exactly."""
    fb_save = net.feedback_scale.clone()
    net.feedback_scale.fill_(1.0)
    rng_val = torch.Generator().manual_seed(VAL_SEED)
    vx, vy = [], []
    for vi in range(N_VAL_BATCHES):
        rx, ry = collect_hmm_readouts(
            net, gen_val,
            batch_size=batch_size, seq_length=seq_length,
            steps_per=steps_per, win_lo=win_lo, win_hi=win_hi,
            step_deg=step_deg, n_ori=n_ori,
            rng=rng_val, device=device,
            model_cfg=model_cfg, train_cfg=train_cfg, stim_cfg=stim_cfg,
            stim_noise_seed=VAL_SEED + vi,
            force_5050=True,
        )
        vx.append(rx); vy.append(ry)
    net.feedback_scale.copy_(fb_save)
    return torch.cat(vx, dim=0), torch.cat(vy, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net-name", required=True, choices=list(PER_NET_CKPT.keys()))
    ap.add_argument("--ckpt", default=None,
                    help="Override; otherwise per-net default.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out-ckpt", default=None)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--log-every", type=int, default=LOG_EVERY)
    ap.add_argument("--save-every", type=int, default=2000,
                    help="Crash-safety snapshot cadence during Stage 2.")
    ap.add_argument("--smoke-stage1-steps", type=int, default=0,
                    help="Debug: override stage1 n_steps (0 = use config default).")
    ap.add_argument("--smoke-stage2-steps", type=int, default=0,
                    help="Debug: override stage2 n_steps (0 = use config default).")
    args = ap.parse_args()

    net_name = args.net_name
    if args.ckpt is None:
        args.ckpt = PER_NET_CKPT[net_name]
    if args.config is None:
        args.config = PER_NET_CFG[net_name]
    if args.out_ckpt is None:
        args.out_ckpt = os.path.join(
            _REPO, f"checkpoints/decoder_a_cotrained_{net_name}.pt")
    if args.out_json is None:
        args.out_json = os.path.join(
            _REPO, f"results/decoder_a_cotrained_training_{net_name}.json")

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}  net={net_name}  ckpt={args.ckpt}  cfg={args.config}", flush=True)

    # ---- Config ----
    model_cfg, train_cfg, stim_cfg = load_config(args.config)
    n_ori = int(model_cfg.n_orientations)
    period = float(model_cfg.orientation_range)
    step_deg = period / n_ori
    steps_on = int(train_cfg.steps_on)
    steps_isi = int(train_cfg.steps_isi)
    steps_per = steps_on + steps_isi
    win_lo, win_hi = READOUT_WIN
    batch_size = int(train_cfg.batch_size)
    seq_length = int(train_cfg.seq_length)

    stage1_n = args.smoke_stage1_steps or int(train_cfg.stage1_n_steps)
    stage2_n = args.smoke_stage2_steps or int(train_cfg.stage2_n_steps)
    stage1_lr = float(train_cfg.stage1_lr)
    stage1_contrast_lo, stage1_contrast_hi = train_cfg.stage1_contrast_range
    burnin_steps = int(train_cfg.stage2_burnin_steps)
    ramp_steps = int(train_cfg.stage2_ramp_steps)

    print(f"[cfg] N={n_ori} step_deg={step_deg:.3f} batch={batch_size} "
          f"seq_length={seq_length} steps_per={steps_per} readout_window=[{win_lo}:{win_hi}]",
          flush=True)
    print(f"[cfg] stage1_n={stage1_n} stage2_n={stage2_n} lr={stage1_lr} "
          f"stage1_contrast={stage1_contrast_lo}-{stage1_contrast_hi} "
          f"burnin={burnin_steps} ramp={ramp_steps} "
          f"amb_frac={train_cfg.ambiguous_fraction} "
          f"task_p_switch={getattr(stim_cfg, 'task_p_switch', 0.0)}",
          flush=True)

    # ---- Frozen network ----
    net = LaminarV1V2Network(model_cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    res = net.load_state_dict(ckpt["model_state"], strict=False)
    print(f"[net] loaded {args.ckpt}  missing={len(res.missing_keys)} "
          f"unexpected={len(res.unexpected_keys)}", flush=True)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    # Assert every net param is frozen — belt-and-braces against any in-line
    # requires_grad flip this codepath could have missed.
    for pname, p in net.named_parameters():
        assert not p.requires_grad, f"param {pname} NOT frozen"
    net.oracle_mode = False
    net.feedback_scale.fill_(1.0)
    frozen_param_count = sum(p.numel() for p in net.parameters())
    print(f"[net] ALL {frozen_param_count:,} parameters frozen (asserted)", flush=True)

    # ---- Fresh Dec A_cotrained ----
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    decoder = nn.Linear(n_ori, n_ori, bias=True).to(device)
    for p in decoder.parameters():
        p.requires_grad_(True)
    decoder_param_count = sum(p.numel() for p in decoder.parameters())
    print(f"[dec] fresh Linear({n_ori}, {n_ori})+bias seed={SEED} "
          f"({decoder_param_count:,} trainable params)", flush=True)

    opt = torch.optim.Adam(decoder.parameters(), lr=stage1_lr)
    print(f"[opt] Adam lr={stage1_lr} (matches train_cfg.stage1_lr)", flush=True)

    # ---- Val pool (built ONCE, before Stage 1) ----
    print(f"[val] building fixed pool: {N_VAL_BATCHES} × {batch_size} × {seq_length} "
          f"= {N_VAL_BATCHES * batch_size * seq_length} readouts (seed={VAL_SEED}, "
          f"50/50 task_state, FB=1.0)", flush=True)
    t0 = time.time()
    gen_val = build_generator(model_cfg, train_cfg, stim_cfg)
    val_X, val_y = build_val_pool(
        net, gen_val,
        batch_size=batch_size, seq_length=seq_length,
        steps_per=steps_per, win_lo=win_lo, win_hi=win_hi,
        step_deg=step_deg, n_ori=n_ori,
        device=device, model_cfg=model_cfg, train_cfg=train_cfg, stim_cfg=stim_cfg,
    )
    n_val = int(val_X.shape[0])
    print(f"[val] pool ready n={n_val} in {time.time()-t0:.1f}s", flush=True)

    @torch.no_grad()
    def eval_val() -> tuple[float, float]:
        decoder.eval()
        logits = decoder(val_X)
        loss = F.cross_entropy(logits, val_y).item()
        acc = (logits.argmax(-1) == val_y).float().mean().item()
        decoder.train()
        return float(loss), float(acc)

    # ---- Stage 1: grating CE on V1-only r_l23 ----
    print(f"\n[stage1] {stage1_n} steps, grating batches, V1-only forward, CE on final r_l23",
          flush=True)
    history = {
        "step": [], "phase": [],
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "feedback_scale": [],
    }
    val_loss0, val_acc0 = eval_val()
    history["step"].append(0)
    history["phase"].append("init")
    history["train_loss"].append(float("nan"))
    history["train_acc"].append(float("nan"))
    history["val_loss"].append(val_loss0)
    history["val_acc"].append(val_acc0)
    history["feedback_scale"].append(1.0)
    print(f"[val @ init] loss={val_loss0:.4f} acc={val_acc0:.4f}", flush=True)

    gen_s1 = torch.Generator(device="cpu").manual_seed(SEED)
    t0 = time.time()
    running_loss = running_correct = running_n = 0.0
    for step in range(1, stage1_n + 1):
        thetas = torch.rand(batch_size, generator=gen_s1) * period
        contrasts = stage1_contrast_lo + (stage1_contrast_hi - stage1_contrast_lo) * torch.rand(
            batch_size, generator=gen_s1)
        stim = generate_grating(
            thetas, contrasts,
            n_orientations=n_ori,
            sigma=model_cfg.sigma_ff,
            n=model_cfg.naka_rushton_n,
            c50=model_cfg.naka_rushton_c50,
            period=period,
        ).to(device)

        # V1-only forward (20 timesteps) under no_grad: net is frozen, decoder is the
        # only trainable path.
        with torch.no_grad():
            r_l23 = _run_v1_only(net, stim, n_timesteps=20)  # [B, N]

        logits = decoder(r_l23)
        targets = (thetas.to(device) / step_deg).round().long() % n_ori
        loss = F.cross_entropy(logits, targets)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        with torch.no_grad():
            running_loss += float(loss.item()) * int(stim.shape[0])
            running_correct += int((logits.argmax(-1) == targets).sum())
            running_n += int(stim.shape[0])

        if step % args.log_every == 0 or step == stage1_n:
            train_loss = running_loss / max(running_n, 1)
            train_acc = running_correct / max(running_n, 1)
            val_loss, val_acc = eval_val()
            elapsed = time.time() - t0
            rate = step / max(elapsed, 1e-6)
            print(f"[s1 {step:5d}/{stage1_n}] train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  "
                  f"elapsed={elapsed:.0f}s rate={rate:.2f} st/s", flush=True)
            history["step"].append(int(step))
            history["phase"].append("stage1")
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["feedback_scale"].append(1.0)
            running_loss = running_correct = running_n = 0.0
    s1_elapsed = time.time() - t0
    print(f"[stage1] done in {s1_elapsed:.1f}s", flush=True)

    # ---- Stage 2: HMM stream CE on full-forward r_l23 ----
    print(f"\n[stage2] {stage2_n} steps, HMM batches, full forward, CE on r_probe  "
          f"(burnin={burnin_steps}, ramp={ramp_steps}, then FB=1.0)", flush=True)
    gen_train = build_generator(model_cfg, train_cfg, stim_cfg)
    rng_train = torch.Generator().manual_seed(SEED)
    t0 = time.time()
    running_loss = running_correct = running_n = 0.0
    for step in range(1, stage2_n + 1):
        # Feedback curriculum: 0 → ramp → 1.0 (replicates original Stage 2).
        if step <= burnin_steps:
            fb = 0.0
        elif step <= burnin_steps + ramp_steps:
            fb = (step - burnin_steps) / max(ramp_steps, 1)
        else:
            fb = 1.0
        net.feedback_scale.fill_(fb)

        rx, ry = collect_hmm_readouts(
            net, gen_train,
            batch_size=batch_size, seq_length=seq_length,
            steps_per=steps_per, win_lo=win_lo, win_hi=win_hi,
            step_deg=step_deg, n_ori=n_ori,
            rng=rng_train, device=device,
            model_cfg=model_cfg, train_cfg=train_cfg, stim_cfg=stim_cfg,
            stim_noise_seed=SEED + 1_000_000 + step,  # deterministic per-step noise
            force_5050=False,  # HMM's own task_state — matches Dec A's original training
        )
        logits = decoder(rx)
        loss = F.cross_entropy(logits, ry)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        with torch.no_grad():
            running_loss += float(loss.item()) * int(rx.shape[0])
            running_correct += int((logits.argmax(-1) == ry).sum())
            running_n += int(rx.shape[0])

        if step % args.log_every == 0 or step == stage2_n:
            train_loss = running_loss / max(running_n, 1)
            train_acc = running_correct / max(running_n, 1)
            val_loss, val_acc = eval_val()
            elapsed = time.time() - t0
            rate = step / max(elapsed, 1e-6)
            print(f"[s2 {step:5d}/{stage2_n}] train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  fb={fb:.3f}  "
                  f"elapsed={elapsed:.0f}s rate={rate:.2f} st/s", flush=True)
            history["step"].append(int(step + stage1_n))
            history["phase"].append("stage2")
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["feedback_scale"].append(fb)
            running_loss = running_correct = running_n = 0.0

        # Periodic crash-safety snapshot during Stage 2 only.
        if args.save_every > 0 and step % args.save_every == 0 and step < stage2_n:
            Path(args.out_ckpt).parent.mkdir(parents=True, exist_ok=True)
            snap_final = args.out_ckpt + f".stage2step{step}"
            torch.save({
                "state_dict": decoder.state_dict(),
                "step": int(step),
                "stage": "stage2",
                "n_steps_target": stage2_n,
            }, snap_final + ".tmp")
            os.replace(snap_final + ".tmp", snap_final)
            print(f"[snap @ s2 {step}] -> {snap_final}", flush=True)
    s2_elapsed = time.time() - t0
    print(f"[stage2] done in {s2_elapsed:.1f}s", flush=True)

    # Ensure final feedback_scale back to 1.0 for post-training eval
    net.feedback_scale.fill_(1.0)

    # ---- Save trained decoder BEFORE the expensive 10k eval (crash-safety) ----
    Path(args.out_ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": decoder.state_dict(),
        "arch": f"Linear({n_ori}, {n_ori})+bias",
        "seed": SEED,
        "val_seed": VAL_SEED,
        "stage1_n_steps": stage1_n,
        "stage2_n_steps": stage2_n,
        "lr": stage1_lr,
        "batch_size": batch_size,
        "seq_length": seq_length,
        "readout_window": [win_lo, win_hi],
        "net_ckpt": args.ckpt,
        "config_path": args.config,
        "final_val_loss": history["val_loss"][-1],
        "final_val_acc": history["val_acc"][-1],
        "protocol": "Dec A cotrained (stage1 grating 2000 steps + stage2 HMM 5000 steps, "
                    "frozen net, decoder-only trainable)",
    }, args.out_ckpt)
    print(f"\n[save] trained decoder -> {args.out_ckpt}", flush=True)

    # ---- 10k HMM eval for apples-to-apples comparison w/ original Dec A ----
    print(f"\n[eval] 10k HMM top-1 (bs={EVAL_BATCH_SIZE}, seed={EVAL_SEED}, "
          f"50/50 task_state, FB=1.0)", flush=True)
    # Load original Dec A for comparison (back-compat fallback for legacy ckpts)
    decA_orig = nn.Linear(n_ori, n_ori, bias=True).to(device)
    decA_state = None
    if isinstance(ckpt.get("loss_heads"), dict) and "orientation_decoder" in ckpt["loss_heads"]:
        decA_state = ckpt["loss_heads"]["orientation_decoder"]
        decA_src = "ckpt.loss_heads.orientation_decoder"
    elif "decoder_state" in ckpt:
        decA_state = ckpt["decoder_state"]
        decA_src = "ckpt.decoder_state"
    if decA_state is not None:
        decA_orig.load_state_dict(decA_state)
        decA_orig.eval()
        print(f"[eval] original Dec A loaded from {decA_src}", flush=True)
    else:
        decA_orig = None
        print(f"[eval] WARN no original Dec A in ckpt — gap will be 'None'", flush=True)

    # Build 10k eval pool (fresh stream, rng seeded by EVAL_SEED)
    eval_gen = build_generator(model_cfg, train_cfg, stim_cfg)
    rng_eval = torch.Generator().manual_seed(EVAL_SEED)
    n_eval_batches = EVAL_N_TRIALS // EVAL_BATCH_SIZE  # 625

    decoder.eval()
    t0 = time.time()
    correct_ret = correct_orig = total = 0
    with torch.no_grad():
        for bi in range(n_eval_batches):
            rx, ry = collect_hmm_readouts(
                net, eval_gen,
                batch_size=EVAL_BATCH_SIZE, seq_length=seq_length,
                steps_per=steps_per, win_lo=win_lo, win_hi=win_hi,
                step_deg=step_deg, n_ori=n_ori,
                rng=rng_eval, device=device,
                model_cfg=model_cfg, train_cfg=train_cfg, stim_cfg=stim_cfg,
                stim_noise_seed=EVAL_SEED + bi,
                force_5050=True,
            )
            pred_ret = decoder(rx).argmax(-1)
            correct_ret += int((pred_ret == ry).sum())
            total += int(rx.shape[0])
            if decA_orig is not None:
                pred_orig = decA_orig(rx).argmax(-1)
                correct_orig += int((pred_orig == ry).sum())
            if (bi + 1) % 100 == 0 or bi == 0:
                print(f"  [eval batch {bi+1}/{n_eval_batches}] elapsed={time.time()-t0:.1f}s",
                      flush=True)
    eval_top1_ret = correct_ret / max(total, 1)
    eval_top1_orig = (correct_orig / max(total, 1)) if decA_orig is not None else None
    print(f"[eval] n_trials={total}", flush=True)
    print(f"[eval] Dec A cotrained    top-1 = {eval_top1_ret:.4f}", flush=True)
    if eval_top1_orig is not None:
        gap = eval_top1_ret - eval_top1_orig
        print(f"[eval] Dec A original    top-1 = {eval_top1_orig:.4f}   "
              f"gap(retrained − orig) = {gap:+.4f}", flush=True)

    # ---- Save training JSON ----
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({
            "label": f"Task #3 — Dec A cotrained on frozen {net_name} net",
            "description": (
                "Fresh Linear(36,36)+bias retrained on r_l23 from a FROZEN fully-trained "
                "net, using Dec A's original co-training data schedule: Stage-1 grating "
                "CE for 2000 steps (V1-only forward, no readout window — final r_l23), "
                "Stage-2 HMM stream CE for 5000 steps (full forward, feedback curriculum "
                "burnin→ramp→1.0, per-presentation readout [onset+9:onset+11]). The net "
                "stays frozen: all param requires_grad_(False) before training, asserted. "
                "The ONLY trainable params are the 36*36+36=1332 decoder weights. Seed 42."
            ),
            "net_name": net_name,
            "net_ckpt": args.ckpt,
            "net_config": args.config,
            "protocol": {
                "stage1_n_steps": stage1_n,
                "stage2_n_steps": stage2_n,
                "stage1_lr": stage1_lr,
                "stage1_contrast_range": [stage1_contrast_lo, stage1_contrast_hi],
                "stage2_burnin_steps": burnin_steps,
                "stage2_ramp_steps": ramp_steps,
                "batch_size": batch_size,
                "seq_length": seq_length,
                "steps_on": steps_on,
                "steps_isi": steps_isi,
                "readout_window_stage2": [win_lo, win_hi],
                "readout_window_stage1": "final timestep of _run_v1_only(n_timesteps=20)",
                "seed": SEED,
                "val_seed": VAL_SEED,
                "n_val_batches": N_VAL_BATCHES,
                "n_val_readouts": n_val,
                "task_state_train": "HMM own distribution (task_p_switch from stim_cfg)",
                "task_state_val": "50/50 focused/routine (Task #25 convention)",
                "frozen_param_count": frozen_param_count,
                "trainable_param_count": decoder_param_count,
            },
            "history": history,
            "final": {
                "val_loss": history["val_loss"][-1],
                "val_acc": history["val_acc"][-1],
            },
            "eval_10k_hmm": {
                "n_trials": total,
                "decA_cotrained_top1": eval_top1_ret,
                "decA_original_top1": eval_top1_orig,
                "gap_cotrained_minus_original":
                    (eval_top1_ret - eval_top1_orig) if eval_top1_orig is not None else None,
            },
            "wall_clock_seconds": {
                "stage1": s1_elapsed,
                "stage2": s2_elapsed,
                "eval_10k": time.time() - t0,
            },
        }, f, indent=2)
    print(f"[save] training JSON -> {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
