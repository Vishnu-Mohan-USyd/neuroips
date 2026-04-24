"""Task #5 — Decoder E: Dec-A-spec linear head trained on natural HMM stream
with the HMM's OWN stochastic task_state (NOT pinned focused, NOT 50/50).

Dec E's relationship to the three existing trained-linear decoders:
  - Dec A  : `Linear(36, 36)` co-trained with L2/3+PV during Stage 1 (moving
             target — L2/3 changes every step). Then continues training in
             Stage 2 alongside V2 (see `src/training/stage2_feedback.py`).
  - Dec A' : Same arch retrained on frozen, fully-trained R1+R2 L2/3 with 50/50
             focused/routine task_state (Task #25 convention).
  - Dec E  : Same arch retrained on frozen, fully-trained R1+R2 L2/3 with the
             HMMSequenceGenerator's OWN task_state distribution — a Markov
             per-presentation process governed by `stim_cfg.task_p_switch = 0.2`.
             Cues left as HMM produces (75 % valid).

Architecture: `Linear(36, 36)` + bias, seed 42.
Training: Adam lr=1e-3, **no weight decay**, CE on (logits, true_ch) at readout
          window `t ∈ [9, 11]` mean. 5000 gradient steps. Val pool seed=1234,
          ~8 k readouts (10 batches × 32 × 25).
Network : frozen (every param `requires_grad_(False)`; asserted at setup),
          `net.eval()`, `feedback_scale = 1.0`.
Outputs : `checkpoints/decoder_e.pt`, `results/decoder_e_training.json`.

R1+R2 only per the brief — per-legacy Dec E is out of scope for this task.
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
from src.stimulus.sequences import HMMSequenceGenerator
from src.training.trainer import build_stimulus_sequence


SEED = 42
VAL_SEED = 1234
N_VAL_BATCHES = 10
READOUT_WIN = (9, 11)
SEQ_LENGTH = 25
BATCH_SIZE = 32
N_GRAD_STEPS = 5_000
LR = 1e-3            # matches stage1_lr (Dec A's effective LR in Stage 2).
WEIGHT_DECAY = 0.0   # Brief: "no weight decay".
LOG_EVERY = 500

R1R2_CKPT_DEFAULT = os.path.join(_REPO, "results/simple_dual/emergent_seed42/checkpoint.pt")
CONFIG_DEFAULT = os.path.join(_REPO, "config/sweep/sweep_rescue_1_2.yaml")
CKPT_OUT_DEFAULT = os.path.join(_REPO, "checkpoints/decoder_e.pt")
JSON_OUT_DEFAULT = os.path.join(_REPO, "results/decoder_e_training.json")


def build_generator(model_cfg, train_cfg, stim_cfg) -> HMMSequenceGenerator:
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
        # Task #5: honor the yaml's task_p_switch (0.2 in sweep_rescue_1_2.yaml) —
        # HMM emits stochastic task_state per presentation. The generator falls
        # back to the default 0.0 (Bernoulli-per-batch) if task_p_switch is not
        # provided via the config — but sweep_rescue_1_2.yaml sets it to 0.2.
        task_p_switch=getattr(stim_cfg, "task_p_switch", 0.0),
    )


def collect_readouts_labels(
    net,
    generator: HMMSequenceGenerator,
    *,
    batch_size: int,
    seq_length: int,
    steps_per: int,
    win_lo: int,
    win_hi: int,
    step_deg: float,
    n_ori: int,
    rng: torch.Generator,
    device,
    model_cfg,
    train_cfg,
    stim_cfg,
    stim_noise_seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one HMM batch through the FROZEN network; return (readouts, true_ch).

    Uses the HMMSequenceGenerator's OWN task_state (stochastic Markov if
    ``task_p_switch > 0`` in stim_cfg, else Bernoulli-per-batch) — NOT a 50/50
    override. This is the key distinction from Dec A'.
    """
    md = generator.generate(batch_size, seq_length, generator=rng)
    # No task_state override — HMM's own distribution is used as-is.

    if stim_noise_seed is not None:
        torch.manual_seed(stim_noise_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(stim_noise_seed)

    stim_seq, cue_seq, ts_seq, _, _, _ = build_stimulus_sequence(
        md, model_cfg, train_cfg, stim_cfg
    )
    stim_seq = stim_seq.to(device)
    cue_seq = cue_seq.to(device)
    ts_seq = ts_seq.to(device)

    with torch.no_grad():
        packed = net.pack_inputs(stim_seq, cue_seq, ts_seq)
        r_l23_all, _, _ = net.forward(packed)

    B = batch_size
    S = seq_length
    N = n_ori
    readouts = torch.empty(B * S, N, device=device)
    for s in range(S):
        onset = s * steps_per
        readouts[s * B:(s + 1) * B] = r_l23_all[:, onset + win_lo:onset + win_hi, :].mean(dim=1)
    true_ori = md.orientations.to(device)
    true_ch = (true_ori / step_deg).round().long() % n_ori
    labels = true_ch.transpose(0, 1).reshape(-1)
    # Also track branch stats (per-presentation focused fraction) for reporting.
    task_states = md.task_states.to(device)  # [B, S, 2]
    focused_frac = task_states[..., 0].mean().item()
    return readouts.detach(), labels.detach(), float(focused_frac)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=R1R2_CKPT_DEFAULT)
    ap.add_argument("--config", default=CONFIG_DEFAULT)
    ap.add_argument("--out-ckpt", default=CKPT_OUT_DEFAULT)
    ap.add_argument("--out-json", default=JSON_OUT_DEFAULT)
    ap.add_argument("--n-steps", type=int, default=N_GRAD_STEPS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--seq-length", type=int, default=SEQ_LENGTH)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--log-every", type=int, default=LOG_EVERY)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}", flush=True)

    model_cfg, train_cfg, stim_cfg = load_config(args.config)
    n_ori = int(model_cfg.n_orientations)
    period = float(model_cfg.orientation_range)
    step_deg = period / n_ori
    steps_on = int(train_cfg.steps_on)
    steps_isi = int(train_cfg.steps_isi)
    steps_per = steps_on + steps_isi
    win_lo, win_hi = READOUT_WIN
    task_p_switch = getattr(stim_cfg, "task_p_switch", 0.0)
    print(
        f"[cfg] N={n_ori} period={period} step_deg={step_deg:.3f} "
        f"seq_length={args.seq_length} batch={args.batch_size} "
        f"steps_per={steps_per} readout_window=[{win_lo}:{win_hi}] "
        f"amb_frac={train_cfg.ambiguous_fraction} "
        f"task_p_switch={task_p_switch} cue_valid_fraction={stim_cfg.cue_valid_fraction}",
        flush=True,
    )

    # FROZEN network
    net = LaminarV1V2Network(model_cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    res = net.load_state_dict(ckpt["model_state"], strict=False)
    print(f"[net] loaded {args.ckpt}  missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}",
          flush=True)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    for n, p in net.named_parameters():
        assert not p.requires_grad, f"param {n} not frozen"
    net.oracle_mode = False
    net.feedback_scale.fill_(1.0)
    print(f"[net] ALL parameters frozen (verified); FB_scale=1.000", flush=True)

    # Dec E head — fresh Linear(36, 36)+bias, seed 42.
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    decE = nn.Linear(n_ori, n_ori, bias=True).to(device)
    for p in decE.parameters():
        p.requires_grad_(True)
    print(f"[decE] fresh Linear({n_ori}, {n_ori}) init seed={SEED}", flush=True)

    opt = torch.optim.Adam(decE.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    print(f"[opt] Adam lr={args.lr} wd={WEIGHT_DECAY} | n_steps={args.n_steps}", flush=True)

    gen_train = build_generator(model_cfg, train_cfg, stim_cfg)
    gen_val = build_generator(model_cfg, train_cfg, stim_cfg)
    rng_train = torch.Generator().manual_seed(SEED)
    rng_val = torch.Generator().manual_seed(VAL_SEED)

    # ---- Val pool (seed 1234, 10 batches × 32 × 25 ≈ 8 000 readouts) ----
    print(f"[val] building fixed pool: {N_VAL_BATCHES} batches × {args.batch_size} × {args.seq_length} "
          f"= {N_VAL_BATCHES * args.batch_size * args.seq_length} readouts (seed={VAL_SEED})",
          flush=True)
    val_X = []
    val_y = []
    val_foc_fracs = []
    for vi in range(N_VAL_BATCHES):
        rx, ry, foc = collect_readouts_labels(
            net, gen_val,
            batch_size=args.batch_size, seq_length=args.seq_length,
            steps_per=steps_per, win_lo=win_lo, win_hi=win_hi,
            step_deg=step_deg, n_ori=n_ori,
            rng=rng_val, device=device,
            model_cfg=model_cfg, train_cfg=train_cfg, stim_cfg=stim_cfg,
            stim_noise_seed=VAL_SEED + vi,
        )
        val_X.append(rx)
        val_y.append(ry)
        val_foc_fracs.append(foc)
    val_X = torch.cat(val_X, dim=0)
    val_y = torch.cat(val_y, dim=0)
    n_val = int(val_X.shape[0])
    val_foc_mean = float(np.mean(val_foc_fracs))
    print(f"[val] pool ready n={n_val}  mean_focused_frac={val_foc_mean:.4f}", flush=True)

    @torch.no_grad()
    def eval_val() -> tuple[float, float]:
        decE.eval()
        logits = decE(val_X)
        loss = F.cross_entropy(logits, val_y).item()
        acc = (logits.argmax(-1) == val_y).float().mean().item()
        decE.train()
        return float(loss), float(acc)

    val_loss0, val_acc0 = eval_val()
    print(f"[val @ step 0] loss={val_loss0:.4f} acc={val_acc0:.4f}", flush=True)

    history = {"step": [0], "train_loss": [float("nan")], "train_acc": [float("nan")],
               "val_loss": [val_loss0], "val_acc": [val_acc0],
               "train_focused_frac_running": []}

    decE.train()
    running_loss = 0.0
    running_correct = 0
    running_n = 0
    running_focused_sum = 0.0
    running_focused_batches = 0
    t0 = time.time()
    for step in range(1, args.n_steps + 1):
        rx, ry, foc = collect_readouts_labels(
            net, gen_train,
            batch_size=args.batch_size, seq_length=args.seq_length,
            steps_per=steps_per, win_lo=win_lo, win_hi=win_hi,
            step_deg=step_deg, n_ori=n_ori,
            rng=rng_train, device=device,
            model_cfg=model_cfg, train_cfg=train_cfg, stim_cfg=stim_cfg,
            stim_noise_seed=SEED + step,
        )
        opt.zero_grad(set_to_none=True)
        logits = decE(rx)
        loss = F.cross_entropy(logits, ry)
        loss.backward()
        opt.step()

        with torch.no_grad():
            running_loss += float(loss.item()) * int(rx.shape[0])
            running_correct += int((logits.argmax(-1) == ry).sum())
            running_n += int(rx.shape[0])
            running_focused_sum += foc
            running_focused_batches += 1

        if step % args.log_every == 0 or step == args.n_steps:
            train_loss = running_loss / max(running_n, 1)
            train_acc = running_correct / max(running_n, 1)
            focused_frac_running = running_focused_sum / max(running_focused_batches, 1)
            val_loss, val_acc = eval_val()
            elapsed = time.time() - t0
            rate = step / max(elapsed, 1e-6)
            print(
                f"[step {step:6d}/{args.n_steps}] "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  "
                f"focused_frac={focused_frac_running:.4f}  "
                f"elapsed={elapsed:.0f}s rate={rate:.2f} st/s",
                flush=True,
            )
            history["step"].append(int(step))
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["train_focused_frac_running"].append(focused_frac_running)
            running_loss = 0.0
            running_correct = 0
            running_n = 0
            running_focused_sum = 0.0
            running_focused_batches = 0

        if args.save_every > 0 and step % args.save_every == 0 and step < args.n_steps:
            Path(args.out_ckpt).parent.mkdir(parents=True, exist_ok=True)
            snap_tmp = args.out_ckpt + ".tmp"
            snap_final = args.out_ckpt + f".step{step}"
            torch.save({
                "state_dict": decE.state_dict(),
                "step": int(step),
                "n_steps_target": args.n_steps,
                "history_so_far": history,
            }, snap_tmp)
            os.replace(snap_tmp, snap_final)
            print(f"[ckpt @ step {step}] snapshot -> {snap_final}", flush=True)

    total_elapsed = time.time() - t0
    print(f"[done] trained {args.n_steps} steps in {total_elapsed:.1f}s "
          f"({args.n_steps/total_elapsed:.2f} st/s)", flush=True)

    # Save FIRST so a post-training comparison crash doesn't lose the model.
    Path(args.out_ckpt).parent.mkdir(parents=True, exist_ok=True)
    pre_cmp_tmp = args.out_ckpt + ".precmp.tmp"
    torch.save({
        "state_dict": decE.state_dict(),
        "arch": f"Linear({n_ori}, {n_ori})+bias",
        "seed": SEED,
        "n_steps": args.n_steps,
        "status": "training_done_cmp_pending",
    }, pre_cmp_tmp)
    os.replace(pre_cmp_tmp, args.out_ckpt + ".precmp")

    # Dec A original comparison on same val pool (back-compat: fall back to
    # decoder_state for legacy ckpts that lack `loss_heads`).
    val_loss_orig = None
    val_acc_orig = None
    decA_state = None
    if isinstance(ckpt.get("loss_heads"), dict) and "orientation_decoder" in ckpt["loss_heads"]:
        decA_state = ckpt["loss_heads"]["orientation_decoder"]
    elif "decoder_state" in ckpt:
        decA_state = ckpt["decoder_state"]
    if decA_state is not None:
        decA_orig = nn.Linear(n_ori, n_ori, bias=True).to(device)
        decA_orig.load_state_dict(decA_state)
        decA_orig.eval()
        with torch.no_grad():
            logits_orig = decA_orig(val_X)
            val_loss_orig = float(F.cross_entropy(logits_orig, val_y).item())
            val_acc_orig = float((logits_orig.argmax(-1) == val_y).float().mean().item())
        print(f"[cmp] original Dec A on SAME val pool: loss={val_loss_orig:.4f} acc={val_acc_orig:.4f}",
              flush=True)
    else:
        print(f"[cmp] WARN: no 'loss_heads.orientation_decoder' or 'decoder_state' in ckpt; "
              f"skipping Dec A comparison", flush=True)

    torch.save({
        "state_dict": decE.state_dict(),
        "arch": f"Linear({n_ori}, {n_ori})+bias",
        "seed": SEED,
        "val_seed": VAL_SEED,
        "n_steps": args.n_steps,
        "lr": args.lr,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": args.batch_size,
        "seq_length": args.seq_length,
        "readout_window": [win_lo, win_hi],
        "net_ckpt": args.ckpt,
        "config_path": args.config,
        "task_p_switch": task_p_switch,
        "cue_valid_fraction": stim_cfg.cue_valid_fraction,
        "final_val_loss": history["val_loss"][-1],
        "final_val_acc": history["val_acc"][-1],
        "val_focused_frac_mean": val_foc_mean,
        "compare_decA_orig_on_val_pool": (
            {"val_loss": val_loss_orig, "val_acc": val_acc_orig}
            if val_loss_orig is not None else None
        ),
    }, args.out_ckpt)
    # Clean up the precmp stash.
    try:
        os.remove(args.out_ckpt + ".precmp")
    except FileNotFoundError:
        pass
    print(f"[save] ckpt -> {args.out_ckpt}", flush=True)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({
            "label": "Task #5 — Dec E training on frozen R1+R2 L2/3 with HMM stochastic task_state",
            "description": (
                "Fresh Linear(36,36)+bias trained on r_l23 from frozen R1+R2 "
                "emergent_seed42 network. Natural HMM stream with the HMM's "
                "OWN task_state (Markov p_switch=0.2 per sweep_rescue_1_2.yaml) — "
                "NOT pinned focused, NOT 50/50. Adam lr=1e-3 no weight decay; "
                "5000 grad steps. Val pool seed=1234."
            ),
            "history": history,
            "config": {
                "n_steps": args.n_steps,
                "lr": args.lr,
                "weight_decay": WEIGHT_DECAY,
                "batch_size": args.batch_size,
                "seq_length": args.seq_length,
                "readout_window": [win_lo, win_hi],
                "seed": SEED,
                "val_seed": VAL_SEED,
                "n_val_batches": N_VAL_BATCHES,
                "n_val_readouts": n_val,
                "steps_on": steps_on,
                "steps_isi": steps_isi,
                "ambiguous_fraction": float(train_cfg.ambiguous_fraction),
                "task_state_mode": ("HMM-own stochastic (Markov task_p_switch="
                                    f"{task_p_switch})"),
                "cue_valid_fraction": float(stim_cfg.cue_valid_fraction),
            },
            "net_ckpt": args.ckpt,
            "net_config": args.config,
            "final": {
                "val_loss": history["val_loss"][-1],
                "val_acc": history["val_acc"][-1],
            },
            "val_focused_frac_mean": val_foc_mean,
            "compare_decA_orig_on_val_pool": (
                {"val_loss": val_loss_orig, "val_acc": val_acc_orig}
                if val_loss_orig is not None else None
            ),
            "wall_clock_seconds": total_elapsed,
        }, f, indent=2)
    print(f"[save] json -> {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
