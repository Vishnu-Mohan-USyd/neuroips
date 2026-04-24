"""Task #1 (Dec A' line) — Train Decoder A' on STABLE (frozen) L2/3.

Dec A is the 36→36 linear orientation decoder that was trained JOINTLY with
L2/3+PV during Stage 1 (cf. ``src/training/stage1_sensory.py:127-163``). This
means Dec A's weights were fit to a moving target: L2/3 representations
changed step-by-step as the network learned.

Dec A' retrains a fresh 36→36 linear decoder on the SAME frozen network (R1+R2
emergent_seed42 ckpt) — i.e. on stable, late-training L2/3. Net is entirely
frozen, gradients flow only through the new decoder weights.

Design (approved by Lead):
  - Checkpoint : results/simple_dual/emergent_seed42/checkpoint.pt
  - Batch       : B=32, seq_length=25 → 800 readouts / grad step
  - Stream      : fresh natural-HMM batch per grad step through frozen net
  - Ambiguous   : kept in (natural stream fidelity)
  - Task state  : 50/50 focused/routine per batch
  - Val pool    : fixed, seed=1234, 10 batches (~8,000 readouts)
  - Optim       : Adam, lr=1e-3, 20,000 gradient steps
  - Seed        : 42 (decoder init + training stream)
  - Eval        : val acc + val loss every 500 steps; last-step checkpoint saved.

Readout window (matches Task #25 / Dec C eval): ``r_l23[:, onset+9:onset+11].mean(1)``.

Outputs:
  checkpoints/decoder_a_prime.pt
  results/decoder_a_prime_training.json   (loss/acc curve, config snapshot)

No gradient flows into the frozen network — verified by ``assert`` at setup and
by iterating all ``net.parameters()`` with ``requires_grad=False``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS_DIR = os.path.dirname(__file__)
_REPO = os.path.abspath(os.path.join(_THIS_DIR, ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, _THIS_DIR)

from src.config import load_config, MechanismType  # noqa: F401  (shim for legacy ckpts)
from src.model.network import LaminarV1V2Network
from src.stimulus.sequences import HMMSequenceGenerator
from src.training.trainer import build_stimulus_sequence


SEED = 42
VAL_SEED = 1234
N_VAL_BATCHES = 10
READOUT_WIN = (9, 11)
SEQ_LENGTH = 25
BATCH_SIZE = 32
N_GRAD_STEPS = 20_000
LR = 1e-3
LOG_EVERY = 500

R1R2_CKPT_DEFAULT = os.path.join(_REPO, "results/simple_dual/emergent_seed42/checkpoint.pt")
CONFIG_DEFAULT = os.path.join(_REPO, "config/sweep/sweep_rescue_1_2.yaml")
CKPT_OUT_DEFAULT = os.path.join(_REPO, "checkpoints/decoder_a_prime.pt")
JSON_OUT_DEFAULT = os.path.join(_REPO, "results/decoder_a_prime_training.json")


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
    )


def set_50_50_task_state(md, batch_size: int, device) -> None:
    """Mutate metadata.task_states to 50/50 focused/routine per trial (Task #25 style)."""
    ts_mode = (torch.arange(batch_size) < batch_size // 2).long()  # first half focused
    new_ts = torch.zeros_like(md.task_states)
    new_ts[..., 0] = ts_mode.float().unsqueeze(-1)        # focused channel
    new_ts[..., 1] = (1 - ts_mode).float().unsqueeze(-1)  # routine channel
    md.task_states = new_ts


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
    """Run one HMM batch through the FROZEN network.

    Returns:
        readouts : [B*seq_length, N]  mean r_l23 over readout window per presentation
        labels   : [B*seq_length]    true channel index per presentation
    """
    md = generator.generate(batch_size, seq_length, generator=rng)
    set_50_50_task_state(md, batch_size, device)

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
        r_l23_all, _, _ = net.forward(packed)   # [B, T_total, N]

    B = batch_size
    S = seq_length
    N = n_ori
    # Extract [win_lo:win_hi].mean(dim=1) per presentation onset.
    # onsets: s * steps_per for s=0..S-1.
    readouts = torch.empty(B * S, N, device=device)
    for s in range(S):
        onset = s * steps_per
        readouts[s * B:(s + 1) * B] = r_l23_all[:, onset + win_lo:onset + win_hi, :].mean(dim=1)
    # True channel per presentation
    true_ori = md.orientations.to(device)                 # [B, S]
    true_ch = (true_ori / step_deg).round().long() % n_ori  # [B, S]
    # Flatten in the same (s * B + b) order as readouts above.
    labels = true_ch.transpose(0, 1).reshape(-1)  # [S*B]
    return readouts.detach(), labels.detach()


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
    ap.add_argument("--save-every", type=int, default=0,
                    help="If >0, atomically save ckpt snapshot every N steps (crash safety).")
    ap.add_argument("--device", default=None, help="cpu|cuda (auto-detect if None)")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}", flush=True)

    # ---- Config ----
    model_cfg, train_cfg, stim_cfg = load_config(args.config)
    n_ori = int(model_cfg.n_orientations)
    period = float(model_cfg.orientation_range)
    step_deg = period / n_ori
    steps_on = int(train_cfg.steps_on)
    steps_isi = int(train_cfg.steps_isi)
    steps_per = steps_on + steps_isi
    win_lo, win_hi = READOUT_WIN
    print(
        f"[cfg] N={n_ori} period={period} step_deg={step_deg:.3f} "
        f"seq_length={args.seq_length} batch={args.batch_size} "
        f"steps_per={steps_per} readout_window=[{win_lo}:{win_hi}] "
        f"amb_frac={train_cfg.ambiguous_fraction}",
        flush=True,
    )

    # ---- Network (FROZEN) ----
    net = LaminarV1V2Network(model_cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    sd = ckpt["model_state"]
    res = net.load_state_dict(sd, strict=False)
    print(
        f"[net] loaded {args.ckpt} "
        f"missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}",
        flush=True,
    )
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    # Sanity — every net param is frozen.
    for n, p in net.named_parameters():
        assert not p.requires_grad, f"param {n} is NOT frozen"
    net.oracle_mode = False
    net.feedback_scale.fill_(1.0)
    print(f"[net] ALL parameters frozen (verified)", flush=True)

    # ---- Decoder A' (fresh Linear(36,36), seed 42) ----
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    decA_prime = nn.Linear(n_ori, n_ori, bias=True).to(device)
    for p in decA_prime.parameters():
        p.requires_grad_(True)
    print(
        f"[decA'] fresh Linear({n_ori}, {n_ori}) init seed={SEED} | "
        f"W_std={decA_prime.weight.std().item():.4f} b_mean={decA_prime.bias.mean().item():.4e}",
        flush=True,
    )

    opt = torch.optim.Adam(decA_prime.parameters(), lr=args.lr)
    print(f"[opt] Adam lr={args.lr} | n_steps={args.n_steps} log_every={args.log_every}", flush=True)

    # ---- HMM generators: train vs val ----
    gen_train = build_generator(model_cfg, train_cfg, stim_cfg)
    gen_val = build_generator(model_cfg, train_cfg, stim_cfg)

    rng_train = torch.Generator().manual_seed(SEED)
    rng_val = torch.Generator().manual_seed(VAL_SEED)

    # ---- Build fixed val pool ----
    print(f"[val] building fixed pool: {N_VAL_BATCHES} batches × {args.batch_size} × {args.seq_length} "
          f"= {N_VAL_BATCHES * args.batch_size * args.seq_length} readouts (seed={VAL_SEED})",
          flush=True)
    val_X = []
    val_y = []
    t0 = time.time()
    for vi in range(N_VAL_BATCHES):
        rx, ry = collect_readouts_labels(
            net, gen_val,
            batch_size=args.batch_size, seq_length=args.seq_length,
            steps_per=steps_per, win_lo=win_lo, win_hi=win_hi,
            step_deg=step_deg, n_ori=n_ori,
            rng=rng_val, device=device,
            model_cfg=model_cfg, train_cfg=train_cfg, stim_cfg=stim_cfg,
            stim_noise_seed=VAL_SEED + vi,  # deterministic val noise
        )
        val_X.append(rx)
        val_y.append(ry)
    val_X = torch.cat(val_X, dim=0)   # [N_val, N]
    val_y = torch.cat(val_y, dim=0)   # [N_val]
    n_val = int(val_X.shape[0])
    print(f"[val] pool ready n={n_val} in {time.time()-t0:.1f}s", flush=True)

    @torch.no_grad()
    def eval_val() -> tuple[float, float]:
        decA_prime.eval()
        logits = decA_prime(val_X)
        loss = F.cross_entropy(logits, val_y).item()
        acc = (logits.argmax(dim=-1) == val_y).float().mean().item()
        decA_prime.train()
        return float(loss), float(acc)

    # ---- Initial val (step 0, untrained Dec A') ----
    val_loss0, val_acc0 = eval_val()
    print(f"[val @ step 0] loss={val_loss0:.4f} acc={val_acc0:.4f}", flush=True)

    # ---- Training loop ----
    decA_prime.train()
    history = {
        "step": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    # Seed the pre-loop entry.
    history["step"].append(0)
    history["train_loss"].append(float("nan"))
    history["train_acc"].append(float("nan"))
    history["val_loss"].append(val_loss0)
    history["val_acc"].append(val_acc0)

    t0 = time.time()
    running_loss = 0.0
    running_correct = 0
    running_n = 0
    for step in range(1, args.n_steps + 1):
        # Stream a fresh batch through frozen net.
        rx, ry = collect_readouts_labels(
            net, gen_train,
            batch_size=args.batch_size, seq_length=args.seq_length,
            steps_per=steps_per, win_lo=win_lo, win_hi=win_hi,
            step_deg=step_deg, n_ori=n_ori,
            rng=rng_train, device=device,
            model_cfg=model_cfg, train_cfg=train_cfg, stim_cfg=stim_cfg,
            stim_noise_seed=SEED + step,  # deterministic per-step noise
        )
        opt.zero_grad(set_to_none=True)
        logits = decA_prime(rx)
        loss = F.cross_entropy(logits, ry)
        loss.backward()
        opt.step()

        with torch.no_grad():
            running_loss += float(loss.item()) * int(rx.shape[0])
            running_correct += int((logits.argmax(dim=-1) == ry).sum())
            running_n += int(rx.shape[0])

        if step % args.log_every == 0 or step == args.n_steps:
            train_loss = running_loss / max(running_n, 1)
            train_acc = running_correct / max(running_n, 1)
            val_loss, val_acc = eval_val()
            elapsed = time.time() - t0
            rate = step / max(elapsed, 1e-6)
            eta = (args.n_steps - step) / max(rate, 1e-6)
            print(
                f"[step {step:6d}/{args.n_steps}] "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
                f"elapsed={elapsed:.0f}s rate={rate:.2f} st/s eta={eta:.0f}s",
                flush=True,
            )
            history["step"].append(int(step))
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            running_loss = 0.0
            running_correct = 0
            running_n = 0

        # Periodic checkpoint snapshot (crash safety).
        if args.save_every > 0 and step % args.save_every == 0 and step < args.n_steps:
            Path(args.out_ckpt).parent.mkdir(parents=True, exist_ok=True)
            snap_tmp = args.out_ckpt + ".tmp"
            snap_final = args.out_ckpt + f".step{step}"
            snap = {
                "state_dict": decA_prime.state_dict(),
                "arch": f"Linear({n_ori}, {n_ori})",
                "seed": SEED,
                "val_seed": VAL_SEED,
                "step": int(step),
                "n_steps_target": args.n_steps,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "seq_length": args.seq_length,
                "readout_window": [win_lo, win_hi],
                "net_ckpt": args.ckpt,
                "config_path": args.config,
                "history_so_far": history,
            }
            torch.save(snap, snap_tmp)
            os.replace(snap_tmp, snap_final)
            print(f"[ckpt @ step {step}] snapshot -> {snap_final}", flush=True)

    total_elapsed = time.time() - t0
    print(f"[done] trained {args.n_steps} steps in {total_elapsed:.1f}s "
          f"({args.n_steps/total_elapsed:.2f} st/s)", flush=True)

    # Save FIRST (pre-cmp stash) so a post-training comparison crash can't lose
    # the trained decoder. Matches the Dec E crash-safety pattern.
    Path(args.out_ckpt).parent.mkdir(parents=True, exist_ok=True)
    pre_cmp_tmp = args.out_ckpt + ".precmp.tmp"
    torch.save({
        "state_dict": decA_prime.state_dict(),
        "arch": f"Linear({n_ori}, {n_ori})",
        "seed": SEED,
        "n_steps": args.n_steps,
        "status": "training_done_cmp_pending",
    }, pre_cmp_tmp)
    os.replace(pre_cmp_tmp, args.out_ckpt + ".precmp")

    # Dec A original comparison on same val pool — back-compat fallback: legacy
    # ckpts (a1/b1/c1/e1) lack ``loss_heads`` but expose ``decoder_state``.
    # If neither is present, skip the comparison rather than crash.
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
            val_acc_orig = float((logits_orig.argmax(dim=-1) == val_y).float().mean().item())
        print(f"[cmp] original Dec A on SAME val pool: loss={val_loss_orig:.4f} acc={val_acc_orig:.4f}",
              flush=True)
    else:
        print(f"[cmp] WARN: no 'loss_heads.orientation_decoder' or 'decoder_state' in ckpt; "
              f"skipping Dec A comparison", flush=True)

    # ---- Save (final) ----
    torch.save({
        "state_dict": decA_prime.state_dict(),
        "arch": f"Linear({n_ori}, {n_ori})",
        "seed": SEED,
        "val_seed": VAL_SEED,
        "n_steps": args.n_steps,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "seq_length": args.seq_length,
        "readout_window": [win_lo, win_hi],
        "net_ckpt": args.ckpt,
        "config_path": args.config,
        "final_val_loss": history["val_loss"][-1],
        "final_val_acc": history["val_acc"][-1],
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
            "label": "Task #1 — Dec A' training on frozen L2/3 (per-net)",
            "description": (
                "Fresh Linear(36,36) trained on r_l23 from the supplied frozen "
                "network ckpt. Natural HMM stream, 50/50 task state, "
                f"{args.n_steps} grad steps Adam lr={args.lr}, val every "
                f"{args.log_every} steps on fixed seed=1234 pool (~8k readouts). "
                "Back-compat Dec A comparison: uses ckpt['loss_heads'] if present, "
                "else ckpt['decoder_state'] (legacy ckpts); skipped if neither."
            ),
            "history": history,
            "config": {
                "n_steps": args.n_steps,
                "lr": args.lr,
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
                "task_state_mode": "50/50 focused/routine per batch",
            },
            "net_ckpt": args.ckpt,
            "net_config": args.config,
            "final": {
                "val_loss": history["val_loss"][-1],
                "val_acc": history["val_acc"][-1],
            },
            "compare_decA_orig_on_val_pool": (
                {"val_loss": val_loss_orig, "val_acc": val_acc_orig}
                if val_loss_orig is not None else None
            ),
            "wall_clock_seconds": total_elapsed,
        }, f, indent=2)
    print(f"[save] json -> {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
