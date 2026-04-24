"""Task #4 retrain — Decoder D (FB-ON neutral) linear heads per checkpoint.

Retrain of the Task #4 Dec D that was invalidated by the earlier FB-off training:
the FB-off ckpt never saw the manifold it was being tested on (ex/unex trials
have feedback ON at inference). This script trains on the NORMAL FB-ON network,
with a balanced mix of expected and unexpected paired-fork trials.

Training data (per checkpoint):
  - NEW-style paired fork: random start, march at 5° steps for N presentations,
    paired ex/unex at probe. Feedback scale = 1.0. Focused task_state. Cue at
    expected-next-step (same cue for ex and unex branches).
  - For unex: D_deg ∈ U[25°, 90°] signed (± uniform), applied at probe; cue
    unchanged (still at expected-next, mismatching the actual probe).
  - Balanced per (target_ch × branch) cell: N_PER_CELL = 1000 trials ⇒
    72 000 total (36 channels × 2 branches). 10 % held-out val (same balance).
  - Each sample produces (r_l23[9:11].mean, target_ch, branch_tag).

Variants (both trained on the SAME pre-generated readouts):
  - D-raw   : Linear(36, 36)+bias on raw r_l23.
  - D-shape : Linear(36, 36)+bias on r_l23 / (r_l23.sum(1) + 1e-8).

Optim: Adam lr=1e-3 wd=1e-4, CE, early-stop patience 3 on balanced-val top-1,
max 30 epochs, seed 42. Crash-safety ckpt snapshot every 1000 global batches.

Outputs:
  checkpoints/decoder_d_fbON_neutral_raw_<net>.pt
  checkpoints/decoder_d_fbON_neutral_shape_<net>.pt
  results/decoder_d_fbON_neutral_<net>.json  (history + test acc)
  /tmp/decD_fbON_neutral_readouts_<net>.pt  (cached readouts for re-runs)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Tuple

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


N_ORI = 36
READOUT_WIN = (9, 11)
CONTRAST_RANGE = (0.4, 1.0)

# Balanced dataset sizes.
N_TRAIN_PER_CELL = 900          # per (target_ch × branch)
N_VAL_PER_CELL = 100            # 10 % of 1000
# Total = 36 × 2 × 1000 = 72 000 (900 train + 100 val per cell).

N_PRE_PROBE_MIN = 4             # N range mirrors NEW (4..10 inclusive)
N_PRE_PROBE_MAX = 10
UNEX_D_CHAN_MIN = 5             # 5 ch × 5° = 25°
UNEX_D_CHAN_MAX = 18            # 18 ch × 5° = 90°

SEED = 42
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_GEN = 32
BATCH_TRAIN = 256
MAX_EPOCHS = 30
PATIENCE = 3
SAVE_EVERY_BATCHES = 1000


def _build_one_batch(
    net: LaminarV1V2Network,
    model_cfg,
    train_cfg,
    device: torch.device,
    target_ch: torch.Tensor,    # [B] int — probe's true channel per trial
    branch_tag: torch.Tensor,   # [B] int — 0 = ex, 1 = unex
    rng: torch.Generator,
) -> torch.Tensor:
    """Forward one paired-fork batch through the frozen FB-ON network; return [B, N] r_l23 readouts.

    Each trial:
      - N_pre_probe ∈ U{4..10}; dir ∈ {+1, −1} uniform.
      - march_end_ch = target_ch (ex) OR (target_ch − D_signed_ch) mod N (unex, D_signed_ch
        sampled uniformly with |D_signed_ch| ∈ [5, 18]).
      - march orientations: start_ch = (march_end_ch − N_pre * dir) mod N; step 1-ch CW/CCW.
      - probe stim: at target_ch.
      - cue: at march expected-next = march_end_ch + dir (same for ex and unex).
      - task_state = focused [1, 0]. Feedback_scale = 1.0 (network default).
      - contrast ∈ U[0.4, 1.0] per presentation.
    """
    B = int(target_ch.shape[0])
    N = int(model_cfg.n_orientations)
    period = float(model_cfg.orientation_range)
    step_deg = period / N
    steps_on = int(train_cfg.steps_on)
    steps_isi = int(train_cfg.steps_isi)
    steps_per = steps_on + steps_isi

    # Per-trial N_pre_probe and dir.
    N_pre_probe = torch.randint(N_PRE_PROBE_MIN, N_PRE_PROBE_MAX + 1, (B,), generator=rng)
    # Use max N_pre_probe across batch so stim tensor has one shape (shorter trials padded with ISI at the end of march).
    N_pre_max = int(N_pre_probe.max().item())
    directions = (torch.randint(0, 2, (B,), generator=rng) * 2 - 1).long()  # ±1

    # For unex trials: D_signed_ch uniform in ±[5, 18].
    D_abs = torch.randint(UNEX_D_CHAN_MIN, UNEX_D_CHAN_MAX + 1, (B,), generator=rng)
    D_sign = (torch.randint(0, 2, (B,), generator=rng) * 2 - 1).long()
    D_signed = D_abs * D_sign  # [B]
    unex_mask = (branch_tag == 1)
    # march_end_ch = target_ch (ex), or target_ch - D_signed (unex).
    march_end_ch = target_ch.clone()
    march_end_ch[unex_mask] = (target_ch[unex_mask] - D_signed[unex_mask]) % N

    # Build per-presentation orientations: for each trial, the march has N_pre_probe[b] steps
    # ending at march_end_ch[b]. Shorter trials pad the first (N_pre_max - N_pre_probe[b])
    # presentations with the same start_ch so the network still sees valid gratings there.
    T_presentations = N_pre_max + 1
    start_ch = (march_end_ch - N_pre_probe * directions) % N   # [B]
    ori_ch = torch.empty(B, T_presentations, dtype=torch.long)
    for b in range(B):
        pad = N_pre_max - int(N_pre_probe[b].item())
        for t in range(T_presentations):
            if t < pad:
                ori_ch[b, t] = start_ch[b]
            elif t < T_presentations - 1:
                step_from_start = t - pad
                ori_ch[b, t] = (start_ch[b] + step_from_start * directions[b]) % N
            else:
                # probe at target_ch (same for both branches)
                ori_ch[b, t] = target_ch[b]

    # Per-presentation contrasts.
    contrasts = CONTRAST_RANGE[0] + (CONTRAST_RANGE[1] - CONTRAST_RANGE[0]) * torch.rand(
        B, T_presentations, generator=rng
    )

    theta_deg = ori_ch.float() * step_deg   # [B, T]
    # Flatten for generate_grating.
    flat_thetas = theta_deg.reshape(-1)
    flat_contrasts = contrasts.reshape(-1)
    gratings_flat = generate_grating(
        flat_thetas, flat_contrasts,
        n_orientations=N,
        sigma=model_cfg.sigma_ff,
        n=model_cfg.naka_rushton_n,
        c50=model_cfg.naka_rushton_c50,
        period=period,
    )
    gratings = gratings_flat.reshape(B, T_presentations, N)    # [B, T, N]

    T_total = T_presentations * steps_per
    stim_seq = torch.zeros(B, T_total, N, device=device)
    cue_seq = torch.zeros(B, T_total, N, device=device)
    task_seq = torch.zeros(B, T_total, 2, device=device)
    task_seq[..., 0] = 1.0   # focused

    # cue: at every presentation, indicate expected-next = march_end + dir*(t-pad+1).
    # For the probe presentation, cue still points to march-next (the HMM "expected" probe).
    cue_contrast = 1.0
    for b in range(B):
        pad = N_pre_max - int(N_pre_probe[b].item())
        for t in range(T_presentations):
            onset = t * steps_per
            # ON phase: grating.
            stim_seq[b, onset:onset + steps_on, :] = gratings[b, t, :].to(device).unsqueeze(0)
            # Cue: next-expected ch from march.
            if t < pad:
                next_ch = start_ch[b]
            else:
                step_from_start = t - pad + 1   # expected next step
                next_ch = (start_ch[b] + step_from_start * directions[b]) % N
            # Cue as a narrow grating bump at next_ch with unit contrast.
            cue_deg = float(next_ch.item()) * step_deg
            cue_bump = generate_grating(
                torch.tensor([cue_deg]), torch.tensor([cue_contrast]),
                n_orientations=N,
                sigma=model_cfg.sigma_ff,
                n=model_cfg.naka_rushton_n,
                c50=model_cfg.naka_rushton_c50,
                period=period,
            )[0].to(device)   # [N]
            cue_seq[b, onset:onset + steps_on, :] = cue_bump.unsqueeze(0)
            # ISI: zeros (already).

    with torch.no_grad():
        packed = net.pack_inputs(stim_seq, cue_seq, task_seq)
        r_l23_all, _, _ = net.forward(packed)
    probe_onset = (T_presentations - 1) * steps_per
    win_lo = probe_onset + READOUT_WIN[0]
    win_hi = probe_onset + READOUT_WIN[1]
    return r_l23_all[:, win_lo:win_hi, :].mean(dim=1).detach()   # [B, N]


def build_balanced_dataset(
    net: LaminarV1V2Network,
    model_cfg,
    train_cfg,
    device: torch.device,
    n_train_per_cell: int,
    n_val_per_cell: int,
    seed: int,
) -> dict:
    """Generate balanced-by-(orientation, branch) (r_win, target_ch, branch) data splits.

    For each (target_ch, branch) cell: n_train_per_cell train + n_val_per_cell val samples.
    Total per split: 36 × 2 × n_{train,val}_per_cell.
    """
    N = int(model_cfg.n_orientations)
    rng = torch.Generator()
    rng.manual_seed(seed)

    def _make_split(n_per_cell: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        total = n_per_cell * N * 2
        # Build (target_ch, branch) assignment: n_per_cell copies of each (ch, branch) cell.
        tc = torch.arange(N).repeat_interleave(n_per_cell).repeat(2)    # [2·N·n_per_cell]
        br = torch.cat([torch.zeros(N * n_per_cell, dtype=torch.long),
                        torch.ones(N * n_per_cell, dtype=torch.long)])
        perm = torch.randperm(total, generator=rng)
        tc = tc[perm]
        br = br[perm]
        all_X = torch.empty(total, N, device=device)
        for i0 in range(0, total, BATCH_GEN):
            i1 = min(i0 + BATCH_GEN, total)
            rb = _build_one_batch(
                net, model_cfg, train_cfg, device,
                target_ch=tc[i0:i1], branch_tag=br[i0:i1], rng=rng,
            )
            all_X[i0:i1] = rb
        return all_X, tc.to(device), br.to(device)

    t0 = time.time()
    print(f"  [data-gen] train ({n_train_per_cell}/cell × 36 × 2)...", flush=True)
    X_tr, y_tr, br_tr = _make_split(n_train_per_cell)
    print(f"  [data-gen] train done: X_tr.shape={tuple(X_tr.shape)}  "
          f"elapsed={time.time()-t0:.1f}s", flush=True)
    print(f"  [data-gen] val ({n_val_per_cell}/cell × 36 × 2)...", flush=True)
    X_va, y_va, br_va = _make_split(n_val_per_cell)
    print(f"  [data-gen] val   done: X_va.shape={tuple(X_va.shape)}  "
          f"elapsed={time.time()-t0:.1f}s", flush=True)
    return {
        "X_train": X_tr, "y_train": y_tr, "branch_train": br_tr,
        "X_val":   X_va, "y_val":   y_va, "branch_val":   br_va,
        "gen_time_s": float(time.time() - t0),
    }


def _balanced_accuracy(pred: torch.Tensor, y: torch.Tensor, n_classes: int) -> float:
    accs = []
    for c in range(n_classes):
        mask = (y == c)
        if mask.sum() == 0:
            continue
        accs.append(float((pred[mask] == c).float().mean().item()))
    return float(np.mean(accs))


def _balanced_per_branch(pred: torch.Tensor, y: torch.Tensor,
                         branch: torch.Tensor, n_classes: int) -> dict:
    out = {}
    for b, lab in enumerate(["ex", "unex"]):
        mask = branch == b
        if mask.sum() == 0:
            out[lab] = None
            continue
        out[lab + "_acc"] = float((pred[mask] == y[mask]).float().mean().item())
        out[lab + "_bal_acc"] = _balanced_accuracy(pred[mask], y[mask], n_classes)
        out[lab + "_n"] = int(mask.sum().item())
    return out


def train_linear(
    decoder: nn.Linear,
    X_tr, y_tr, br_tr, X_va, y_va, br_va,
    n_classes: int,
    save_snapshot_path: str | None = None,
) -> dict:
    opt = torch.optim.Adam(decoder.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    n_tr = int(X_tr.shape[0])
    best_val_bal = -1.0
    best_state = None
    best_epoch = -1
    epochs_since = 0
    history = {"epoch": [], "train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": [], "val_bal_acc": [],
               "val_ex_acc": [], "val_unex_acc": []}
    global_batch = 0

    for epoch in range(MAX_EPOCHS):
        decoder.train()
        perm = torch.randperm(n_tr)
        tr_losses = []
        tr_correct = 0
        for i0 in range(0, n_tr, BATCH_TRAIN):
            idx = perm[i0:i0 + BATCH_TRAIN]
            xb = X_tr[idx]
            yb = y_tr[idx]
            opt.zero_grad(set_to_none=True)
            logits = decoder(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            tr_losses.append(float(loss.item()) * int(xb.shape[0]))
            tr_correct += int((logits.argmax(dim=-1) == yb).sum())
            global_batch += 1
            if save_snapshot_path and SAVE_EVERY_BATCHES and (global_batch % SAVE_EVERY_BATCHES == 0):
                tmp = save_snapshot_path + ".tmp"
                torch.save({"state_dict": decoder.state_dict(),
                            "global_batch": int(global_batch),
                            "epoch": int(epoch)}, tmp)
                os.replace(tmp, save_snapshot_path + f".batch{global_batch}")

        tr_loss = sum(tr_losses) / n_tr
        tr_acc = tr_correct / n_tr

        decoder.eval()
        with torch.no_grad():
            logits_va = decoder(X_va)
            va_loss = float(F.cross_entropy(logits_va, y_va).item())
            va_pred = logits_va.argmax(dim=-1)
            va_acc = float((va_pred == y_va).float().mean().item())
            va_bal = _balanced_accuracy(va_pred, y_va, n_classes)
            va_branch = _balanced_per_branch(va_pred, y_va, br_va, n_classes)

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["val_bal_acc"].append(va_bal)
        history["val_ex_acc"].append(va_branch.get("ex_acc"))
        history["val_unex_acc"].append(va_branch.get("unex_acc"))

        improved = va_bal > best_val_bal + 1e-6
        if improved:
            best_val_bal = va_bal
            best_state = {k: v.detach().clone() for k, v in decoder.state_dict().items()}
            best_epoch = epoch
            epochs_since = 0
        else:
            epochs_since += 1

        flag = " *" if improved else ""
        print(f"    epoch {epoch:02d}  tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f}  "
              f"va_loss={va_loss:.4f} va_acc={va_acc:.4f} va_bal={va_bal:.4f}  "
              f"ex={va_branch.get('ex_acc'):.4f} unex={va_branch.get('unex_acc'):.4f}"
              f"{flag}", flush=True)

        if epochs_since >= PATIENCE:
            print(f"    early stop @ epoch {epoch} (no improvement for {PATIENCE} epochs)",
                  flush=True)
            break

    if best_state is not None:
        decoder.load_state_dict(best_state)
    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_bal_acc": best_val_bal,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net-name", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", default=os.path.join(_REPO, "checkpoints"))
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--readouts-cache", default=None)
    ap.add_argument("--n-train-per-cell", type=int, default=N_TRAIN_PER_CELL)
    ap.add_argument("--n-val-per-cell", type=int, default=N_VAL_PER_CELL)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device} net={args.net_name}", flush=True)

    model_cfg, train_cfg, stim_cfg = load_config(args.config)
    n_ori = int(model_cfg.n_orientations)

    # Frozen network, FB ON (default).
    net = LaminarV1V2Network(model_cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    sd = ckpt.get("model_state", ckpt)
    res = net.load_state_dict(sd, strict=False)
    print(f"[net] missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}",
          flush=True)
    net.eval()
    net.oracle_mode = False
    net.feedback_scale.fill_(1.0)    # FB ON (training under normal feedback)
    for p in net.parameters():
        p.requires_grad_(False)
    # Verify frozen.
    for n_, p in net.named_parameters():
        assert not p.requires_grad, f"param {n_} not frozen"
    print(f"[net] frozen, FB_scale={float(net.feedback_scale.item()):.3f}", flush=True)

    cache_path = args.readouts_cache or f"/tmp/decD_fbON_neutral_readouts_{args.net_name}.pt"
    if Path(cache_path).exists():
        print(f"[cache] loading readouts from {cache_path}", flush=True)
        data = torch.load(cache_path, map_location=device, weights_only=False)
    else:
        print(f"[cache] building FB-ON paired-fork readouts → {cache_path}", flush=True)
        data = build_balanced_dataset(
            net, model_cfg, train_cfg, device,
            n_train_per_cell=args.n_train_per_cell,
            n_val_per_cell=args.n_val_per_cell,
            seed=SEED,
        )
        torch.save({k: v.cpu() if torch.is_tensor(v) else v for k, v in data.items()},
                   cache_path)
        print(f"[cache] saved → {cache_path}", flush=True)

    X_tr = data["X_train"].to(device).float()
    y_tr = data["y_train"].to(device).long()
    br_tr = data["branch_train"].to(device).long()
    X_va = data["X_val"].to(device).float()
    y_va = data["y_val"].to(device).long()
    br_va = data["branch_val"].to(device).long()

    def _shape(X): return X / (X.sum(dim=1, keepdim=True) + 1e-8)
    X_tr_shape = _shape(X_tr)
    X_va_shape = _shape(X_va)

    print(f"[data] train n={int(X_tr.shape[0])} (ex={(br_tr==0).sum().item()} "
          f"unex={(br_tr==1).sum().item()})  "
          f"val n={int(X_va.shape[0])} (ex={(br_va==0).sum().item()} "
          f"unex={(br_va==1).sum().item()})", flush=True)

    # Train D-raw
    print(f"\n=== {args.net_name} / D-raw (FB-ON) ===", flush=True)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    dec_raw = nn.Linear(n_ori, n_ori, bias=True).to(device)
    snap_raw = os.path.join(args.out_dir, f"decoder_d_fbON_neutral_raw_{args.net_name}.pt")
    raw_res = train_linear(dec_raw, X_tr, y_tr, br_tr, X_va, y_va, br_va,
                           n_classes=n_ori, save_snapshot_path=snap_raw)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": dec_raw.state_dict(),
        "variant": "D-raw-fbON-neutral",
        "net_name": args.net_name,
        "net_ckpt": args.ckpt,
        "config": args.config,
        "seed": SEED,
        "best_val_bal_acc": raw_res["best_val_bal_acc"],
        "best_epoch": raw_res["best_epoch"],
        "n_train_per_cell": args.n_train_per_cell,
        "n_val_per_cell": args.n_val_per_cell,
        "readout_window": list(READOUT_WIN),
    }, snap_raw)
    print(f"[save] D-raw → {snap_raw}", flush=True)
    print(f"[D-raw {args.net_name}] best_val_bal={raw_res['best_val_bal_acc']:.4f}",
          flush=True)

    # Train D-shape
    print(f"\n=== {args.net_name} / D-shape (FB-ON) ===", flush=True)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    dec_shape = nn.Linear(n_ori, n_ori, bias=True).to(device)
    snap_shape = os.path.join(args.out_dir, f"decoder_d_fbON_neutral_shape_{args.net_name}.pt")
    shape_res = train_linear(dec_shape, X_tr_shape, y_tr, br_tr,
                             X_va_shape, y_va, br_va,
                             n_classes=n_ori, save_snapshot_path=snap_shape)
    torch.save({
        "state_dict": dec_shape.state_dict(),
        "variant": "D-shape-fbON-neutral",
        "net_name": args.net_name,
        "net_ckpt": args.ckpt,
        "config": args.config,
        "seed": SEED,
        "best_val_bal_acc": shape_res["best_val_bal_acc"],
        "best_epoch": shape_res["best_epoch"],
        "n_train_per_cell": args.n_train_per_cell,
        "n_val_per_cell": args.n_val_per_cell,
        "readout_window": list(READOUT_WIN),
    }, snap_shape)
    print(f"[save] D-shape → {snap_shape}", flush=True)
    print(f"[D-shape {args.net_name}] best_val_bal={shape_res['best_val_bal_acc']:.4f}",
          flush=True)

    out_json = args.out_json or os.path.join(
        _REPO, f"results/decoder_d_fbON_neutral_{args.net_name}.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({
            "label": f"Task #4 retrain — Dec D FB-ON neutral linear ({args.net_name})",
            "net_name": args.net_name,
            "net_ckpt": args.ckpt,
            "config": args.config,
            "seed": SEED,
            "design": {
                "n_train_per_cell": args.n_train_per_cell,
                "n_val_per_cell": args.n_val_per_cell,
                "feedback_scale": 1.0,
                "paired_fork_design": "march at 5° for N∈{4..10}, probe at target_ch; "
                                      "unex = D_signed_ch ∈ ±[5, 18]",
                "readout_window": list(READOUT_WIN),
                "contrast_range": list(CONTRAST_RANGE),
                "cue": "expected-next per march (same cue for ex and unex)",
                "task_state": "focused [1, 0]",
                "max_epochs": MAX_EPOCHS,
                "patience": PATIENCE,
                "batch_size_train": BATCH_TRAIN,
                "batch_size_gen": BATCH_GEN,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
            },
            "data_gen_time_s": float(data.get("gen_time_s", -1.0)),
            "d_raw": raw_res,
            "d_shape": shape_res,
        }, f, indent=2)
    print(f"[json] → {out_json}", flush=True)


if __name__ == "__main__":
    main()
