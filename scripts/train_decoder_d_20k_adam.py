"""Task #8 — Dec D (FB-ON paired-fork balanced) re-train at 20,000 Adam steps.

Disambiguates the 20k Dec A' result on a1/b1 (Δ = +0.21/+0.18) — is it genuine
sharpening from r_l23, or natural-HMM prior-bias overfitting?

Dec D's training data is paired-fork ex+unex BALANCED per (target_ch × branch)
cell — by construction there is no natural-HMM prior bias to exploit. If 20k
Dec D-raw on a1/b1 also reports Δ > 0, the sharpening signal is genuinely in
r_l23 (Task #7 retraction goes wrong direction). If 20k Dec D-raw stays at
Δ ≈ 0 (or negative like 5k Dec D), then 20k Dec A' was overfitting prior bias.

Differences from `scripts/train_decoder_d_fbON_neutral.py`:
  - Trains for EXACTLY n_steps Adam updates (default 20,000), NO early stopping.
  - Optimizer: Adam lr=1e-3, weight_decay=0 (matches Dec A' 20k regime).
  - Reuses cached paired-fork readouts from `/tmp/decD_fbON_neutral_readouts_<net>.pt`
    (saves ~4 min per net of data-gen).
  - Saves the FINAL (not best-val) decoder state to match the Dec A' 20k convention.
  - Logs val_loss / val_acc / val_bal_acc every 500 steps.
  - Trains both Dec D-raw and Dec D-shape variants on the same generated readouts.

Outputs:
  checkpoints/decoder_d_20k_raw_<net>.pt       (final state, 20k steps Adam)
  checkpoints/decoder_d_20k_shape_<net>.pt     (final state, 20k steps Adam)
  results/decoder_d_20k_training_<net>.json    (history + wall_clock per variant)
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

# Reuse data-gen functions from the 5k script.
from train_decoder_d_fbON_neutral import (  # noqa: E402
    N_ORI, READOUT_WIN, BATCH_TRAIN,
    _balanced_accuracy, _balanced_per_branch,
    build_balanced_dataset,
)


SEED = 42
LR = 1e-3                   # matches Dec A' 20k
WEIGHT_DECAY = 0.0          # matches Dec A' 20k (no regularization)
N_STEPS_DEFAULT = 20_000    # 20k Adam steps, NO early stopping
LOG_EVERY = 500


def train_linear_n_steps(
    decoder: nn.Linear,
    X_tr: torch.Tensor, y_tr: torch.Tensor, br_tr: torch.Tensor,
    X_va: torch.Tensor, y_va: torch.Tensor, br_va: torch.Tensor,
    *,
    n_classes: int,
    n_steps: int,
    log_every: int,
    label: str,
) -> dict:
    """Train Linear(N,N)+bias for exactly `n_steps` Adam updates. No early stopping.

    Each step is one batch of size BATCH_TRAIN drawn uniformly from the train set
    via random sampling (NOT epoch-based shuffling).
    """
    opt = torch.optim.Adam(decoder.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    n_tr = int(X_tr.shape[0])

    history = {
        "step": [], "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_bal_acc": [],
        "val_ex_acc": [], "val_unex_acc": [],
    }

    # Initial val
    with torch.no_grad():
        decoder.eval()
        logits_va = decoder(X_va)
        v_loss = float(F.cross_entropy(logits_va, y_va).item())
        v_pred = logits_va.argmax(dim=-1)
        v_acc = float((v_pred == y_va).float().mean().item())
        v_bal = _balanced_accuracy(v_pred, y_va, n_classes)
        v_branch = _balanced_per_branch(v_pred, y_va, br_va, n_classes)
        history["step"].append(0)
        history["train_loss"].append(float("nan"))
        history["train_acc"].append(float("nan"))
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)
        history["val_bal_acc"].append(v_bal)
        history["val_ex_acc"].append(v_branch.get("ex_acc"))
        history["val_unex_acc"].append(v_branch.get("unex_acc"))
        decoder.train()

    print(f"  [{label}] step 0   val_loss={v_loss:.4f}  val_acc={v_acc:.4f}  "
          f"val_bal={v_bal:.4f}  ex={v_branch.get('ex_acc'):.4f} "
          f"unex={v_branch.get('unex_acc'):.4f}",
          flush=True)

    rng = torch.Generator(device=X_tr.device).manual_seed(SEED)

    t0 = time.time()
    running_loss = 0.0
    running_correct = 0
    running_n = 0
    for step in range(1, n_steps + 1):
        idx = torch.randint(0, n_tr, (BATCH_TRAIN,), generator=rng, device=X_tr.device)
        xb = X_tr[idx]
        yb = y_tr[idx]
        opt.zero_grad(set_to_none=True)
        logits = decoder(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        opt.step()
        running_loss += float(loss.item()) * int(xb.shape[0])
        running_correct += int((logits.argmax(dim=-1) == yb).sum())
        running_n += int(xb.shape[0])

        if step % log_every == 0 or step == n_steps:
            tr_loss = running_loss / max(running_n, 1)
            tr_acc = running_correct / max(running_n, 1)
            decoder.eval()
            with torch.no_grad():
                logits_va = decoder(X_va)
                v_loss = float(F.cross_entropy(logits_va, y_va).item())
                v_pred = logits_va.argmax(dim=-1)
                v_acc = float((v_pred == y_va).float().mean().item())
                v_bal = _balanced_accuracy(v_pred, y_va, n_classes)
                v_branch = _balanced_per_branch(v_pred, y_va, br_va, n_classes)
            decoder.train()
            history["step"].append(int(step))
            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["val_loss"].append(v_loss)
            history["val_acc"].append(v_acc)
            history["val_bal_acc"].append(v_bal)
            history["val_ex_acc"].append(v_branch.get("ex_acc"))
            history["val_unex_acc"].append(v_branch.get("unex_acc"))
            elapsed = time.time() - t0
            rate = step / max(elapsed, 1e-6)
            eta = (n_steps - step) / max(rate, 1e-6)
            print(f"  [{label}] step {step:6d}/{n_steps}  "
                  f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f}  "
                  f"val_loss={v_loss:.4f} val_acc={v_acc:.4f} val_bal={v_bal:.4f}  "
                  f"ex={v_branch.get('ex_acc'):.4f} unex={v_branch.get('unex_acc'):.4f}  "
                  f"elapsed={elapsed:.0f}s rate={rate:.1f} st/s eta={eta:.0f}s",
                  flush=True)
            running_loss = 0.0
            running_correct = 0
            running_n = 0

    total_elapsed = time.time() - t0
    print(f"  [{label}] done {n_steps} steps in {total_elapsed:.1f}s "
          f"({n_steps/total_elapsed:.2f} st/s)", flush=True)

    return {
        "history": history,
        "n_steps": int(n_steps),
        "wall_clock_seconds": float(total_elapsed),
        "final_val_acc": float(history["val_acc"][-1]),
        "final_val_bal_acc": float(history["val_bal_acc"][-1]),
        "final_val_ex_acc": history["val_ex_acc"][-1],
        "final_val_unex_acc": history["val_unex_acc"][-1],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net-name", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", default=os.path.join(_REPO, "checkpoints"))
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--readouts-cache", default=None,
                    help="If unset, defaults to /tmp/decD_fbON_neutral_readouts_<net>.pt "
                         "— REUSED from Task #4 5k Dec D run when present.")
    ap.add_argument("--n-steps", type=int, default=N_STEPS_DEFAULT)
    ap.add_argument("--log-every", type=int, default=LOG_EVERY)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}  net={args.net_name}  n_steps={args.n_steps}",
          flush=True)
    print(f"[setup] lr={LR}  wd={WEIGHT_DECAY}  no_early_stop=True", flush=True)

    model_cfg, train_cfg, _stim_cfg = load_config(args.config)
    n_ori = int(model_cfg.n_orientations)
    assert n_ori == N_ORI, f"expected n_orientations={N_ORI}, got {n_ori}"

    cache_path = args.readouts_cache or f"/tmp/decD_fbON_neutral_readouts_{args.net_name}.pt"
    if Path(cache_path).exists():
        print(f"[cache] loading readouts from {cache_path}", flush=True)
        data = torch.load(cache_path, map_location=device, weights_only=False)
    else:
        # Build network only when needed (cache miss).
        print(f"[cache] miss — building paired-fork readouts → {cache_path}", flush=True)
        net = LaminarV1V2Network(model_cfg).to(device)
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        sd = ckpt.get("model_state", ckpt)
        res = net.load_state_dict(sd, strict=False)
        print(f"[net] missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}",
              flush=True)
        net.eval()
        net.oracle_mode = False
        net.feedback_scale.fill_(1.0)
        for p in net.parameters():
            p.requires_grad_(False)
        data = build_balanced_dataset(
            net, model_cfg, train_cfg, device,
            n_train_per_cell=900, n_val_per_cell=100, seed=SEED,
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

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # ---- Train Dec D-raw 20k ----
    print(f"\n=== {args.net_name} / D-raw (FB-ON, 20k Adam steps, no early stop) ===",
          flush=True)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    dec_raw = nn.Linear(n_ori, n_ori, bias=True).to(device)
    raw_res = train_linear_n_steps(
        dec_raw, X_tr, y_tr, br_tr, X_va, y_va, br_va,
        n_classes=n_ori, n_steps=args.n_steps, log_every=args.log_every,
        label="raw",
    )
    snap_raw = os.path.join(args.out_dir, f"decoder_d_20k_raw_{args.net_name}.pt")
    torch.save({
        "state_dict": dec_raw.state_dict(),
        "variant": "D-raw-fbON-neutral-20k",
        "net_name": args.net_name,
        "net_ckpt": args.ckpt,
        "config": args.config,
        "seed": SEED,
        "n_steps": args.n_steps,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "early_stop": False,
        "final_val_acc": raw_res["final_val_acc"],
        "final_val_bal_acc": raw_res["final_val_bal_acc"],
        "final_val_ex_acc": raw_res["final_val_ex_acc"],
        "final_val_unex_acc": raw_res["final_val_unex_acc"],
        "wall_clock_seconds": raw_res["wall_clock_seconds"],
        "readout_window": list(READOUT_WIN),
    }, snap_raw)
    print(f"[save] D-raw  → {snap_raw}", flush=True)

    # ---- Train Dec D-shape 20k ----
    print(f"\n=== {args.net_name} / D-shape (FB-ON, 20k Adam steps, no early stop) ===",
          flush=True)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    dec_shape = nn.Linear(n_ori, n_ori, bias=True).to(device)
    shape_res = train_linear_n_steps(
        dec_shape, X_tr_shape, y_tr, br_tr, X_va_shape, y_va, br_va,
        n_classes=n_ori, n_steps=args.n_steps, log_every=args.log_every,
        label="shape",
    )
    snap_shape = os.path.join(args.out_dir, f"decoder_d_20k_shape_{args.net_name}.pt")
    torch.save({
        "state_dict": dec_shape.state_dict(),
        "variant": "D-shape-fbON-neutral-20k",
        "net_name": args.net_name,
        "net_ckpt": args.ckpt,
        "config": args.config,
        "seed": SEED,
        "n_steps": args.n_steps,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "early_stop": False,
        "final_val_acc": shape_res["final_val_acc"],
        "final_val_bal_acc": shape_res["final_val_bal_acc"],
        "final_val_ex_acc": shape_res["final_val_ex_acc"],
        "final_val_unex_acc": shape_res["final_val_unex_acc"],
        "wall_clock_seconds": shape_res["wall_clock_seconds"],
        "readout_window": list(READOUT_WIN),
    }, snap_shape)
    print(f"[save] D-shape → {snap_shape}", flush=True)

    # ---- Save combined per-net JSON ----
    out_json = args.out_json or os.path.join(
        _REPO, f"results/decoder_d_20k_training_{args.net_name}.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({
            "label": f"Task #8 — Dec D 20k Adam (no early stop, no wd) on {args.net_name}",
            "net_name": args.net_name,
            "net_ckpt": args.ckpt,
            "config": args.config,
            "seed": SEED,
            "design": {
                "n_steps": args.n_steps,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "batch_size_train": BATCH_TRAIN,
                "early_stop": False,
                "feedback_scale": 1.0,
                "paired_fork_design": (
                    "march at 5° for N∈{4..10}, probe at target_ch; "
                    "unex = D_signed_ch ∈ ±[5, 18]"
                ),
                "readout_window": list(READOUT_WIN),
                "cue": "expected-next per march (same cue for ex and unex)",
                "task_state": "focused [1, 0]",
            },
            "data": {
                "n_train": int(X_tr.shape[0]),
                "n_val":   int(X_va.shape[0]),
                "n_train_per_cell": 900,
                "n_val_per_cell": 100,
            },
            "d_raw": raw_res,
            "d_shape": shape_res,
        }, f, indent=2)
    print(f"[save] json → {out_json}", flush=True)


if __name__ == "__main__":
    main()
