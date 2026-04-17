"""Task #11 — train standalone Decoder C on synthetic single+multi orientation data.

Architecture: Linear(36, 36), softmax/cross-entropy. Bias on. No hidden layers.

Training data (100k examples, 50/50 single/multi):
  Single: A in [0.1, 2.0] uniform, sigma=3 ch, mu in {0..35} uniform; noise N(0, 0.02).
  Multi : K in {2,3} uniform, K distinct mu's; per-bump A in [0.1, 2.0]; strict-max
          amplitude resolves the label; same noise.

Training: Adam LR 1e-3, batch 256, up to 30 epochs, early stopping on val acc
patience=3. 90/10 train/val split. Seed 42.

Validation:
  (1) held-out synthetic test  (target >=95%)
  (2) real-network natural HMM forward, r_l23 -> Decoder C vs true probe channel
       (target >=50%, flag if <30%)
  (3) Pass A / Pass B emulation from Task #10 (both should decode to ch18)

Outputs:
  checkpoints/decoder_c.pt
  results/decoder_c_validation.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_THIS_DIR, ".."))
sys.path.insert(0, _THIS_DIR)

from src.config import load_config
from src.model.network import LaminarV1V2Network
from src.stimulus.sequences import HMMSequenceGenerator
from src.training.trainer import build_stimulus_sequence

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
N = 36
SIGMA = 3.0
A_LOW, A_HIGH = 0.1, 2.0
NOISE_STD = 0.02
N_TOTAL = 100_000
N_SINGLE = 50_000
N_MULTI = 50_000
SEED = 42
BATCH = 256
LR = 1e-3
EPOCHS = 30
PATIENCE = 3
VAL_FRAC = 0.10

CKPT_OUT = Path("checkpoints/decoder_c.pt")
JSON_OUT = Path("results/decoder_c_validation.json")
NET_CKPT = Path("results/simple_dual/emergent_seed42/checkpoint.pt")
NET_CFG = "config/sweep/sweep_rescue_1_2.yaml"


# -----------------------------------------------------------------------------
# Synthetic data
# -----------------------------------------------------------------------------
def gaussian_template(n: int = N, sigma: float = SIGMA) -> torch.Tensor:
    """Build (n, n) lookup g[mu, i] = exp(-0.5 * (circ(i, mu)/sigma)**2)."""
    chs = torch.arange(n, dtype=torch.float64)
    diff = chs.unsqueeze(0) - chs.unsqueeze(1)               # mu rows, ch cols
    diff = (diff + n / 2) % n - n / 2                        # signed circular
    g = torch.exp(-0.5 * (diff / sigma) ** 2)
    return g.float()                                         # (n, n)


def make_single_dataset(n: int, g: torch.Tensor, gen: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
    mus = torch.randint(0, N, (n,), generator=gen)            # [n]
    As = torch.empty(n).uniform_(A_LOW, A_HIGH, generator=gen) # [n]
    bumps = g[mus] * As.unsqueeze(1)                          # [n, N]
    noise = torch.randn(n, N, generator=gen) * NOISE_STD
    X = bumps + noise
    y = mus.long()
    return X, y


def make_multi_dataset(n: int, g: torch.Tensor, gen: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
    """K in {2,3}; K distinct mus; strict max A determines label."""
    Ks = torch.randint(2, 4, (n,), generator=gen)             # 2 or 3
    X = torch.zeros(n, N)
    y = torch.zeros(n, dtype=torch.long)
    n_redraws = 0
    for i in range(n):
        K = int(Ks[i])
        # Sample K distinct mus (rejection on duplicates).
        mus = torch.randperm(N, generator=gen)[:K]            # distinct by construction
        # Sample K As; redraw if not strictly maximal.
        while True:
            As = torch.empty(K).uniform_(A_LOW, A_HIGH, generator=gen)
            sorted_, idx = torch.sort(As, descending=True)
            if sorted_[0].item() > sorted_[1].item():
                break
            n_redraws += 1
        bump_sum = (g[mus] * As.unsqueeze(1)).sum(dim=0)      # [N]
        X[i] = bump_sum
        y[i] = int(mus[idx[0]])
    X = X + torch.randn(n, N, generator=gen) * NOISE_STD
    if n_redraws:
        print(f"  multi-dataset redraws (tied amplitudes): {n_redraws}")
    return X, y


def build_dataset(seed: int = SEED) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (X_train, y_train, X_val, y_val) with 90/10 split."""
    print(f"Building dataset: {N_SINGLE} single + {N_MULTI} multi = {N_TOTAL} total")
    g = gaussian_template()
    gen = torch.Generator().manual_seed(seed)

    Xs, ys = make_single_dataset(N_SINGLE, g, gen)
    Xm, ym = make_multi_dataset(N_MULTI, g, gen)
    X = torch.cat([Xs, Xm], dim=0)
    y = torch.cat([ys, ym], dim=0)

    # Shuffle then split 90/10
    perm = torch.randperm(N_TOTAL, generator=gen)
    X = X[perm]
    y = y[perm]
    n_val = int(round(N_TOTAL * VAL_FRAC))
    n_train = N_TOTAL - n_val
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_val = X[n_train:]
    y_val = y[n_train:]
    print(f"  split: train n={n_train}, val n={n_val}")
    return X_train, y_train, X_val, y_val


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def train_decoder(X_train, y_train, X_val, y_val) -> Tuple[nn.Linear, dict]:
    torch.manual_seed(SEED)
    decoder = nn.Linear(N, N, bias=True)
    opt = torch.optim.Adam(decoder.parameters(), lr=LR)

    n_train = X_train.shape[0]
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    best_state = None
    best_epoch = -1
    epochs_since_improve = 0

    print(f"Training: epochs<={EPOCHS}, batch={BATCH}, LR={LR}, patience={PATIENCE}")
    for epoch in range(EPOCHS):
        decoder.train()
        perm = torch.randperm(n_train)
        train_losses = []
        train_correct = 0
        for i in range(0, n_train, BATCH):
            idx = perm[i:i + BATCH]
            xb = X_train[idx]
            yb = y_train[idx]
            opt.zero_grad()
            logits = decoder(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            train_losses.append(float(loss.item()) * xb.shape[0])
            train_correct += int((logits.argmax(dim=1) == yb).sum())
        train_loss = sum(train_losses) / n_train
        train_acc = train_correct / n_train

        decoder.eval()
        with torch.no_grad():
            logits_val = decoder(X_val)
            val_loss = float(F.cross_entropy(logits_val, y_val).item())
            val_acc = float((logits_val.argmax(dim=1) == y_val).float().mean().item())

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        improved = val_acc > best_val_acc + 1e-6
        if improved:
            best_val_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in decoder.state_dict().items()}
            best_epoch = epoch
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        flag = " *" if improved else ""
        print(f"  epoch {epoch:02d}  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}{flag}")

        if epochs_since_improve >= PATIENCE:
            print(f"  early stop @ epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

    if best_state is not None:
        decoder.load_state_dict(best_state)
    print(f"Best val_acc={best_val_acc:.4f} @ epoch {best_epoch}")
    return decoder, {
        "history": history,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
    }


# -----------------------------------------------------------------------------
# Validation 1 — held-out synthetic
# -----------------------------------------------------------------------------
def validate_synth(decoder: nn.Linear, X_val: torch.Tensor, y_val: torch.Tensor) -> dict:
    print("\n=== Validation 1: held-out synthetic ===")
    decoder.eval()
    with torch.no_grad():
        logits = decoder(X_val)
        pred = logits.argmax(dim=1)
        acc = float((pred == y_val).float().mean().item())
    print(f"  n={int(y_val.shape[0])}  acc={acc:.4f}")
    return {"n": int(y_val.shape[0]), "acc": acc, "target": 0.95, "passed": acc >= 0.95}


# -----------------------------------------------------------------------------
# Validation 2 — real-network natural HMM forward
# -----------------------------------------------------------------------------
def validate_real_network(decoder: nn.Linear) -> dict:
    print("\n=== Validation 2: real-network natural HMM forward ===")
    if not NET_CKPT.exists():
        return {"available": False, "path": str(NET_CKPT), "note": "ckpt missing"}
    if not Path(NET_CFG).exists():
        return {"available": False, "path": NET_CFG, "note": "config missing"}

    device = torch.device("cpu")
    print(f"  loading network: {NET_CKPT}")
    model_cfg, train_cfg, stim_cfg = load_config(NET_CFG)
    net = LaminarV1V2Network(model_cfg).to(device)
    ckpt = torch.load(NET_CKPT, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["model_state"])
    net.eval()
    net.oracle_mode = False
    net.feedback_scale.fill_(1.0)

    Nch = model_cfg.n_orientations
    period = model_cfg.orientation_range
    step_deg = period / Nch

    seq_length = train_cfg.seq_length
    batch_size = train_cfg.batch_size
    steps_on = train_cfg.steps_on
    steps_isi = train_cfg.steps_isi
    steps_per = steps_on + steps_isi
    W_START, W_END = 9, 11
    print(f"  N={Nch}, seq_length={seq_length}, batch={batch_size}, "
          f"steps_per={steps_per}, readout window=[{W_START}:{W_END}]")

    gen = HMMSequenceGenerator(
        n_orientations=Nch,
        p_self=stim_cfg.p_self,
        p_transition_cw=stim_cfg.p_transition_cw,
        p_transition_ccw=stim_cfg.p_transition_ccw,
        n_anchors=stim_cfg.n_anchors,
        jitter_range=stim_cfg.jitter_range,
        transition_step=stim_cfg.transition_step,
        period=period,
        contrast_range=tuple(train_cfg.stage2_contrast_range),
        ambiguous_fraction=train_cfg.ambiguous_fraction,
        ambiguous_offset=stim_cfg.ambiguous_offset,
        cue_dim=stim_cfg.cue_dim,
        n_states=stim_cfg.n_states,
        cue_valid_fraction=stim_cfg.cue_valid_fraction,
    )

    rng = torch.Generator().manual_seed(SEED)

    n_batches = 4  # 4 * 32 trials * 25 presentations ~= ~3200 readouts
    all_pred = []
    all_true = []
    all_pred_ambdrop = []
    all_true_ambdrop = []

    decoder.eval()
    with torch.no_grad():
        for bi in range(n_batches):
            metadata = gen.generate(batch_size, seq_length, generator=rng)
            stim_seq, cue_seq, ts_seq, _, _, _ = build_stimulus_sequence(
                metadata, model_cfg, train_cfg, stim_cfg
            )
            stim_seq = stim_seq.to(device)
            cue_seq = cue_seq.to(device)
            ts_seq = ts_seq.to(device)

            true_ori = metadata.orientations.to(device)        # [B, S]
            is_amb = metadata.is_ambiguous.to(device)          # [B, S] bool
            true_ch = (true_ori / step_deg).round().long() % Nch  # [B, S]

            packed = net.pack_inputs(stim_seq, cue_seq, ts_seq)
            r_l23, _, _ = net.forward(packed)                  # [B, T_total, N]

            # Per presentation: average r_l23 across the readout window of the ON phase.
            B, T_total, _ = r_l23.shape
            for s in range(seq_length):
                onset = s * steps_per
                r_win = r_l23[:, onset + W_START:onset + W_END, :].mean(dim=1)  # [B, N]
                logits_c = decoder(r_win)
                pred = logits_c.argmax(dim=1)                                  # [B]
                tc = true_ch[:, s]                                              # [B]
                all_pred.append(pred.cpu().numpy())
                all_true.append(tc.cpu().numpy())
                amb_mask = is_amb[:, s].cpu().numpy().astype(bool)
                keep = ~amb_mask
                if keep.any():
                    all_pred_ambdrop.append(pred.cpu().numpy()[keep])
                    all_true_ambdrop.append(tc.cpu().numpy()[keep])
            print(f"  batch {bi+1}/{n_batches} done")

    pred = np.concatenate(all_pred)
    true = np.concatenate(all_true)
    pred_nd = np.concatenate(all_pred_ambdrop)
    true_nd = np.concatenate(all_true_ambdrop)

    def acc_at(pred, true, tol):
        d = np.minimum(np.abs(pred - true), Nch - np.abs(pred - true))
        return float((d <= tol).mean())

    out = {
        "available": True,
        "n_readouts_total": int(pred.size),
        "n_readouts_nonamb": int(pred_nd.size),
        "acc_pm0_all": acc_at(pred, true, 0),
        "acc_pm1_all": acc_at(pred, true, 1),
        "acc_pm2_all": acc_at(pred, true, 2),
        "acc_pm0_nonamb": acc_at(pred_nd, true_nd, 0),
        "acc_pm1_nonamb": acc_at(pred_nd, true_nd, 1),
        "acc_pm2_nonamb": acc_at(pred_nd, true_nd, 2),
        "target": 0.50,
        "flag_threshold": 0.30,
        "n_batches": n_batches,
        "batch_size": int(batch_size),
        "seq_length": int(seq_length),
        "readout_window": [int(W_START), int(W_END)],
        "checkpoint": str(NET_CKPT),
        "config": NET_CFG,
    }
    a0 = out["acc_pm0_nonamb"]
    out["passed_target"] = a0 >= 0.50
    out["below_flag_threshold"] = a0 < 0.30
    print(f"  acc (non-amb, +-0): {a0:.4f}   acc (+-1): {out['acc_pm1_nonamb']:.4f}   "
          f"acc (+-2): {out['acc_pm2_nonamb']:.4f}")
    print(f"  acc (all,     +-0): {out['acc_pm0_all']:.4f}   acc (+-1): {out['acc_pm1_all']:.4f}   "
          f"acc (+-2): {out['acc_pm2_all']:.4f}")
    print(f"  target=0.50  passed_target={out['passed_target']}  "
          f"below_flag={out['below_flag_threshold']}")
    return out


# -----------------------------------------------------------------------------
# Validation 3 — Pass A / Pass B emulation
# -----------------------------------------------------------------------------
def validate_passAB(decoder: nn.Linear) -> dict:
    print("\n=== Validation 3: Pass A / Pass B synthetic emulation ===")
    g = gaussian_template()
    # Pass A: A=0.68 at mu=18
    xA = g[18] * 0.68
    # Pass B: 0.44 at mu=18 + 0.20 at mu=0 (ring opposite +90deg)
    xB = g[18] * 0.44 + g[0] * 0.20
    decoder.eval()
    with torch.no_grad():
        zA = decoder(xA.unsqueeze(0))[0]
        zB = decoder(xB.unsqueeze(0))[0]
    out = {}
    for label, x, z in [("pass_A_synth", xA, zA), ("pass_B_synth", xB, zB)]:
        order = z.argsort(descending=True)
        top1 = int(order[0])
        top2 = int(order[1])
        top3 = int(order[2])
        m12 = float(z[top1] - z[top2])
        m23 = float(z[top2] - z[top3])
        print(f"  {label}: argmax=ch{top1} (logit {float(z[top1]):+.4f}), "
              f"top2=ch{top2} (logit {float(z[top2]):+.4f}), "
              f"margin top1-top2={m12:.4f}")
        out[label] = {
            "argmax": top1,
            "argmax_at_18": top1 == 18,
            "top1_logit": float(z[top1]),
            "top2_class": top2,
            "top2_logit": float(z[top2]),
            "top3_class": top3,
            "top3_logit": float(z[top3]),
            "margin_top1_top2": m12,
            "margin_top2_top3": m23,
            "all_logits": [float(v) for v in z],
        }
    out["both_decode_to_18"] = (
        out["pass_A_synth"]["argmax_at_18"] and out["pass_B_synth"]["argmax_at_18"]
    )
    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    X_train, y_train, X_val, y_val = build_dataset(seed=SEED)
    decoder, train_meta = train_decoder(X_train, y_train, X_val, y_val)

    CKPT_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": decoder.state_dict(),
        "arch": "Linear(36, 36)",
        "train_meta": train_meta,
        "seed": SEED,
        "sigma": SIGMA,
        "amplitude_range": [A_LOW, A_HIGH],
        "noise_std": NOISE_STD,
        "n_total": N_TOTAL,
        "n_single": N_SINGLE,
        "n_multi": N_MULTI,
    }, CKPT_OUT)
    print(f"\nSaved checkpoint -> {CKPT_OUT}")

    val1 = validate_synth(decoder, X_val, y_val)
    val2 = validate_real_network(decoder)
    val3 = validate_passAB(decoder)

    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(JSON_OUT, "w") as f:
        json.dump({
            "label": "Task #11 — Decoder C training + validation (R1+R2 reference)",
            "checkpoint_path": str(CKPT_OUT),
            "training": train_meta,
            "data_config": {
                "N": N, "sigma": SIGMA, "A_range": [A_LOW, A_HIGH],
                "noise_std": NOISE_STD, "n_total": N_TOTAL,
                "n_single": N_SINGLE, "n_multi": N_MULTI,
                "seed": SEED, "val_frac": VAL_FRAC,
            },
            "training_config": {
                "lr": LR, "batch": BATCH, "epochs_max": EPOCHS, "patience": PATIENCE,
                "optimizer": "Adam",
            },
            "validation_1_synthetic": val1,
            "validation_2_real_network": val2,
            "validation_3_passAB": val3,
        }, f, indent=2)
    print(f"Wrote validation JSON -> {JSON_OUT}")


if __name__ == "__main__":
    main()
