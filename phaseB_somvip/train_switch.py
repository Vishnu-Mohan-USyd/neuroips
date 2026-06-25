#!/usr/bin/env python3
"""
train_switch.py — train ONE context-switching SOM/VIP network (Phase B) and save the
{'net', 'read', 'cfg'} wrapper checkpoint.

This is the generator for phaseB_somvip/ckpt_ctxswitch_seed1.pt (the committed seed-1
checkpoint that the rest of Phase B reloads). The run_seed() training body is copied
VERBATIM from train_switch_seed0_gate.py (the seed-0 kill-gate), so the saved network is
apples-to-apples with the gate / robustness / audit scripts. The only differences here
are (a) run_seed returns the trained modules and (b) __main__ saves the wrapper.

Recipe (one network, one weight-set; regime chosen at RUNTIME by a +/-1 context bit):
  - phase1: 2000 steps, learn the static L4 -> L2/3 representation with feedback off.
  - freeze W_ff + decoder, then 4000 steps of context-trained next-step prediction.
    Each step draws a random +/-1 context bit per sequence:
        ctx = +1  ATTEND : minimise the NOISY-readout CE  -> sharpen the expected channel.
        ctx = -1  SAVE   : minimise activity energy         -> dampen the expected channel.
  The sharpen<->dampen switch is carried entirely by a learned context gain on VIP
  (g_ctx); see proveout_switch.py for the K1-K4 refutation battery.

Fixed hyperparameters: sigma=1.0 (readout noise), lam_read=1.0, lam_save=0.2, lr=1e-3.
RNG order per step is load-bearing for reproducibility:
  make_sequences -> torch.rand (context bit) -> forward_seq (no RNG) -> randn_like (noise).

Usage:
  python train_switch.py                          # seed 1 -> canonical ckpt_ctxswitch_seed1.pt
  python train_switch.py --seed 1 --out other.pt  # write elsewhere (keeps the committed copy)
GPU strongly recommended (phase1 2000 + ctx-train 4000 steps).
NOTE: with no --out, this OVERWRITES the canonical committed checkpoint. Pass --out to a
scratch path if you want to preserve the shipped copy and md5-compare afterwards.
"""
import argparse, os, sys
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # find package-root simple_net.py
from simple_net import SimpleNet, forward_seq, phase1, make_sequences, chan, N, STEP_DEG, device

_HERE = os.path.dirname(os.path.abspath(__file__))   # resolve checkpoint path relative to THIS script, not CWD
SIGMA = 1.0; LAM_READ = 1.0; LAM_SAVE = 0.2          # fixed hyperparams
VELS = torch.tensor([-3, -2, -1, 1, 2, 3], device=device); K = 4; B = 8000


@torch.no_grad()
def probe(net, cv):
    """End-of-training confirmation probe (does NOT affect saved weights; runs after training).
    Returns (expected-channel r with fb on, fb-off floor, held-out next-step accuracy %)."""
    c0 = torch.randint(0, N, (B,), device=device); v = VELS[torch.randint(0, 6, (B,), device=device)]
    tt = torch.arange(K, device=device)[None, :]; ctxd = ((c0[:, None] + v[:, None] * tt) % N).float() * STEP_DEG
    e = (c0 + v * K) % N; th = torch.cat([ctxd, (e.float() * STEP_DEG)[:, None]], 1); idx = e.view(-1, 1)
    cb = torch.full((B, 1), float(cv), device=device)
    _, ron = forward_seq(net, th, 1.0, ctx=cb); _, roff = forward_seq(net, th, 0.0, ctx=cb)
    th2 = make_sequences(B, 12, mode='momentum', p_stay=0.9)
    p2, _ = forward_seq(net, th2, 1.0, ctx=torch.full((B, 1), float(cv), device=device))
    acc = (p2[:, :-1].argmax(-1) == chan(th2[:, 1:])).float().mean().item() * 100
    return ron[:, K, :].gather(1, idx).mean().item(), roff[:, K, :].gather(1, idx).mean().item(), acc


def run_seed(seed):
    """Train one context-switching net. VERBATIM recipe from train_switch_seed0_gate.py
    (only the return value differs: this yields (net, read, switched) so __main__ can save).
    Returns: (net, read, switched_bool)."""
    torch.manual_seed(seed)
    net = SimpleNet(use_circuit=True, context=True).to(device)
    read = nn.Linear(N, N).to(device)
    phase1(net, steps=2000)
    for p in list(net.W_ff.parameters()) + list(net.decoder.parameters()): p.requires_grad_(False)
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad] + list(read.parameters()), lr=1e-3)
    for s in range(1, 4001):
        theta = make_sequences(128, 12, mode='momentum', p_stay=0.9)
        ctxb = torch.where(torch.rand(128, 1, device=device) < 0.5, torch.ones(128, 1, device=device), -torch.ones(128, 1, device=device))
        preds, r_all = forward_seq(net, theta, 1.0, ctx=ctxb)
        ce = F.cross_entropy(preds[:, :-1, :].reshape(-1, N), chan(theta[:, 1:]).reshape(-1), reduction='none').reshape(128, 11).mean(1)
        r_noisy = r_all + SIGMA * torch.randn_like(r_all)
        rd = F.cross_entropy(read(r_noisy).reshape(-1, N), chan(theta).reshape(-1), reduction='none').reshape(128, 12).mean(1)
        energy = r_all.abs().mean(dim=(1, 2))
        att = (ctxb.squeeze(1) > 0).float()
        loss = ce.mean() + LAM_READ * (rd * att).sum() / att.sum().clamp(min=1) + LAM_SAVE * (energy * (1 - att)).sum() / (1 - att).sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 1000 == 0: print(f"  step {s} loss {loss.item():.3f} g_ctx {F.softplus(net.g_ctx_raw).item():.3f}", flush=True)
    out = {}
    for cv in (1.0, -1.0):
        re, rf, acc = probe(net, cv); out[cv] = (re, rf, acc)
        print(f"  [seed{seed} ctx={cv:+.0f} {'attend' if cv > 0 else 'save'}] exp_r={re:.3f} floor={rf:.3f} exp-floor={re-rf:+.3f} held={acc:.1f}% -> {'SHARPEN' if re > rf else 'DAMPEN'}", flush=True)
    gctx = F.softplus(net.g_ctx_raw).item()
    sw = out[1.0][0] > out[1.0][1] and out[-1.0][0] < out[-1.0][1]
    print(f"  seed{seed}: g_ctx={gctx:.3f}  SWITCH={'YES' if sw else 'NO'} (attend-margin {out[1.0][0]-out[1.0][1]:+.2f}, save-margin {out[-1.0][0]-out[-1.0][1]:+.2f})", flush=True)
    return net, read, sw


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Train a Phase B context-switching SOM/VIP net and save the wrapper checkpoint.")
    ap.add_argument('--seed', type=int, default=1, help="training seed (committed checkpoint is seed 1)")
    ap.add_argument('--out', default=os.path.join(_HERE, 'ckpt_ctxswitch_seed1.pt'),
                    help="output checkpoint path (default: canonical committed path)")
    args = ap.parse_args()
    print(f"=== train_switch: training seed {args.seed} on {device} -> {args.out} ===", flush=True)
    net, read, sw = run_seed(args.seed)
    cfg = {'sigma': SIGMA, 'lam_read': LAM_READ, 'lam_save': LAM_SAVE, 'seed': args.seed, 'ctx_coding': '+1 attend / -1 save'}
    torch.save({'net': net.state_dict(), 'read': read.state_dict(), 'cfg': cfg}, args.out)
    print(f"SAVED {args.out}  cfg={cfg}  SWITCH={'YES' if sw else 'NO'}", flush=True)
    print("TRAIN_SWITCH_DONE", flush=True)
