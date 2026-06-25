#!/usr/bin/env python3
"""
INDEPENDENT AUDIT of the noisy-readout single-knob context switch (#42 Phase B).
Debugger job: PROVE the seed-0 'switch' is a genuine regime difference or FIND the artifact/bug.

Training recipe is copied VERBATIM from train_switch_seed0_gate.py run_seed (so any reproduction is
apples-to-apples). On each trained net we run an ARTIFACT-KILLING battery the Lead's runs did NOT:

  T0  Lead-parity probe (fresh RNG, exactly the Lead's probe) -> reproduce seed-0 numbers (anchor).
  T1  EXTENDED probe: also unexp_r (opposite channel) + mean|r|@K + held-out fb_ON and fb_OFF.
        - unexp_r ~ floor in BOTH ctx  => switch is EXPECTATION-SPECIFIC, not a global gain knob.
        - save mean|r|@K NOT ~0        => 'save 0.000' is a SELECTIVE floor, not a global collapse.
        - held fb_ON ~80% but fb_OFF craters in BOTH ctx => prediction genuinely USES feedback.
  T2  FIXED-input floor parity: same c0/v for ctx=+1 and ctx=-1.
        floor(+1)==floor(-1) to ~0  => floor = relu(drive), ctx-INDEPENDENT (no floor artifact).
        and the switch holds on IDENTICAL inputs (attend exp_r>floor, save exp_r<floor).
  T3  ctx=0 knob-neutral probe: regime with NO context drive (the shared-gain default).
  T4  g_ctx LESION (softplus(g_ctx)->~0): ctx's ONLY path is via g_ctx, so the switch MUST collapse
        (|attend-save| exp-floor -> ~0). If it does NOT collapse, ctx leaks elsewhere => BUG.

GPU recommended (trains two nets from scratch). Does NOT touch phaseA_somvip/ saved nets (trains fresh in-memory).
Usage: python audit_switch.py            # seeds 0 and 7
"""
import torch, torch.nn as nn, torch.nn.functional as F, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # find package-root simple_net.py
from simple_net import (SimpleNet, forward_seq, phase1, make_sequences, chan,
                        N, STEP_DEG, device)

SIGMA = 1.0; LAM_READ = 1.0; LAM_SAVE = 0.2          # Lead's fixed hyperparams
VELS = torch.tensor([-3, -2, -1, 1, 2, 3], device=device); K = 4; B = 8000


def train(seed):
    """VERBATIM copy of train_switch_seed0_gate.py run_seed training (phase1 2000 + ctx-train 4000)."""
    torch.manual_seed(seed)
    net = SimpleNet(use_circuit=True, context=True).to(device)
    read = nn.Linear(N, N).to(device)
    phase1(net, steps=2000)
    for p in list(net.W_ff.parameters()) + list(net.decoder.parameters()):
        p.requires_grad_(False)
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad] + list(read.parameters()), lr=1e-3)
    for s in range(1, 4001):
        theta = make_sequences(128, 12, mode='momentum', p_stay=0.9)
        ctxb = torch.where(torch.rand(128, 1, device=device) < 0.5,
                           torch.ones(128, 1, device=device), -torch.ones(128, 1, device=device))
        preds, r_all = forward_seq(net, theta, 1.0, ctx=ctxb)
        ce = F.cross_entropy(preds[:, :-1, :].reshape(-1, N), chan(theta[:, 1:]).reshape(-1),
                             reduction='none').reshape(128, 11).mean(1)
        rd = F.cross_entropy(read(r_all + SIGMA * torch.randn_like(r_all)).reshape(-1, N),
                             chan(theta).reshape(-1), reduction='none').reshape(128, 12).mean(1)
        energy = r_all.abs().mean(dim=(1, 2)); att = (ctxb.squeeze(1) > 0).float()
        loss = (ce.mean()
                + LAM_READ * (rd * att).sum() / att.sum().clamp(min=1)
                + LAM_SAVE * (energy * (1 - att)).sum() / (1 - att).sum().clamp(min=1))
        opt.zero_grad(); loss.backward(); opt.step()
        if seed == 0 and s % 1000 == 0:
            print(f"    [train seed0] step {s} loss {loss.item():.3f} g_ctx {F.softplus(net.g_ctx_raw).item():.3f}", flush=True)
    return net


@torch.no_grad()
def probe_full(net, cv, seed_fix=None):
    """Phase-A-identical probe (forward_seq r[:,K,:] at channel e, fb on vs off) PLUS unexp/mean/held.
    If seed_fix is set, the c0/v draw is reproducible so two ctx values see IDENTICAL inputs."""
    if seed_fix is not None:
        torch.manual_seed(seed_fix)
    c0 = torch.randint(0, N, (B,), device=device); v = VELS[torch.randint(0, 6, (B,), device=device)]
    tt = torch.arange(K, device=device)[None, :]
    ctxd = ((c0[:, None] + v[:, None] * tt) % N).float() * STEP_DEG
    e = (c0 + v * K) % N; u = (e + N // 2) % N
    th = torch.cat([ctxd, (e.float() * STEP_DEG)[:, None]], 1)
    idx = e.view(-1, 1); idu = u.view(-1, 1)
    cb = torch.full((B, 1), float(cv), device=device)
    _, ron = forward_seq(net, th, 1.0, ctx=cb)
    _, roff = forward_seq(net, th, 0.0, ctx=cb)
    exp_r = ron[:, K, :].gather(1, idx).mean().item()
    floor = roff[:, K, :].gather(1, idx).mean().item()
    unexp_r = ron[:, K, :].gather(1, idu).mean().item()
    mean_r = ron[:, K, :].abs().mean().item()
    th2 = make_sequences(B, 12, mode='momentum', p_stay=0.9)
    p_on, _ = forward_seq(net, th2, 1.0, ctx=cb)
    p_off, _ = forward_seq(net, th2, 0.0, ctx=cb)
    acc_on = (p_on[:, :-1].argmax(-1) == chan(th2[:, 1:])).float().mean().item() * 100
    acc_off = (p_off[:, :-1].argmax(-1) == chan(th2[:, 1:])).float().mean().item() * 100
    return dict(exp_r=exp_r, floor=floor, unexp_r=unexp_r, mean_r=mean_r, acc_on=acc_on, acc_off=acc_off)


def report(net, seed):
    print(f"\n==================== SEED {seed} ====================", flush=True)
    gctx = F.softplus(net.g_ctx_raw).item()
    gains = [round(x, 3) for x in F.softplus(net.circ_raw).detach().cpu().tolist()]
    print(f"  g_ctx(softplus)={gctx:.4f}   gains[g_v,g_s,g_sv,g_e,g_ps]={gains}", flush=True)

    # T0/T1: Lead-parity probe (fresh RNG, no seed_fix) + extended observables
    res = {}
    for cv in (1.0, -1.0):
        d = probe_full(net, cv); res[cv] = d
        reg = 'SHARPEN' if d['exp_r'] > d['floor'] else 'DAMPEN'
        print(f"  [ctx={cv:+.0f} {'attend' if cv > 0 else 'save  '}] "
              f"exp_r={d['exp_r']:.3f} floor={d['floor']:.3f} exp-floor={d['exp_r']-d['floor']:+.3f} | "
              f"unexp_r={d['unexp_r']:.3f} mean|r|@K={d['mean_r']:.3f} | "
              f"held fb_on={d['acc_on']:.1f}% fb_off={d['acc_off']:.1f}%  -> {reg}", flush=True)
    sw = res[1.0]['exp_r'] > res[1.0]['floor'] and res[-1.0]['exp_r'] < res[-1.0]['floor']
    print(f"  SWITCH(fresh RNG)={'YES' if sw else 'NO'}", flush=True)

    # T2: FIXED-input floor parity + switch on identical inputs
    a = probe_full(net, 1.0, seed_fix=12345); s = probe_full(net, -1.0, seed_fix=12345)
    print(f"  [T2 fixed-input] floor(ctx+1)={a['floor']:.5f} floor(ctx-1)={s['floor']:.5f} "
          f"|Δfloor|={abs(a['floor']-s['floor']):.2e}  (expect ~0 => floor is ctx-INDEPENDENT)", flush=True)
    print(f"           attend exp_r={a['exp_r']:.3f}>{a['floor']:.3f}? {a['exp_r']>a['floor']} ; "
          f"save exp_r={s['exp_r']:.3f}<{s['floor']:.3f}? {s['exp_r']<s['floor']}  (switch on IDENTICAL inputs)", flush=True)

    # T3: ctx=0 knob-neutral
    z = probe_full(net, 0.0, seed_fix=12345)
    print(f"  [T3 ctx=0 neutral] exp_r={z['exp_r']:.3f} floor={z['floor']:.3f} exp-floor={z['exp_r']-z['floor']:+.3f} "
          f"-> {'SHARPEN' if z['exp_r']>z['floor'] else 'DAMPEN'}  (shared-gain default with NO context)", flush=True)

    # T4: g_ctx LESION -> switch MUST collapse (ctx has no other path)
    saved = net.g_ctx_raw.data.clone()
    net.g_ctx_raw.data.fill_(-20.0)                          # softplus(-20) ~ 2e-9 ~ 0
    la = probe_full(net, 1.0, seed_fix=12345); ls = probe_full(net, -1.0, seed_fix=12345)
    da = la['exp_r'] - la['floor']; ds = ls['exp_r'] - ls['floor']
    print(f"  [T4 g_ctx LESION~0] attend exp-floor={da:+.3f} save exp-floor={ds:+.3f} "
          f"|attend-save|={abs(da-ds):.2e}  (expect ~0 => switch is 100% carried by g_ctx, no leak)", flush=True)
    net.g_ctx_raw.data.copy_(saved)                          # restore

    print(f"  VERDICT-INPUTS seed{seed}: switch={sw} | floor_ctx_indep={abs(a['floor']-s['floor'])<1e-3} | "
          f"lesion_collapses={abs(da-ds)<1e-3} | save_alive(mean|r|@K={res[-1.0]['mean_r']:.2f}) | "
          f"unexp~floor(att {res[1.0]['unexp_r']:.2f}/{res[1.0]['floor']:.2f}, sav {res[-1.0]['unexp_r']:.2f}/{res[-1.0]['floor']:.2f})", flush=True)


if __name__ == '__main__':
    print("=== INDEPENDENT AUDIT: noisy-readout context switch (seeds 0 anchor, 7 fresh) ===", flush=True)
    for seed in (0, 7):
        net = train(seed)
        report(net, seed)
    print("AUDIT_DONE", flush=True)
