#!/usr/bin/env python3
"""
proveout_switch.py — pre-registered K1-K4 prove-out for the Phase B runtime context switch.

Reloads the committed wrapper checkpoint (ckpt_ctxswitch_seed1.pt = {'net','read','cfg'})
and runs a REFUTATION battery on the FROZEN net (no training; CPU-verifiable). The battery
is designed to KILL the "one net switches sharpen<->dampen by a context bit" claim, not to
confirm it:

  K1  ATTEND (ctx=+1) SHARPENS the expected channel:   exp_r > floor.
  K2  SAVE   (ctx=-1) DAMPENS  the expected channel:   exp_r < floor.
  K3  held-out next-step prediction >= 75% in BOTH contexts  (save trades energy for a
      little fidelity; it does NOT 'ignore the input' -> the prediction task survives).
  K4  CRUX (fixed weights, IDENTICAL inputs, flip ONLY the context bit):
        (a) the regime flips        : +1 -> sharpen AND -1 -> dampen;
        (b) the fb-OFF floor is ctx-independent : |floor(+1) - floor(-1)| ~ 0
            (so the switch is NOT a feedforward-floor artifact); and
        (c) a g_ctx LESION (softplus(g_ctx) -> ~0) COLLAPSES the switch
            (|attend - save| exp-floor ~ 0) -> the switch is carried 100% by the
            context gain, with no leak through any other pathway.
  GO only if K1 AND K2 AND K3 AND K4 all hold.

Optionally also trains a FRESH seed end-to-end (same recipe, via train_switch.run_seed) and
runs the identical battery, to show the switch is a reproducible property of the recipe and
not an idiosyncrasy of the one saved net.

Usage:
  python proveout_switch.py                       # K1-K4 on the committed seed-1 ckpt (CPU ok)
  python proveout_switch.py --ckpt other.pt       # prove-out a different wrapper checkpoint
  python proveout_switch.py --fresh 11            # ALSO train seed 11 fresh and prove-out (GPU)
"""
import argparse, os, sys
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # find package-root simple_net.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                   # find sibling train_switch.py
from simple_net import SimpleNet, forward_seq, make_sequences, chan, N, STEP_DEG, device

_HERE = os.path.dirname(os.path.abspath(__file__))   # resolve checkpoint path relative to THIS script, not CWD
VELS = torch.tensor([-3, -2, -1, 1, 2, 3], device=device); K = 4; B = 8000


@torch.no_grad()
def probe_full(net, cv, seed_fix=None):
    """Phase-A-identical probe (forward_seq r[:,K,:] at the expected channel e, fb on vs off)
    plus held-out next-step accuracy (fb on / off). If seed_fix is set the c0/v draw is
    reproducible, so two context values are evaluated on IDENTICAL inputs (the K4 crux)."""
    if seed_fix is not None:
        torch.manual_seed(seed_fix)
    c0 = torch.randint(0, N, (B,), device=device); v = VELS[torch.randint(0, 6, (B,), device=device)]
    tt = torch.arange(K, device=device)[None, :]
    ctxd = ((c0[:, None] + v[:, None] * tt) % N).float() * STEP_DEG
    e = (c0 + v * K) % N
    th = torch.cat([ctxd, (e.float() * STEP_DEG)[:, None]], 1)
    idx = e.view(-1, 1)
    cb = torch.full((B, 1), float(cv), device=device)
    _, ron = forward_seq(net, th, 1.0, ctx=cb)
    _, roff = forward_seq(net, th, 0.0, ctx=cb)
    exp_r = ron[:, K, :].gather(1, idx).mean().item()
    floor = roff[:, K, :].gather(1, idx).mean().item()
    th2 = make_sequences(B, 12, mode='momentum', p_stay=0.9)
    p_on, _ = forward_seq(net, th2, 1.0, ctx=cb)
    acc_on = (p_on[:, :-1].argmax(-1) == chan(th2[:, 1:])).float().mean().item() * 100
    return dict(exp_r=exp_r, floor=floor, acc_on=acc_on)


def gate(net, tag):
    """Run K1-K4 on a (frozen) net. Returns True iff all four kills are survived (GO)."""
    print(f"\n----- GATE: {tag} -----", flush=True)
    net.eval()
    # K1/K2: fresh-RNG probe, both contexts.
    res = {}
    for cv in (1.0, -1.0):
        d = probe_full(net, cv); res[cv] = d
        reg = 'SHARPEN' if d['exp_r'] > d['floor'] else 'DAMPEN'
        print(f"  ctx={cv:+.0f} {'attend' if cv > 0 else 'save  '}: "
              f"exp_r={d['exp_r']:.3f} floor={d['floor']:.3f} exp-floor={d['exp_r']-d['floor']:+.3f} "
              f"held={d['acc_on']:.1f}% -> {reg}", flush=True)
    K1 = res[1.0]['exp_r'] > res[1.0]['floor']         # attend sharpens
    K2 = res[-1.0]['exp_r'] < res[-1.0]['floor']        # save dampens
    K3 = res[1.0]['acc_on'] >= 75.0 and res[-1.0]['acc_on'] >= 75.0
    # K4: fixed-input frozen-weight context flip + floor parity + g_ctx lesion.
    a = probe_full(net, 1.0, seed_fix=12345); s = probe_full(net, -1.0, seed_fix=12345)
    flip = (a['exp_r'] > a['floor']) and (s['exp_r'] < s['floor'])
    floor_indep = abs(a['floor'] - s['floor']) < 1e-3
    saved = net.g_ctx_raw.data.clone()
    net.g_ctx_raw.data.fill_(-20.0)                     # softplus(-20) ~ 2e-9 ~ 0  (kill the context gain)
    la = probe_full(net, 1.0, seed_fix=12345); ls = probe_full(net, -1.0, seed_fix=12345)
    da = la['exp_r'] - la['floor']; ds = ls['exp_r'] - ls['floor']
    lesion_collapses = abs(da - ds) < 1e-3
    net.g_ctx_raw.data.copy_(saved)                     # restore
    K4 = flip and floor_indep and lesion_collapses
    print(f"  [K4 fixed-input] flip(+1 sharpen={a['exp_r']>a['floor']}, -1 dampen={s['exp_r']<s['floor']}) "
          f"| |Δfloor|={abs(a['floor']-s['floor']):.2e} (ctx-indep<1e-3? {floor_indep}) "
          f"| g_ctx-lesion |attend-save|={abs(da-ds):.2e} (collapse<1e-3? {lesion_collapses})", flush=True)
    verdict = K1 and K2 and K3 and K4
    print(f"  ===> {tag}: {'PASS (GO)' if verdict else 'FAIL (NO-GO)'}  "
          f"[K1={K1} K2={K2} K3={K3} K4={K4}]", flush=True)
    return verdict


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Pre-registered K1-K4 prove-out for the Phase B context switch.")
    ap.add_argument('--ckpt', default=os.path.join(_HERE, 'ckpt_ctxswitch_seed1.pt'),
                    help="wrapper checkpoint to prove out (default: committed seed-1)")
    ap.add_argument('--fresh', type=int, default=None,
                    help="if set, ALSO train this seed from scratch and prove it out (GPU recommended)")
    args = ap.parse_args()
    print("=== PRE-REGISTERED PROVE-OUT: K1-K4 (context-switch sharpen/dampen) ===", flush=True)

    ck = torch.load(args.ckpt, map_location=device)
    net = SimpleNet(use_circuit=True, context=True).to(device)
    net.load_state_dict(ck['net']); net.eval()
    print(f"reloaded {os.path.basename(args.ckpt)}  cfg={ck.get('cfg')}  "
          f"g_ctx={F.softplus(net.g_ctx_raw).item():.3f}", flush=True)
    saved_pass = gate(net, f"saved {os.path.basename(args.ckpt)} (reload-from-disk)")

    fresh_pass = None
    if args.fresh is not None:
        from train_switch import run_seed
        netf, _read, _sw = run_seed(args.fresh)
        fresh_pass = gate(netf, f"fresh retrain seed-{args.fresh}")

    summary = f"PROVEOUT_SUMMARY  saved_PASS={saved_pass}"
    if fresh_pass is not None:
        summary += f"  fresh_seed{args.fresh}_PASS={fresh_pass}"
    print(summary, flush=True)
    print("PROVEOUT_DONE", flush=True)
