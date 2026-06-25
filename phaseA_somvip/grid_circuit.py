import torch, torch.nn.functional as F, copy, os, sys, contextlib, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # find package-root simple_net.py
from simple_net import SimpleNet, phase1, phase2, forward_seq, make_sequences, chan, device, N, STEP_DEG

_HERE = os.path.dirname(os.path.abspath(__file__))   # resolve checkpoint paths relative to THIS script, not CWD

# Where regenerated checkpoints are written. DEFAULT is a fresh grid_output/ subdir so a bare run does
# NOT overwrite the committed ckpt_circuit_{sharpen,dampen}.pt. To regenerate the shipped checkpoints in
# place, opt in explicitly with --out pointing at the phaseA_somvip directory (e.g. --out phaseA_somvip).
ap = argparse.ArgumentParser(description="Phase-A energy-vs-task grid: train the SOM/VIP circuit across a "
                                         "lam_energy x ce_weight grid and regenerate the sharpen/dampen checkpoints.")
ap.add_argument('--out', default=os.path.join(_HERE, 'grid_output'),
                help="directory for regenerated ckpt_circuit_{sharpen,dampen}.pt "
                     "(default: phaseA_somvip/grid_output/, which leaves the shipped checkpoints untouched). "
                     "Pass --out <phaseA_somvip dir> to regenerate the canonical shipped checkpoints in place.")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

@contextlib.contextmanager
def quiet():
    with open(os.devnull, 'w') as dn:
        old = sys.stdout; sys.stdout = dn
        try: yield
        finally: sys.stdout = old

torch.manual_seed(0)
base = SimpleNet(use_circuit=True).to(device)
with quiet(): phase1(base, steps=2000)
frozen = copy.deepcopy(base.state_dict())

VELS = torch.tensor([-3, -2, -1, 1, 2, 3], device=device); K = 4; B = 8000
@torch.no_grad()
def probe(net):
    c0 = torch.randint(0, N, (B,), device=device); v = VELS[torch.randint(0, 6, (B,), device=device)]
    tt = torch.arange(K, device=device)[None, :]; ctx = ((c0[:, None] + v[:, None] * tt) % N).float() * STEP_DEG
    e = (c0 + v * K) % N; u = (e + N // 2) % N
    th_e = torch.cat([ctx, (e.float() * STEP_DEG)[:, None]], 1); th_u = torch.cat([ctx, (u.float() * STEP_DEG)[:, None]], 1)
    _, re_on = forward_seq(net, th_e, 1.0); _, ru_on = forward_seq(net, th_u, 1.0); _, re_off = forward_seq(net, th_e, 0.0)
    re = re_on[:, K, :].gather(1, e.view(-1, 1)).mean().item()
    ru = ru_on[:, K, :].gather(1, u.view(-1, 1)).mean().item()
    rf = re_off[:, K, :].gather(1, e.view(-1, 1)).mean().item()
    return re, ru, rf
def regime(net):
    g = F.softplus(net.circ_raw).detach()
    som_eff = torch.relu(g[1] - g[2] * g[0]); gain = (g[3] - g[4] * som_eff).item()
    return gain, [round(x, 2) for x in g.tolist()]
@torch.no_grad()
def heldout(net, B=8192, S=12):
    th = make_sequences(B, S, mode='momentum', p_stay=0.9); preds, _ = forward_seq(net, th, 1.0)
    c = chan(th); return (preds[:, :-1].argmax(-1) == c[:, 1:]).float().mean().item() * 100

LAMS = [0.0, 0.02, 0.05, 0.1, 0.2]; CES = [0.5, 1.0, 2.0]
print("lam    ce   ratio  net_fb_gain regime   exp_r   unexp_r  floor  exp-floor  held%  gains[g_v,g_s,g_sv,g_e,g_ps]", flush=True)
for ce in CES:
    for lam in LAMS:
        try:
            net = SimpleNet(use_circuit=True).to(device); net.load_state_dict(frozen)
            with quiet(): phase2(net, mode='momentum', steps=4000, S=12, p_stay=0.9, lam_energy=lam, ce_weight=ce)
            gain, gains = regime(net); re, ru, rf = probe(net); ho = heldout(net)
            reg = 'SHARPEN' if gain > 0 else 'DAMPEN'; ratio = lam / ce
            print(f"{lam:<6.2f} {ce:<4.1f} {ratio:<6.3f} {gain:+9.3f}  {reg:<7} {re:7.3f} {ru:7.3f} {rf:6.3f} {re-rf:+8.3f}  {ho:5.1f}  {gains}", flush=True)
            if ce == 1.0 and lam == 0.0:  torch.save(net.state_dict(), os.path.join(args.out, 'ckpt_circuit_sharpen.pt'))
            if ce == 1.0 and lam == 0.2:  torch.save(net.state_dict(), os.path.join(args.out, 'ckpt_circuit_dampen.pt'))
        except Exception as ex:
            print(f"{lam:<6.2f} {ce:<4.1f} ERROR {type(ex).__name__}: {ex}", flush=True)
print("GRID_DONE", flush=True)
