import torch, torch.nn.functional as F, copy, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # find package-root simple_net.py
from simple_net import SimpleNet, phase1, phase2, forward_seq, device, N, STEP_DEG

torch.manual_seed(0)
base = SimpleNet(use_circuit=True).to(device)
phase1(base, steps=2000)                                   # shared frozen substrate (same as additive/subtractive)
frozen = copy.deepcopy(base.state_dict())

VELS = torch.tensor([-3, -2, -1, 1, 2, 3], device=device); K = 4; B = 8000
@torch.no_grad()
def probe(net):
    c0 = torch.randint(0, N, (B,), device=device); v = VELS[torch.randint(0, 6, (B,), device=device)]
    tt = torch.arange(K, device=device)[None, :]
    ctx = ((c0[:, None] + v[:, None] * tt) % N).float() * STEP_DEG
    e = (c0 + v * K) % N; u = (e + N // 2) % N
    th_e = torch.cat([ctx, (e.float() * STEP_DEG)[:, None]], 1)
    th_u = torch.cat([ctx, (u.float() * STEP_DEG)[:, None]], 1)
    _, re_on  = forward_seq(net, th_e, 1.0)                 # expected,   fb on
    _, ru_on  = forward_seq(net, th_u, 1.0)                 # unexpected, fb on
    _, re_off = forward_seq(net, th_e, 0.0)                 # expected,   fb off (FF floor)
    re = re_on[:, K, :].gather(1, e.view(-1, 1)).mean().item()
    ru = ru_on[:, K, :].gather(1, u.view(-1, 1)).mean().item()
    rf = re_off[:, K, :].gather(1, e.view(-1, 1)).mean().item()
    return re, ru, rf

def regime(net):
    g = F.softplus(net.circ_raw).detach()
    som_eff = torch.relu(g[1] - g[2] * g[0])               # SOM drive per unit fb after VIP disinhibition
    gain = (g[3] - g[4] * som_eff).item()                  # net feedback gain at the predicted channel
    return gain, [round(x, 3) for x in g.tolist()]

for lam in (0.0, 0.2):
    net = SimpleNet(use_circuit=True).to(device); net.load_state_dict(frozen)
    phase2(net, mode='momentum', steps=4000, S=12, p_stay=0.9, lam_energy=lam, ce_weight=1.0)
    gain, gains = regime(net)
    re, ru, rf = probe(net)
    print(f"\n[lam={lam}] gains g_v,g_s,g_sv,g_e,g_ps = {gains}", flush=True)
    print(f"[lam={lam}] net_fb_gain (predicted ch)  = {gain:+.3f}  -> {'SHARPEN' if gain > 0 else 'DAMPEN'}", flush=True)
    print(f"[lam={lam}] expected r={re:.3f}  unexpected r={ru:.3f}  FF-floor={rf:.3f}  | exp-floor={re - rf:+.3f}", flush=True)
print("PROVEOUT_DONE", flush=True)
