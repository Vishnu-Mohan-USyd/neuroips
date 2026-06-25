import torch, torch.nn.functional as F, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # find package-root simple_net.py
from simple_net import SimpleNet, l4_code, forward_seq, device, N, STEP_DEG

_HERE = os.path.dirname(os.path.abspath(__file__))   # resolve checkpoint paths relative to THIS script, not CWD

def load(path):
    net = SimpleNet(use_circuit=True).to(device)
    sd = torch.load(path, map_location=device)
    if isinstance(sd, dict) and 'state_dict' in sd: sd = sd['state_dict']
    net.load_state_dict(sd); net.eval(); return net

VELS = torch.tensor([-3, -2, -1, 1, 2, 3], device=device); K = 4; B = 8000

@torch.no_grad()
def run(net, theta, fb_scale=1.0, kill=None, want_full=False):
    # Faithful replica of forward_seq's loop, but exposes the VIP/SOM/drive/r populations at step K,
    # and can causally silence VIP or SOM (kill in {None,'vip','som'}).
    g = F.softplus(net.circ_raw); Bb, S = theta.shape
    h = torch.zeros(Bb, net.hidden, device=device); pred_down = torch.zeros(Bb, N, device=device)
    rec = None; r_seq = []
    for t in range(S):
        drive = net.W_ff(l4_code(theta[:, t])); fb = fb_scale * pred_down
        vip = F.relu(g[0] * fb)
        if kill == 'vip': vip = torch.zeros_like(vip)
        som = F.relu(g[1] * fb - g[2] * vip)
        if kill == 'som': som = torch.zeros_like(som)
        r = F.relu(drive + g[3] * fb - g[4] * som)
        if t == K: rec = dict(drive=drive, fb=fb, vip=vip, som=som, r=r)
        r_seq.append(r); h = net.gru(r, h); pred_down = F.relu(net.W_fb(h))
    return (rec, torch.stack(r_seq, 1)) if want_full else rec

@torch.no_grad()
def probe(net, kill=None, check=False):
    c0 = torch.randint(0, N, (B,), device=device); v = VELS[torch.randint(0, 6, (B,), device=device)]
    tt = torch.arange(K, device=device)[None, :]; ctx = ((c0[:, None] + v[:, None] * tt) % N).float() * STEP_DEG
    e = (c0 + v * K) % N; th = torch.cat([ctx, (e.float() * STEP_DEG)[:, None]], 1); idx = e.view(-1, 1)
    if check:
        rec, r_seq = run(net, th, 1.0, kill=kill, want_full=True)
        _, r_ref = forward_seq(net, th, 1.0)
        print(f"   [sanity] instrumented r vs forward_seq r at step K: max|diff|={ (r_seq[:,K]-r_ref[:,K]).abs().max().item():.2e} (expect ~0)", flush=True)
    else:
        rec = run(net, th, 1.0, kill=kill)
    return {k: rec[k].gather(1, idx).mean().item() for k in ['drive', 'fb', 'vip', 'som', 'r']}

for name, path in [('SHARPEN', os.path.join(_HERE, 'ckpt_circuit_sharpen.pt')), ('DAMPEN', os.path.join(_HERE, 'ckpt_circuit_dampen.pt'))]:
    net = load(path); print(f"\n===== {name}  ({path}) =====", flush=True)
    base = probe(net, check=True)
    print(f"[{name}] predicted channel, fb ON:  drive={base['drive']:.3f}  fb={base['fb']:.3f}  VIP={base['vip']:.3f}  SOM={base['som']:.3f}  r={base['r']:.3f}", flush=True)
    kv = probe(net, kill='vip'); ks = probe(net, kill='som')
    print(f"[{name}] KILL VIP -> SOM {base['som']:.3f}=>{kv['som']:.3f} ,  r {base['r']:.3f}=>{kv['r']:.3f}", flush=True)
    print(f"[{name}] KILL SOM -> r {base['r']:.3f}=>{ks['r']:.3f}   (drive/floor={base['drive']:.3f})", flush=True)
print("\nMECH_DONE", flush=True)
