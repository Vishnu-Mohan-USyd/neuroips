import torch, torch.nn.functional as F, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # find package-root simple_net.py
from simple_net import SimpleNet, phase1, phase2, forward_seq, l4_code, device, N, STEP_DEG

_HERE = os.path.dirname(os.path.abspath(__file__))   # resolve checkpoint paths relative to THIS script, not CWD

# 1) REGRESSION: use_circuit=False still loads old additive ckpt + reproduces FF center ~1.23
torch.manual_seed(0)
old = SimpleNet().to(device)
sd = torch.load(os.path.join(_HERE, 'ckpt_momentum.pt'), map_location=device); sd = sd['state_dict'] if (isinstance(sd, dict) and 'state_dict' in sd) else sd
old.load_state_dict(sd); old.eval()
VELS = torch.tensor([-3, -2, -1, 1, 2, 3], device=device); K = 4
c0 = torch.randint(0, N, (4000,), device=device); v = VELS[torch.randint(0, 6, (4000,), device=device)]
tt = torch.arange(K, device=device)[None, :]; ctx = ((c0[:, None] + v[:, None] * tt) % N).float() * STEP_DEG
e = (c0 + v * K) % N; th = torch.cat([ctx, (e.float() * STEP_DEG)[:, None]], 1)
with torch.no_grad(): _, r = forward_seq(old, th, 0.0)
print("[regression] use_circuit=False loads old ckpt; FF center = %.3f (expect ~1.23)" % r[:, K, :].gather(1, e.view(-1, 1)).mean().item())

# 2) circuit construction + shapes
torch.manual_seed(0)
net = SimpleNet(use_circuit=True).to(device)
rr = net.l23(l4_code(torch.zeros(8, device=device)), torch.ones(8, N, device=device))
print("[shape] l23 ->", tuple(rr.shape), "(expect (8,%d)) | circ_raw" % N, tuple(net.circ_raw.shape), "| gru in", net.gru.weight_ih.shape[1])
print("[init] softplus(circ_raw) g_v,g_s,g_sv,g_e,g_ps =", [round(x, 3) for x in F.softplus(net.circ_raw).tolist()])

# 3) phase1 must NOT move gains (fb=0); phase2 MUST move them (trainable via optimizer)
g0 = F.softplus(net.circ_raw).detach().clone()
phase1(net, steps=300)
gp1 = F.softplus(net.circ_raw).detach().clone()
phase2(net, mode='momentum', steps=400, S=12, p_stay=0.9, lam_energy=0.02, ce_weight=1.0)
g1 = F.softplus(net.circ_raw).detach()
print("[phase1 effect on gains] max|delta| = %.4f (expect ~0)" % (gp1 - g0).abs().max().item())
print("[gains after short phase2] g_v,g_s,g_sv,g_e,g_ps =", [round(x, 3) for x in g1.tolist()])
print("[gains moved from init]   max|delta| = %.4f (expect >0 -> trainable)" % (g1 - g0).abs().max().item())
print("SMOKE_OK")
