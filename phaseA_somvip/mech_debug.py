import torch, torch.nn.functional as F, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # find package-root simple_net.py
from simple_net import SimpleNet, l4_code, forward_seq, device, N, STEP_DEG

_HERE = os.path.dirname(os.path.abspath(__file__))   # resolve checkpoint paths relative to THIS script, not CWD

def load(path, circ):
    net = SimpleNet(use_circuit=circ).to(device)
    sd = torch.load(path, map_location=device)
    if isinstance(sd, dict) and 'state_dict' in sd: sd = sd['state_dict']
    net.load_state_dict(sd); net.eval(); return net

add = load(os.path.join(_HERE, 'ckpt_momentum.pt'), False)        # additive baseline (use_circuit=False)
shp = load(os.path.join(_HERE, 'ckpt_circuit_sharpen.pt'), True)
dmp = load(os.path.join(_HERE, 'ckpt_circuit_dampen.pt'), True)

# 1) Is the frozen phase-1 W_ff actually identical between additive and circuit checkpoints?
print("W_ff weight equal (additive vs circuit-sharpen):", torch.equal(add.W_ff.weight, shp.W_ff.weight))
print("W_ff weight equal (circuit-sharpen vs dampen)   :", torch.equal(shp.W_ff.weight, dmp.W_ff.weight))

torch.manual_seed(123)
VELS = torch.tensor([-3, -2, -1, 1, 2, 3], device=device); K = 4; B = 8000
c0 = torch.randint(0, N, (B,), device=device); v = VELS[torch.randint(0, 6, (B,), device=device)]
tt = torch.arange(K, device=device)[None, :]; ctx = ((c0[:, None] + v[:, None] * tt) % N).float() * STEP_DEG
e = (c0 + v * K) % N; th = torch.cat([ctx, (e.float() * STEP_DEG)[:, None]], 1); idx = e.view(-1, 1)

@torch.no_grad()
def report(net, tag):
    drive_K = net.W_ff(l4_code(th[:, K]))                       # FF drive at the LAST step (expected stim e)
    di = drive_K.gather(1, idx).mean().item()
    pk = drive_K.max(1).values.mean().item()
    am = (drive_K.argmax(1) == e).float().mean().item()
    _, r_off = forward_seq(net, th, 0.0)                        # grid 'floor' = fb-off r at e
    fo = r_off[:, K, :].gather(1, idx).mean().item()
    matches = torch.allclose(r_off[:, K], F.relu(drive_K), atol=1e-4)
    print(f"[{tag}] drive@e={di:.3f}  drive_peak={pk:.3f}  argmax==e frac={am:.3f}  | fb-off r@e (floor)={fo:.3f}  | floor==relu(drive)? {matches}", flush=True)

report(add, 'ADDITIVE   ')
report(shp, 'CIRC-SHARP ')
report(dmp, 'CIRC-DAMP  ')
print("DEBUG_DONE", flush=True)
