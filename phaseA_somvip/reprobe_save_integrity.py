import torch, torch.nn.functional as F, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # find package-root simple_net.py
from simple_net import SimpleNet, forward_seq, N, STEP_DEG, device

_HERE = os.path.dirname(os.path.abspath(__file__))   # resolve checkpoint paths relative to THIS script, not CWD
VELS = torch.tensor([-3,-2,-1,1,2,3], device=device); K=4; B=8000
torch.manual_seed(0)
@torch.no_grad()
def probe(ckpt):
    net = SimpleNet(use_circuit=True).to(device)
    net.load_state_dict(torch.load(ckpt, map_location=device))
    c0 = torch.randint(0,N,(B,),device=device); v = VELS[torch.randint(0,6,(B,),device=device)]
    tt = torch.arange(K,device=device)[None,:]; ctxd=((c0[:,None]+v[:,None]*tt)%N).float()*STEP_DEG
    e=(c0+v*K)%N; th=torch.cat([ctxd,(e.float()*STEP_DEG)[:,None]],1); idx=e.view(-1,1)
    u=((e+N//2)%N).view(-1,1)
    _, r_on  = forward_seq(net, th, 1.0)
    _, r_off = forward_seq(net, th, 0.0)
    re  = r_on [:,K,:].gather(1,idx).mean().item()
    rf  = r_off[:,K,:].gather(1,idx).mean().item()
    reu = r_on [:,K,:].gather(1,u).mean().item()
    return re, rf, reu
for name,ck in [('sharpen',os.path.join(_HERE,'ckpt_circuit_sharpen.pt')),('dampen',os.path.join(_HERE,'ckpt_circuit_dampen.pt'))]:
    re,rf,reu = probe(ck)
    reg = 'SHARPEN' if re>rf else 'DAMPEN'
    print(f"[{name:7s}] expected_fb_on={re:7.3f}  floor(fb_off)={rf:6.3f}  unexpected_fb_on={reu:6.3f}  exp/floor={re/rf:5.2f}x  -> {reg}", flush=True)
print("DONE", flush=True)
