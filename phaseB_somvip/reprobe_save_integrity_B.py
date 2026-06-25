"""Save-integrity: reload ckpt_ctxswitch_seed1.pt from disk and confirm the runtime
sharpen/dampen switch still reproduces (no training, pure forward probe). Mirrors
Phase A's reprobe_save_integrity.py."""
import torch, torch.nn as nn, torch.nn.functional as F, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # find package-root simple_net.py
from simple_net import SimpleNet, forward_seq, make_sequences, chan, N, STEP_DEG, device

_HERE = os.path.dirname(os.path.abspath(__file__))   # resolve checkpoint path relative to THIS script, not CWD
VELS=torch.tensor([-3,-2,-1,1,2,3],device=device); K=4; B=8000
@torch.no_grad()
def probe(net,cv):
    c0=torch.randint(0,N,(B,),device=device); v=VELS[torch.randint(0,6,(B,),device=device)]
    tt=torch.arange(K,device=device)[None,:]; ctxd=((c0[:,None]+v[:,None]*tt)%N).float()*STEP_DEG
    e=(c0+v*K)%N; th=torch.cat([ctxd,(e.float()*STEP_DEG)[:,None]],1); idx=e.view(-1,1)
    cb=torch.full((B,1),float(cv),device=device)
    _,ron=forward_seq(net,th,1.0,ctx=cb); _,roff=forward_seq(net,th,0.0,ctx=cb)
    th2=make_sequences(B,12,mode='momentum',p_stay=0.9)
    p2,_=forward_seq(net,th2,1.0,ctx=torch.full((B,1),float(cv),device=device))
    acc=(p2[:,:-1].argmax(-1)==chan(th2[:,1:])).float().mean().item()*100
    return ron[:,K,:].gather(1,idx).mean().item(), roff[:,K,:].gather(1,idx).mean().item(), acc
ck=torch.load(os.path.join(_HERE, 'ckpt_ctxswitch_seed1.pt'),map_location=device)
net=SimpleNet(use_circuit=True,context=True).to(device); net.load_state_dict(ck['net']); net.eval()
print(f"reloaded ckpt_ctxswitch_seed1.pt  cfg={ck['cfg']}  g_ctx={F.softplus(net.g_ctx_raw).item():.3f}",flush=True)
sw=[]
for cv in (1.0,-1.0):
    re,rf,acc=probe(net,cv)
    reg='SHARPEN' if re>rf else 'DAMPEN'; sw.append(reg)
    print(f"  [reload ctx={cv:+.0f} {'attend' if cv>0 else 'save '}] exp_r={re:.3f} floor={rf:.3f} exp-floor={re-rf:+.3f} held={acc:.1f}% -> {reg}",flush=True)
print(f"SWITCH_FROM_DISK={'YES' if sw==['SHARPEN','DAMPEN'] else 'NO'}",flush=True)
print("REPROBEDONE",flush=True)
