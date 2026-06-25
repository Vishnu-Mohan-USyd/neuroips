import torch, torch.nn as nn, torch.nn.functional as F, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # find package-root simple_net.py
from simple_net import SimpleNet, forward_seq, phase1, make_sequences, chan, N, STEP_DEG, device

SIGMA=1.0; LAM_READ=1.0; LAM_SAVE=0.2   # fixed starting hyperparams
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

def run_seed(seed):
    torch.manual_seed(seed)
    net=SimpleNet(use_circuit=True,context=True).to(device)
    read=nn.Linear(N,N).to(device)
    phase1(net,steps=2000)
    for p in list(net.W_ff.parameters())+list(net.decoder.parameters()): p.requires_grad_(False)
    opt=torch.optim.Adam([p for p in net.parameters() if p.requires_grad]+list(read.parameters()),lr=1e-3)
    for s in range(1,4001):
        theta=make_sequences(128,12,mode='momentum',p_stay=0.9)
        ctxb=torch.where(torch.rand(128,1,device=device)<0.5, torch.ones(128,1,device=device), -torch.ones(128,1,device=device))
        preds,r_all=forward_seq(net,theta,1.0,ctx=ctxb)
        ce=F.cross_entropy(preds[:,:-1,:].reshape(-1,N),chan(theta[:,1:]).reshape(-1),reduction='none').reshape(128,11).mean(1)
        r_noisy=r_all+SIGMA*torch.randn_like(r_all)
        rd=F.cross_entropy(read(r_noisy).reshape(-1,N),chan(theta).reshape(-1),reduction='none').reshape(128,12).mean(1)
        energy=r_all.abs().mean(dim=(1,2))
        att=(ctxb.squeeze(1)>0).float()
        loss=ce.mean()+LAM_READ*(rd*att).sum()/att.sum().clamp(min=1)+LAM_SAVE*(energy*(1-att)).sum()/(1-att).sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); opt.step()
        if seed==0 and s%1000==0: print(f"  step {s} loss {loss.item():.3f} g_ctx {F.softplus(net.g_ctx_raw).item():.3f}",flush=True)
    out={}
    for cv in (1.0,-1.0):
        re,rf,acc=probe(net,cv); out[cv]=(re,rf,acc)
        print(f"  [seed{seed} ctx={cv:+.0f} {'attend' if cv>0 else 'save'}] exp_r={re:.3f} floor={rf:.3f} exp-floor={re-rf:+.3f} held={acc:.1f}% -> {'SHARPEN' if re>rf else 'DAMPEN'}",flush=True)
    gctx=F.softplus(net.g_ctx_raw).item()
    sw = out[1.0][0]>out[1.0][1] and out[-1.0][0]<out[-1.0][1]
    print(f"  seed{seed}: g_ctx={gctx:.3f}  SWITCH={'YES' if sw else 'NO'} (attend-margin {out[1.0][0]-out[1.0][1]:+.2f}, save-margin {out[-1.0][0]-out[-1.0][1]:+.2f})",flush=True)
    return sw

print("=== NOISY-READOUT FIX: seed-0 KILL-GATE (sigma=1.0, lam_read=1.0, lam_save=0.2) ===",flush=True)
s0=run_seed(0)
print(f"SEED0_SWITCH={s0}",flush=True)
print("GATE0DONE",flush=True)
