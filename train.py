"""Percepta: heuristic pre-training + continuous refinement + gate + retrieval + destabilization."""
import argparse,time,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
from copy import deepcopy
from env import MuJoCoPlayground

device='cuda' if torch.cuda.is_available() else 'cpu'
torch.set_float32_matmul_precision('high')

def sf(obs,task):
    ti=np.argmax(task)
    vt=obs[6+ti*6:8+ti*6]-obs[0:2];vg=obs[24:26]-obs[6+ti*6:8+ti*6]
    dt=np.linalg.norm(vt);dg=np.linalg.norm(vg)
    behind=1.0 if dt>0.1 and np.dot(vt/(dt+1e-6),vg/(dg+1e-6))<-0.3 else 0.0
    return np.array([vt[0],vt[1],vg[0],vg[1],dt,dg,obs[27+ti],behind],dtype=np.float32)

def direction(sf):
    vt,vg,dt=sf[:2],sf[2:4],sf[4]
    if dt>0.5: d=vt/(dt+1e-6)
    else: dg=sf[5]; d=vg/(dg+1e-6) if dg>0.01 else np.zeros(2)
    return np.array([d[0],d[1],-0.3],dtype=np.float32)

def heuristic_action(obs,task):
    ti=np.argmax(task)
    op=obs[6+ti*6:9+ti*6];ap=obs[0:3];gp=obs[24:27]
    pdv=op[:2]-gp[:2];pd=np.linalg.norm(pdv);pu=pdv/(pd+1e-6)
    dt_vec=(op[:2]+pu*0.6)-ap[:2];dt=np.linalg.norm(dt_vec)
    if dt>0.3: a2d=dt_vec/(dt+1e-6)*min(1.0,dt*2.0)
    else:
        gdv=gp[:2]-op[:2];gn=np.linalg.norm(gdv)
        a2d=gdv/(gn+1e-6)*min(0.4,gn*0.3) if gn>0.01 else np.zeros(2)
    return np.array([a2d[0],a2d[1],-0.3])

def compute_tg(act,sp): return float(np.clip(np.dot(act[:2],direction(sp)[:2])/(np.linalg.norm(direction(sp)[:2])+1e-6),0,1))

class GainNet(nn.Module):
    def __init__(self): super().__init__();self.net=nn.Sequential(nn.Linear(8,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self,sf): return torch.sigmoid(self.net(sf))

class EpisodicMemory:
    def __init__(self,cap=500):
        self.k,self.v,self.f,self.cap=[],[],[],cap
    def add(self,k,v):
        self.k.append(k.copy());self.v.append(v);self.f.append(1)
        if len(self.k)>self.cap:
            i=int(np.argmin(self.f));self.k.pop(i);self.v.pop(i);self.f.pop(i)
    def retrieve(self,q,k=5):
        if not self.k: return[]
        qn=q/(np.linalg.norm(q)+1e-8)
        sims=[(np.dot(qn,ki/(np.linalg.norm(ki)+1e-8)),i) for i,ki in enumerate(self.k)]
        sims.sort(key=lambda x:-x[0]);res=[]
        for sim,i in sims[:k]:
            self.f[i]+=1
            if sim>0.5: res.append((self.v[i],sim))
        return res

def predict(fast,slow,sf_t,mem,sp):
    with torch.no_grad(): fg,sg=fast(sf_t).item(),slow(sf_t).item()
    ret=mem.retrieve(sp,5)
    if ret: tw=sum(r[1] for r in ret);eg=sum(r[0]*r[1] for r in ret)/(tw+1e-8)
    else: eg=sg
    return float(np.clip(sg*0.3+fg*0.3+eg*0.4,0,1)),fg,sg,eg

def evaluate(fast,slow,mem,n=20):
    env=MuJoCoPlayground(max_steps=300,force_scale=50.0,curriculum_dist=3.0);g=0
    for _ in range(n):
        o,i=env.reset();t=np.zeros(3,dtype=np.float32);t[i['target_object']]=1.0;done=False
        while not done:
            sp=sf(o,t);sf_t=torch.as_tensor(sp,device=device).unsqueeze(0)
            g_,_,_,_=predict(fast,slow,sf_t,mem,sp)
            o,_,done,tr,_=env.step(np.clip(direction(sp)*g_,-1,1))
            if done: g+=1
            if tr: break
    env.close();return g/n

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--steps',type=int,default=100000)
    p.add_argument('--lr',type=float,default=1e-3)
    args=p.parse_args()

    fast=GainNet().to(device)
    slow=GainNet().to(device);slow.load_state_dict(fast.state_dict())
    mem=EpisodicMemory(500)
    opt=torch.optim.Adam(fast.parameters(),lr=args.lr)

    env=MuJoCoPlayground(max_steps=300,force_scale=50.0,curriculum_dist=3.0)

    # PHASE 1: Heuristic pre-training (abundant data)
    print('Phase 1: Heuristic pre-training...')
    demo_sf,demo_tg=[],[]
    for ep in range(50):
        o,i=env.reset();t=np.zeros(3,dtype=np.float32);t[i['target_object']]=1.0;done=False
        while not done:
            ha=heuristic_action(o,t);sp=sf(o,t)
            demo_sf.append(sp.copy());demo_tg.append(compute_tg(ha,sp))
            o,_,done,tr,_=env.step(ha)
            if tr: break
    sf_b=torch.as_tensor(np.stack(demo_sf),device=device)
    tg_b=torch.tensor(demo_tg,device=device,dtype=torch.float32)
    for _ in range(500):
        idx=torch.randperm(len(sf_b))[:128]
        loss=F.mse_loss(fast(sf_b[idx]).view(-1),tg_b[idx])
        opt.zero_grad();loss.backward();opt.step()
    slow.load_state_dict(fast.state_dict())
    gr=evaluate(fast,slow,mem,20)
    print(f'  After heuristic: {gr*100:.0f}%')

    # Fill memory with heuristic data
    for i in range(len(demo_sf)): mem.add(demo_sf[i],demo_tg[i])

    # PHASE 2: Continuous refinement
    print('\nPhase 2: Continuous refinement...')
    o,i=env.reset();t=np.zeros(3,dtype=np.float32);t[i['target_object']]=1.0
    goals=0;episodes=0;prev_sp=None;t0=time.time()
    buf_sf,buf_tg=[],[]

    for s in range(1,args.steps+1):
        sp=sf(o,t)
        sf_t=torch.as_tensor(sp,device=device).unsqueeze(0)
        g_,_,sg,_=predict(fast,slow,sf_t,mem,sp)
        act=np.clip(direction(sp)*g_,-1,1).astype(np.float32)
        no,r,term,trunc,info=env.step(act)
        nt=np.zeros(3,dtype=np.float32);nt[info['target_object']]=1.0

        if prev_sp is not None:
            imp=prev_sp[5]-sf(no,nt)[5]
            if imp>0.005:
                tg=compute_tg(act,sp)
                buf_sf.append(sp.copy());buf_tg.append(tg)
                mem.add(sp,tg)

                # Destabilization
                if sg>0.5 and tg<0.3:
                    with torch.no_grad():
                        corr=sg*0.6+tg*0.4
                        ld=F.mse_loss(slow(sf_t).view(-1),torch.tensor([corr],device=device))
                    od=torch.optim.SGD(slow.parameters(),lr=args.lr)
                    od.zero_grad();ld.backward();od.step()

        prev_sp=sp;o,t=no,nt
        if term: goals+=1
        if term or trunc:
            o,i=env.reset();t=np.zeros(3,dtype=np.float32);t[i['target_object']]=1.0
            episodes+=1

        if s%100==0 and len(buf_sf)>=16:
            sf_b=torch.as_tensor(np.stack(buf_sf),device=device)
            tg_b=torch.tensor(buf_tg,device=device,dtype=torch.float32)
            for _ in range(20):
                idx=torch.randperm(len(sf_b))[:min(64,len(sf_b))]
                loss=F.mse_loss(fast(sf_b[idx]).view(-1),tg_b[idx])
                opt.zero_grad();loss.backward();opt.step()

            # Gate
            with torch.no_grad():
                se=F.mse_loss(slow(sf_b).view(-1),tg_b).item()
                fe=F.mse_loss(fast(sf_b).view(-1),tg_b).item()
            if fe<se*0.8 and se>0.01:
                shadow=deepcopy(slow)
                sho=torch.optim.Adam(shadow.parameters(),lr=args.lr)
                for _ in range(30):
                    idx=torch.randperm(len(sf_b))[:min(64,len(sf_b))]
                    ls=F.mse_loss(shadow(sf_b[idx]).view(-1),tg_b[idx])
                    sho.zero_grad();ls.backward();sho.step()
                with torch.no_grad():
                    ne=F.mse_loss(shadow(sf_b).view(-1),tg_b).item()
                    if ne<se*0.8: slow.load_state_dict(shadow.state_dict())
            buf_sf,buf_tg=[],[]

        if s%(args.steps//3)==0:
            gr=evaluate(fast,slow,mem,20)
            print(f'Step {s:7d} | Eps={episodes:4d} | Goals={goals:3d} | Eval={gr*100:.0f}% | T={time.time()-t0:.0f}s')

    print(f'\nDone | Goals={goals} | Eps={episodes} | {time.time()-t0:.0f}s')
    gr=evaluate(fast,slow,mem,40)
    print(f'Final: {gr*100:.0f}% goal rate')

if __name__=='__main__': main()
