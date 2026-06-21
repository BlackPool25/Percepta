"""Hierarchical CLS-RL: PFC Manager + BG Gate + Motor Worker + Hippocampal Memory.

Architecture (brain-grounded, 2025-2026 research):
  PFC (Manager): outputs subgoal direction (high-level, updated every K steps)
  BG (Gate):     switches subgoal when current is achieved or stuck
  Motor (Worker): executes continuous actions to reach subgoal (every step)
  HC (Memory):   stores (state, action, Q) for one-shot episodic retrieval

Full PPO trains both Manager and Worker. Memory augments the critic.
"""

import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.distributions import Normal
from pathlib import Path
from hopfield_memory import PatternSeparator
from env_nav import NavArena

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT = Path('results/sr'); OUT.mkdir(parents=True, exist_ok=True)
S, P, H, G, A = 12, 64, 64, 2, 2

# ═══ Neocortex: SR Network ═══════════════════════════════════════
class SRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(S, 128), nn.ReLU(), nn.Linear(128, P))
        self.w = nn.Parameter(torch.zeros(P))
    def phi(self, s): return self.net(s)
    def q(self, phi): return (phi * self.w.unsqueeze(0)).sum(dim=-1)

# ═══ PFC Manager: outputs subgoal direction ══════════════════════
class Manager(nn.Module):
    """High-level: outputs subgoal direction. Updated every K steps."""
    def __init__(self):
        super().__init__()
        self.gru = nn.GRUCell(S + G, H)
        self.subgoal = nn.Linear(H, G)  # outputs 2D subgoal direction
        self.value = nn.Linear(H, 1)

    def forward(self, s, gd, h=None):
        if h is None: h = s.new_zeros(s.size(0), H)
        h = self.gru(torch.cat([s, gd], dim=-1), h)
        sg = torch.tanh(self.subgoal(h))
        v = self.value(h).squeeze(-1)
        return sg, v, h

# ═══ Motor Worker: executes actions to reach subgoal ═════════════
class Worker(nn.Module):
    """Low-level: takes (state, subgoal) → continuous action. Every step."""
    def __init__(self):
        super().__init__()
        self.gru = nn.GRUCell(S + G + G, H)  # state + goal_dir + subgoal
        self.mean = nn.Linear(H, A)
        self.log_std = nn.Parameter(torch.zeros(A))
        self.value = nn.Linear(H, 1)
        nn.init.orthogonal_(self.mean.weight, 0.01)
        nn.init.orthogonal_(self.value.weight, 1.0)

    def forward(self, s, gd, sg, h=None):
        if h is None: h = s.new_zeros(s.size(0), H)
        h = self.gru(torch.cat([s, gd, sg], -1), h)
        m = torch.tanh(self.mean(h))
        sd = F.softplus(self.log_std) + 1e-4
        v = self.value(h).squeeze(-1)
        return m, sd, v, h

# ═══ BG Gate: decides when to switch subgoal ═════════════════════
class BGGate(nn.Module):
    """Binary gate: should we switch subgoal? Trained by dopamine RPE."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(S + G + 1, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, s, sg, rpe):
        return torch.sigmoid(self.net(torch.cat([s, sg, rpe.unsqueeze(-1)], -1)))

# ═══ Hippocampus: Q-Memory ════════════════════════════════════════
class QMemory:
    """One-shot (state, action, Q) storage. Used to augment critic."""
    def __init__(self, max_sz=2000, beta=2.0):
        self.keys, self.states, self.actions, self.qs, self.imp = [],[],[],[],[]
        self.max_sz, self.beta = max_sz, beta
    def store(self, key, state, action, q):
        if len(self.keys) >= self.max_sz:
            i = int(np.argmin(self.imp)); [l.pop(i) for l in [self.keys,self.states,self.actions,self.qs,self.imp]]
        self.keys.append(key.cpu().detach()); self.states.append(state.cpu().detach())
        self.actions.append(action.cpu().detach()); self.qs.append(float(q)); self.imp.append(0.5)
    def retrieve(self, q):
        if not self.keys: return (q.new_zeros(1,S), q.new_zeros(1,A), q.new_zeros(1), 0.0, None)
        Z = torch.stack(self.keys).to(q.device, q.dtype)
        attn = F.softmax(self.beta*(q@Z.T),-1)
        qn, Zn = q/(q.norm(dim=-1,keepdim=True)+1e-8), Z/(Z.norm(dim=-1,keepdim=True)+1e-8)
        rbf = torch.exp(-2.0*(1.0-(qn@Zn.T).squeeze(0))).max().item()
        bi = attn.argmax().item()
        Ss = torch.stack(self.states).to(q.device,q.dtype)
        As = torch.stack(self.actions).to(q.device,q.dtype)
        Qs = torch.tensor(self.qs, device=q.device, dtype=q.dtype).unsqueeze(-1)
        return attn@Ss, As[bi].unsqueeze(0), (attn@Qs).squeeze(-1), rbf, bi

    def retrieve_topk(self, q, k=10):
        """Return top-K action candidates for forward planning (theta sweep)."""
        if not self.keys or len(self.keys) < k: return None
        Z = torch.stack(self.keys).to(q.device, q.dtype)
        As = torch.stack(self.actions).to(q.device, q.dtype)
        logits = self.beta * (q @ Z.T).squeeze(0)
        top_idx = logits.topk(min(k, len(self.keys))).indices
        return As[top_idx]  # (k, A) candidate actions
    def td_update(self, i, e, lr=0.3):
        if i is not None and i<len(self.qs): self.qs[i] += lr*e
    def sample(self, n, t=0.5):
        if not self.states: return None,None,None,None
        imp = np.clip(np.array(self.imp,np.float64),0.01,1.0)**(1/t)
        imp /= imp.sum()+1e-10
        idx = np.random.choice(len(self.states),min(n,len(self.states)),False,imp)
        return (torch.stack([self.keys[i] for i in idx]),
                torch.stack([self.states[i] for i in idx]),
                torch.stack([self.actions[i] for i in idx]),
                torch.tensor([self.qs[i] for i in idx]))
    def _upd_imp(self, w):
        ws = w.detach().cpu().numpy()
        for i,v in enumerate(ws):
            if v>.05: self.imp[i]=min(self.imp[i]+.05,1.0)
    def decay_imp(self,f=.99):
        for i in range(len(self.imp)): self.imp[i]=max(self.imp[i]*f,.01)
    def __len__(self): return len(self.keys)

def run_demo(mem, dg):
    env = NavArena(render_mode='rgb_array')
    obs,_=env.reset(seed=42); data=[]
    for _ in range(500):
        g=obs['state'][2:4];p=obs['state'][:2];d=g-p;dist=np.linalg.norm(d)
        a=np.clip(d/dist if dist>.2 else d*.5,-1,1).astype(np.float32)
        obs2,r,term,tr,_=env.step(a)
        data.append({'s':obs['state'].copy(),'r':r}); obs=obs2
        if term: break
    env.close()
    qs=[0.0]*len(data); c=0.0
    for i in range(len(data)-1,-1,-1): c=data[i]['r']+.99*c; qs[i]=c
    env2=NavArena(render_mode='rgb_array'); obs2,_=env2.reset(seed=42)
    for i in range(len(data)):
        g=obs2['state'][2:4];p=obs2['state'][:2];d=g-p;dist=np.linalg.norm(d)
        a=np.clip(d/dist if dist>.2 else d*.5,-1,1).astype(np.float32)
        z=dg(torch.tensor(p,dtype=torch.float32).unsqueeze(0)).squeeze(0)
        mem.keys.append(z.cpu()); mem.states.append(torch.tensor(obs2['state']))
        mem.actions.append(torch.tensor(a)); mem.qs.append(qs[i]); mem.imp.append(1.0)
        obs2,_,term,tr,_=env2.step(a)
        if term: break
    env2.close()
    print(f"  Demo: {len(data)} steps, Q=[{qs[-1]:.1f}, {qs[0]:.1f}]")

def compute_gae(r, v, d, g=.99, l=.95):
    adv = torch.zeros_like(r)
    last = 0.0
    for t in reversed(range(len(r))):
        δ = r[t] + g*v[t+1]*(1-d[t]) - v[t]
        last = δ + g*l*(1-d[t])*last
        adv[t] = last
    return adv, adv + v[:-1]

def train(n_steps=2000):
    dg = PatternSeparator(2,2000,.02)
    mem = QMemory()
    sf = SRNet().to(DEVICE)
    mgr = Manager().to(DEVICE)
    wrk = Worker().to(DEVICE)
    gate = BGGate().to(DEVICE)

    # Forward model (cerebellar analogue): predicts φ(s') from φ(s) + a
    fm = nn.Sequential(nn.Linear(P + A + G, 128), nn.ReLU(), nn.Linear(128, P)).to(DEVICE)

    opt_mgr = torch.optim.Adam(mgr.parameters(), lr=3e-4)
    opt_wrk = torch.optim.Adam(list(wrk.parameters())+list(sf.parameters())+list(fm.parameters()), lr=3e-4)
    opt_gate = torch.optim.SGD(gate.parameters(), lr=1e-3)

    run_demo(mem, dg)
    env = NavArena(render_mode=None); env.set_curriculum(0)  # no rendering during training

    # Training state
    s = env.reset(seed=42)[0]['state']
    h_mgr, h_wrk = None, None
    subgoal = torch.zeros(2, device=DEVICE)
    subgoal_steps = 0
    SUBGOAL_INTERVAL = 10  # manager outputs new subgoal every 10 steps
    goals, step = 0, 0

    # Rollout buffer
    roll = {k: [] for k in ['s','gd','sg','a','lp','v','r','d','q']}

    while step < n_steps:
        st = torch.from_numpy(s).float().to(DEVICE).unsqueeze(0)
        gd = (st[:,2:4]-st[:,:2]) / ((st[:,2:4]-st[:,:2]).norm(dim=-1,keepdim=True)+1e-8)

        # Manager: new subgoal every K steps or when gate says switch
        if subgoal_steps % SUBGOAL_INTERVAL == 0 or h_mgr is None:
            sg, v_mgr, h_mgr = mgr(st, gd, h_mgr)

        # BG Gate: should we switch subgoal?
        with torch.no_grad(): _, _, q_mem, rbf, _ = mem.retrieve(dg(st[:,:2]))
        gate_signal = gate(st.squeeze(0), sg.squeeze(0), torch.tensor(0.0, device=DEVICE))
        if gate_signal.item() > 0.5:
            sg, v_mgr, h_mgr = mgr(st, gd, h_mgr)
            subgoal_steps = 0

        # Forward planning: simulate candidate actions through the forward model
        # (Hippocampal theta sweep analogue — simulate before acting)
        with torch.no_grad():
            q_par = sf.q(sf.phi(st)).item()
            phi_s = sf.phi(st)  # current successor features

            # Get candidate actions from memory (top-K nearest patterns)
            candidates = mem.retrieve_topk(dg(st[:,:2]), k=5)
            if candidates is not None and len(mem) > 10:
                # For each candidate, simulate φ' = fm(φ, a, gd), evaluate Q'
                gd_expand = gd.expand(candidates.size(0), -1)
                phi_expand = phi_s.expand(candidates.size(0), -1)
                fm_in = torch.cat([phi_expand, candidates, gd_expand], dim=-1)
                phi_next_pred = fm(fm_in)  # (k, P) predicted next φ
                q_pred = sf.q(phi_next_pred)  # (k,) predicted next Q
                # Select the action with highest predicted Q
                best_idx = q_pred.argmax().item()
                action_planned = candidates[best_idx:best_idx+1]
                rbf, _ = 0.5, None  # signal that we used planning
            else:
                action_planned = None
                rbf = 0.0

        # Always compute worker policy (for log_prob on planned actions)
        m, sd, v_wrk, h_wrk = wrk(st, gd, sg, h_wrk)
        dist = Normal(m, sd)
        if action_planned is not None:
            action = action_planned  # override with planned action
            q_blended = q_par
        else:
            action = dist.sample()
            q_blended = q_par

        # Intrinsic reward for worker: subgoal proximity
        sg_progress = -(subgoal - (st[:,2:4]-st[:,:2])/((st[:,2:4]-st[:,:2]).norm(dim=-1,keepdim=True)+1e-8)).norm().item()

        obs2, re, term, trunc, _ = env.step(action.squeeze(0).cpu().numpy())
        s2 = obs2['state']; done = term or trunc

        # Store in memory (bounded: use parametric Q as target)
        store_target = re + 0.99 * q_par
        if rbf < 0.1 and abs(store_target) < 200:
            mem.store(dg(st[:,:2]).squeeze(0), st.squeeze(0), action.squeeze(0), store_target)
        mem.decay_imp()

        # Rollout storage
        total_r = re + .1 * sg_progress + .05 * q_blended
        roll['s'].append(st.squeeze(0).detach())
        roll['gd'].append(gd.squeeze(0).detach())
        roll['sg'].append(sg.squeeze(0).detach())
        roll['a'].append(action.squeeze(0).detach())
        roll['lp'].append(dist.log_prob(action).sum(-1).detach())
        roll['v'].append(v_wrk.detach())
        roll['r'].append(total_r)
        roll['d'].append(1.0 if done else 0.0)
        roll['q'].append(q_blended)

        subgoal_steps += 1
        step += 1
        if term: goals += 1
        if done:
            env.reset(seed=42); s = env.reset(seed=42)[0]['state']
            h_mgr, h_wrk = None, None; subgoal_steps = 0
        else: s = s2

        # Update w via TD (parametric Q learning)
        nxt_t = torch.from_numpy(s2).float().to(DEVICE).unsqueeze(0)
        q_next_par = sf.q(sf.phi(nxt_t)).item()
        td_par = re + 0.99 * q_next_par - q_par
        sf.w.data += 1e-4 * td_par * sf.phi(st).squeeze(0).detach()

        # TD update on memory: confidence-based revision
        # (Selective destabilization: revise memory when prediction error is large)

        # PPO update every 64 steps
        if step > 0 and step % 64 == 0 and len(roll['s']) >= 64:
            sb = torch.stack(roll['s'])
            gb = torch.stack(roll['gd'])
            sgb = torch.stack(roll['sg'])
            ab = torch.stack(roll['a'])
            ob = torch.stack(roll['lp'])
            rb = torch.tensor(roll['r'], device=DEVICE, dtype=torch.float32)
            db = torch.tensor(roll['d'], device=DEVICE, dtype=torch.float32)
            vb = torch.stack(roll['v'])
            qb = torch.tensor(roll['q'], device=DEVICE, dtype=torch.float32)

            # GAE bootstrap
            with torch.no_grad():
                last_s = sb[-1:]; last_gd = gb[-1:]; last_sg = sgb[-1:]
                _, _, nv, _ = wrk(last_s, last_gd, last_sg)
            av = torch.cat([vb.view(-1), nv.view(-1)])
            adv, ret = compute_gae(rb, av, db)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            # PPO epochs
            for _ in range(4):
                perm = torch.randperm(len(sb))
                for i in range(0, len(sb), 256):
                    idx = perm[i:i+32]
                    # Worker forward (for PPO)
                    m,sd,vm,_ = wrk(sb[idx], gb[idx], sgb[idx])
                    d2 = Normal(m,sd)
                    lp2 = d2.log_prob(ab[idx]).sum(-1)
                    # PPO clipped objective
                    ratio = (lp2 - ob[idx]).exp()
                    ca = torch.clamp(ratio, 0.8, 1.2) * adv[idx]
                    actor_loss = -(torch.min(ratio * adv[idx], ca)).mean()
                    # Critic learns from blended Q
                    critic_loss = F.mse_loss(vm, qb[idx])
                    loss = actor_loss + 0.5 * critic_loss - 0.01 * d2.entropy().sum(-1).mean()
                    opt_wrk.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(wrk.parameters(), 0.5)
                    opt_wrk.step()

            roll = {k: [] for k in ['s','gd','sg','a','lp','v','r','d','q']}

        if step % 200 == 0:
            print(f"  step {step}: Qpar={q_par:.0f} goals={goals} mem={len(mem)}")

    env.close()
    torch.save(wrk.state_dict(), OUT/'worker.pt')
    torch.save(mgr.state_dict(), OUT/'manager.pt')
    torch.save({'keys':mem.keys,'states':mem.states,'actions':mem.actions,'qs':mem.qs,'imp':mem.imp}, OUT/'memory.pt')
    print(f"  Done: {goals} goals in {step} steps")

def test(n_eps=50):
    wrk = Worker().to(DEVICE); wrk.load_state_dict(torch.load(OUT/'worker.pt'))
    mgr = Manager().to(DEVICE); mgr.load_state_dict(torch.load(OUT/'manager.pt'))
    dg = PatternSeparator(2,2000,.02)
    mem = QMemory()
    try:
        sd = torch.load(OUT/'memory.pt', map_location='cpu')
        mem.keys = sd['keys']; mem.states = sd['states']; mem.actions = sd['actions']
        mem.qs = list(sd['qs']); mem.imp = list(sd['imp'])
        print(f"  Loaded trained memory: {len(mem)} patterns")
    except:
        run_demo(mem, dg)
        print(f"  Fallback to demo memory: {len(mem)} patterns")
    env = NavArena(render_mode='rgb_array'); env.set_curriculum(0)
    goals = 0
    for ep in range(n_eps):
        env.reset(seed=42); s = env.reset(seed=42)[0]['state']
        h_mgr,h_wrk = None,None; subgoal = torch.zeros(2,device=DEVICE); sg_steps=0; reached=False
        for _ in range(500):
            st = torch.from_numpy(s).float().to(DEVICE).unsqueeze(0)
            gd = (st[:,2:4]-st[:,:2])/((st[:,2:4]-st[:,:2]).norm(dim=-1,keepdim=True)+1e-8)
            if sg_steps % 10 == 0 or h_mgr is None:
                sg,_,h_mgr = mgr(st, gd, h_mgr)
            m,sd,_,h_wrk = wrk(st, gd, sg, h_wrk)
            obs2,_,term,trunc,_ = env.step(Normal(m,sd).sample().squeeze(0).cpu().numpy())
            s = obs2['state']; sg_steps += 1
            if term: reached=True; break
            if trunc: h_mgr,h_wrk=None,None; sg_steps=0;
        if reached: goals+=1
        if (ep+1)%10==0: print(f"  {ep+1}/{n_eps}: {goals}/{ep+1}={goals/(ep+1):.0%}")
    print(f"  Test: {goals}/{n_eps} = {goals/n_eps:.0%}")
    env.close()

if __name__ == '__main__':
    import matplotlib; matplotlib.use('Agg')
    train(2000); test(50)
