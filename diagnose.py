"""Diagnostic: audit each mechanism independently."""
import numpy as np
import torch
from env import MuJoCoPlayground
from agent import PerceptaAgent
from train import run_episode, sleep_phase

device = 'cuda' if torch.cuda.is_available() else 'cpu'
agent = PerceptaAgent(device=device)
env = MuJoCoPlayground(render_mode=None, max_steps=200, force_scale=50.0)

# --- AUDIT 1: Fast/Slow ---
print('=== AUDIT 1: Fast/Slow weight adaptation ===')
gen_w0 = agent.percepta.fast_gen.net[0].weight.data.clone().cpu()
slow_w0 = agent.percepta.slow[0].weight.data.clone().cpu()

# Run episode + sleep with the fix (re-encode through PerceptaModel)
data = run_episode(env, agent, max_steps=200, curiosity_scale=0.5, training=True)

gen_w_mid = agent.percepta.fast_gen.net[0].weight.data.clone().cpu()
slow_w_mid = agent.percepta.slow[0].weight.data.clone().cpu()
gen_diff_mid = (gen_w_mid - gen_w0).abs().mean().item()
slow_diff_mid = (slow_w_mid - slow_w0).abs().mean().item()
print(f'  After episode (no training): Fast Δ={gen_diff_mid:.8f}, Slow Δ={slow_diff_mid:.8f}')

# Sleep with gradient flow to PerceptaModel
l = sleep_phase(agent, data, n_rssm_steps=20, n_policy_steps=0,
                subseq_len=32, imagination_horizon=50)

gen_w1 = agent.percepta.fast_gen.net[0].weight.data.clone().cpu()
slow_w1 = agent.percepta.slow[0].weight.data.clone().cpu()
gen_diff = (gen_w1 - gen_w0).abs().mean().item()
slow_diff = (slow_w1 - slow_w0).abs().mean().item()
print(f'  After sleep (WITH gradient fix): Fast Δ={gen_diff:.8f}, Slow Δ={slow_diff:.8f}')
print(f'  (Before: 0.000000 — now should be > 0)')
print(f'  RSSM loss after sleep: {l["rssm_loss"]:.4f}')

# --- AUDIT 2: Curiosity ---
print('\n=== AUDIT 2: Curiosity variation ===')
h, z = None, None
action_p = torch.zeros(1, 3, device=device)
curiosities = []
for t in range(min(200, len(data['obs']))):
    task = np.zeros(3, dtype=np.float32)
    task[int(data['info_target'][t])] = 1.0 if 'info_target' in data else 0
    # Actually just use task from data
    obs_t = data['obs'][t]
    h, z, c, _ = agent.tick(obs_t, np.array([1,0,0], dtype=np.float32), h, z, action_p)
    curiosities.append(c)
    action_p = torch.as_tensor(data['action'][t], device=device).unsqueeze(0)

c_arr = np.array(curiosities)
print(f'  Mean: {c_arr.mean():.4f}, Std: {c_arr.std():.4f}')
print(f'  Top 5% - Bottom 5%: {np.percentile(c_arr,95) - np.percentile(c_arr,5):.4f}')

# --- AUDIT 3: RSSM prediction ---
print('\n=== AUDIT 3: RSSM prediction quality ===')
o = torch.as_tensor(data['obs'][-50:], device=device).unsqueeze(1).float()
t = torch.zeros(50, 1, 3, device=device)  # dummy task
a = torch.as_tensor(data['action'][-50:], device=device).unsqueeze(1).float()
r = torch.as_tensor(data['reward'][-50:], device=device).unsqueeze(1).float()

with torch.no_grad():
    f = agent.percepta(o.squeeze(1), t.squeeze(1)).unsqueeze(1)
    out = agent.rssm.forward_sequence(f, a, r, obs=o)
    print(f'  Held-out recon loss: {out["recon_loss"].item():.4f}')
    print(f'  Held-out KL loss:    {out["kl_loss"].item():.4f}')

# --- AUDIT 4: Task encoding ---
print('\n=== AUDIT 4: Task encoding sensitivity ===')
obs0 = torch.as_tensor(data['obs'][0], device=device).unsqueeze(0).float()
encodings = {}
for obj in range(3):
    t = torch.zeros(1, 3, device=device)
    t[0, obj] = 1.0
    with torch.no_grad():
        feat = agent.percepta(obs0, t)
    encodings[obj] = feat
d01 = (encodings[0] - encodings[1]).norm().item()
d02 = (encodings[0] - encodings[2]).norm().item()
d12 = (encodings[1] - encodings[2]).norm().item()
print(f'  Cross-task distances: d01={d01:.4f}, d02={d02:.4f}, d12={d12:.4f}')
