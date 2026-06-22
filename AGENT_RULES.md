# Percepta — Rules for the Next Agent

**Read this FIRST before making any changes.** Our architecture is NOVEL. No existing paper or codebase combines DG pattern separation + cerebellar sparse expansion + dopamine 3-factor plasticity + compositional trajectory replay in the same system. Approaches that work for other architectures (Dreamer, PPO, SAC, EWC, etc.) may NOT work for ours.

---

## Rule 1: Never Import Solutions from Other Architectures Without Analysis

Before implementing anything from a paper or another codebase, you must answer:

- **What assumption does this method make about the learning system?** (e.g., "uses replay buffer," "has a single policy network," "uses MSE loss")
- **Does our architecture satisfy that assumption?** (e.g., we don't use a replay buffer — we use DG + CA3. We don't have a single policy — we have hippocampus retrieval + dopamine policy + RawFM)
- **If not, how would the method need to change to work with our specific components?**

**Examples of what NOT to do:**
- Don't add EWC — we already proved diagonal Fisher is structurally insufficient (Spearman ρ=0.34 for dense layers) in our own experiments
- Don't add Dreamer-style planning — we already proved the RawFM compounds prediction errors over multiple steps
- Don't add PPO clipping or GAE — we replaced PPO with dopamine-modulated REINFORCE for a reason
- Don't add transformer attention — our DG + CA3 already provides content-addressable retrieval

---

## Rule 2: Our Components and What They Actually Do

| Component | What It Actually Does | NOT What You Might Think |
|-----------|----------------------|--------------------------|
| **DG (PatternSeparator)** | Fixed random projection + k-WTA sparsity. Maps 10-dim state → 2000-dim sparse binary code (2% active). NEVER learned. | It's NOT a learned embedding. NOT a neural network layer. It's a FIXED random projection. |
| **CA3Memory** | One-shot Hebbian storage. Stores sparse code + associated (state, action, reward, next_state, episode_id). Retrieval via softmax attention over stored patterns. | It's NOT a neural network. NOT gradient-based. NOT a transformer. It's a content-addressable memory. |
| **CerebellarModel (RawFM)** | PatternSeparator(14→5000, 2%) + MLP(5000→128→13). Predicts Δs = s' - s (efference copy). Trained via MSE on every transition. | It's NOT accurate enough for multi-step planning. It's a SINGLE-STEP physics predictor. Use it for single-step evaluation, not multi-step simulation. |
| **Policy** | MLP(10→128→128→2). Takes state ONLY (no goal). Outputs action via tanh. Trained by BC (sleep) + dopamine REINFORCE (wake). | It does NOT know the goal. It does NOT receive goal direction. It learns purely from past (state → action) pairs stored in hippocampus. |
| **Dopamine REINFORCE** | Δθ = α · δ · ∇_θ log π(a|s). δ = r + γV(s') - V(s). LR modulated by |δ| and phasic boost. | It's NOT PPO. No importance sampling, no clipping, no GAE. It's the brain's 3-factor plasticity rule. |
| **Hippocampus retrieval** | softmax(β · z_q @ Z.T) @ actions. Weighted average of actions from 10 most similar stored states. | It's NOT learned. NOT gradient-based. It directly uses stored experience. |
| **Trajectory retrieval** | Retrieves next K actions from the same episode, not just the nearest transition. | The episode library is a DIFFERENTIABLE KNOWLEDGE BASE. Use it for multi-step actions without compounding error. |
| **ACC** | Velocity-based stuck detection. If vel < 0.05 AND action > 0.5 for 3+ steps → detour. | It does NOT know the goal position. It only knows whether the agent is moving. |
| **Compositional sleep replay** | Stitches trajectory segments from different episodes at similar states. Generates NOVEL (state, action) pairs. | This is NOT random data augmentation. It recombines REAL experiences to create plausible novel transitions. |

---

## Rule 3: The RawFM is the Neocortex. The Policy is the Striatum.

**This is the most important architectural insight.**

| Component | Brain Region | Learning Type | What It Learns |
|-----------|-------------|---------------|----------------|
| **RawFM** | **Neocortex** | **Unsupervised predictive coding** | **Physics rules:** "force → acceleration → position change" |
| **Policy** | **Striatum** | **Dopamine-modulated REINFORCE** | **Action selection:** "given this state, which action leads to reward?" |

**What this means for building:**
- Sleep should train the RawFM MORE (it learns universal physics)
- The policy learns from dopamine during wake (it learns task-specific behavior)
- Do NOT train the policy via supervised BC during sleep — that's memorization, not abstraction
- The RawFM's physics knowledge generalizes to ANY goal; the policy's action knowledge is task-specific

**The barrier to true generalization is the RawFM's physics accuracy, not the policy's action memory.** A better RawFM → better predictions → better curiosity signal → better exploration → better hippocampus patterns → better policy.

---

## Rule 4: Never Remove a Component Because It "Seems Unnecessary"

Every component in this architecture exists because it solved a specific problem that other approaches couldn't. Before removing or changing anything:

1. Read the decision log in NEW_ARCHITECTURE.md (section 2)
2. Understand WHY that component was added
3. Understand WHAT failure mode it prevents
4. Only change if you understand both the component AND its failure mode

**Components that seem redundant but are critical:**
- **DG pattern separation** — Without it, the hippocampus can't distinguish similar states → catastrophic interference
- **Separate policy/value optimizers** — Without it, dopamine boost destabilizes value learning
- **No goal in state** — Without it, the agent cheats by reading goal coordinates (not brain-like)
- **Cerebellar sparse expansion** — Without it, the RawFM can't learn physics (128 hidden units vs 5000 sparse codes)
- **Episode trajectory tracking** — Without it, you can't retrieve multi-step action sequences without compounding error

---

## Rule 5: The Three Missing Pieces Are Uses of Existing Components

The three mechanisms needed for full autonomy and generalization are NOT new brain regions. They are new USES of the components we already have:

| Missing Piece | Component to Modify | What to Change | Lines |
|---------------|-------------------|----------------|-------|
| **Curiosity** | RawFM + Dopamine REINFORCE | Add RawFM prediction error to reward signal: `reward = task_reward + β * ||RawFM(s,a) - s'||²` | ~5 |
| **Better RawFM** | RawFM training loop | Increase sleep iterations from 30 to 1000+. The RawFM is the neocortex — it learns universal physics rules. | ~2 |
| **Memory compression** | CA3Memory sleep | Cluster DG patterns by active unit overlap, keep prototypes, evict redundant patterns | ~50 |

**Do NOT add new network architectures. Do NOT import world model implementations from Dreamer. Do NOT add transformers. The solution is already in the code — it just needs to be used more intensively.**

---

## Rule 6: Test Methodology

When testing changes, use these exact commands:

```bash
# Quick test (1 min)
.venv/bin/python3 train_sr.py  # 2000 steps + 10 episode test

# Full generalization test (10 min)
.venv/bin/python3 test_generalization.py  # 4 phases, 50 episodes each

# Custom test script (for specific experiments)
.venv/bin/python3 -c "from train_sr import train; hc, pi, raw_fm = train(5000)"
```

**What to track:**
- Phase 0: should always be 90-100% (if not, something is broken)
- Phase 3 (walls): target is 40-66% (improving from trajectory retrieval)
- Phase 4 (massive random mazes): target is 38-60%
- RawFM init loss: should be 70-90 after 200 iterations (if >150, the RawFM isn't learning)
- Sleep BC loss: should decrease over training (if it stays flat, nothing is being learned)

**Never test on only one seed.** Run at least 3 seeds to measure variance.

---

## Rule 7: Ensure the Model Doesn't Cheat — Verify It Actually Learns

Our architecture is vulnerable to CHEATING — the agent finding shortcuts that look like learning but aren't. Every change must be verified against these known cheating patterns:

### Known Cheating Patterns (From Our Experience)

| Cheating Pattern | How It Happens | How to Detect | How to Fix |
|-----------------|---------------|---------------|------------|
| **Goal leaking through state** | Goal position included in observation. Agent reads coordinates directly. | Remove goal from state. If performance drops → it was cheating. | State must be ONLY agent position + velocity + objects + contacts. NO goal. |
| **Goal leaking through reward** | Reward = -0.1*dist_to_goal. Agent can INFER goal position from reward differences. | Test with goal at (-3,-3): if agent succeeds without ever seeing that goal, reward is leaking. | Use sparse reward (0 or 1, no distance signal). Or verify reward diff is <0.01 per step. |
| **Policy memorizing (s→a) instead of learning rules** | Sleep BC on (state → action) pairs. Policy just memorizes the training data. | Test on NOVEL states (different positions, different goals). If cosim on train=0.999 but test fails → memorization. | Train RawFM on prediction (s,a → s') instead of BC on action. Physics rules generalize. |
| **Hippocampus retrieving exact match instead of generalizing** | Retrieval returns action from nearest stored state. Works perfectly for training states, fails for novel ones. | Test on states OUTSIDE the 5×5 demo grid (e.g., position (2.7, -1.3)). If it fails but works on grid points → retrieval without generalization. | Add compositional sleep replay to force generalization. Interleave episodes during retrieval. |
| **RawFM memorizing specific transitions instead of learning physics** | RawFM learns (s, a → s') for TRAINING states but can't predict for novel states. | Test RawFM prediction on held-out (s, a) pairs. If loss low on training data but high on held-out → memorization. | Increase granule cell expansion (5000→10000+). Add more diverse training data. |
| **Skip connection hiding failure** | Policy output = gd + correction. Even if MLP outputs 0, skip connection steers toward goal. | Remove skip connection. If performance drops → policy was useless. The skip connection was doing all the work. | Train without skip connection. Verify the MLP actually learns. |
| **Dopamine REINFORCE not actually learning** | RPE = r + γV(s') - V(s). If V(s) ≈ V(s') for all states, RPE ≈ r. Policy just maximizes immediate reward. | Check value function: V(s) should vary across states. If V(s) is constant for all s, value learning has collapsed. | Verify value loss decreases during training. Add entropy bonus to policy. |
| **Demos providing too much information** | Demo trajectories cover the entire state space. The agent never needs to generalize because it has a stored experience for every situation. | Test with FEWER demos (5 instead of 25). If performance drops sharply → agent was relying on dense demo coverage, not learning. | Reduce demo count. Add curiosity to drive self-exploration. |

### The Cheating Audit Checklist

Before declaring any result valid, run this checklist:

- [ ] **Goal removed from state?** State should be 10-dim: [pos, vel, objects, contacts]. No goal coordinates.
- [ ] **Goal not in policy input?** Policy takes state ONLY. No goal direction, no goal position.
- [ ] **Reward leakage measured?** Test with goal at (-3,-3). If success rate > random (26%), reward is leaking goal info.
- [ ] **Skip connection removed or verified?** Policy should work without it.
- [ ] **Held-out state test passed?** Test on states outside the demo grid. Agent should still succeed.
- [ ] **Fewer demos test passed?** Reduce demos from 25 to 5. Performance should degrade gracefully, not catastrophically.
- [ ] **Value function varies?** V(s) should be different for different states. If constant, nothing is being learned.
- [ ] **RawFM generalizes?** RawFM loss on held-out (s, a) should be similar to training loss. If much higher, RawFM is memorizing.

### The Golden Rule

**If the agent succeeds but you don't know why, assume it's cheating until proven otherwise.** Disable components one by one. Remove the skip connection. Remove the goal from state. Test with random actions as control (26% on Phase 0). The agent should only be credited with learning if it fails when the learning mechanism is removed.

---

## Rule 8: The Final Goal

The architecture, when all three missing pieces are implemented and scaled:

1. **No demos needed** — the agent explores via curiosity (RawFM prediction error)
2. **Learns physics** — the RawFM becomes an accurate single-step physics predictor through intensive sleep training
3. **Generalizes to ANY goal** — physics rules are universal; the hippocampus stores experiences, the RawFM learns the underlying laws
4. **Scales to lifelong learning** — memory compression via DG prototype clustering prevents capacity overflow
5. **Acts autonomously** — dopamine REINFORCE with curiosity + task reward drives all behavior

The architecture is complete. The missing pieces are not inventions — they are amplifications of what already exists. The RawFM error is already computed. The DG patterns are already clusterable. The sleep training loop is already running. Use them more.
