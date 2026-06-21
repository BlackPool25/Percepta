# Percepta — Training Autopsy: What Broke and Why

## Summary

Everything we've tried fails. The agent doesn't learn. This document explains why.

---

## Attempt 1: Dreamer-Style RL (Imagination Training)

**What we did:**
- RSSM world model trained via JEPA
- Actor trained in imagination (Dreamer's imagine_sequence)
- Curiosity reward from ensemble disagreement
- Exponential proximity reward

**Results:**
- Lost explosion (42 → 8143): encoder + RSSM joint training unstable
- Fixed with JEPA, autoencoding, frozen features: stable but agent didn't learn
- Collapse at ep 150: reward dropped from 664 to 16 in 50 episodes

**Why it failed:**
- Joint PerceptaModel + RSSM training is architecturally unstable
- PPO/REINFORCE on a single episode is too noisy
- The actor has NO good data to learn from — it never reaches the goal
- Catastrophic policy update destroys progress

**Moved away because:** The Dreamer approach requires the RSSM to be accurate enough for imagination, but the RSSM needs diverse data, which requires exploration, which requires a good policy. Circular dependency.

---

## Attempt 2: Behavioral Cloning from Heuristic (BC)

**What we did:**
- Heuristic action: position behind object, push toward goal
- Train actor via MSE to match heuristic's (h, z) → action mapping
- 30 episodes of heuristic data, 100+ BC iterations

**Results:**
- BC loss reached 0.001 (near-perfect match on training data)
- But ACTOR EVAL: 0% goal rate (never reaches the goal)
- Actor outputs different actions than heuristic on the SAME state

**Why it failed:**
- **Distribution mismatch.** The actor was trained on (h, z) from HEURISTIC rollouts, but evaluated on (h, z) from ACTOR rollouts. Different → wrong actions.
- With 300+ steps of closed-loop control, tiny errors compound into complete divergence.

**Moved away because:** Pure BC can't overcome distribution mismatch without DAgger.

---

## Attempt 3: DAgger (Dataset Aggregation)

**What we did:**
- Actor acts → collects (h, z) from its OWN rollouts
- Heuristic corrects: computes what it WOULD have done
- Train actor on (h, z, heuristic_action) from actor's own distribution
- Accumulates 7500+ samples, 25+ iterations

**Results:**
- BC loss bottomed at 0.10-0.26 (never below 0.1)
- ACTOR EVAL: 0% goal rate
- Actor learns to match average behavior, not precise actions

**Why it failed:**
- MSE = 0.26 → average error per action dim = sqrt(0.26/3) = 0.29
- With actions in [-1, 1], 0.29 error means the agent moves in the WRONG direction ~30% of the time
- At 500 steps, this compounds into never reaching the goal
- **The actor's network capacity is too small** (10K params, 2×64 hidden) for the 96-dim (h, z) input
- **The (h, z) representation from the RSSM may not contain precise spatial position information** needed for pushing

**Moved away because:** The actor simply cannot learn the precise (h, z) → action mapping with its current architecture and inputs.

---

## Attempt 4: REINFORCE / PPO with Confidence Modulation

**What we did:**
- Traditional actor-critic RL on REAL experience (not imagination)
- PPO clipping to prevent large updates
- Bi-directional confidence: high confidence → slow learning
- Curiosity reward + exponential task reward

**Results:**
- Reward varied wildly: 13 → 106 → 663 → 16 → 8 (no convergence)
- Eval: 0-10% goal rate
- Min distance sometimes improved (4.98 → 2.74) then collapsed (→ 8.31)

**Why it failed:**
- REINFORCE with a single episode per update has enormous variance
- The exponential reward creates high-variance returns (0 → 20+ per step)
- The critic can't learn a stable baseline with 1 episode per update
- A single bad episode destroys the policy (confidence modulation helps but doesn't prevent collapse)

**Moved away because:** 500 timesteps per episode × 1 episode per update is not enough data for any RL algorithm to converge.

---

## Attempt 5: Residual Learning (Heuristic + Actor Blend)

**What we did:**
- Final action = heuristic × α + actor × (1-α)
- α decays from 1.0 → 0.0 over 30 episodes
- Actor trained via BC on heuristic actions
- When α=0, actor has full control

**Results:**
- α=0.97: Goal reached, eval=20% (heuristic working)
- α=0.33: Goal not reached, eval=20% (blend still works)
- α=0.00: Goal not reached, eval=0% (actor fails)
- Reward and min distance improve slowly but goal never reached

**Why it failed:**
- The actor never LEARNS the task. It just averages heuristic actions.
- When α=0 and the actor has full control, it produces near-zero actions
- The BC training learns the AVERAGE heuristic action, but the correct action depends on the CURRENT state — the actor can't learn this conditional mapping from MSE alone

**Moved away because:** The actor doesn't learn the pushing skill. It just produces a blurred average of all heuristic actions it's seen.

---

## THE ROOT CAUSE

Every approach fails because of the same fundamental problem:

**The actor has no usable learning signal.**

1. **The reward is too sparse.** The goal is almost never reached (0% in actor-only mode). Without experiencing the goal, the agent can't learn to pursue it.

2. **The (h, z) representation is wrong for action selection.** The RSSM is trained via JEPA to predict PERCEPTAMODEL FEATURES, not to preserve spatial positions. The actor tries to learn (h, z) → action, but (h, z) might not even contain the precise (x, y) coordinates needed for pushing.

3. **The actor's network is too small.** 10K params for a 96-dim input mapping to 3-dim output, requiring precise spatial reasoning — insufficient capacity.

4. **The BC target is the wrong thing to learn.** Even with perfect (h, z) representation, learning to match the heuristic via MSE produces a "blurred average" that doesn't capture the state-conditioned nature of the policy. Two different states near the goal vs far from the goal require very different actions.

5. **ALL approaches converge on "do nothing"** because the action penalty (-0.001 × action²) plus zero reward for staying still is better than taking actions that might move the agent away from the goal.

---

## WHAT NEEDS TO CHANGE

The architecture needs a fundamental redesign of the action selection and learning pathway:

**Problem 1: The actor can't see the information it needs.**
- The actor takes (h, z) from the RSSM, not the raw observation
- (h, z) is optimized for JEPA prediction, not for action selection
- **Fix:** Give the actor DIRECT access to the raw observation (positions of agent, objects, goal)

**Problem 2: The learning signal is wrong.**
- BC trains the actor to match the heuristic's AVERAGE action (wrong)
- RL trains the actor from a reward that's never experienced (sparse)
- **Fix:** Train the actor to predict the DIRECTION from agent to target object and from object to goal — simple spatial relationships that can be computed directly from the observation

**Problem 3: None of the Percepta mechanisms are actually used.**
- The three fast paths compute features but they're never trained by gradient descent
- The bi-directional confidence is computed but never modulates anything
- The gate copies the entire actor, not just the slow weights
- The episodic buffer, priority replay, destabilization — all dead code
- **Fix:** Actually use the fast/slow mechanism. The fast paths should modulate the action. The confidence should gate the learning rate. The gate should consolidate slow weights from fast adaptations.

---

## PROPOSED REDESIGN

Build one thing at a time, validate each:

1. **Direct actor** — Actor takes (obs, task) directly (not (h, z)). Train to imitate heuristic via DAgger. Validate: 60%+ goal rate.

2. **Add PerceptaModel features** — Actor takes (obs, task, percepta_features). The fast/slow features provide within-episode adaptation. Validate: same performance.

3. **Add confidence modulation** — PerceptaModel's confidence gates the actor's learning rate. When slow is confident, actor learns slowly. Validate: no catastrophic collapse.

4. **Add gate consolidation** — After successful episodes, copy actor's fast-adapted weights to a "slow" copy. The gate validates: does this improve without forgetting? Validate: retained knowledge across episodes.

5. **Add curiosity** — Ensemble disagreement as exploration bonus. Validate: agent explores diverse states.

6. **Add RSSM world model** — JEPA for curiosity computation (not for action selection). Validate: curiosity drives exploration.

Step 1 is the foundation. Without it, nothing else matters.
