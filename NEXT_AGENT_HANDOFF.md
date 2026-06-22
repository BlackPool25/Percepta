# Next Agent Handoff — Percepta Project

## Your Mission

Continue building Percepta — a brain-inspired autonomous learning agent. The architecture is novel and complete. Your job is to implement the three missing mechanisms that will transform it from a memorizing navigation system into a truly autonomous, generalizing agent.

## Critical: Read These Files First (in order)

1. **`AGENT_RULES.md`** — The rules of this project. Read this ENTIRELY before anything else. It contains:
   - Why you cannot import solutions from other architectures without analysis
   - What each component actually does (and doesn't do)
   - The neuroscience fidelity of each component (which are brain-like and which aren't)
   - The known cheating patterns and how to detect them
   - The three missing mechanisms and how to implement them

2. **`NEW_ARCHITECTURE.md`** — Complete architecture reference with all decisions, results, and the path forward.

3. **`train_sr.py`** — The main codebase (~830 lines). Read it to understand the implementation.

## The Three Things You Need to Build

### 1. Curiosity (Intrinsic Motivation)
**Status:** NOT YET IMPLEMENTED
**What to do:** Add RawFM prediction error as an intrinsic reward signal.
**How:** Modify the reward in the training loop:
```python
curiosity = F.mse_loss(sp, s2_t.detach())  # already computed!
reward_total = re + 0.1 * curiosity.item()
```
**Risk:** The curiosity coefficient (0.1) needs tuning. Too high → agent never exploits. Too low → agent never explores.
**Verification:** Agent should explore without demos. Test with zero task reward — agent should still explore.

### 2. Sleep Trains RawFM More (Neocortical Abstraction)
**Status:** PARTIALLY IMPLEMENTED (sleep trains RawFM for 30 iterations)
**What to do:** Increase RawFM training during sleep from 30 to 1000+ iterations. The RawFM is the neocortex — it learns universal physics rules through prediction.
**Where:** In the sleep section, change:
```python
for _ in range(30):  # CHANGE TO 1000+
    sp, rp = raw_fm(hs, ha)
    ...
```
**Risk:** More training = slower sleep. But physics rules are UNIVERSAL — a well-trained RawFM works for ANY goal.
**Verification:** RawFM prediction error (MSE) should decrease significantly. Test: after sleep, RawFM should predict held-out transitions accurately.

### 3. Memory Compression (Lifelong Scaling)
**Status:** NOT YET IMPLEMENTED
**What to do:** Add prototype clustering to CA3Memory. During sleep, cluster DG patterns by active unit overlap, keep one prototype per cluster, discard redundant patterns.
**How:** In CA3Memory, add:
```python
def compress(self, max_prototypes=200):
    # Cluster patterns by Jaccard similarity of active units
    # Keep only prototype (state, action) per cluster
    # Store prototypes in a separate SchemaBank
```
**Risk:** Overly aggressive compression loses information. Conservative compression doesn't help scaling.
**Verification:** After compression, test performance should remain similar (within 5%). With compression, the hippocampus can scale to millions of experiences.

## Before You Build Anything

### Research Online First

Before touching any code, research these topics. Use Google Scholar, ArXiv, and the search tool.

1. **Curiosity as intrinsic motivation in RL** — Search for "prediction error as intrinsic reward" and "curiosity-driven exploration." Understand the theory, then adapt it to OUR architecture (which uses a RawFM, not a separate world model).

2. **Sleep consolidation in brains** — Search for "hippocampal replay neocortical consolidation during sleep." Understand HOW the brain replays experiences during sleep to consolidate memory. Then apply it to OUR architecture (which already has a sleep phase — it just needs to use it more effectively).

3. **Prototype clustering for memory compression** — Search for "gist extraction from episodic memory" and "hippocampal schema formation." Understand how the brain compresses similar experiences into schemas. Then implement it for OUR DG patterns.

### After Research, Before Code

For EACH of the three mechanisms, write a brief plan:
- What does the brain do?
- How does our architecture approximate it?
- What exactly will you change in the code (which file, which lines)?
- How will you verify it works?

Show the plan before writing code.

## After Building

For each mechanism, run the full test suite:
```bash
.venv/bin/python3 train_sr.py              # Quick test (1 min)
.venv/bin/python3 test_generalization.py    # Full test (10 min)
```

Track these metrics:
- Phase 0: should stay 90-100%
- Phase 3 (walls): currently 40% — target is 50%+
- Phase 4 (massive random mazes): currently 38% — target is 50%+
- RawFM init loss: should be 70-90 after 200 iters
- Sleep BC loss: should consistently decrease

Run at least 3 seeds to measure variance.

## What NOT to Do

- Do NOT add PPO, SAC, Dreamer, or any other existing RL algorithm
- Do NOT add transformers, attention layers, or any new network architecture
- Do NOT add EWC, SI, or any weight regularization
- Do NOT change the DG (fixed random projection + k-WTA) — it's correct
- Do NOT make the RawFM do multi-step planning — it's a single-step forward model
- Do NOT add goal position back to the state — the agent should navigate from memory

## The Goal

When all three mechanisms are implemented and working:

1. **No demos needed** — curiosity drives exploration
2. **Learns physics** — RawFM becomes an accurate single-step predictor
3. **Scales to lifelong** — memory compression prevents capacity overflow
4. **Generalizes** — physics rules apply to ANY goal, ANY maze

The architecture is complete. The three missing pieces are amplifications of what already exists — not new inventions. Build them.
