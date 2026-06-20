# Percepta — Continuation Document

**Source transcript:** `Claude-Adaptive AI model with continuous learning in simulated worlds.md`
**Handoff created:** 6/20/2026

---

## 1. Project Overview

Building an AI agent that learns continuously from experience in a simulated world — without frozen weights, without transformer-scale cost, and without catastrophic forgetting. The core architectural innovations are:

- A **fast/slow weight split** where a cheap fast layer adapts moment-to-moment and a protected slow layer (cortex analogue) accumulates corroborated knowledge
- A **two-stage promotion gate** (cheap frequency/persistent-surprise filter → expensive Fisher-validated integration) deciding what gets consolidated
- A **recall-destabilization mechanism** (modeled on systems reconsolidation in neuroscience) where retrieved consolidated beliefs can be unlocked to integrate contradictory evidence
- An **uncertainty-reduction curiosity reward** (not raw surprise/novelty) to drive exploration
- A **Fisher-importance mask** reused for both consolidation validation and belief addressing

The project sits on open research frontiers — no published system combines all these elements.

---

## 2. Current Direction

The project is currently at the **prototype v1 design stage**. The plan is to build and test the core consolidation mechanism in isolation before adding any other component.

**Immediate next target:** A minimal prototype testing the two-stage promotion gate (stage1: frequency/persistent-surprise filter → stage2: Fisher-validated integration) against two baselines (naive fine-tuning and standard EWC) on Split-MNIST.

**Deferred (not yet building):** curiosity reward, world model, recall-destabilization, fast-weight layer, simulated 3D environment, foundation model.

---

## 3. Final Decisions

| # | Decision | Context |
|---|----------|---------|
| D1 | **Fast/slow core scoped as first build target** | Chosen over trying to build all 4 pillars at once (curiosity, world model, foundation model, memory). Rationale: most foundational, everything else depends on it. |
| D2 | **No foundation model / frozen-weight common sense** | Rejected because: (a) foundation models are the expensive transformers we wanted to avoid; (b) frozen weights at the knowledge level contradict the goal of continuous learning. "Frozen physics" priors (hardcoded rules like "objects fall") are acceptable but must be innate priors, not learned-then-locked weights. |
| D3 | **Promotion trigger = staged pipeline, not weighted blend** | Stage 1 (cheap): frequency OR persistent retrieval-surprise. Stage 2 (expensive): tentative integration + forgetting check via Fisher proxy. Rationale: expensive validation should only run on candidates already promising by cheap measures. Mirrors biological molecular-timer cascade. |
| D4 | **Recall-destabilization = separate mechanism from promotion gate** | Chosen over reusing the same gate. Rationale: cleaner separation of concerns even though it means more to build. |
| D5 | **Fisher mask reused as belief-addressing scheme for destabilization** | Chosen over building explicit modular structure. Rationale: cheap reuse of something already needed for Stage 2 validation. Caveat: distributed representations mean imperfect isolation — belief bleed is expected (consistent with biology). |
| D6 | **Fisher importance harvested as byproduct of Stage 2 validation pass** | Not computed as a separate step. The forward/backward pass for tentative integration already generates the gradients needed. |
| D7 | **Uncertainty-reduction (information gain) reward, NOT raw surprise/novelty** | Raw surprise leads to the noisy-TV problem (agent parks on unpredictable noise). Rewarding reduction in prediction error over time gives the fix — noise can't be reduced so reward dries up. |
| D8 | **Experience replay, NOT Progressive Neural Networks** | Progressive Networks grow unboundedly — violates the efficiency constraint. Experience replay is the defensible choice, grounded in brain-replay research. |
| D9 | **Prototype v1 uses small CNN (no fast-weight layer yet)** | The fast/slow split efficiency question is deferred. V1 tests only: does the gate+masking logic protect old knowledge? |
| D10 | **ROCm on Linux (Ubuntu 24.04)** for GPU compute | 7900 GRE is officially supported by ROCm 7.2.4 (gfx1100 architecture). Windows/WSL2 involves manual workarounds. Note: for v1 prototype (MNIST), GPU barely matters — CPU would suffice. |

---

## 4. Decision History and Why

### 4.1 Project framing
- **Initial idea:** continuous learning agent with curiosity, world model, memory, exploring a simulated world
- **Intervention:** Assistant flagged 3 stacked unsolved problems — catastrophic forgetting, noisy-TV problem in curiosity reward, frozen-vs-learning world model dilemma
- **Resolution:** Scoped down to tackle one mechanism at a time, starting with memory/consolidation

### 4.2 Neuroscience grounding
- Complementary Learning Systems theory (hippocampus fast/sparse → cortex slow/dense) validated as correct neuroscience analogy
- 2025-2026 research added important complications:
  - Consolidation is a multi-stage gated molecular cascade (Nature 2025), not a single "keep/discard" signal
  - Recall destabilizes memories — retrieval makes them editable again, requiring fresh resources (neurogenesis), with precision loss as side effect
  - Consolidation actively reorganizes/generalizes, not just copies
- **Key gap for AI:** No existing system models recall-destabilization or multi-stage gating

### 4.3 Foundation model rejection
- **Considered:** Using a VLM/LLM as a "common sense" core
- **Rejected because:** Foundation models are the expensive transformers we're trying to avoid. Would reintroduce frozen weights at the knowledge level. Contradicts the project's core premise.
- **Alternative:** Hardcoded physics priors (innate, not learned) as the only "frozen" knowledge

### 4.4 Curiosity reward choice
- **Option A — raw surprise/novelty:** Maximize prediction error → noisy-TV failure mode
- **Option B — uncertainty reduction:** Reward = reduction in prediction error over time → noise becomes unrewarding (can't be reduced) — **SELECTED**

### 4.5 Promotion gate architecture
- **Option A — weighted blend:** Single combined score of frequency + surprise + validation. Rejected because expensive validation would run on everything.
- **Option B — staged pipeline:** Cheap pre-filter (frequency OR persistent-surprise) → expensive validation (Fisher-based forgetting check) — **SELECTED**. Matches biology: cheap checkpoints gate access to costly protein synthesis / neurogenesis.

### 4.6 Recall-destabilization mechanism
- **Option A — reuse promotion gate:** Treat contradicted recall as just another candidate for consolidation. Rejected: chosen to be separate mechanism for cleaner separation.
- **Option B — separate mechanism:** Fisher-mask-based selective unlocking, with magnitude-scaled trigger (strong contradiction → immediate unlock, weak → requires recurrence), cooldown/budget to prevent rumination, restabilization = recompute Fisher mask. — **SELECTED**

### 4.7 Belief addressing
- **Option A — explicit modular structure:** Build separate modules/slots per belief. Rejected: excessive complexity.
- **Option B — Fisher mask reuse:** Reuse the importance mask for addressing. — **SELECTED**. Caveat: distributed representations mean imperfect isolation (some bleed into nearby knowledge is expected and has biological precedent).

### 4.8 "Beat LLMs" calibration
- Asked whether the project could beat current LLMs
- **Honest assessment:** At general knowledge/language — no chance (data scale mismatch, foundation model rejected). At continuous online adaptation cheaply — meaningful gap exists, LLMs are genuinely bad at this. Honest framing: open research direction, not a known-solvable engineering task.

---

## 5. Rejected Options

| Option | Why Rejected |
|--------|-------------|
| Foundation model for common sense | Expensive transformer; re-creates frozen weights at knowledge level |
| Progressive Neural Networks | Memory size grows without bound — violates efficiency constraint |
| Weighted-blend promotion trigger | Expensive validation would run on every experience |
| Reuse promotion gate for recall-destabilization | Less clean architecturally |
| Explicit modular belief addressing | Too complex; Fisher mask reuse is cheaper |
| Curiosity = raw surprise/novelty | Noisy-TV problem: agent fixates on unpredictable noise |
| Windows/WSL2 for ROCm | Linux is better-supported for compute workloads |
| Test full 5-mechanism stack at once | Can't isolate failure causes |
| Self-attention for within-input selectivity | Reintroduces transformer cost; not needed at prototype scale (MNIST) |

---

## 6. Prototype Plan (v1 — verbatim)

### 6.1 Hypothesis
Does the two-stage promotion gate (cheap frequency/persistent-surprise filter → Fisher-validated integration) reduce catastrophic forgetting better than (a) naive sequential fine-tuning with no protection, and (b) plain single-mask EWC?

### 6.2 Scope
- **Environment:** Split-MNIST (5 tasks, 2 classes each: 0/1, 2/3, 4/5, 6/7, 8/9), presented sequentially. Standard continual-learning benchmark with known baselines.
- **Model:** One small CNN (2 conv layers + 1 FC head). This is the "slow" layer. No fast-weight adapter yet.
- **No curiosity reward, no world model, no recall-destabilization, no RL loop, no 3D environment.**

### 6.3 Project layout
```
project/
  data.py       # Split-MNIST task stream
  model.py      # slow network (small CNN)
  buffer.py     # episodic buffer + clustering
  gate.py       # stage 1 + stage 2 promotion logic
  baselines.py  # naive fine-tune, standard EWC
  train.py      # experiment harness
  metrics.py    # ACC / forgetting computation
```

### 6.4 Setup (ROCm)
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm<version>
```
Check pytorch.org for the exact current ROCm wheel tag. Verify with `torch.cuda.is_available()` and `torch.cuda.get_device_name(0)` (ROCm PyTorch exposes through `torch.cuda` namespace).

### 6.5 Components

**data.py** — Split-MNIST: 5 sequential tasks, 2 digit classes each.

**model.py** — Small CNN (2 conv + 1 FC). "Slow" layer for v1.

**buffer.py** — Episodic buffer:
```python
class BufferEntry:
    cluster_id: int        # v1 simplification: cluster = task/class id
    inputs: Tensor
    labels: Tensor
    pred_error_history: list[float]   # recomputed each recheck
    seen_count: int
```
Clustering by class/task id is a deliberate simplification to sidestep the "what counts as one belief" question for now.

**gate.py** — Two-stage promotion:

Stage 1 (cheap filter, runs periodically over buffer clusters):
```python
def is_candidate(entry, freq_threshold, persist_window):
    frequent = entry.seen_count >= freq_threshold
    recent_errors = entry.pred_error_history[-persist_window:]
    persistently_surprising = not is_decreasing(recent_errors)
    return frequent or persistently_surprising
```
`is_decreasing` fits a simple linear trend on recent error history; flat-or-rising means the model isn't explaining it away.

Stage 2 (expensive validation):
```python
def validate_and_commit(model, candidate, fisher_masks, eps_gain, eps_forget):
    shadow = copy.deepcopy(model)
    fine_tune(shadow, candidate.inputs, candidate.labels, steps=K, lr=low_lr)

    gain = old_error(model, candidate) - new_error(shadow, candidate)
    forgetting = eval_replay_sample(model) - eval_replay_sample(shadow)

    if gain > eps_gain and forgetting < eps_forget:
        commit(model, shadow)
        fisher_masks[candidate.cluster_id] = compute_fisher_diag(shadow, candidate)
        return True
    return False  # candidate stays in buffer, can retry later
```
- `compute_fisher_diag` reuses gradients from `fine_tune` — no separate computation.
- `eval_replay_sample` checks a small fixed sample drawn across previously-committed clusters.
- EWC-style penalty for future updates: `loss += λ * Σ F_i * (θ_i - θ*_i)²` using merged Fisher masks (running max across clusters per parameter, standard multi-task EWC accumulation).

**baselines.py**:
- **Naive**: plain sequential SGD fine-tuning, no protection.
- **Standard EWC**: one cumulative Fisher penalty over all tasks, applied unconditionally — no gate, no candidacy filter.

**metrics.py**:
- **ACC**: average accuracy across all tasks seen so far, after each task.
- **BWT**: for each task, accuracy now minus accuracy right after it was first learned, averaged. The actual forgetting number.

### 6.6 Run plan
- Same architecture, same task order
- 5 seeds per config, 3 configs (naive / EWC / two-stage gate)
- Plot ACC and BWT curves across the task sequence for all three on one chart

### 6.7 Outcome interpretation

| Outcome | Meaning |
|---------|---------|
| Gate beats EWC on BWT, comparable/better ACC | Staged gating has real value. Green light for next layer. |
| Gate ≈ EWC | Added complexity doesn't earn its cost at this scale. Could be clean task boundaries (need noisier stream) or extra machinery genuinely unnecessary. |
| Gate worse than EWC | Likely tuning issue: thresholds too loose (naive behavior) or too strict (nothing promotes), or replay sample unrepresentative. |
| High variance across seeds | Mechanism unstable at this scale; need more seeds or larger benchmark. |

---

## 7. Constraints and Assumptions

### Hard Constraints
1. **No transformer-scale cost.** Training, inference, and continuous updating must all be cheap. Self-attention is explicitly avoided.
2. **No frozen weights for knowledge that should grow.** "Frozen physics" priors OK only as hardcoded rules (innate priors), not learned-then-locked weights.
3. **No foundation model.** Rejected for cost and philosophical reasons. Agent's "common sense" comes from its own simulated experience, not internet-scale pretraining.
4. **ROCm on Linux (Ubuntu 24.04)** for GPU compute (AMD 7900 GRE, gfx1100 architecture).
5. **Online learning.** The agent updates from its own experience stream, not episodic batch retraining.

### Assumptions (untested)
1. Fisher mask from Step 2's tentative fine-tune pass is a sufficient proxy for weight importance for that candidate (inherits EWC's known limitations — Fisher assumes local quadratic approximation).
2. Clustering by class/task id is a valid simplification that doesn't mask gate mechanism failures.
3. The staged pipeline (cheap filter → expensive validation) will actually reduce compute versus running validation on everything.
4. Distributed representation bleed during destabilization will be tolerable, not catastrophic — the biological precedent (precision loss in nearby memories) is acceptable.
5. Uncertainty-reduction reward won't plateau before the environment is rich enough.

---

## 8. Open Questions

| # | Question | Priority |
|---|----------|----------|
| Q1 | Does the two-stage gate actually beat plain EWC on forgetting? (v1 prototype answer) | **Critical** — gates all future work |
| Q2 | What counts as one "belief" for addressing purposes? Cluster granularity is unresolved — currently simplified to task/class, but this matters for real environments. | High |
| Q3 | How should the recall-destabilization trigger threshold be set? Strong contradiction → immediate? Weak → recurring? What's the boundary? | High |
| Q4 | What's the budget/cooldown for destabilization to prevent rumination? | High |
| Q5 | Is the simulated environment cheap enough to run the volume of steps online RL needs? (economic constraint) | Medium |
| Q6 | What is "growth" measured against? Without a metric, exploration vs drift can't be distinguished. | Medium |
| Q7 | Should the world model itself be updated online or frozen? (recursive-why from early discussion) | Medium |
| Q8 | Does the curiosity reward plateau in a finite environment = success ("agent grows up") or a problem? | Medium |
| Q9 | Should retrieval-destabilization share a cooldown budget with the promotion gate, or have its own? | Low |

---

## 9. Dependencies and Risks

### Dependency chain
```
v1: promotion gate + Fisher consolidation (Split-MNIST, no fast layer)
  → validates: does staged gating reduce forgetting vs EWC?
  ↓ if passes
v2: add fast-weight layer (test: online adaptation speed, same benchmark)
  → validates: can fast/slow split work together
  ↓ if passes
v3: add recall-destabilization (test: concept-drift setting, old beliefs contradicted)
  → validates: Fisher-mask addressing + selective unlock
  ↓ if passes
v4: add curiosity reward + exploration environment (Crafter → later 3D)
  → closes the full loop from the original vision
```

### Key risks
1. **v1 fails** — staged gating provides no benefit over EWC. Mitigation: test on noisier stream before abandoning.
2. **Recall-destabilization collapses** — distributed representations cause belief bleed so severe that "selective" destabilization is meaningless.
3. **Curiosity reward plateaus** — finite environments exhaust novelty; agent stops learning.
4. **Scale mismatch** — the real environment may need to be very rich to sustain open-ended learning; NetHack/Craftax remain unsolved by anyone.
5. **Compute compounds** — each mechanism is individually cheap, but stacking them could break the efficiency promise.

---

## 10. Next Steps

### Immediate
1. **Build v1 prototype** following the plan in Section 6. Implement in this order:
   - `data.py` — Split-MNIST task stream
   - `model.py` — small CNN
   - `buffer.py` — episodic buffer with class-clustered entries
   - `gate.py` — two-stage promotion logic
   - `baselines.py` — naive fine-tune and standard EWC
   - `train.py` — experiment harness running all 3 configs
   - `metrics.py` — ACC and BWT computation
2. **Run** 5 seeds per config, collect ACC/BWT curves.
3. **Evaluate** — which outcome from Section 6.7 was observed?

### If v1 passes
4. Design v2: integrate a fast-weight layer (small adapter/Hebbian weights)
5. Test on Split-MNIST with online adaptation speed metrics

### If v1 is inconclusive
6. Try a noisier benchmark (e.g., permuted MNIST or a stream with concept drift)
7. Adjust threshold tuning (freq_threshold, persist_window, eps_gain, eps_forget)

---

## 11. Verbatim Critical Excerpts

> *"The promotion gate + Fisher consolidation: solid theoretical footing. EWC/SI-style protection is proven to reduce forgetting versus naive fine-tuning. Whether your staged version beats plain EWC is genuinely unknown — that's literally what your v1 prototype tests."*

> *"Recall-destabilization via reused Fisher masks: this is the part with no existing implementation anywhere. It's a genuine bet, not an engineering certainty."*

> *"The actual fork to settle next is what triggers promotion from the fast layer into the slow layer — that's the single decision everything else above depends on."*

> *"The catch: AdA doesn't update weights on the fly at all — adaptation happens in-context, inside a huge attention-based memory, during a single episode. That's exactly the transformer-heavy, expensive route you told me you wanted to avoid."*

> *"Nobody has shown the cheap, continuously-weight-updating, non-catastrophically-forgetting version you're after. You're not late to a finished party — you're poking at something genuinely open."*

> *"Each piece has a foundation. The full assembly has never been built or tested by anyone. 'Theoretically sound in its parts, unproven as a whole' is the accurate sentence — not 'this will work' and not 'this won't work.'"*

> *"The actual tension: each piece above is individually cheap-ish. Stacked — fast/slow core + external memory + staged gating + recall-triggered restabilization — the costs compound, and 'efficient' stops being a property of any one component and becomes a property of how they interact, which is unproven."*

---

## 12. Handoff Notes for Next Agent

- The user is building this from scratch on an **AMD 7900 GRE GPU** (ROCm, Ubuntu 24.04). They indicated they will write the code themselves from the prototype plan.
- The user's strong preference is for **efficiency** — every design choice was made to avoid transformer-scale cost. Do not suggest architectures that reintroduce this cost.
- The user has explicitly **rejected foundation models, frozen weights for deduced knowledge, Progressive Neural Networks, and self-attention**.
- The assistant persona in the conversation was an **anti-pleasing, rigorous ideation partner** — the user values honest calibration over encouragement. Expect this interaction style.
- The communication style in the rest of discussion should be direct, rigorous, and grounded — no excessive validation, no manufactured confidence, name unknowns explicitly.
- Neuroscience findings (multi-stage molecular cascade, recall-destabilization, systems reconsolidation) are the project's primary source of architectural inspiration — keep this grounding visible.
- The project name is **Percepta** (based on the directory name).
