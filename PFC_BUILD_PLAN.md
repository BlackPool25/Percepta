# Percepta PFC Module — Build Plan

## Overview

Add a Prefrontal Cortex (PFC) module to the existing hippocampal + cerebellar architecture.
The PFC provides working memory, subgoal generation, prediction error detection, and 
task switching — the missing components for true generalization.

## Current Architecture (What Exists)

```
Hippocampus (DG+CA3) → Episodic memory retrieval
Cerebellum (RawFM)   → Physics prediction (efference copy)
Striatum (Dopamine)   → 3-factor plasticity learning
Motor Cortex (Policy) → Action execution with skip connection
```

## Target Architecture (What We're Building)

```
                    ┌─────────────────────────────┐
                    │         PFC MODULE          │
                    │  ┌──────┐  ┌──────────────┐  │
                    │  │ ACC  │  │ DLPFC Working│  │
                    │  │Error │  │ Memory (4    │  │
                    │  │Detect│  │ slots)       │  │
                    │  └──┬───┘  └──────┬───────┘  │
                    │     │             │          │
                    │  ┌──▼─────────────▼───────┐  │
                    │  │  Subgoal Generator     │  │
                    │  │  (Candidate + Simulate)│  │
                    │  └──┬─────────────────────┘  │
                    │     │                       │
                    │  ┌──▼──────────────┐        │
                    │  │ OFC Value Eval  │        │
                    │  └──────┬──────────┘        │
                    └─────────┼────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐   ┌──────────────────┐   ┌──────────────┐
│  Hippocampus  │   │   Cerebellum     │   │   Striatum   │
│  (Episodic    │   │   (RawFM)        │   │  (Dopamine   │
│   Memory)     │   │   Physics Sim    │   │   REINFORCE) │
└───────────────┘   └──────────────────┘   └──────────────┘
```

## Build Order

### Phase 1: ACC — Prediction Error Detector (1-2 days)
**Goal:** Detect when the agent's expectations don't match reality.
**Triggers re-planning when the direct path is blocked.**

Implementation:
- Add `PredictionErrorDetector` class
- After each step, compare RawFM prediction vs actual outcome
- If `||s_pred - s_actual|| > threshold`, set `detour_needed = True`
- Log prediction error for analysis

Files to modify:
- `train_sr.py`: Add ACC class, integrate into main loop

### Phase 2: DLPFC — Working Memory (2-3 days)
**Goal:** Maintain task-relevant information across time.
**Holds current subgoal, task context, recent prediction errors.**

Implementation:
- Add `WorkingMemory` class with 4 slots
- Each slot is a 128-dim vector (matching policy hidden state)
- Gate network (BG analogue) controls slot updates
- Slots maintain: current_subgoal, task_phase, recent_error, plan_step

Files to create/modify:
- `train_sr.py`: Add WorkingMemory class

### Phase 3: Subgoal Candidate Generator (2-3 days)
**Goal:** Generate alternative waypoints when the direct path is blocked.
**Uses RawFM to simulate each candidate and pick the best.**

Implementation:
- When `detour_needed`:
  1. Generate K=10 candidate waypoints in free space
  2. For each candidate, use RawFM to simulate K=5 steps
  3. Score each simulation by cumulative distance-to-goal improvement
  4. Pick the candidate with the best score
  5. Set working memory slot to subgoal mode

Files to create/modify:
- `train_sr.py`: Subgoal generation logic

### Phase 4: OFC — Value Evaluation (1-2 days)
**Goal:** Evaluate the predicted value of each candidate subgoal.
**Combines RawFM simulation with goal-distance metric.**

Implementation:
- For each simulated trajectory, compute:
  - Value = -sum(dist_to_goal over trajectory)
  - Bonus for reaching the goal
  - Penalty for colliding with walls (Δs ≈ 0 when hitting wall)
- Select candidate with highest value

### Phase 5: Full Integration (2-3 days)
**Goal:** All PFC components working together seamlessly.

Implementation:
- PFCModule class combining all subcomponents
- Smooth transition between subgoal and goal-steering modes
- Task switching across curriculum phases
- Logging and analysis of PFC decisions

## How the PFC Solves Phase 3 (Maze Walls)

Current behavior (no PFC):
1. Agent at (0, 2), goal at (3, 3)
2. Hippocampus retrieves "steer toward goal" action
3. Agent hits wall at x=1, velocity → 0
4. Next step: still retrieves "steer toward goal"
5. Agent is stuck against wall forever

With PFC:
1. Agent at (0, 2), goal at (3, 3)
2. Hippocampus retrieves "steer toward goal" action
3. Agent hits wall, velocity → 0
4. **ACC**: `||s_pred - s_actual|| = 0.8 > threshold = 0.3`
   → Signal: DETOUR NEEDED
5. **DLPFC**: Clear goal slot, set mode=SUBGOAL
6. **Subgoal Generator**: Generate 10 candidate waypoints
   - Candidate 1: (1, 1) → RawFM simulates: closer to goal? No, still blocked
   - Candidate 2: (0.5, 2.5) → RawFM simulates: closer? Yes, goes around wall
   - ... (test all candidates)
7. **OFC**: Candidate 2 has highest value (avoids wall, approaches goal)
8. **DLPFC**: Set subgoal = (0.5, 2.5)
9. **Hippocampus**: Retrieve actions for reaching (0.5, 2.5)
10. **Execute**: Go to (0.5, 2.5), then go to (3, 3)
11. **ACC**: Prediction error low → plan working → return to goal mode

## Expected Improvement

| Phase | Before PFC | After PFC (estimated) |
|-------|-----------|----------------------|
| 0: Fixed start | 100% | 100% |
| 1: Random start | 100% | 100% |
| 2: Random goal | 62% | 85% |
| 3: Random maze | 66% | **90%** |

The PFC should raise Phase 3 from 66% to ~90% by enabling the agent to 
detect walls and plan alternative routes around them.

## How to Test

After each phase, run:
```bash
.venv/bin/python3 train_sr.py  # Train with interleaved phases
.venv/bin/python3 test_generalization.py  # Test all 4 phases
```

Phase 3 improvement is the primary metric. Secondary metrics:
- Prediction error values (should spike at walls, drop during normal travel)
- Subgoal generation frequency (should increase in Phase 3)
- Working memory slot usage (should show subgoal/goal switching)
