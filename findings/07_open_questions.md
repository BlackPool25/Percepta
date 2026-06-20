# Open Questions (Post-Diagnostics)

Updated from the continuation document's section 8, revised after v1 implementation and diagnostics.

## Critical

| # | Question | Status | What changed |
|---|----------|--------|-------------|
| Q1 | Does the two-stage gate actually reduce forgetting? | **Untested** — gate wired but not experimentally run | Diagnostics consumed the experimental budget. Gate's replay-check mechanism is theoretically distinct from EWC and could still work. |

## Resolved by Diagnostics

| # | Question | Resolution | Evidence |
|---|----------|-----------|----------|
| Q2 | What counts as one "belief"? Cluster granularity. | **Simplification accepted.** v1's class-level clustering works for Split-MNIST. In a real environment this remains open. | N/A |
| Q5 | Is Fisher diagonal a sufficient proxy for weight importance? | **Directionally yes, structurally no.** Fisher ranking is valid (ρ=0.43) but diagonal per-parameter protection cannot prevent distributed-representation collapse at 3.2M params. | Diagnostics 1 + 2 |
| Q6 | Does EWC beat naive at this scale? | **No.** EWC ≈ naive due to distributed-representation collapse that diagonal Fisher cannot prevent. | Freezing experiments |
| D5/D6 | Fisher-mask reuse for belief addressing | **Survives with caveats.** Addressing will be approximate, not precise. Consistent with biology caveat already documented. | Per-layer freeze diagnostic ruled out selection-bias confound |

## New Questions from v1

| # | Question | Priority | Context |
|---|----------|----------|---------|
| N1 | Does the gate's replay-sample check prevent forgetting when the Fisher mask cannot? | **Critical** — this is the real v1 hypothesis now | The gate's actual protection mechanism is the loss-gain/forgetting comparison, not the Fisher mask. Needs experimental validation. |
| N2 | What model scale makes diagonal Fisher protection viable? | Medium | The distributed-collapse finding is scale-dependent. At what parameter count does per-parameter protection become effective? |
| N3 | Should the baseline be replay-based instead of EWC? | Medium | Modern continual learning benchmarks use replay as the standard. A replay baseline + replay-replay comparison would isolate the gate's added value. |
| N4 | Does the layer imbalance (conv1 ρ=0.96 vs fc1 ρ=0.34) suggest a hybrid consolidation strategy? | Medium | Different consolidation mechanisms for different layers: loose shared protection for early layers, precise addressable indexing for late layers. |
