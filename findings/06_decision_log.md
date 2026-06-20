# Decision Log

Chronological record of architectural decisions and rationale.

## D1 — Project Setup

- **Decision:** Use `uv` for environment management, Python 3.12, AMD ROCm wheels
- **Context:** System has ROCm 7.2.4 + AMD 7900 GRE. Official AMD wheels at repo.radeon.com are tested and stable. Nightly PyTorch ROCm builds skipped due to user's stability requirement.
- **Alternatives considered:** Nightly ROCm wheels, CPU-only (spec says CPU suffices for MNIST)

## D2 — Fisher Normalization

- **Decision:** Per-layer max normalization (each layer's max Fisher scaled to 1.0)
- **Context:** Raw Fisher values span 8 orders of magnitude. Per-layer normalization preserves within-layer ranking and cross-layer comparability.
- **Risks identified:** Cross-mask comparability for multi-task EWC accumulation — every mask gets a "1.0" somewhere, potentially over-protecting weakly-important params from high-noise clusters.

## D3 — EWC Implementation

- **Decision:** Compute Fisher on post-training model using task data, store as separate per-task masks/theta_stars
- **Context:** Original approach accumulated gradients during current-task training (measuring what's being learned, not what should be protected). Fixed to compute Fisher on just-trained task's data at convergence.
- **Outcome:** Still produced naive-identical results. Root cause identified as distributed-representation collapse (see finding 04).

## D4 — EWC Comparison Validity

- **Decision:** Document that EWC ≈ naive at this scale due to diagonal Fisher limitations. Comparison is effectively "gate vs naive."
- **Context:** Freezing experiments confirmed distributed collapse — per-parameter protection structurally cannot preserve distributed representations regardless of normalization or λ.
- **Implications for hypothesis:** The v1 hypothesis "gate beats EWC" is technically underdetermined because EWC is structurally incapable of outperforming naive at this model scale. A meaningful EWC comparison requires a smaller model or a different importance mechanism.

## D5 — Fisher-Based Addressing (from spec D5/D6)

- **Decision:** The diagonal Fisher is directionally correct but structurally insufficient for precise per-belief addressing. D5/D6 survive with the caveat that addressability is approximate — belief bleed is inherent, not a tuning issue.
- **Evidence:** Fisher freeze diagnostics show distributed representation collapse even with per-layer stratified protection including fc2 neurons. The diagnostic confound (global vs per-layer selection) was ruled out.
- **Architecture implication:** The gate's forgetting-prevention relies on the replay-sample check (loss comparison), not the Fisher mask. The mask provides weak directional bias.

## D6 — Scope Boundary

- **Decision:** No scope expansion. Explicitly deferred: fast-weight layer, recall-destabilization, curiosity reward, world model, RL, transformers, foundation models.
- **Result confirmed:** All v1 modules are built. Two diagnostics completed. EWC baseline documented. Gate mechanism wired but not yet experimentally validated across 5 seeds.
