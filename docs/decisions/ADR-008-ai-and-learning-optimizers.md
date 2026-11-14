# ADR-008: AI and Learning Optimizers Are Verified, Not Trusted

- **Status**: ACCEPTED (v0.2)
- **Context**: Phase 0 listed RQ4 (runtime feedback) and RQ5 (LLM recommendations vs empirical results). Learned and LLM components can be wrong in ways that are hard to see.
- **Decision**:
  1. Learning/AI optimizers subclass `CostBasedOptimizer` and override only one decision each (estimator, cardinality source, or join order) via hooks (`_make_annotator`, `_join_output_rows`, `_dp_best_join_order`). Everything else stays identical, so comparisons isolate one variable.
  2. LLM output never reaches the executor unvalidated: join orders must be exact permutations and are costed against the DP optimum; rewrites are executed and result multisets compared; NL→SQL must pass the engine's parser/validator.
  3. Every AI feature has a deterministic offline path; tests use a scripted fake client (no network).
  4. LLM completions are cached on disk and the model identity is written to experiment provenance, so RQ5 runs are replayable.
  5. Optimizers are named by parametrised specs (`cost_based[qerr=10,mode=under]`) through `optimizer.registry`; unknown ids are errors, never silent fallbacks.
- **Consequences**: Small per-query overhead (feedback recording, one-off model training). Learned models are per table and cover numeric conjunctions only; other predicates fall back to histograms under independence.
