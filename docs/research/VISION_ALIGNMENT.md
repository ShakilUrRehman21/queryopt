# Vision Alignment — v0.2

Gap analysis of the Phase 0 documents (`Documents/`) against the code, and what v0.2 changed.

| Phase 0 commitment | Before v0.2 | v0.2 |
|---|---|---|
| Phase 6 — Adaptive optimizer, feedback collection, W10-style workload, RQ4 testable | Not started | `optimizer/feedback.py`, `optimizer/signature.py`, `optimizer_adaptive.py`; every SELECT records feedback; W9; `rq4_adaptive_feedback.yaml` |
| Gap 4 — cardinality error as a *configurable* variable | Only measured post hoc | `query_stats/noisy.py`, spec syntax `cost_based[qerr=..,mode=..]`; `rq2b_error_injection.yaml` |
| Persona 1 — inject a new cardinality estimator and compare | Estimator hard-wired in CBO | Estimator injectable; self-supervised learned estimator (`query_stats/learned.py`, `learned` optimizer); `rq3_correlated_estimation.yaml` |
| RQ3 — skew/correlation effect on plan selection | No correlated data | `CORRELATED` distribution, W8/W9 |
| RQ5 — LLM recommendations vs empirical results (optional) | Not started | `ai/llm_optimizer.py` with cost-regret metric, cached + provenance; `rq5_llm_join_ordering.yaml` |
| Gap 3 / Persona 4 — optimizer rationale + human-readable explanations | Plan graphs only | `ai/narrator.py` (facts → narrative, optional grounded LLM prose); AI Lab tab; `queryopt ai explain` |
| Reproducibility (ADR-006) | Generator used salted `hash()` → data differed per process | Stable `zlib.crc32` seeding; regression test across `PYTHONHASHSEED` |
| Predicate pushdown across joins (W2) | Whole WHERE pushed or nothing | Conjuncts split and pushed per side |
| Unknown optimizer id | Silently ran rule-based | Error everywhere (engine, API, experiments) |
| Docker / CI (phase0_deployment.md) | Missing | `docker/Dockerfile.api`, `.github/workflows/ci.yml` |

## Still open
- Literature review (`docs/research/literature.md`) before any novelty claim — required by Phase 0 before publication.
- Real-LLM RQ5 data (shipped results use the offline stand-in).
- Larger scales (Phase 7 Option B) and statistical tests across seeds.
- Bushy join trees, index scans, mid-query re-optimization (deliberately out of scope per OQ5).
- Tests write to `./data/default` (pre-existing); consider isolating with `DATA_DIR`.
