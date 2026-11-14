# ADR-004: Volcano Iterator Execution Model

- **Status**: ACCEPTED
- **Date**: 2026-09-21
- **Deciders**: Systems Architect

## Context
Query execution engines commonly use pull-based (Volcano/Iterator), push-based, or vectorized models. We need a clean, standard execution model that allows operator-level instrumentation of row counts, timing, and memory consumption.

## Decision
We implement the pull-based iterator model (`open()`, `next()`, `close()`) pioneered by Goetz Graefe (Volcano, 1994).

## Rationale
- Composable and straightforward to implement and trace.
- Memory efficient: streams tuples one by one without buffering full intermediate tables (except for pipeline breakers like Sort or HashJoin build side).
- Facilitates per-operator instrumentation (`actual_rows`, `execution_time_ms`, cardinality estimation error ratios).
