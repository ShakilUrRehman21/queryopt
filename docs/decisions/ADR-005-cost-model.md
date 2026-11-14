# ADR-005: Parameterized and Documented Cost Model

- **Status**: ACCEPTED
- **Date**: 2026-09-21
- **Deciders**: Systems Architect, Research Engineer

## Context
Cost-based optimizers require a mathematical cost formula combining CPU instructions (tuple comparisons, hash evaluations) and I/O costs (page reads). Hardcoding arbitrary "magic numbers" makes optimizer decisions opaque and untestable.

## Decision
We implement a parameter-driven cost model where every cost factor (page I/O cost, tuple CPU cost, operator startup cost) is exposed in configuration, documented with explicit theoretical justification, and modifiable for ablation studies.
