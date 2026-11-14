# ADR-003: Explicit Logical and Physical Plan Separation

- **Status**: ACCEPTED
- **Date**: 2026-09-21
- **Deciders**: Systems Architect

## Context
Query optimizers must explore algebraic equivalence classes (e.g. filter pushdown, join reordering) independently of the physical operator choices (e.g. SeqScan vs IndexScan, NestedLoopJoin vs HashJoin). Mixing logical relational algebra with physical execution details prevents modular plan enumeration.

## Decision
We enforce an explicit two-stage plan representation:
1. `LogicalPlanTree`: Composed of algebraic operators (`LogicalScan`, `LogicalFilter`, `LogicalProject`, `LogicalJoin`, `LogicalAggregate`, `LogicalSort`, `LogicalLimit`).
2. `PhysicalPlanTree`: Composed of Volcano iterator execution operators (`SeqScan`, `Filter`, `Project`, `NestedLoopJoin`, `HashJoin`, `HashAggregate`, `Sort`, `Limit`).

## Rationale
- Allows rule-based optimizers to rewrite relational algebra without committing to physical algorithms.
- Enables cost-based optimizers to enumerate multiple physical implementations for a single logical node.
