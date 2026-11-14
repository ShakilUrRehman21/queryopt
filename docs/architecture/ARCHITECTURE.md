# QueryOpt Lab — Architecture Specification

## Overview

QueryOpt Lab is an open-source experimental platform for studying, executing, visualizing, benchmarking, and evaluating relational query optimization strategies.

## System Architecture

The engine is organized into distinct, decoupled packages:

```
                  ┌─────────────────────────────────────┐
                  │              CLI / API              │
                  │   queryopt.cli / apps.api.main      │
                  └──────────────────┬──────────────────┘
                                     │
                  ┌──────────────────▼──────────────────┐
                  │             QueryEngine             │
                  │      Unified pipeline entrypoint    │
                  └─────────┬─────────────────┬─────────┘
                            │                 │
             ┌──────────────▼──────┐   ┌──────▼──────────────┐
             │       Parser        │   │       Storage       │
             │ SQLGlot -> Internal │   │ PageStore, Catalog, │
             │ AST -> Validator    │   │ TableHandle, Stats  │
             └──────────────┬──────┘   └─────────────────────┘
                            │
             ┌──────────────▼──────┐
             │       Planner       │
             │ AST -> Canonical    │
             │ LogicalPlanTree     │
             └──────────────┬──────┘
                            │
             ┌──────────────▼───────────────────────────┐
             │                Optimizer                 │
             │  RuleBased (Rewriting), CostBased, Naive │
             │  Compiles LogicalNode -> PhysicalOperator│
             └──────────────┬───────────────────────────┘
                            │
             ┌──────────────▼───────────────────────────┐
             │                Executor                  │
             │  Volcano Pull Iterator: open/next/close  │
             │  SeqScan, Filter, Project, Joins, Agg... │
             └──────────────────────────────────────────┘
```

## Packages

| Package | Purpose |
|---|---|
| `packages/common` | Domain primitives (`Schema`, `Column`, `DataType`, `Row`, `QueryResult`), error hierarchy, config |
| `packages/storage` | JSON-backed page store, schema catalog, table handles, column statistics |
| `packages/parser` | SQLGlot-based parser, AST node definitions, catalog validation |
| `packages/planner` | Canonical logical plan builder, plan tree printer (`explain`) |
| `packages/query_stats` | Cardinality and selectivity estimation, uniform estimator, equi-width and equi-depth histograms |
| `packages/optimizer` | Optimizer interface, heuristic rewrite rules, Cost-Based Optimizer (DP join reordering), cardinality annotator, analytical cost model, physical planner |
| `packages/executor` | Volcano model execution engine, physical operators (`SeqScan`, `Filter`, `Project`, `NestedLoopJoin`, `HashJoin`, `HashAggregate`, `Sort`, `Limit`) |
| `packages/benchmark` | Synthetic dataset generator (Uniform, Normal, Zipfian distributions), standard research workloads (W1–W7) |
| `packages/experiment` | Matrix experiment runner, ADR-006 YAML config parser, metric extraction (latency, q-error, plan cost), provenance & reproducibility serialization |
| `packages/queryopt` | Engine orchestrator and Click/Rich CLI |

## Design Guarantees

1. **Separation of Concerns**: The planner and executor never import SQLGlot.
2. **Correctness Oracle**: Every optimized query produces identical semantic results to the naive baseline.
3. **Reproducibility**: Parameterized cost models, deterministic seeds, and versioned experiment configs.
