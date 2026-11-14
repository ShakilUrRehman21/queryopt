# ADR-001: Python for Database Engine

- **Status**: ACCEPTED
- **Date**: 2026-09-21
- **Deciders**: Systems Architect, Research Engineer

## Context
QueryOpt Lab is an experimental research platform designed to evaluate and visualize query optimization algorithms (rule-based, cost-based, adaptive) under controlled, reproducible conditions. The system is designed for researchers and students, not high-throughput production OLTP or web-scale data warehousing.

## Decision
We implement the core database engine in Python (3.12+ / 3.13) rather than C++, Rust, or Go.

## Rationale
1. **Research Velocity & Extensibility**: Writing new optimization rules or dynamic cost functions takes minutes in Python dataclasses vs hours of compile-heavy metaprogramming in C++.
2. **Ecosystem**: Direct access to scientific Python packages (NumPy, SciPy, Pandas, Matplotlib, Jupyter, Hypothesis).
3. **Transparency**: Engine internals are readable, inspectable, and easy to trace through interactive debugging.

## Consequences & Trade-offs
- **Throughput**: Single-threaded interpreted Python execution is ~10-100x slower than compiled C++/Rust engines.
- **Mitigation**: All research claims and benchmark comparisons are bounded to Python-scale workloads (e.g. 10K-1M rows) with explicit disclaimers that wall-clock execution times are comparative across optimizers, not competitive with DuckDB/PostgreSQL.
