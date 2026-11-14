# ADR-006: YAML Experiment Configurations

- **Status**: ACCEPTED
- **Date**: 2026-09-21
- **Deciders**: DevOps Lead, Research Engineer

## Context
Scientific reproducibility requires that benchmarks and optimization experiments can be executed deterministically from version-controlled configuration files without manual GUI interaction or hidden script parameters.

## Decision
All experiments are defined as declarative YAML specification files validated against strict Pydantic schemas. Every experiment artifact records environment parameters (git commit SHA, CPU model, OS version, timestamps, random seed).
