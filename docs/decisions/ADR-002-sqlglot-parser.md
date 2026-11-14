# ADR-002: SQLGlot as SQL Parser

- **Status**: ACCEPTED
- **Date**: 2026-09-21
- **Deciders**: Systems Architect

## Context
A query optimizer requires an unambiguous abstract syntax tree (AST) representation of SQL statements (SELECT, JOIN, WHERE, GROUP BY, ORDER BY, LIMIT, CREATE TABLE, INSERT). Writing a handwritten SQL grammar or using unmaintained Lex/Yacc wrappers creates significant maintenance overhead.

## Decision
We adopt `SQLGlot` as our SQL tokenizer and parser, wrapped behind an internal QueryOpt Lab AST adapter (`packages/parser/`).

## Rationale
1. **Pure Python & Optional C Acceleration**: Zero complex native compilation dependencies, with optional `sqlglotc` acceleration.
2. **Multi-Dialect Coverage**: Supports standard ANSI SQL and dialect conversions.
3. **Strict Decoupling**: Planner and optimizer packages never import SQLGlot directly; they only interact with QueryOpt Lab's internal `ast_types.py` dataclasses.
