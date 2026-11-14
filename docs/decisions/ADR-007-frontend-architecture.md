# ADR-007: React + TypeScript + React Flow Frontend

- **Status**: ACCEPTED
- **Date**: 2026-09-21
- **Deciders**: Frontend Lead, Systems Architect

## Context
Researchers and educators need to visually inspect query plan trees, compare alternative plan graphs side-by-side, inspect operator cost distributions, and monitor benchmark metrics.

## Decision
We build the web visualization client using React 18+, TypeScript, Vite, Tailwind CSS, React Flow (for interactive DAG plan visualization), and Recharts (for benchmark time and memory series). Communication with the backend is mediated via FastAPI REST endpoints.
