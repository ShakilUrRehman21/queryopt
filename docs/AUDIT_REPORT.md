# QueryOpt SQL Optimizer — Audit, Test & Remediation Report

Date: 2026-11-14 · Scope: the product optimizer (`packages/sqlopt`, `POST /api/optimize`, `apps/web`).
Validation server: **MySQL 8.0.46** (local), plus DuckDB 2.0 as a second engine.

## 1. Modules inspected

| Module | Role | Notes from the audit |
|---|---|---|
| `packages/sqlopt/optimizer.py` | parsing (sqlglot), rule pipeline, applied vs suggested changes, API result | rules mutate a copy of the AST; each rule isolated by try/except; output pretty-printed only when something was applied |
| `packages/sqlopt/structural.py` | date-range rewrites, implied-bound removal, outer-join simplification, IN/EXISTS set cleanup, index planner, correlated-subquery consolidation, repeated-scan, function, join and GROUP BY analysis | the main source of the bugs below |
| `packages/sqlopt/schema.py` | optional DDL parsing: columns, types, PK/UNIQUE, indexes | used for index de-duplication, `SELECT *` expansion, GROUP BY dependency |
| `packages/sqlopt/ai.py` | optional LLM pass, always labelled unverified, never merged into the optimized SQL | unchanged |
| `packages/sqlopt/verify.py` | **new** — semantic-equivalence harness (MySQL or DuckDB) | — |
| `apps/api/routes/optimize.py` | REST endpoint | unchanged apart from schema input |
| `apps/web/src/Optimizer.tsx` | single-page UI | verified by browser tests |
| Database integrations | none in the product path (static analysis, by design); MySQL only in tests/benchmarks | — |
| Research engine (`packages/optimizer`, `executor`, …) | not in the product path | covered by the existing 115 tests, all still passing |

Performance of the platform itself: optimizing a 400-predicate query takes well under a second (browser test); no bottleneck found.

## 2. Scopes tested & 3. test counts

Corpus: `tests/semantic/corpus.py` — **88 MySQL cases**, 60 of which are *must-not-rewrite* cases.

| Scope | Cases | Covers |
|---|---:|---|
| A. Basic | 16 | SELECT *, AND/OR→IN, IN/NOT IN, EXISTS, BETWEEN, LIKE (leading/trailing %), DISTINCT, `= NULL`, `col = col`, collation-dependent constants, constant arithmetic |
| B. Function filters | 20 | YEAR/MONTH/DAY/DATE with =,<,<=,>,>=,BETWEEN, NOT, OR; DATE vs datetime literal; LOWER/UPPER; CAST/CONVERT; COALESCE/IFNULL; implied bounds |
| C. Joins | 12 | comma joins, CROSS, LEFT→INNER (null-rejecting vs IS NULL / COALESCE / OR / ON-clause filters), RIGHT, multi-join, join used only in ON |
| D. Aggregation | 9 | HAVING→WHERE (incl. alias and WITH ROLLUP traps), DISTINCT+GROUP BY, COUNT(*) vs COUNT(col), redundant GROUP BY columns |
| E. Subqueries | 14 | COUNT>0→EXISTS (incl. COUNT(col), COUNT(DISTINCT)), DISTINCT/UNION inside IN and NOT IN, ORDER BY in derived tables with/without LIMIT, unused/recursive CTEs, 6-way correlated consolidation, non-mergeable cases |
| F. Indexes | 5 + unit | composite ordering, existing/PK/low-cardinality detection, prefix overlap |
| G. Sorting & pagination | 6 | ORDER BY with/without LIMIT, large OFFSET, ORDER BY RAND(), sorting by computed column |
| H. Advanced | 6 | UNION vs UNION ALL, window functions, nested CASE, multi-condition EXISTS, derived tables, mixed predicates |
| I. Edge cases | built into the data | NULLs (20%), duplicates, empty tables, 23:59:59 / 00:00:00 boundaries, leap day, mixed-case strings, DECIMAL boundaries |

| Suite | Tests | Result |
|---|---:|---|
| Corpus static expectations (applied / suggested / must-not-rewrite / index content) | 88 | pass |
| ORDER BY / LIMIT preservation on every rewrite | 88 | pass |
| Semantic equivalence on **MySQL 8.0.46** (every rewrite + the consolidation suggestion; 7 data sets each) | 29 | pass |
| Semantic equivalence on DuckDB (same pairs) | 29 | pass |
| `tests/unit/test_sqlopt.py` | 36 | pass |
| Browser tests (`tests/e2e`, Chromium) | 8 | pass |
| Pre-existing research-engine tests | 115 | pass |
| **Total** | **394** | **394 passed, 0 failed** (CI without MySQL/browser skips 37) |

## 4–7. Bugs found, root causes, fixes

All five semantic bugs were **reproduced on real MySQL** (wrong results) before fixing; each has a regression case in the corpus.

| # | Severity | Symptom (reproduced) | Root cause | Fix | Regression |
|---|---|---|---|---|---|
| 1 | **Critical** | `WHERE 'a' = 'A' AND …` lost every row | constant folder evaluated string equality case-sensitively; MySQL's default collation is case-insensitive | only fold conjuncts with no columns, no string literals, no functions | A15 |
| 2 | **Critical** | `col = col` could be dropped as a tautology | same folder; `NULL = NULL` is not TRUE | same restriction | A14 |
| 3 | **Critical** | `COUNT(col) > 0` / `COUNT(DISTINCT x) >= 1` → EXISTS returned extra rows | rule accepted any COUNT; EXISTS ignores NULLs | only `COUNT(*)` / `COUNT(constant)`, no LIMIT | E02, E03, E04 |
| 4 | **Critical** | `HAVING` moved to `WHERE` under `WITH ROLLUP` changed totals | rule ignored ROLLUP | skip ROLLUP/CUBE/GROUPING SETS, select aliases, subqueries | D03, D04 |
| 5 | **Critical** | `DATE(col) = '2025-12-31 10:00:00'` rewritten to a whole-day range | literal truncated to 10 chars | DATE() rewrite only for exact `YYYY-MM-DD` literals | B08 |
| 6 | High | 6-subquery consolidation silently skipped | `COUNT(DISTINCT …)` crashed the rule (swallowed exception) | DISTINCT aggregates supported; subqueries with ORDER/LIMIT/GROUP/DISTINCT excluded | E12, E14 |
| 7 | High | implied-bound removal could pick the wrong string bound | Python string order ≠ collation order | only merge numeric or pure ISO-date bounds | unit |
| 8 | Medium | recommended indexes hurt or did nothing on MySQL (GROUP BY on `status` 2× slower; IN-list and `quantity > 0` indexes unused) | planner added single-column indexes for long IN lists and filter-only indexes on lookup tables | IN lists > 2 values no longer drive indexes; filter-only indexes on joined tables only with equality filters or PK-keyed joins; warning added | F01–F05, benchmark |
| 9 | Medium | impact labels overstated (OR→IN, DISTINCT in IN marked medium/high) | labels were assumptions | recalibrated from MySQL measurements (now low) | — |
| 10 | Low | harness: DuckDB rejected `TINYINT(1)` | display widths | stripped in the DuckDB backend | all DuckDB tests |

New capabilities added while closing gaps found by the corpus: `YEAR(col) [NOT] BETWEEN` and all comparison operators for YEAR/DATE; LEFT JOIN → INNER JOIN when WHERE is null-rejecting (applied, verified); DISTINCT removal and UNION → UNION ALL inside IN / NOT IN / EXISTS (applied, verified); "join used only in its ON clause" and "redundant GROUP BY columns" (suggestions).

## 8. Files modified / added

Modified: `packages/sqlopt/optimizer.py`, `packages/sqlopt/structural.py`, `tests/unit/test_sqlopt.py`, `pyproject.toml` (dev deps: duckdb, pymysql, playwright).
Added: `packages/sqlopt/verify.py`, `tests/semantic/{corpus.py,test_corpus.py}`, `tests/e2e/test_ui.py`, `scripts/mysql_benchmark.py`, `scripts/bench_user_query.sql`, `experiments/results/mysql_perf/{report.md,results.json}`, this report.
Untouched: API contract, UI, research engine.

## 9. Regression tests

Every bug above has a named corpus case; every *applied* rewrite in the corpus is executed on MySQL and DuckDB over 7 generated data sets (empty table, 3 seeds × 8 rows, 3 seeds × 60 rows), so any future rule change that alters results fails the suite.

## 10. Performance measured (real MySQL 8.0.46)

Full tables: `experiments/results/mysql_perf/report.md`. Median of 5 warm runs; results compared row-for-row (all identical).
At 200,000 orders / 400,000 items / 20,000 customers:

| Change | Base indexes | With recommended indexes | Evidence class |
|---|---|---|---|
| `DATE(col) <= …` → range | 23.8 → 21.8 ms | **22.6 → 7.3 ms (3.1×)** | measured, plan uses index |
| `YEAR(col) = 2025` → range | ~same | 24.2 → 20.8 ms | measured |
| `COUNT(*) > 0` → EXISTS | **51.3 → 30.8 ms (1.7×)** | 58.1 → 34.0 ms | measured |
| User's report query (full pipeline) | **214 → 168 ms (1.3×)** | 101 → 91 ms | measured |
| 6 correlated subqueries → one derived table (suggestion) | **751 → 471 ms (1.6×)** | 544 → 475 ms | measured; can lose once indexes make per-row lookups cheap (an earlier 200k run with indexes: 405 → 452 ms) |
| HAVING → WHERE | 104 → 100 ms | ~same | small |
| OR → IN, DISTINCT inside IN, LEFT→INNER | within noise | within noise | syntactic; MySQL 8 already plans these the same |

Run-to-run noise at this scale is roughly ±15–30% on sub-50 ms queries; single comparisons below ~1.3× should not be read as wins. Cross-query interaction observed: `orders (status, created_at)` halves the YEAR query but made `GROUP BY status` ~2.5× slower because MySQL chose it for grouping.

## 11. Remaining limitations

- Static analysis only in the product: the UI cannot show "verified by EXPLAIN" for a user's own database; that needs an opt-in connection (not built, by your earlier choice).
- Index advice has no table statistics (selectivity, row counts) and cannot predict cross-query effects like the GROUP BY regression above.
- Semantic tests prove equivalence on generated data, not for all data; the generator targets the known risk areas (NULLs, duplicates, boundaries, collation, empty tables).
- Not covered: time-zone conversion of TIMESTAMP across sessions, SQL modes other than the 8.0 default, MySQL 5.7, transactions/locking — none of the rewrites depend on them, but they were not executed.
- Order of tied rows under ORDER BY is not compared (legitimately plan-dependent); instead rewrites are asserted never to touch a top-level ORDER BY / LIMIT.
- Consolidation and LEFT→INNER only handle the clear, provably-equivalent shapes; everything else stays a suggestion.

## 12. Recommendations

1. Opt-in, read-only MySQL connection to run `EXPLAIN ANALYZE` on original vs optimized and on hypothetical indexes, and to read real statistics (`information_schema.STATISTICS`, histograms).
2. When a schema is supplied, run `sqlopt.verify` automatically in the API and demote any applied change that fails to a suggestion.
3. Add CI jobs with a MySQL 8.0 and 5.7 service container so the MySQL tests stop being optional.
4. Extend consolidation to subqueries that share a JOIN, using pre-aggregation per join key.

## Readiness

The rewrite engine is **ready for use as an advisory tool**. Every automatic rewrite in the 88-case corpus is proven equivalent on real MySQL and DuckDB. Every known semantic bug is fixed and covered by a regression test. Applied changes and suggestions are strictly separated in the API and UI, and the UI is browser-tested. Index advice and speed claims are evidence-labelled. Index advice is still heuristic, so treat it as a starting point and confirm it with `EXPLAIN ANALYZE`.
