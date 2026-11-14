"""
packages.optimizer.rewriter — Automated SQL Query Optimization and Rewrite Advisor.

Analyzes input SQL queries for structural performance bottlenecks, anti-patterns,
and sub-optimal constructs, and automatically synthesizes a rewritten, optimized SQL
query along with plain-English rationales and DDL index recommendations.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.simplify import simplify


@dataclass
class AntiPattern:
    id: str
    title: str
    severity: str  # "HIGH", "MEDIUM", "INFO"
    description: str
    impact: str


@dataclass
class OptimizationAction:
    rule_name: str
    summary: str
    before_snippet: str
    after_snippet: str
    explanation: str


@dataclass
class IndexRecommendation:
    table: str
    columns: list[str]
    ddl: str
    reason: str


@dataclass
class RewriteResult:
    original_sql: str
    optimized_sql: str
    has_improvements: bool
    anti_patterns: list[AntiPattern]
    optimizations: list[OptimizationAction]
    index_recommendations: list[IndexRecommendation]
    score_before: int
    score_after: int
    summary_heading: str


def _get_conjuncts(expr: exp.Expression | None) -> list[exp.Expression]:
    if expr is None:
        return []
    if isinstance(expr, exp.And):
        return _get_conjuncts(expr.this) + _get_conjuncts(expr.expression)
    return [expr]


class QueryRewriteAdvisor:
    """
    Automated SQL Query Optimizer and Rewrite Advisor.
    Takes an unoptimized SQL query and generates a cleaner, faster rewritten query.
    """

    def analyze_and_rewrite(self, sql_str: str, dialect: str = "postgres") -> RewriteResult:
        original_sql = sql_str.strip()
        try:
            parsed = sqlglot.parse_one(original_sql, read=dialect)
        except Exception:
            # Fallback to generic parsing
            try:
                parsed = sqlglot.parse_one(original_sql)
            except Exception as e:
                return RewriteResult(
                    original_sql=original_sql,
                    optimized_sql=original_sql,
                    has_improvements=False,
                    anti_patterns=[
                        AntiPattern(
                            id="syntax_error",
                            title="SQL Syntax Parse Error",
                            severity="HIGH",
                            description=str(e),
                            impact="Query could not be parsed into relational AST.",
                        )
                    ],
                    optimizations=[],
                    index_recommendations=[],
                    score_before=30,
                    score_after=30,
                    summary_heading="Syntax Error",
                )

        anti_patterns: list[AntiPattern] = []
        optimizations: list[OptimizationAction] = []
        index_recommendations: list[IndexRecommendation] = []

        # Clone AST for rewriting
        optimized_ast = parsed.copy()

        # -------------------------------------------------------------
        # 1. Detect SELECT * (Over-fetching)
        # -------------------------------------------------------------
        has_wildcard = any(isinstance(node, exp.Star) for node in optimized_ast.find_all(exp.Star))
        if has_wildcard:
            anti_patterns.append(
                AntiPattern(
                    id="select_star",
                    title="Unconstrained SELECT * Wildcard",
                    severity="MEDIUM",
                    description="Fetching all columns forces full row deserialization, inflates memory and network I/O, and prevents covering index-only scans.",
                    impact="Prevents index-only scans and increases memory & network transfer overhead.",
                )
            )

        # -------------------------------------------------------------
        # 2. Detect & Simplify Constant Expressions & Tautologies
        # -------------------------------------------------------------
        before_simplify_sql = optimized_ast.sql()
        try:
            simplified = simplify(optimized_ast)
            after_simplify_sql = simplified.sql()
            if before_simplify_sql != after_simplify_sql:
                optimized_ast = simplified
                optimizations.append(
                    OptimizationAction(
                        rule_name="ConstantFoldingAndTautologyPruning",
                        summary="Folded constant expressions and pruned tautological predicates",
                        before_snippet=before_simplify_sql,
                        after_snippet=after_simplify_sql,
                        explanation="Evaluated compile-time arithmetic and eliminated redundant boolean conditions (e.g., 1 = 1) so database engine skips runtime condition checks.",
                    )
                )
        except Exception:
            pass

        # -------------------------------------------------------------
        # 3. Detect & Rewrite Comma Joins (Implicit Cartesian Join)
        # -------------------------------------------------------------
        from_node = optimized_ast.find(exp.From)
        joins = optimized_ast.args.get("joins", [])
        where_node = optimized_ast.args.get("where")

        comma_joins = [
            j for j in joins
            if j.args.get("on") is None and (not j.args.get("kind") or j.args.get("kind") == "")
        ]

        if (from_node and len(from_node.expressions) > 0) or comma_joins:
            anti_patterns.append(
                AntiPattern(
                    id="implicit_comma_join",
                    title="Legacy Comma-Join (Implicit Cross Join)",
                    severity="HIGH",
                    description="Tables are separated by commas in the FROM clause. This relies on WHERE clause filtering and risks unintentional Cartesian products.",
                    impact="Can trigger massive intermediate Cartesian product sizes and confuses query planner join ordering.",
                )
            )

            if from_node and where_node:
                primary_table = from_node.this
                primary_alias = (primary_table.alias_or_name or primary_table.name).lower()
                joined_aliases = {primary_alias}

                conjuncts = _get_conjuncts(where_node.this)
                new_conjuncts = list(conjuncts)
                applied_rewrites = []

                for j in comma_joins:
                    sec_alias = (j.this.alias_or_name or j.this.name).lower()
                    matched_conj = None
                    for c in conjuncts:
                        if isinstance(c, exp.EQ):
                            cols = list(c.find_all(exp.Column))
                            tbls = {col.table.lower() for col in cols if col.table}
                            if sec_alias in tbls and any(t in joined_aliases for t in tbls):
                                matched_conj = c
                                break
                    if matched_conj:
                        j.set("kind", "INNER")
                        j.set("on", matched_conj.copy())
                        if matched_conj in new_conjuncts:
                            new_conjuncts.remove(matched_conj)
                        joined_aliases.add(sec_alias)
                        applied_rewrites.append((sec_alias, matched_conj.sql()))

                if applied_rewrites:
                    if new_conjuncts:
                        cur = new_conjuncts[0]
                        for rem in new_conjuncts[1:]:
                            cur = exp.And(this=cur, expression=rem)
                        optimized_ast.set("where", exp.Where(this=cur))
                    else:
                        optimized_ast.set("where", None)

                    optimizations.append(
                        OptimizationAction(
                            rule_name="AnsiJoinStandardization",
                            summary="Converted legacy comma-joins to explicit ANSI INNER JOINs",
                            before_snippet="FROM ..., ... WHERE condition",
                            after_snippet="INNER JOIN ... ON ...",
                            explanation="Explicit ANSI INNER JOINs allow relational optimizers to evaluate join orders early and avoid accidental Cartesian enumeration.",
                        )
                    )

        # -------------------------------------------------------------
        # 4. Detect Subquery in WHERE IN (...) and Rewrite to EXISTS
        # -------------------------------------------------------------
        in_subqueries = []
        for in_node in optimized_ast.find_all(exp.In):
            query = in_node.args.get("query")
            sub_select = (
                query.this if isinstance(query, exp.Subquery)
                else (query if isinstance(query, exp.Select) else None)
            )
            if sub_select:
                in_subqueries.append((in_node, query, sub_select))

        if in_subqueries:
            anti_patterns.append(
                AntiPattern(
                    id="uncorrelated_in_subquery",
                    title="Subquery inside WHERE IN (...) Filter",
                    severity="HIGH",
                    description="Using WHERE col IN (SELECT ...) often materializes the entire subquery result set or triggers unindexed nested loop probes.",
                    impact="Forces sub-optimal subquery materialization or un-indexed nested loop probes.",
                )
            )

            for in_node, query, sub_select in in_subqueries:
                sub_selects = sub_select.expressions
                sub_from = sub_select.find(exp.From)
                if sub_selects and sub_from:
                    target_col = in_node.this
                    sub_col = sub_selects[0]
                    target_table = sub_from.this
                    target_table_name = target_table.alias_or_name or target_table.name
                    exists_cond = exp.EQ(
                        this=target_col.copy(),
                        expression=exp.Column(this=sub_col.this.copy(), table=target_table_name),
                    )
                    sub_where = sub_select.args.get("where")
                    if sub_where:
                        combined = exp.And(this=sub_where.this.copy(), expression=exists_cond)
                    else:
                        combined = exists_cond
                    sub_select.set("where", exp.Where(this=combined))
                    sub_select.set("expressions", [exp.Literal.number(1)])
                    
                    exists_expr = exp.Exists(
                        this=query if isinstance(query, exp.Subquery) else exp.Subquery(this=sub_select)
                    )
                    before_snippet = in_node.sql()
                    in_node.replace(exists_expr)

                    optimizations.append(
                        OptimizationAction(
                            rule_name="InToExistsConversion",
                            summary="Converted WHERE IN (SELECT ...) to WHERE EXISTS (SELECT 1 ...)",
                            before_snippet=before_snippet,
                            after_snippet=exists_expr.sql(),
                            explanation="EXISTS short-circuits as soon as the first matching row is found, avoiding full intermediate table materialization and hash builds.",
                        )
                    )

        # -------------------------------------------------------------
        # 5. Detect & Rewrite Non-SARGable Predicates
        # -------------------------------------------------------------
        # Rewrite YEAR(col) = YYYY into col >= 'YYYY-01-01' AND col < 'YYYY+1-01-01'
        for eq_node in list(optimized_ast.find_all(exp.EQ)):
            left, right = eq_node.this, eq_node.expression
            func_node = None
            lit_node = None
            if isinstance(left, exp.Year) and isinstance(right, exp.Literal) and right.is_number:
                func_node, lit_node = left, right
            elif isinstance(right, exp.Year) and isinstance(left, exp.Literal) and left.is_number:
                func_node, lit_node = right, left

            if func_node and lit_node:
                try:
                    yr = int(lit_node.this)
                    col = func_node.this
                    start_date = exp.Literal.string(f"{yr}-01-01")
                    end_date = exp.Literal.string(f"{yr+1}-01-01")
                    sargable_cond = exp.And(
                        this=exp.GTE(this=col.copy(), expression=start_date),
                        expression=exp.LT(this=col.copy(), expression=end_date),
                    )
                    before_snippet = eq_node.sql()
                    eq_node.replace(sargable_cond)

                    anti_patterns.append(
                        AntiPattern(
                            id="non_sargable_date_function",
                            title=f"Non-SARGable YEAR() Function Call",
                            severity="HIGH",
                            description=f"Using YEAR({col.sql()}) = {yr} prevents B-Tree index lookups and causes a full table scan.",
                            impact="Invalidates B-Tree index seeks on this column.",
                        )
                    )

                    optimizations.append(
                        OptimizationAction(
                            rule_name="SargableDateRangeRewrite",
                            summary=f"Rewrote YEAR({col.sql()}) = {yr} to bounded date range scan",
                            before_snippet=before_snippet,
                            after_snippet=sargable_cond.sql(),
                            explanation=f"Transformed non-SARGable function to '{col.sql()} >= \"{yr}-01-01\" AND {col.sql()} < \"{yr+1}-01-01\"' allowing direct index seek.",
                        )
                    )
                except Exception:
                    pass

        # Check for other functions wrapping columns in WHERE
        for func in optimized_ast.find_all(exp.Anonymous, exp.Func):
            func_name = (func.key or getattr(func, "name", "")).upper()
            if func_name in ("MONTH", "DAY", "UPPER", "LOWER", "SUBSTRING", "DATE"):
                parent = func.parent
                if parent and isinstance(parent, (exp.EQ, exp.GT, exp.GTE, exp.LT, exp.LTE)):
                    col_arg = next((arg for arg in func.args.values() if isinstance(arg, exp.Column)), None)
                    col_name = col_arg.name if col_arg else "column"
                    anti_patterns.append(
                        AntiPattern(
                            id="non_sargable_function",
                            title=f"Non-SARGable Function Call: {func_name}({col_name})",
                            severity="HIGH",
                            description=f"Wrapping column '{col_name}' with function '{func_name}()' prevents B-Tree index lookups and forces sequential scan.",
                            impact="Index cannot be used for direct seek unless a functional/expression index is created.",
                        )
                    )

        # Detect leading wildcard in LIKE '%xyz'
        for like in optimized_ast.find_all(exp.Like):
            right = like.expression
            if isinstance(right, exp.Literal) and right.is_string:
                val = right.this
                if val.startswith("%"):
                    anti_patterns.append(
                        AntiPattern(
                            id="leading_wildcard_like",
                            title=f"Leading Wildcard in LIKE '{val}'",
                            severity="MEDIUM",
                            description=f"Prefix wildcards like '{val}' cannot utilize standard B-Tree index range scans, triggering an O(N) table scan.",
                            impact="Full scan required for string evaluation. Consider Postgres pg_trgm or Full-Text Search.",
                        )
                    )

        # -------------------------------------------------------------
        # 6. Automatic Index Recommendations
        # -------------------------------------------------------------
        filter_cols_by_table: dict[str, list[str]] = {}
        for binary in optimized_ast.find_all(exp.Binary):
            if isinstance(binary, (exp.EQ, exp.GT, exp.GTE, exp.LT, exp.LTE)):
                for col in binary.find_all(exp.Column):
                    tbl = col.table
                    if not tbl:
                        ancestor_sel = col.find_ancestor(exp.Select)
                        if ancestor_sel:
                            from_expr = ancestor_sel.find(exp.From)
                            if from_expr and hasattr(from_expr, "this") and from_expr.this:
                                tbl = getattr(from_expr.this, "alias_or_name", None) or getattr(from_expr.this, "name", None)
                    tbl = tbl or "target_table"
                    col_name = col.name
                    if tbl not in filter_cols_by_table:
                        filter_cols_by_table[tbl] = []
                    if col_name and col_name not in filter_cols_by_table[tbl]:
                        filter_cols_by_table[tbl].append(col_name)

        for join in optimized_ast.find_all(exp.Join):
            on = join.args.get("on")
            if on:
                for col in on.find_all(exp.Column):
                    tbl = col.table or (join.this.alias_or_name if hasattr(join.this, "alias_or_name") else "joined_table")
                    col_name = col.name
                    if tbl not in filter_cols_by_table:
                        filter_cols_by_table[tbl] = []
                    if col_name and col_name not in filter_cols_by_table[tbl]:
                        filter_cols_by_table[tbl].append(col_name)

        for tbl, cols in filter_cols_by_table.items():
            if cols:
                col_list_str = ", ".join(cols)
                clean_cols = "_".join(c.lower() for c in cols[:2])
                idx_name = f"idx_{tbl.lower()}_{clean_cols}"
                index_recommendations.append(
                    IndexRecommendation(
                        table=tbl,
                        columns=cols,
                        ddl=f"CREATE INDEX {idx_name} ON {tbl} ({col_list_str});",
                        reason=f"Accelerates predicate filtering and equi-join lookups on columns ({col_list_str}).",
                    )
                )

        # Format optimized SQL
        pretty_optimized_sql = optimized_ast.sql(pretty=True)
        has_improvements = (
            pretty_optimized_sql.strip() != original_sql.strip()
            or len(anti_patterns) > 0
            or len(optimizations) > 0
        )

        # Calculate heuristic score (100 is optimal)
        score_before = max(20, 100 - len(anti_patterns) * 20)
        score_after = min(98, score_before + len(optimizations) * 25 + (10 if len(optimizations) > 0 else 0))
        if score_after <= score_before and has_improvements:
            score_after = min(100, score_before + 15)
        elif not has_improvements:
            score_before = 95
            score_after = 95

        summary_heading = (
            f"Query Optimized (+{score_after - score_before}% Estimated Efficiency Gain)"
            if has_improvements and score_after > score_before
            else ("Query Anti-Patterns Detected" if anti_patterns else "Query is Well-Structured")
        )

        return RewriteResult(
            original_sql=original_sql,
            optimized_sql=pretty_optimized_sql,
            has_improvements=has_improvements,
            anti_patterns=anti_patterns,
            optimizations=optimizations,
            index_recommendations=index_recommendations,
            score_before=score_before,
            score_after=score_after,
            summary_heading=summary_heading,
        )
