"""
sqlopt.ai — Optional LLM pass on top of the rule-based result.

The model sees the already-optimized SQL and may propose one further rewrite.
The proposal is accepted only if it parses in the chosen dialect and differs
from the input; it is always labelled "ai" because nothing is executed, so
equivalence cannot be checked here.
"""
from __future__ import annotations

import sqlglot

from ai.llm import LLMCallError, LLMUnavailableError, extract_json, get_llm_client
from sqlopt.optimizer import DIALECTS, Change, OptimizeResult

SYSTEM = """You are a senior database engineer optimizing SQL for {dialect}.
You get a query that has already had safe mechanical rewrites applied.
Propose a faster version ONLY if you are confident it returns exactly the same rows
for any data (same columns, same multiset of rows, same order if ORDER BY is present).
Typical wins: correlated subqueries -> joins/window functions, repeated subqueries -> one CTE,
avoiding unnecessary DISTINCT/sorts, pushing filters into derived tables.
Do not invent tables or columns. Keep the dialect.
Reply with ONLY JSON:
{{"sql": "<optimized query, or the input unchanged>", "changes": [{{"title": "<short>", "why": "<one sentence>", "impact": "high|medium|low"}}]}}
If there is nothing worthwhile, return the input unchanged with an empty changes list."""


def ai_available() -> bool:
    return get_llm_client().available


def add_ai_pass(result: OptimizeResult) -> OptimizeResult:
    """Mutates and returns `result`. Never raises for model/network problems."""
    if not result.ok:
        return result
    client = get_llm_client()
    if not client.available:
        return result
    try:
        resp = client.complete(SYSTEM.format(dialect=DIALECTS[result.dialect]), result.optimized_sql, max_tokens=1500)
        obj = extract_json(resp.text)
        proposal = str(obj.get("sql", "")).strip().rstrip(";")
        parsed = sqlglot.parse_one(proposal, read=result.dialect)
        new_sql = parsed.sql(dialect=result.dialect, pretty=True)
        old_norm = sqlglot.parse_one(result.optimized_sql, read=result.dialect).sql(dialect=result.dialect)
        if parsed.sql(dialect=result.dialect) == old_norm:
            return result
    except (LLMCallError, LLMUnavailableError, ValueError, sqlglot.errors.ParseError, Exception):
        return result
    for c in obj.get("changes", [])[:5]:
        result.changes.append(Change(
            kind="ai",
            title=str(c.get("title", "AI rewrite"))[:120],
            why=str(c.get("why", ""))[:400] + " Suggested by a language model and not verified by running it; test before use.",
            impact=c.get("impact") if c.get("impact") in ("high", "medium", "low") else "medium",
        ))
    result.ai_sql = new_sql
    return result
