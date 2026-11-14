"""
ai.rewrite_verifier — "Trust, but verify" for AI-suggested query rewrites.

A rewrite suggestion — from an LLM, from the heuristic QueryRewriteAdvisor, or
typed by a person — is only accepted after the engine has

1. parsed and validated it,
2. executed original and candidate on the same data and optimizer,
3. compared the result multisets (order-insensitive, floats rounded), and
4. timed both over several repetitions (median).

Verdicts: ``faster`` | ``equivalent`` (same rows, no measurable gain) |
``slower`` | ``not_equivalent`` (rejected) | ``invalid`` (doesn't run).

Equivalence here is *empirical on this database instance* — evidence, not a
proof. The response says so explicitly so nobody mistakes testing for
verification.
"""
from __future__ import annotations

import statistics
from collections import Counter
from typing import TYPE_CHECKING, Any

from ai.context import SUPPORTED_SQL, schema_prompt
from ai.llm import LLMClient, LLMUnavailableError, extract_json, extract_sql, get_llm_client

if TYPE_CHECKING:
    from queryopt.engine import QueryEngine

REWRITE_SYSTEM = f"""You rewrite SQL queries to run faster on QueryOpt Lab while returning exactly
the same multiset of rows. The engine is a row-at-a-time (Volcano) executor with hash joins,
predicate/projection pushdown and a cost-based join orderer; it has NO indexes.
Good rewrites: remove redundant or tautological predicates, turn comma joins into explicit
equi-joins, add implied predicates (transitivity) that let filters apply earlier, drop
unused columns, avoid non-equality join conditions.
{SUPPORTED_SQL}
Reply with ONLY a JSON object: {{"sql": "<rewritten SELECT>", "rationale": "<one or two sentences>"}}
If no beneficial rewrite exists, return the original SQL and say so in the rationale."""

_SPEEDUP_THRESHOLD = 1.10


def _normalize(rows: list[tuple]) -> Counter:
    def norm(v: Any) -> Any:
        return round(v, 6) if isinstance(v, float) else v

    return Counter(tuple(norm(v) for v in r) for r in rows)


def propose_rewrite(engine: QueryEngine, sql: str, client: LLMClient | None = None) -> dict[str, Any]:
    client = client or get_llm_client()
    if not client.available:
        raise LLMUnavailableError("No language model configured.")
    prompt = f"SCHEMA:\n{schema_prompt(engine.db.catalog)}\n\nQUERY:\n{sql}"
    resp = client.complete(REWRITE_SYSTEM, prompt, max_tokens=700)
    obj = extract_json(resp.text)
    return {
        "sql": extract_sql(str(obj.get("sql", ""))),
        "rationale": str(obj.get("rationale", "")),
        "source": "llm",
        **resp.provenance(),
    }


def heuristic_rewrite(sql: str) -> dict[str, Any]:
    from optimizer.rewriter import QueryRewriteAdvisor

    r = QueryRewriteAdvisor().analyze_and_rewrite(sql)
    return {
        "sql": r.optimized_sql,
        "rationale": "; ".join(o.summary for o in r.optimizations) or "No heuristic rewrite applies.",
        "source": "heuristic",
    }


def verify_rewrite(
    engine: QueryEngine,
    original_sql: str,
    candidate_sql: str,
    optimizer_id: str = "cost_based",
    repetitions: int = 5,
) -> dict[str, Any]:
    from common.types import ExecutionStatus

    def run(sql: str) -> tuple[Any, list[float]]:
        times: list[float] = []
        res = None
        for _ in range(max(1, repetitions)):
            res = engine.execute(sql, optimizer_id=optimizer_id)
            if res.status != ExecutionStatus.SUCCESS:
                return res, []
            times.append(res.execution_time_ms + res.planning_time_ms)
        return res, times

    orig, t_orig = run(original_sql)
    if not t_orig:
        return {"verdict": "invalid", "reason": f"Original query failed: {orig.error_message}"}
    cand, t_cand = run(candidate_sql)
    if not t_cand:
        return {"verdict": "invalid", "reason": f"Candidate failed to run: {cand.error_message}"}

    same = _normalize(orig.rows) == _normalize(cand.rows)
    m_orig, m_cand = statistics.median(t_orig), statistics.median(t_cand)
    speedup = m_orig / m_cand if m_cand > 0 else 1.0
    if not same:
        verdict = "not_equivalent"
    elif speedup >= _SPEEDUP_THRESHOLD:
        verdict = "faster"
    elif speedup <= 1 / _SPEEDUP_THRESHOLD:
        verdict = "slower"
    else:
        verdict = "equivalent"
    return {
        "verdict": verdict,
        "results_match": same,
        "original_rows": len(orig.rows),
        "candidate_rows": len(cand.rows),
        "original_median_ms": round(m_orig, 3),
        "candidate_median_ms": round(m_cand, 3),
        "speedup": round(speedup, 3),
        "repetitions": len(t_orig),
        "optimizer_id": optimizer_id,
        "caveat": (
            "Equivalence was tested by comparing result multisets on the current data, "
            "not proven for all possible databases."
        ),
    }
