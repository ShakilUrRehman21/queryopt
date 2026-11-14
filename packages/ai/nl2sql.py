"""
ai.nl2sql — Natural language → SQL with an engine-in-the-loop repair cycle.

The model writes SQL against the *live* catalog. Every candidate is parsed and
semantically validated by QueryOpt Lab's own parser/validator; on failure the
exact engine error is fed back and the model gets another attempt. Only SQL
the engine accepts is ever returned as `valid`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ai.context import SUPPORTED_SQL, schema_prompt
from ai.llm import LLMClient, LLMUnavailableError, extract_json, extract_sql, get_llm_client

if TYPE_CHECKING:
    from queryopt.engine import QueryEngine

SYSTEM = f"""You translate questions into SQL for QueryOpt Lab, a small research database.
{SUPPORTED_SQL}
Reply with ONLY a JSON object: {{"sql": "<one SELECT statement>", "explanation": "<one sentence>"}}"""


def _check(engine: QueryEngine, sql: str) -> str | None:
    from parser.ast_types import SelectStatement
    from parser.parser import parse
    from parser.validator import validate

    try:
        stmt = parse(sql).statement
        if not isinstance(stmt, SelectStatement):
            return "Only SELECT statements are allowed."
        validate(stmt, engine.db.catalog, engine.db_name)
        return None
    except Exception as exc:  # parser/validator errors carry useful messages
        return getattr(exc, "message", None) or str(exc)


def nl_to_sql(
    engine: QueryEngine,
    question: str,
    client: LLMClient | None = None,
    max_attempts: int = 3,
) -> dict[str, Any]:
    client = client or get_llm_client()
    if not client.available:
        raise LLMUnavailableError(
            "Natural-language querying needs a language model. Set ANTHROPIC_API_KEY or "
            "QUERYOPT_LLM_BASE_URL and restart the API."
        )
    schema = schema_prompt(engine.db.catalog)
    attempts: list[dict[str, Any]] = []
    feedback = ""
    for _ in range(max_attempts):
        prompt = f"SCHEMA:\n{schema}\n\nQUESTION: {question.strip()}"
        if feedback:
            prompt += f"\n\nYOUR PREVIOUS ATTEMPTS WERE REJECTED BY THE ENGINE:\n{feedback}\nFix the SQL."
        resp = client.complete(SYSTEM, prompt, max_tokens=600)
        try:
            obj = extract_json(resp.text)
            sql = extract_sql(str(obj.get("sql", "")))
            explanation = str(obj.get("explanation", ""))
        except Exception:
            sql, explanation = extract_sql(resp.text), ""
        error = _check(engine, sql) if sql else "Empty SQL."
        attempts.append({"sql": sql, "error": error, "cached": resp.cached})
        if error is None:
            return {
                "valid": True,
                "sql": sql,
                "explanation": explanation,
                "attempts": attempts,
                "provider": resp.provider,
                "model": resp.model,
            }
        feedback += f"- {sql}\n  ERROR: {error}\n"
    return {
        "valid": False,
        "sql": attempts[-1]["sql"] if attempts else "",
        "explanation": "",
        "attempts": attempts,
        "provider": client.provider,
        "model": client.model,
    }
