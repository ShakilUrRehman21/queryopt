"""
ai.context — Compact, factual descriptions of the database for prompts and UIs.

Everything a model is told about the data comes from the catalog and the
collected statistics, never from guesses, so explanations can be audited.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from storage.catalog import Catalog

SUPPORTED_SQL = """Supported SQL subset (anything else is rejected by the parser):
- SELECT <columns | aggregates> FROM t [JOIN u ON t.a = u.b ...] [WHERE ...]
  [GROUP BY ...] [ORDER BY ... ASC|DESC] [LIMIT n [OFFSET m]]
- Use explicit JOIN ... ON (INNER or LEFT). Comma joins (FROM a, b), RIGHT and FULL JOIN are not supported.
- Predicates: =, !=, <, <=, >, >=, AND, OR, NOT, IS NULL, IS NOT NULL, + - * /.
- Aggregates: COUNT, SUM, AVG, MIN, MAX. Scalar: ABS, LOWER, UPPER, LENGTH.
- NOT supported: subqueries, CTEs (WITH), UNION, IN (...), LIKE, BETWEEN, CASE,
  window functions, HAVING, DISTINCT, column aliases in WHERE.
- Always qualify columns with their table name when more than one table is used."""


def schema_facts(catalog: Catalog, tables: list[str] | None = None) -> list[dict[str, Any]]:
    from storage.statistics import TableStatistics

    out = []
    names = tables or catalog.list_tables()
    for name in names:
        if not catalog.has_table(name):
            continue
        meta = catalog.get_table(name)
        stats = TableStatistics.load(catalog.db_dir, name)
        cols = []
        for c in meta.schema.columns:
            entry: dict[str, Any] = {"name": c.name, "type": c.data_type.name}
            cs = stats.get_column(c.name) if stats else None
            if cs is not None:
                entry.update(
                    distinct=cs.distinct_count,
                    min=cs.min_value,
                    max=cs.max_value,
                    null_fraction=round(cs.null_fraction, 3) if cs.row_count else 0.0,
                )
            if c.primary_key:
                entry["primary_key"] = True
            cols.append(entry)
        out.append({"table": name, "rows": meta.row_count, "columns": cols})
    return out


def schema_prompt(catalog: Catalog, tables: list[str] | None = None) -> str:
    """Human-readable schema block for LLM prompts."""
    lines = []
    for t in schema_facts(catalog, tables):
        lines.append(f"TABLE {t['table']}  ({t['rows']} rows)")
        for c in t["columns"]:
            extra = []
            if "distinct" in c:
                extra.append(f"distinct={c['distinct']}")
            if c.get("min") is not None:
                extra.append(f"range=[{c['min']}, {c['max']}]")
            if c.get("primary_key"):
                extra.append("PK")
            lines.append(f"  - {c['name']} {c['type']}" + (f"  ({', '.join(extra)})" if extra else ""))
    return "\n".join(lines)
