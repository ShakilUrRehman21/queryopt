"""
sqlopt.optimizer — Static SQL optimizer for any dialect.

Input: a SQL string and a dialect. Output: an optimized SQL string plus a list
of changes, each with a plain-language reason and an estimated impact.

Two kinds of change:

* **applied**     rewrites that return exactly the same rows on every database
                  (semantics-preserving), so they are made automatically;
* **suggestion**  improvements that depend on data, schema, NULLs or intent, so
                  they are explained but never applied silently.

Nothing is executed. Impact is an estimate based on what the rewrite usually
enables (index use, less sorting, filtering earlier), not a measurement.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import asdict, dataclass, field
from typing import Callable

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError
from sqlglot.optimizer.simplify import simplify

DIALECTS = {
    "postgres": "PostgreSQL",
    "mysql": "MySQL",
    "sqlite": "SQLite",
    "tsql": "SQL Server",
    "oracle": "Oracle",
    "bigquery": "BigQuery",
    "snowflake": "Snowflake",
    "duckdb": "DuckDB",
}


@dataclass
class Change:
    kind: str  # "applied" | "suggestion" | "ai"
    title: str
    why: str
    impact: str  # "high" | "medium" | "low"
    before: str = ""
    after: str = ""


@dataclass
class OptimizeResult:
    ok: bool
    dialect: str
    original_sql: str
    optimized_sql: str
    changed: bool
    changes: list[Change] = field(default_factory=list)
    error: str | None = None
    ai_sql: str | None = None  # separate, unverified LLM proposal (never merged into optimized_sql)

    def to_dict(self) -> dict:
        return asdict(self)


class _Ctx:
    def __init__(self, dialect: str, schema=None) -> None:  # noqa: ANN001
        from sqlopt.schema import Schema

        self.dialect = dialect
        self.changes: list[Change] = []
        self.schema: Schema = schema if schema is not None else Schema()

    def sql(self, node: exp.Expression) -> str:
        return node.sql(dialect=self.dialect)

    def add(self, kind: str, title: str, why: str, impact: str, before: str = "", after: str = "") -> None:
        self.changes.append(Change(kind, title, why, impact, before, after))


# ── helpers ────────────────────────────────────────────────────────────────


def _conjuncts(e: exp.Expression | None) -> list[exp.Expression]:
    if e is None:
        return []
    e = e.unnest()
    if isinstance(e, exp.And):
        return _conjuncts(e.left) + _conjuncts(e.right)
    return [e]


def _and(parts: list[exp.Expression]) -> exp.Expression | None:
    if not parts:
        return None
    def wrap(x: exp.Expression) -> exp.Expression:
        return exp.Paren(this=x) if isinstance(x, exp.Or) else x

    out = wrap(parts[0])
    for p in parts[1:]:
        out = exp.And(this=out, expression=wrap(p))
    return out


def _flatten_conditions(tree: exp.Expression, ctx: _Ctx) -> None:
    """Final tidy-up: rewrite every WHERE/HAVING/ON as a flat AND list (no nested parentheses)."""
    for clause in tree.find_all(exp.Where, exp.Having):
        clause.set("this", _and([c.copy() for c in _conjuncts(clause.this)]))
    for j in tree.find_all(exp.Join):
        if j.args.get("on") is not None:
            j.set("on", _and([c.copy() for c in _conjuncts(j.args["on"])]))


def _in_where(node: exp.Expression) -> bool:
    p = node.parent
    while p is not None and not isinstance(p, exp.Select):
        if isinstance(p, (exp.Where, exp.Join, exp.Having)):
            return True
        p = p.parent
    return False


def _has_limit(sel: exp.Select) -> bool:
    return bool(sel.args.get("limit") or sel.args.get("offset") or sel.args.get("fetch"))


# ── applied (semantics-preserving) rules ───────────────────────────────────


def _simplify_conditions(tree: exp.Expression, ctx: _Ctx) -> None:
    """Drop always-true and duplicate AND-terms and fold constants, keeping the original order."""
    for clause in list(tree.find_all(exp.Where, exp.Having)):
        before = ctx.sql(clause.this)
        kept: list[exp.Expression] = []
        dropped: list[str] = []
        seen: set[str] = set()
        for c in _conjuncts(clause.this):
            # Only conjuncts without columns and without string literals are evaluated:
            # `col = col` is NOT a tautology (NULL rows), and string equality depends on
            # the collation ('a' = 'A' is TRUE in MySQL's default collation).
            if c.find(exp.Column) is None and not any(l.is_string for l in c.find_all(exp.Literal)) \
                    and c.find(exp.Subquery, exp.Select, exp.Anonymous, exp.Func) is None:
                try:
                    folded = simplify(c.copy())
                except Exception:
                    folded = c
                if isinstance(folded, exp.Boolean) and folded.this is True:
                    dropped.append(ctx.sql(c))
                    continue
            key = c.sql()
            if key in seen:
                dropped.append(ctx.sql(c))
                continue
            seen.add(key)
            kept.append(c.copy())
        if [k.sql() for k in kept] == [c.sql() for c in _conjuncts(clause.this)]:
            continue
        if kept:
            clause.set("this", _and(kept))
            after = ctx.sql(clause.this)
        else:
            clause.pop()
            after = "(condition removed)"
        ctx.add("applied", "Removed redundant conditions",
                "These conditions are always true or repeat another one, so they only cost time.",
                "low", " AND ".join(dropped), "(removed)")


def _or_to_in(tree: exp.Expression, ctx: _Ctx) -> None:
    for node in list(tree.find_all(exp.Or)):
        if isinstance(node.parent, exp.Or):
            continue  # handle each OR chain once, from its top
        terms, stack = [], [node]
        while stack:
            n = stack.pop().unnest()
            if isinstance(n, exp.Or):
                stack += [n.right, n.left]
            else:
                terms.append(n)
        cols = {t.this.sql() for t in terms if isinstance(t, exp.EQ) and isinstance(t.this, exp.Column)
                and isinstance(t.expression, exp.Literal)}
        if len(terms) >= 2 and len(cols) == 1 and all(
            isinstance(t, exp.EQ) and isinstance(t.this, exp.Column) and isinstance(t.expression, exp.Literal)
            for t in terms
        ):
            col = terms[0].this.copy()
            new = exp.In(this=col, expressions=[t.expression.copy() for t in terms])
            before = ctx.sql(node)
            target = node.parent if isinstance(node.parent, exp.Paren) else node
            target.replace(new)
            ctx.add("applied", "Combined OR conditions into IN",
                    "Several equality checks on the same column become one IN list. Same rows and easier "
                    "to read; MySQL 8 already plans both forms the same way, so expect little speed change.",
                    "low", before, ctx.sql(new))


def _sargable_dates(tree: exp.Expression, ctx: _Ctx) -> None:
    """YEAR(col) = 2024  ->  col >= '2024-01-01' AND col < '2025-01-01'."""
    for eq in list(tree.find_all(exp.EQ)):
        if not _in_where(eq):
            continue
        fn, lit = eq.this, eq.expression
        if isinstance(fn, exp.Literal):
            fn, lit = lit, fn
        if not isinstance(lit, exp.Literal):
            continue
        if isinstance(fn, exp.Extract) and fn.this.name.upper() == "YEAR":
            fn = exp.Year(this=fn.expression)
        col = fn.this if isinstance(fn, (exp.Year, exp.Date, exp.TsOrDsToDate)) else None
        while isinstance(col, (exp.TsOrDsToDate, exp.Cast, exp.TryCast)):  # MySQL wraps YEAR's argument
            col = col.this
        if not isinstance(col, exp.Column):
            continue
        try:
            if isinstance(fn, exp.Year):
                y = int(lit.this)
                lo, hi = _dt.date(y, 1, 1), _dt.date(y + 1, 1, 1)
            else:
                lo = _dt.date.fromisoformat(str(lit.this)[:10])
                hi = lo + _dt.timedelta(days=1)
        except (ValueError, TypeError):
            continue
        new = exp.and_(
            exp.GTE(this=col.copy(), expression=exp.Literal.string(lo.isoformat())),
            exp.LT(this=col.copy(), expression=exp.Literal.string(hi.isoformat())),
        )
        before = ctx.sql(eq)
        eq.replace(exp.Paren(this=new) if isinstance(eq.parent, (exp.Or, exp.Not)) else new)
        ctx.add("applied", "Made a date filter index-friendly",
                f"Wrapping {ctx.sql(col)} in a function hides it from indexes, so every row is checked. "
                "Comparing the raw column against a range returns the same rows and lets an index be used.",
                "high", before, ctx.sql(new))


def _having_to_where(tree: exp.Expression, ctx: _Ctx) -> None:
    for sel in list(tree.find_all(exp.Select)):
        having = sel.args.get("having")
        group = sel.args.get("group")
        if not having or not group:
            continue
        if group.args.get("rollup") or group.args.get("cube") or group.args.get("grouping_sets") \
                or "ROLLUP" in group.sql().upper():
            continue  # filtering before ROLLUP changes the super-aggregate rows
        aliases = {p.alias.lower() for p in sel.expressions if p.alias}
        grouped = {g.sql() for g in group.expressions}
        keep, move = [], []
        for c in _conjuncts(having.this):
            has_agg = any(True for _ in c.find_all(exp.AggFunc))
            cols = [col.sql() for col in c.find_all(exp.Column)]
            names = {col.name.lower() for col in c.find_all(exp.Column)}
            if not has_agg and cols and all(col in grouped for col in cols) and not (names & aliases) \
                    and c.find(exp.Subquery, exp.Select) is None:
                move.append(c.copy())
            else:
                keep.append(c.copy())
        if not move:
            continue
        before = ctx.sql(having)
        where = sel.args.get("where")
        sel.set("where", exp.Where(this=_and(_conjuncts(where.this if where else None) + move)))
        sel.set("having", exp.Having(this=_and(keep)) if keep else None)
        ctx.add("applied", "Moved a filter from HAVING to WHERE",
                "HAVING runs after grouping; a condition on a grouping column can run before it, "
                "so fewer rows are grouped. Same result.", "medium", before,
                "WHERE " + " AND ".join(ctx.sql(m) for m in move))


def _count_to_exists(tree: exp.Expression, ctx: _Ctx) -> None:
    """(SELECT COUNT(*) FROM ...) > 0  ->  EXISTS (SELECT 1 FROM ...)."""
    for cmp in list(tree.find_all(exp.GT, exp.GTE, exp.NEQ)):
        sub, lit = cmp.this, cmp.expression
        if isinstance(sub, exp.Paren):
            sub = sub.this
        if isinstance(sub, exp.Subquery):
            sub = sub.this
        if not (isinstance(sub, exp.Select) and isinstance(lit, exp.Literal) and not lit.is_string):
            continue
        n = lit.this
        if not ((isinstance(cmp, (exp.GT, exp.NEQ)) and n == "0") or (isinstance(cmp, exp.GTE) and n == "1")):
            continue
        projs = sub.expressions
        if len(projs) != 1 or not isinstance(projs[0], exp.Count) or sub.args.get("group") or sub.args.get("having"):
            continue
        arg = projs[0].this
        # COUNT(col) ignores NULLs and COUNT(DISTINCT ...) is a different question; only
        # COUNT(*) / COUNT(<non-null constant>) is equivalent to "a row exists".
        if not (arg is None or isinstance(arg, exp.Star) or (isinstance(arg, exp.Literal) and not arg.is_string)):
            continue
        if sub.args.get("limit") or sub.args.get("offset"):
            continue
        inner = sub.copy()
        inner.set("expressions", [exp.Literal.number(1)])
        new = exp.Exists(this=inner)
        before = ctx.sql(cmp)
        cmp.replace(new)
        ctx.add("applied", "Replaced COUNT(...) > 0 with EXISTS",
                "Counting every matching row just to check that one exists is wasted work. "
                "EXISTS stops at the first match.", "high", before, ctx.sql(new))


def _distinct_with_group_by(tree: exp.Expression, ctx: _Ctx) -> None:
    for sel in list(tree.find_all(exp.Select)):
        group = sel.args.get("group")
        if not sel.args.get("distinct") or not group:
            continue
        grouped = {g.sql() for g in group.expressions}
        plain = [e.unalias().sql() for e in sel.expressions if not any(True for _ in e.find_all(exp.AggFunc))]
        if grouped and grouped <= set(plain):
            sel.set("distinct", None)
            ctx.add("applied", "Removed redundant DISTINCT",
                    "GROUP BY already makes each group appear once, so DISTINCT only adds an extra "
                    "de-duplication step.", "low", "SELECT DISTINCT ...", "SELECT ...")


def _order_in_subquery(tree: exp.Expression, ctx: _Ctx) -> None:
    for sel in list(tree.find_all(exp.Select)):
        if sel is tree or not sel.args.get("order") or _has_limit(sel):
            continue
        parent = sel.parent
        if isinstance(parent, (exp.Subquery, exp.CTE, exp.In, exp.Exists)) or isinstance(parent, exp.Paren):
            before = ctx.sql(sel.args["order"])
            sel.set("order", None)
            ctx.add("applied", "Removed ORDER BY from a subquery",
                    "Rows from a subquery have no guaranteed order unless it also has LIMIT, so this sort "
                    "costs time without changing the result.", "medium", before, "(removed)")


def _unused_ctes(tree: exp.Expression, ctx: _Ctx) -> None:
    with_ = tree.args.get("with_") or tree.args.get("with")
    if not with_ or with_.args.get("recursive"):
        return
    for cte in list(with_.expressions):
        name = cte.alias.lower()
        if not any(t.name.lower() == name for t in tree.find_all(exp.Table)):
            before = f"{cte.alias} AS (...)"
            cte.pop()
            ctx.add("applied", f"Removed unused CTE '{cte.alias}'",
                    "This WITH block is never referenced, and some databases still compute it.", "medium",
                    before, "(removed)")
    if not with_.expressions:
        tree.set("with_" if "with_" in tree.args else "with", None)


def _comma_joins(tree: exp.Expression, ctx: _Ctx) -> None:
    for sel in list(tree.find_all(exp.Select)):
        joins = sel.args.get("joins") or []
        commas = [j for j in joins if not j.args.get("on") and not j.args.get("using")
                  and not j.args.get("kind") and not j.args.get("side") and isinstance(j.this, exp.Table)]
        frm = sel.args.get("from_") or sel.args.get("from")
        if not commas or not frm:
            continue
        where = sel.args.get("where")
        conds = _conjuncts(where.this if where else None)
        seen = {frm.this.alias_or_name.lower()}
        moved_any = False
        for j in joins:
            name = j.this.alias_or_name.lower()
            if j in commas:
                mine = []
                for c in list(conds):
                    tabs = {col.table.lower() for col in c.find_all(exp.Column) if col.table}
                    if name in tabs and len(tabs) >= 2 and tabs <= seen | {name}:
                        mine.append(c)
                        conds.remove(c)
                if mine:
                    j.set("on", _and([m.copy() for m in mine]))
                    j.set("kind", None)
                    moved_any = True
            seen.add(name)
        if moved_any:
            sel.set("where", exp.Where(this=_and([c.copy() for c in conds])) if conds else None)
            ctx.add("applied", "Rewrote comma joins as explicit JOIN ... ON",
                    "Join conditions now sit next to the tables they connect. Same rows; much harder to "
                    "accidentally drop a condition and create a cross join.", "low")


def _applied_rules() -> list[Callable[[exp.Expression, _Ctx], None]]:
    from sqlopt import structural as st

    return [
        _simplify_conditions, _unused_ctes, _comma_joins, _having_to_where, _count_to_exists,
        st.date_ranges, st.year_between, st.merge_ranges, st.left_to_inner, st.subquery_set_cleanup, _or_to_in, _distinct_with_group_by, _order_in_subquery,
        _flatten_conditions,
    ]


# ── suggestions (never applied automatically) ─────────────────────────────


def _index_advice(tree: exp.Expression, ctx: _Ctx) -> None:
    """Columns that are filtered, joined or sorted on -> candidate single-column indexes."""
    aliases: dict[str, str] = {}
    for t in tree.find_all(exp.Table):
        aliases[(t.alias or t.name).lower()] = t.name
    single_table = len({v.lower() for v in aliases.values()}) == 1
    picks: dict[tuple[str, str], str] = {}

    def table_of(col: exp.Column) -> str | None:
        if col.table:
            return aliases.get(col.table.lower())
        if single_table:
            return next(iter(aliases.values()))
        sel = col.find_ancestor(exp.Select)
        tabs = [t.name for t in sel.find_all(exp.Table) if t.find_ancestor(exp.Select) is sel] if sel else []
        return tabs[0] if len(tabs) == 1 else None

    def add(col: exp.Column, why: str) -> None:
        tab = table_of(col)
        if tab:
            picks.setdefault((tab, col.name), why)

    for cmp in tree.find_all(exp.EQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.In):
        if not _in_where(cmp):
            continue
        cols = [c for c in (cmp.this, cmp.args.get("expression")) if isinstance(c, exp.Column)]
        if isinstance(cmp, exp.In) and isinstance(cmp.this, exp.Column):
            add(cmp.this, "lookup")
        elif len(cols) == 1:
            add(cols[0], "filter")
        elif len(cols) == 2:
            for c in cols:
                add(c, "join")
    for order in tree.find_all(exp.Order):
        for o in order.expressions:
            if isinstance(o.this, exp.Column):
                add(o.this, "sort")
    if not picks:
        return
    ddl = [f"CREATE INDEX idx_{t}_{c} ON {t} ({c});" for (t, c) in list(picks)[:6]]
    ctx.add("suggestion", "Indexes that would help this query",
            "These columns are filtered, joined or sorted on. If they are not indexed yet, the database "
            "has to scan the whole table. Check existing indexes before creating new ones.",
            "high", "\n".join(ddl))


def _suggest(tree: exp.Expression, ctx: _Ctx) -> None:
    from sqlopt import structural as st

    for fn in (st.index_plan, st.consolidate_scalars, st.repeated_scans, st.function_filters,
               st.redundant_joins, st.redundant_group_by):
        try:
            fn(tree, ctx)
        except Exception:
            continue
    merged = any(c.title.startswith("Merge ") for c in ctx.changes)

    for i in tree.find_all(exp.In):
        if i.args.get("query") is not None and not isinstance(i.parent, exp.Not):
            ctx.add("suggestion", "IN (subquery) can often be a JOIN or EXISTS",
                    "Most modern planners already treat this as a semi-join, but older MySQL versions and "
                    "some engines run the subquery repeatedly. EXISTS or a JOIN (with DISTINCT if the "
                    "subquery can repeat keys) is a safe alternative.", "medium", ctx.sql(i)[:200])
            break

    for sel in tree.find_all(exp.Select):
        if not any(isinstance(e, exp.Star) for e in sel.expressions):
            continue
        cols: list[str] = []
        for t in sel.find_all(exp.Table):
            if t.find_ancestor(exp.Select) is sel and t.name.lower() in ctx.schema.columns:
                prefix = f"{t.alias_or_name}." if len(list(sel.find_all(exp.Table))) > 1 else ""
                cols += [prefix + c for c in ctx.schema.columns[t.name.lower()]]
        ctx.add("suggestion", "List the columns you need instead of SELECT *",
                "Reading every column moves more data and prevents index-only scans. Replace * with the "
                "columns the caller actually uses"
                + (" (full list from your schema below; delete the ones you don't need)." if cols else "."),
                "medium", "SELECT *" if cols else "", ("SELECT " + ", ".join(cols)) if cols else "")
        break

    for n in tree.find_all(exp.Not):
        if isinstance(n.this, exp.In) and n.this.args.get("query"):
            ctx.add("suggestion", "Use NOT EXISTS instead of NOT IN (subquery)",
                    "If the subquery can return NULL, NOT IN returns no rows at all, and many planners "
                    "handle NOT EXISTS better. Not applied automatically because it changes results when "
                    "NULLs are present.", "high", ctx.sql(n))

    for cmp in tree.find_all(exp.EQ, exp.NEQ):
        if isinstance(cmp.expression, exp.Null) or isinstance(cmp.this, exp.Null):
            ctx.add("suggestion", "Comparison with NULL is never true",
                    "`= NULL` and `<> NULL` always evaluate to unknown, so this filter matches nothing. "
                    "You probably meant IS NULL / IS NOT NULL.", "high", ctx.sql(cmp))

    for like in tree.find_all(exp.Like, exp.ILike):
        pat = like.expression
        if isinstance(pat, exp.Literal) and pat.is_string and pat.this.startswith("%"):
            ctx.add("suggestion", "Leading wildcard in LIKE",
                    "A pattern starting with % can't use a normal index, so every row is scanned. "
                    "Consider a full-text or trigram index if this is a frequent search.", "medium", ctx.sql(like))

    for u in tree.find_all(exp.Union):
        if u.args.get("distinct"):
            ctx.add("suggestion", "UNION removes duplicates; UNION ALL does not",
                    "If the two sides can't overlap, or duplicates are fine, UNION ALL skips a sort or hash "
                    "step over the whole result.", "medium")
            break

    for sel in tree.find_all(exp.Select):
        for proj in sel.expressions:
            if not merged and (proj.find(exp.Subquery) or isinstance(proj.unalias(), exp.Subquery)):
                ctx.add("suggestion", "Subquery in the SELECT list",
                        "A correlated subquery here can run once per output row. A JOIN to a grouped "
                        "subquery usually computes the same values in one pass.", "high",
                        ctx.sql(proj)[:160])
                break

        joins = sel.args.get("joins") or []
        for j in joins:
            if j.args.get("kind") == "CROSS" or (not j.args.get("on") and not j.args.get("using")
                                                  and not j.args.get("side") and isinstance(j.this, exp.Table)):
                ctx.add("suggestion", f"No join condition for {j.this.alias_or_name}",
                        "Without a condition every row is paired with every row of the other table "
                        "(a cross join). Add the missing ON clause if that wasn't intended.", "high")

        off = sel.args.get("offset")
        if off is not None and isinstance(off.expression, exp.Literal):
            try:
                if int(off.expression.this) >= 1000:
                    ctx.add("suggestion", "Large OFFSET",
                            "The database still reads and discards every skipped row. Keyset pagination "
                            "(WHERE id > last_seen_id ORDER BY id LIMIT n) stays fast on later pages.", "medium")
            except ValueError:
                pass

        order = sel.args.get("order")
        if order and any(isinstance(o.this, (exp.Rand,)) or (isinstance(o.this, exp.Anonymous)
                         and o.this.name.upper() in ("RANDOM", "RAND", "NEWID")) for o in order.expressions):
            ctx.add("suggestion", "ORDER BY random sorts the whole table",
                    "Every row gets a random value and is sorted just to keep a few. Sampling "
                    "(TABLESAMPLE) or picking random ids is far cheaper on large tables.", "medium")

        if order and not sel.args.get("limit") and sel is tree:
            ctx.add("suggestion", "Sorting the full result",
                    "ORDER BY without LIMIT sorts every matching row. If the caller only shows the first "
                    "page, add LIMIT so the database can stop early.", "low")

        if sel.args.get("limit") and not order and sel is tree:
            ctx.add("suggestion", "LIMIT without ORDER BY",
                    "Which rows come back is not defined and can change between runs. Add ORDER BY if the "
                    "choice matters.", "low")


# ── entry point ───────────────────────────────────────────────────────────


def optimize_sql(sql: str, dialect: str = "postgres", schema_ddl: str = "") -> OptimizeResult:
    dialect = dialect if dialect in DIALECTS else "postgres"
    text = (sql or "").strip()
    if not text:
        return OptimizeResult(False, dialect, sql, sql, False, error="Paste a SQL query first.")
    try:
        statements = [s for s in sqlglot.parse(text, read=dialect) if s is not None]
    except ParseError as e:
        msg = str(e).split("\n")[0]
        return OptimizeResult(False, dialect, sql, sql, False, error=f"Could not parse this as {DIALECTS[dialect]}: {msg}")
    if not statements:
        return OptimizeResult(False, dialect, sql, sql, False, error="No SQL statement found.")

    allowed = (exp.Select, exp.Union, exp.Intersect, exp.Except, exp.Insert, exp.Update, exp.Delete,
               exp.Create, exp.Merge)
    if not all(isinstance(s, allowed) for s in statements):
        return OptimizeResult(False, dialect, sql, sql, False,
                              error=f"This doesn't look like a {DIALECTS[dialect]} query. Check the first keyword.")

    from sqlopt.schema import parse_schema

    schema, schema_error = parse_schema(schema_ddl, dialect)
    if schema_error:
        return OptimizeResult(False, dialect, sql, sql, False, error=schema_error)
    ctx = _Ctx(dialect, schema)
    out_sql = []
    for stmt in statements:
        tree = stmt.copy()
        n_before = len(ctx.changes)
        if isinstance(tree, (exp.Select, exp.Union, exp.Insert, exp.Update, exp.Delete, exp.Create)):
            for rule in _applied_rules():
                try:
                    rule(tree, ctx)
                except Exception:  # a rule must never break the whole run
                    continue
            try:
                _suggest(tree, ctx)
            except Exception:
                pass
        applied_here = any(c.kind == "applied" for c in ctx.changes[n_before:])
        out_sql.append(tree.sql(dialect=dialect, pretty=True) if applied_here else stmt.sql(dialect=dialect, pretty=True))

    changed = any(c.kind == "applied" for c in ctx.changes)
    optimized = (";\n\n".join(out_sql) + (";" if len(out_sql) > 1 else "")) if changed else text
    order = {"applied": 0, "suggestion": 1}
    rank = {"high": 0, "medium": 1, "low": 2}
    ctx.changes.sort(key=lambda c: (order.get(c.kind, 2), rank.get(c.impact, 3)))
    return OptimizeResult(True, dialect, sql, optimized, changed, ctx.changes)
