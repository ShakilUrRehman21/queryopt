"""
sqlopt.structural — Query-level analysis beyond single-pattern rules.

* date_ranges          YEAR()/DATE()/EXTRACT(YEAR) compared with =, <, <=, >, >= -> raw-column ranges (applied)
* merge_ranges         drop bounds already implied by a tighter bound on the same column (applied)
* function_filters     every function-wrapped column in WHERE / JOIN ON / HAVING (suggestion)
* index_plan           composite indexes per query block, prefix-overlap removed (suggestion)
* consolidate_scalars  repeated correlated aggregate subqueries -> one grouped derived table (suggestion,
                       with a full rewritten query)
* repeated_scans       tables read many times across the query (suggestion)

Everything here is static. Nothing is executed, so suggestions are never marked as verified.
"""
from __future__ import annotations

import datetime as _dt
import re
from collections import OrderedDict, defaultdict

from sqlglot import exp

from sqlopt.optimizer import _and, _conjuncts, _Ctx, _in_where

_FLIP = {exp.GT: exp.LT, exp.GTE: exp.LTE, exp.LT: exp.GT, exp.LTE: exp.GTE, exp.EQ: exp.EQ}
_CMP = (exp.EQ, exp.GT, exp.GTE, exp.LT, exp.LTE)


def _unwrap_col(node: exp.Expression | None) -> exp.Column | None:
    while isinstance(node, (exp.TsOrDsToDate, exp.Cast, exp.TryCast)):
        node = node.this
    return node if isinstance(node, exp.Column) else None


def _filter_clauses(tree: exp.Expression) -> list[exp.Expression]:
    out: list[exp.Expression] = [w.this for w in tree.find_all(exp.Where, exp.Having)]
    out += [j.args["on"] for j in tree.find_all(exp.Join) if j.args.get("on") is not None]
    return out


# ── applied: date functions -> ranges ─────────────────────────────────────


def date_ranges(tree: exp.Expression, ctx: _Ctx) -> None:
    for cmp in list(tree.find_all(*_CMP)):
        if not _in_where(cmp):
            continue
        fn, lit, op = cmp.this, cmp.expression, type(cmp)
        if isinstance(fn, exp.Literal):
            fn, lit, op = lit, fn, _FLIP[op]
        if not isinstance(lit, exp.Literal):
            continue
        if isinstance(fn, exp.Extract) and fn.this.name.upper() == "YEAR":
            fn = exp.Year(this=fn.expression)
        if not isinstance(fn, (exp.Year, exp.Date, exp.TsOrDsToDate)):
            continue
        col = _unwrap_col(fn.this)
        if col is None:
            continue
        try:
            if isinstance(fn, exp.Year):
                y = int(lit.this)
                lo, hi = _dt.date(y, 1, 1), _dt.date(y + 1, 1, 1)
            else:
                if not (lit.is_string and re.fullmatch(r"\d{4}-\d{2}-\d{2}", lit.this)):
                    continue  # '2025-12-31 10:00' compared with DATE() has engine-specific semantics
                lo = _dt.date.fromisoformat(lit.this)
                hi = lo + _dt.timedelta(days=1)
            if isinstance(fn, exp.Year) and not (lit.is_int or (lit.is_string and lit.this.isdigit())):
                continue
        except (ValueError, TypeError, OverflowError):
            continue

        def ge(d: _dt.date) -> exp.Expression:
            return exp.GTE(this=col.copy(), expression=exp.Literal.string(d.isoformat()))

        def lt(d: _dt.date) -> exp.Expression:
            return exp.LT(this=col.copy(), expression=exp.Literal.string(d.isoformat()))

        new = {
            exp.EQ: lambda: exp.And(this=ge(lo), expression=lt(hi)),
            exp.GTE: lambda: ge(lo),
            exp.GT: lambda: ge(hi),
            exp.LT: lambda: lt(lo),
            exp.LTE: lambda: lt(hi),
        }[op]()
        before = ctx.sql(cmp)
        cmp.replace(exp.Paren(this=new) if isinstance(new, exp.And) and isinstance(cmp.parent, (exp.Or, exp.Not)) else new)
        ctx.add("applied", "Made a date filter index-friendly",
                f"Wrapping {ctx.sql(col)} in a function hides it from indexes, so every row is checked. "
                "Comparing the raw column against a range returns the same rows and lets an index be used.",
                "high", before, ctx.sql(new))


def year_between(tree: exp.Expression, ctx: _Ctx) -> None:
    """YEAR(col) BETWEEN 2024 AND 2025  ->  col >= '2024-01-01' AND col < '2026-01-01'."""
    for b in list(tree.find_all(exp.Between)):
        fn = b.this
        if isinstance(fn, exp.Extract) and fn.this.name.upper() == "YEAR":
            fn = exp.Year(this=fn.expression)
        if not _in_where(b) or not isinstance(fn, exp.Year):
            continue
        col = _unwrap_col(fn.this)
        lo_l, hi_l = b.args.get("low"), b.args.get("high")
        if col is None or not all(isinstance(x, exp.Literal) and x.is_int for x in (lo_l, hi_l)):
            continue
        try:
            lo, hi = _dt.date(int(lo_l.this), 1, 1), _dt.date(int(hi_l.this) + 1, 1, 1)
        except (ValueError, OverflowError):
            continue
        new = exp.And(this=exp.GTE(this=col.copy(), expression=exp.Literal.string(lo.isoformat())),
                      expression=exp.LT(this=col.copy(), expression=exp.Literal.string(hi.isoformat())))
        before = ctx.sql(b)
        b.replace(exp.Paren(this=new))
        ctx.add("applied", "Made a date filter index-friendly",
                f"YEAR({ctx.sql(col)}) BETWEEN ... hides the column from indexes. The equivalent range on the "
                "raw column returns the same rows and can use an index.", "high", before, ctx.sql(new))


# ── applied: drop implied bounds ──────────────────────────────────────────


def _bound(c: exp.Expression) -> tuple[str, str, object, bool, exp.Column] | None:
    """col OP literal -> (col_sql, 'upper'|'lower', value, strict, col)."""
    if not isinstance(c, (exp.GT, exp.GTE, exp.LT, exp.LTE)):
        return None
    col, lit, op = c.this, c.expression, type(c)
    if isinstance(col, exp.Literal):
        col, lit, op = lit, col, _FLIP[op]
    if not isinstance(col, exp.Column) or not isinstance(lit, exp.Literal):
        return None
    if lit.is_string and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", lit.this):
        return None  # arbitrary strings order by collation, not by Python comparison
    val: object = lit.this if lit.is_string else float(lit.this)
    side = "upper" if op in (exp.LT, exp.LTE) else "lower"
    return col.sql(), side, val, op in (exp.LT, exp.GT), col


def merge_ranges(tree: exp.Expression, ctx: _Ctx) -> None:
    for holder in list(tree.find_all(exp.Where, exp.Having)) + [j for j in tree.find_all(exp.Join) if j.args.get("on")]:
        key = "this" if not isinstance(holder, exp.Join) else "on"
        parts = _conjuncts(holder.args[key])
        best: dict[tuple[str, str, bool], tuple[int, object, bool]] = {}
        for i, c in enumerate(parts):
            b = _bound(c)
            if not b:
                continue
            col, side, val, strict, _ = b
            k = (col, side, isinstance(val, str))
            if k not in best:
                best[k] = (i, val, strict)
                continue
            _, bval, bstrict = best[k]
            tighter = (val < bval or (val == bval and strict and not bstrict)) if side == "upper" else \
                      (val > bval or (val == bval and strict and not bstrict))  # type: ignore[operator]
            if tighter:
                best[k] = (i, val, strict)
        keep_idx = {v[0] for v in best.values()}
        dropped = [p for i, p in enumerate(parts) if _bound(p) and i not in keep_idx]
        if not dropped:
            continue
        kept = [p.copy() for i, p in enumerate(parts) if not _bound(p) or i in keep_idx]
        holder.set(key, _and(kept))
        ctx.add("applied", "Removed a condition already implied by another",
                "A tighter bound on the same column already guarantees this one, so it only adds work.",
                "low", " AND ".join(ctx.sql(d) for d in dropped), "(removed)")


# ── suggestions ───────────────────────────────────────────────────────────

_WRAPPERS = (exp.Lower, exp.Upper, exp.Trim, exp.Cast, exp.TryCast, exp.Substring, exp.Coalesce,
             exp.Year, exp.Month, exp.Day, exp.Date, exp.Abs, exp.Concat, exp.DateTrunc, exp.Anonymous)


def function_filters(tree: exp.Expression, ctx: _Ctx) -> None:
    found: OrderedDict[str, None] = OrderedDict()
    for clause in _filter_clauses(tree):
        for cmp in clause.find_all(*_CMP, exp.NEQ, exp.Like, exp.ILike, exp.In):
            for side in (cmp.this, cmp.args.get("expression")):
                if isinstance(side, _WRAPPERS) and side.find(exp.Column) is not None:
                    found[ctx.sql(cmp)] = None
    if found:
        ctx.add("suggestion", f"Function applied to a filtered column ({len(found)}×)",
                "A function around a column hides it from ordinary indexes, so these conditions are checked "
                "row by row. Store the value already normalized (e.g. lower-case), or add an expression "
                "index on exactly this expression.", "medium", "\n".join(found))


def _block_tables(sel: exp.Select) -> dict[str, str]:
    """alias -> table name for tables that belong to this SELECT (not its subqueries)."""
    out = {}
    for t in sel.find_all(exp.Table):
        if t.find_ancestor(exp.Select) is sel:
            out[(t.alias or t.name).lower()] = t.name
    return out


def index_plan(tree: exp.Expression, ctx: _Ctx) -> None:
    candidates: list[tuple[str, tuple[str, ...]]] = []
    for sel in tree.find_all(exp.Select):
        tabs = _block_tables(sel)
        if not tabs:
            continue

        def own(col: exp.Column) -> str | None:
            if col.table:
                return tabs.get(col.table.lower())
            return next(iter(tabs.values())) if len(tabs) == 1 else None

        eq: dict[str, list[str]] = defaultdict(list)    # column = literal / IN (list)
        rng: dict[str, list[str]] = defaultdict(list)   # column </>/BETWEEN literal
        keys: dict[str, list[str]] = defaultdict(list)  # join or correlation keys
        joined: set[str] = set()                        # tables whose key comes from a join in this block
        aliases = {p.alias.lower() for p in sel.expressions if p.alias}

        def put(bucket: dict[str, list[str]], col: exp.Column) -> None:
            t = own(col)
            if t and col.name not in bucket[t]:
                bucket[t].append(col.name)

        def key_pair(a: exp.Column, b: exp.Column, lookup_side: str | None = None) -> None:
            ta, tb = own(a), own(b)
            for col, t, other in ((a, ta, tb), (b, tb, ta)):
                if t is None:
                    continue
                if lookup_side and (col.table or "").lower() != lookup_side:
                    continue
                put(keys, col)
                if other is not None:
                    joined.add(t)

        where = sel.args.get("where")
        for c in _conjuncts(where.this if where else None):
            a, b = c.this, c.args.get("expression")
            if isinstance(c, exp.In) and isinstance(a, exp.Column) and not c.args.get("query"):
                # a long IN list usually matches a large share of rows (measured: no gain on MySQL 8)
                if len(c.expressions) <= 2:
                    put(eq, a)
            elif isinstance(c, exp.EQ) and isinstance(a, exp.Column) and isinstance(b, exp.Column):
                key_pair(a, b)
            elif isinstance(c, exp.Between) and isinstance(a, exp.Column):
                put(rng, a)
            elif isinstance(c, _CMP):
                cols = [x for x in (a, b) if isinstance(x, exp.Column)]
                if len(cols) == 1:
                    put(eq if isinstance(c, exp.EQ) else rng, cols[0])
        for j in sel.args.get("joins") or []:
            jt = j.this
            if not isinstance(jt, exp.Table) or j.args.get("on") is None:
                continue
            jalias = (jt.alias or jt.name).lower()
            for c in _conjuncts(j.args["on"]):
                a, b = c.this, c.args.get("expression")
                if isinstance(c, exp.EQ) and isinstance(a, exp.Column) and isinstance(b, exp.Column):
                    key_pair(a, b, lookup_side=jalias)
        for t in set(eq) | set(rng) | set(keys):
            filt = eq[t][:3] + [r for r in rng[t] if r not in eq[t]][:1]
            if keys[t]:
                # look up by key, then filter inside the same index
                candidates.append((t, tuple(dict.fromkeys(keys[t][:1] + filt))))
            if filt and (not keys[t] or (t in joined and (eq[t] or (ctx.schema and ctx.schema.unique_prefix(t, tuple(keys[t][:1])))))):
                # the table may also be the one the plan starts from, driven by its own filters
                candidates.append((t, tuple(dict.fromkeys(filt))))
        order = sel.args.get("order")
        if order and sel is tree:
            for o in order.expressions:
                if isinstance(o.this, exp.Column) and o.this.name.lower() not in aliases and own(o.this):
                    candidates.append((own(o.this), (o.this.name,)))  # type: ignore[arg-type]

    # dedupe and drop indexes that are a prefix of another index on the same table
    uniq = list(dict.fromkeys(candidates))
    final = [
        (t, cols) for t, cols in uniq
        if not any(t2 == t and len(c2) > len(cols) and c2[: len(cols)] == cols for t2, c2 in uniq)
    ]
    sch = ctx.schema
    existing_notes: list[str] = []
    if sch:
        keep = []
        for t, cols in final:
            uniq_key = sch.unique_prefix(t, cols)
            if uniq_key:
                existing_notes.append(f"-- {t} ({', '.join(cols)}): not needed, ({', '.join(uniq_key)}) is a primary/unique key")
                continue
            if len(cols) == 1 and sch.low_cardinality(t, cols[0]):
                existing_notes.append(f"-- {t} ({cols[0]}): skipped, a yes/no column rarely benefits from its own index")
                continue
            have = sch.covered(t, cols)
            if have:
                existing_notes.append(f"-- {t} ({', '.join(cols)}): already covered by existing index ({', '.join(have)})")
                continue
            # an existing index that is a prefix of this one can be widened instead of adding another
            prefix = next((ix for ix in sch.indexes.get(t.lower(), []) if cols[: len(ix)] == ix), None)
            if prefix:
                existing_notes.append(f"-- {t}: consider widening existing index ({', '.join(prefix)}) to the one below")
            keep.append((t, cols))
        final = keep
    if not final and not existing_notes:
        return
    final.sort(key=lambda x: (x[0], -len(x[1])))
    ddl = [f"CREATE INDEX idx_{t}_{'_'.join(c)} ON {t} ({', '.join(c)});" for t, c in final[:8]]
    removed = len(uniq) - len(set(final)) - (len(existing_notes) if sch else 0)
    if not final:
        ctx.add("suggestion", "Your existing indexes already cover this query",
                "Every index this query could use already exists in the schema you provided.", "low",
                "\n".join(existing_notes))
        return
    ctx.add("suggestion", "Indexes that would help this query",
            "Equality and join columns come first, then one range column, which is the order an index can "
            "use them in. "
            + (f"{removed} narrower index(es) were dropped because a wider one starts with the same columns. " if removed > 0 else "")
            + "Each index speeds up reads but costs disk space and slows INSERT/UPDATE, and on a column with "
            "few distinct values MySQL may pick an index that is slower than a table scan (measured on a "
            "GROUP BY over `status`). Create them one at a time and compare EXPLAIN ANALYZE. "
            + ("Indexes already in your schema were checked and skipped." if sch else
               "Paste your CREATE TABLE / CREATE INDEX statements to have existing indexes skipped."),
            "high", "\n".join(existing_notes + ddl))


def _strip_alias(node: exp.Expression, alias: str, table: str) -> str:
    n = node.copy()
    for c in n.find_all(exp.Column):
        if c.table and c.table.lower() == alias.lower():
            c.set("table", exp.to_identifier(table))
    return n.sql()


def consolidate_scalars(tree: exp.Expression, ctx: _Ctx) -> None:
    """Correlated aggregate subqueries in a SELECT list that read the same table with the same
    correlation are merged into one grouped derived table using conditional aggregation."""
    for sel in list(tree.find_all(exp.Select)):
        outer = _block_tables(sel)
        groups: dict[tuple, list] = defaultdict(list)
        for proj in sel.expressions:
            sub = proj.unalias()
            if not isinstance(sub, exp.Subquery) or not isinstance(sub.this, exp.Select) or not proj.alias:
                continue
            inner = sub.this
            if any(inner.args.get(k) for k in ("joins", "group", "having", "order", "limit", "offset", "distinct", "with_", "with")) \
                    or len(inner.expressions) != 1:
                continue
            agg = inner.expressions[0].unalias()
            if not isinstance(agg, (exp.Sum, exp.Count, exp.Max, exp.Min, exp.Avg)):
                continue
            frm = inner.args.get("from_") or inner.args.get("from")
            if not frm or not isinstance(frm.this, exp.Table):
                continue
            itab = frm.this
            ialias = (itab.alias or itab.name).lower()
            where = inner.args.get("where")
            corr, filters, ok = None, [], True
            for c in _conjuncts(where.this if where else None):
                outer_cols = [x for x in c.find_all(exp.Column) if x.table and x.table.lower() in outer and x.table.lower() != ialias]
                if outer_cols:
                    if corr is None and isinstance(c, exp.EQ) and isinstance(c.this, exp.Column) and isinstance(c.expression, exp.Column):
                        a, b = c.this, c.expression
                        inner_c, outer_c = (a, b) if (a.table or "").lower() == ialias else (b, a)
                        corr = (inner_c.name, outer_c.sql())
                    else:
                        ok = False
                else:
                    filters.append(c)
            if not ok or corr is None:
                continue
            groups[(itab.name.lower(), corr)].append((proj, agg, filters, ialias, itab.name))

        for (tname, (icol, ocol)), members in groups.items():
            if len(members) < 2:
                continue
            dalias = f"{tname}_agg"
            cols = [icol]
            replacements = []
            for proj, agg, filters, ialias, real in members:
                cond = " AND ".join(_strip_alias(f, ialias, "t") for f in filters)
                distinct = isinstance(agg.this, exp.Distinct)
                target = agg.this.expressions[0] if distinct and agg.this.expressions else agg.this
                if isinstance(agg, exp.Count) and (target is None or isinstance(target, exp.Star)):
                    arg = "1"
                else:
                    arg = _strip_alias(target, ialias, "t")
                fn = type(agg).__name__.upper()
                body = f"CASE WHEN {cond} THEN {arg} END" if cond else arg
                cols.append(f"{fn}({'DISTINCT ' if distinct else ''}{body}) AS {proj.alias}")
                ref = f"{dalias}.{proj.alias}"
                replacements.append((proj, f"COALESCE({ref}, 0) AS {proj.alias}" if fn == "COUNT" else f"{ref} AS {proj.alias}"))
            derived = f"SELECT t.{icol}, " + ", ".join(cols[1:]) + f" FROM {members[0][4]} AS t GROUP BY t.{icol}"

            rewritten = tree.copy()
            target = next(s for s in rewritten.find_all(exp.Select) if s.sql() == sel.sql())
            for proj, new_sql in replacements:
                for p in target.expressions:
                    if p.alias == proj.alias:
                        p.replace(exp.maybe_parse(new_sql, dialect=ctx.dialect))
                        break
            target.join(f"({derived}) AS {dalias}", on=f"{dalias}.{icol} = {ocol}", join_type="left", copy=False,
                        dialect=ctx.dialect)
            names = ", ".join(m[0].alias for m in members)
            ctx.add("suggestion", f"Merge {len(members)} correlated subqueries on {tname} into one pass",
                    f"{names} each re-read {tname} once per output row. One grouped derived table computes all "
                    "of them in a single scan, using CASE inside each aggregate to keep the original filters. "
                    "Same results, including rows with no matches (COUNT becomes 0, others NULL). Best when the "
                    "outer query returns many rows; if it returns only a few, indexed per-row lookups can be "
                    "just as fast. Shown as a suggestion because nothing was run to confirm it on your data.",
                    "high", "\n".join(ctx.sql(m[0]) for m in members)[:600],
                    rewritten.sql(dialect=ctx.dialect, pretty=True))


def repeated_scans(tree: exp.Expression, ctx: _Ctx) -> None:
    counts: dict[str, int] = defaultdict(int)
    for t in tree.find_all(exp.Table):
        counts[t.name.lower()] += 1
    heavy = {t: n for t, n in counts.items() if n >= 3}
    if heavy:
        ctx.add("suggestion", "The same table is read many times",
                "Each mention is a separate read, often once per output row when it sits in a correlated "
                "subquery. Computing the shared part once (a CTE or grouped derived table) usually removes "
                "most of these reads.", "medium",
                ", ".join(f"{t} ×{n}" for t, n in sorted(heavy.items(), key=lambda x: -x[1])))


# ── applied: outer join simplification ────────────────────────────────────

_NULL_REJECTING = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Like, exp.ILike, exp.In, exp.Between)


def _rejects_null_for(c: exp.Expression, alias: str) -> bool:
    """True when conjunct `c` is FALSE/UNKNOWN whenever every column of `alias` is NULL."""
    if isinstance(c, exp.Not) and isinstance(c.this, exp.Is) and isinstance(c.this.this, exp.Column):
        return (c.this.this.table or "").lower() == alias and isinstance(c.this.expression, exp.Null)
    if not isinstance(c, _NULL_REJECTING) or isinstance(c, exp.NullSafeEQ):
        return False
    subject = c.this
    if not (isinstance(subject, exp.Column) and (subject.table or "").lower() == alias):
        subject = c.args.get("expression")
        if not (isinstance(subject, exp.Column) and (subject.table or "").lower() == alias):
            return False
    # the other operands must not mention the alias and must not be NULL-tolerant expressions
    for other in c.iter_expressions():
        if other is subject:
            continue
        if any((x.table or "").lower() == alias for x in other.find_all(exp.Column)):
            return False
        if other.find(exp.Null, exp.Coalesce, exp.If, exp.Case, exp.Subquery) is not None:
            return False
    return True


def left_to_inner(tree: exp.Expression, ctx: _Ctx) -> None:
    for sel in list(tree.find_all(exp.Select)):
        joins = sel.args.get("joins") or []
        if any((j.args.get("side") or "").upper() in ("RIGHT", "FULL") for j in joins):
            continue
        where = sel.args.get("where")
        conds = _conjuncts(where.this if where else None)
        for j in joins:
            if (j.args.get("side") or "").upper() != "LEFT" or not isinstance(j.this, exp.Table):
                continue
            alias = (j.this.alias or j.this.name).lower()
            rejecting = next((c for c in conds if _rejects_null_for(c, alias)), None)
            if rejecting is None:
                continue
            j.set("side", None)
            j.set("kind", None)
            ctx.add("applied", f"Turned LEFT JOIN {j.this.alias_or_name} into an inner JOIN",
                    f"The WHERE condition {ctx.sql(rejecting)} throws away every row where {j.this.alias_or_name} "
                    "had no match, so the LEFT JOIN already behaves like an inner join. Saying so lets the "
                    "optimizer choose a better join order.", "medium", "LEFT JOIN " + ctx.sql(j.this), "JOIN " + ctx.sql(j.this))


# ── applied: set semantics inside IN / EXISTS ─────────────────────────────


def subquery_set_cleanup(tree: exp.Expression, ctx: _Ctx) -> None:
    """IN (...) and EXISTS (...) only ask *whether* a value/row exists, so duplicates inside are
    irrelevant: DISTINCT can go and UNION can become UNION ALL (also under NOT IN / NOT EXISTS)."""
    for holder in list(tree.find_all(exp.In, exp.Exists)):
        q = holder.args.get("query") if isinstance(holder, exp.In) else holder.this
        if isinstance(q, exp.Subquery):
            q = q.this
        if q is None:
            continue
        if isinstance(q, exp.Select) and q.args.get("distinct") and not q.args.get("limit") and not q.args.get("group"):
            q.set("distinct", None)
            ctx.add("applied", "Removed DISTINCT inside an IN/EXISTS subquery",
                    "IN and EXISTS only check whether a match exists, so removing duplicates first is "
                    "wasted work. Same result. MySQL 8 often removes these duplicates itself, so the gain is small.", "low", "SELECT DISTINCT ...", "SELECT ...")
        if isinstance(q, exp.Union) and q.args.get("distinct") and not q.args.get("limit"):
            for u in q.find_all(exp.Union):
                u.set("distinct", False)
            ctx.add("applied", "Used UNION ALL inside an IN/EXISTS subquery",
                    "UNION sorts or hashes every row to drop duplicates, but IN/EXISTS don't care about "
                    "duplicates. UNION ALL gives the same answer with less work.", "medium", "UNION", "UNION ALL")


# ── suggestions: redundant joins and grouping ─────────────────────────────


def redundant_joins(tree: exp.Expression, ctx: _Ctx) -> None:
    for sel in tree.find_all(exp.Select):
        for j in sel.args.get("joins") or []:
            if (j.args.get("side") or "") or j.args.get("on") is None or not isinstance(j.this, exp.Table):
                continue
            alias = (j.this.alias or j.this.name).lower()
            used_elsewhere = any(
                (c.table or "").lower() == alias and not c.find_ancestor(exp.Join) is j
                for c in sel.find_all(exp.Column)
            )
            if not used_elsewhere:
                ctx.add("suggestion", f"Join to {j.this.name} is only used in its ON clause",
                        f"No column of {j.this.alias_or_name} is selected, filtered or sorted on. The join "
                        "still removes rows without a match and repeats rows with several matches, so it is "
                        "not removed automatically. If every row is guaranteed exactly one match (a foreign "
                        "key to a unique key), drop it; if you only need 'has a match', use EXISTS.", "medium",
                        ctx.sql(j))


def redundant_group_by(tree: exp.Expression, ctx: _Ctx) -> None:
    if not ctx.schema:
        return
    for sel in tree.find_all(exp.Select):
        group = sel.args.get("group")
        if not group:
            continue
        tabs = _block_tables(sel)
        cols = [g for g in group.expressions if isinstance(g, exp.Column) and g.table]
        for alias, table in tabs.items():
            mine = [g for g in cols if g.table.lower() == alias]
            keys = [u for u in ctx.schema.unique.get(table.lower(), []) if len(u) == 1]
            key = next((g for g in mine if (g.name.lower(),) in keys), None)
            extra = [g for g in mine if g is not key]
            if key is not None and extra:
                ctx.add("suggestion", "Redundant GROUP BY columns",
                        f"{ctx.sql(key)} is a unique key, so it already determines "
                        f"{', '.join(ctx.sql(e) for e in extra)}. MySQL 5.7+ accepts grouping by the key alone "
                        "(functional dependency), which makes the grouping key smaller. Kept as a suggestion "
                        "because older versions and ONLY_FULL_GROUP_BY setups on other engines may reject it.",
                        "low", ctx.sql(group))
