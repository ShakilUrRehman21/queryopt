"""Tests for the product optimizer (static, dialect-aware)."""
from __future__ import annotations

import pytest
import sqlglot
from starlette.testclient import TestClient

from sqlopt import optimize_sql


def titles(r, kind=None):
    return [c.title for c in r.changes if kind is None or c.kind == kind]


def test_untouched_query_is_returned_verbatim():
    q = "SELECT id, name FROM users WHERE id = 7 ORDER BY id LIMIT 1"
    r = optimize_sql(q)
    assert r.ok and not r.changed and r.optimized_sql == q


@pytest.mark.parametrize("q,expect,fragment", [
    ("SELECT a FROM t WHERE 1 = 1 AND b = 2 AND b = 2", "Removed redundant conditions", "b = 2"),
    ("SELECT a FROM t WHERE x = 1 OR x = 2 OR x = 3", "Combined OR conditions into IN", "IN (1, 2, 3)"),
    ("SELECT a FROM t WHERE EXTRACT(YEAR FROM d) = 2024", "Made a date filter index-friendly", "d < '2025-01-01'"),
    ("SELECT g, COUNT(*) FROM t GROUP BY g HAVING g > 3", "Moved a filter from HAVING to WHERE", "WHERE"),
    ("SELECT a FROM t WHERE (SELECT COUNT(*) FROM u WHERE u.k = t.k) > 0", "Replaced COUNT(...) > 0 with EXISTS", "EXISTS"),
    ("SELECT DISTINCT g, COUNT(*) FROM t GROUP BY g", "Removed redundant DISTINCT", "SELECT"),
    ("SELECT * FROM (SELECT a FROM t ORDER BY a) AS s", "Removed ORDER BY from a subquery", "FROM t"),
    ("WITH x AS (SELECT 1), y AS (SELECT 2) SELECT * FROM y", "Removed unused CTE 'x'", "y AS"),
    ("SELECT t.a FROM t, u WHERE t.id = u.tid AND t.a > 1", "Rewrote comma joins as explicit JOIN ... ON", "JOIN u"),
])
def test_safe_rewrites(q, expect, fragment):
    r = optimize_sql(q)
    assert expect in titles(r, "applied")
    assert fragment in r.optimized_sql
    sqlglot.parse_one(r.optimized_sql, read="postgres")  # output is valid SQL


def test_having_with_aggregate_stays():
    r = optimize_sql("SELECT g FROM t GROUP BY g HAVING COUNT(*) > 3")
    assert "HAVING" in r.optimized_sql or not r.changed


@pytest.mark.parametrize("q,expect", [
    ("SELECT * FROM t", "List the columns you need instead of SELECT *"),
    ("SELECT a FROM t WHERE a NOT IN (SELECT b FROM u)", "Use NOT EXISTS instead of NOT IN (subquery)"),
    ("SELECT a FROM t WHERE a = NULL", "Comparison with NULL is never true"),
    ("SELECT a FROM t WHERE a LIKE '%x'", "Leading wildcard in LIKE"),
    ("SELECT a FROM t UNION SELECT a FROM u", "UNION removes duplicates; UNION ALL does not"),
    ("SELECT a FROM t, u", "No join condition for u"),
])
def test_risky_changes_are_only_suggested(q, expect):
    r = optimize_sql(q)
    assert expect in titles(r, "suggestion")
    assert not r.changed


@pytest.mark.parametrize("dialect", ["postgres", "mysql", "sqlite", "tsql", "oracle", "bigquery", "snowflake", "duckdb"])
def test_dialects_round_trip(dialect):
    r = optimize_sql("SELECT a FROM t WHERE 1 = 1 AND b = 2", dialect)
    assert r.ok and r.changed
    sqlglot.parse_one(r.optimized_sql, read=dialect)


def test_errors_are_friendly():
    assert not optimize_sql("").ok
    assert "doesn't look like" in optimize_sql("SELEC x").error
    assert "Could not parse" in optimize_sql("SELECT FROM WHERE (").error


def test_api():
    from apps.api.main import app

    with TestClient(app) as c:
        assert "postgres" in c.get("/api/optimize/meta").json()["dialects"]
        d = c.post("/api/optimize", json={"sql": "SELECT a FROM t WHERE x = 1 OR x = 2", "use_ai": False}).json()
        assert d["ok"] and "IN" in d["optimized_sql"]


# ── structural analysis ─────────────────────────────────────────────────────

SIX = """SELECT c.customer_id,
 (SELECT COUNT(*) FROM orders o1 WHERE o1.customer_id = c.customer_id) AS n_orders,
 (SELECT COUNT(*) FROM orders o2 WHERE o2.customer_id = c.customer_id AND o2.status = 'paid') AS n_paid,
 (SELECT SUM(o3.total) FROM orders o3 WHERE o3.customer_id = c.customer_id AND o3.status = 'paid') AS paid_total,
 (SELECT AVG(o4.total) FROM orders o4 WHERE o4.customer_id = c.customer_id) AS avg_total,
 (SELECT MAX(o5.created_at) FROM orders o5 WHERE o5.customer_id = c.customer_id) AS last_order,
 (SELECT MIN(o6.created_at) FROM orders o6 WHERE o6.customer_id = c.customer_id AND o6.created_at >= '2025-01-01') AS first_2025
FROM customers c WHERE c.active = 1"""


def _merge(r):
    return next(c for c in r.changes if c.title.startswith("Merge "))


def test_six_correlated_subqueries_are_consolidated():
    r = optimize_sql(SIX, "sqlite")
    m = _merge(r)
    assert m.kind == "suggestion" and "6 correlated subqueries" in m.title
    assert m.after.count("FROM orders") == 1 and "GROUP BY" in m.after
    assert not r.changed  # structural rewrites are suggested, never applied silently


def test_consolidated_rewrite_returns_identical_rows():
    """Execute original and suggested rewrite on random data (SQLite) and compare."""
    import random
    import sqlite3

    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE customers (customer_id INT, active INT)")
    db.execute("CREATE TABLE orders (customer_id INT, status TEXT, total REAL, created_at TEXT)")
    rnd = random.Random(1)
    db.executemany("INSERT INTO customers VALUES (?, ?)", [(i, rnd.choice([0, 1])) for i in range(60)])
    db.executemany("INSERT INTO orders VALUES (?, ?, ?, ?)", [
        (rnd.randrange(50), rnd.choice(["paid", "open", None]), rnd.choice([None, round(rnd.random() * 100, 2)]),
         f"202{rnd.randrange(3, 6)}-0{rnd.randrange(1, 9)}-1{rnd.randrange(0, 9)}") for _ in range(400)])
    rewritten = _merge(optimize_sql(SIX, "sqlite")).after
    norm = lambda rows: sorted((tuple(round(v, 6) if isinstance(v, float) else v for v in r) for r in rows), key=repr)  # noqa: E731
    assert norm(db.execute(SIX).fetchall()) == norm(db.execute(rewritten).fetchall())


def test_subqueries_with_joins_or_extra_correlation_are_not_merged():
    q = """SELECT c.id,
      (SELECT SUM(i.qty) FROM orders o JOIN items i ON i.oid = o.id WHERE o.cid = c.id) AS a,
      (SELECT COUNT(*) FROM orders o2 WHERE o2.cid = c.id AND o2.region = c.region) AS b,
      (SELECT MAX(o3.d) FROM orders o3 WHERE o3.cid = c.id) AS m
    FROM customers c"""
    assert not any(c.title.startswith("Merge ") for c in optimize_sql(q).changes)


def test_date_function_with_range_operator_and_implied_bound():
    r = optimize_sql("SELECT a FROM o WHERE YEAR(d) = 2025 AND DATE(d) <= '2025-12-31'", "mysql")
    assert r.optimized_sql.count("'2026-01-01'") == 1
    assert "Removed a condition already implied by another" in titles(r, "applied")
    assert "YEAR" not in r.optimized_sql and "DATE(" not in r.optimized_sql


def test_all_function_filters_are_reported():
    q = "SELECT a FROM t JOIN u ON LOWER(u.k) = t.k WHERE LOWER(t.c) = 'x' AND (SELECT 1 FROM p WHERE UPPER(p.z) = 'Y') = 1"
    f = next(c for c in optimize_sql(q).changes if c.title.startswith("Function applied"))
    assert "(3×)" in f.title


def test_index_plan_is_composite_and_without_prefix_duplicates():
    q = "SELECT c.id, (SELECT COUNT(*) FROM orders o WHERE o.cid = c.id AND o.d >= '2025-01-01') AS n FROM customers c ORDER BY n"
    idx = next(c for c in optimize_sql(q).changes if c.title.startswith("Indexes")).before
    assert "orders (cid, d)" in idx
    assert "orders (cid);" not in idx and "customers (n)" not in idx


# ── schema context ──────────────────────────────────────────────────────────

DDL = """CREATE TABLE customers (customer_id INT PRIMARY KEY, email VARCHAR(100), is_active TINYINT(1));
CREATE TABLE orders (order_id INT PRIMARY KEY, customer_id INT, created_at DATETIME, KEY (customer_id));
CREATE INDEX ix_orders_cd ON orders (customer_id, created_at);"""


def test_schema_skips_existing_primary_and_low_cardinality_indexes():
    q = ("SELECT c.email FROM customers c JOIN orders o ON o.customer_id = c.customer_id "
         "WHERE c.is_active = 1 AND o.created_at >= '2025-01-01'")
    without = next(c for c in optimize_sql(q, "mysql").changes if c.title.startswith(("Index", "Your"))).before
    with_ = next(c for c in optimize_sql(q, "mysql", DDL).changes if c.title.startswith(("Index", "Your"))).before
    assert "CREATE INDEX" in without
    assert "already covered" in with_ and "yes/no column" in with_
    assert "idx_orders_customer_id" not in with_  # covered by ix_orders_cd


def test_schema_expands_select_star():
    r = optimize_sql("SELECT * FROM orders", "mysql", DDL)
    star = next(c for c in r.changes if "SELECT *" in c.title)
    assert star.after == "SELECT order_id, customer_id, created_at"


def test_bad_schema_is_reported():
    assert "schema" in optimize_sql("SELECT 1", "mysql", "CREATE TABLE (((").error.lower()
