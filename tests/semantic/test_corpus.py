"""
Audit Phases 2–3: every corpus case is checked statically (what was applied vs suggested)
and every rewrite is executed against generated data to prove it returns the same rows.

Backends: DuckDB always; a real MySQL server as well when QUERYOPT_MYSQL_SOCKET or
QUERYOPT_MYSQL_HOST is set (CI without MySQL skips those tests, it does not fake them).
"""
from __future__ import annotations

import os

import pytest
import sqlglot
from sqlglot import exp

from sqlopt import optimize_sql
from sqlopt.verify import DuckDBBackend, check_equivalent
from tests.semantic.corpus import CASES, SCHEMA

IDS = [c["id"] for c in CASES]
HAVE_MYSQL = bool(os.environ.get("QUERYOPT_MYSQL_SOCKET") or os.environ.get("QUERYOPT_MYSQL_HOST"))


def _run(case):
    return optimize_sql(case["sql"], "mysql", SCHEMA)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_static_expectations(case):
    r = _run(case)
    assert r.ok, r.error
    applied = [c.title for c in r.changes if c.kind == "applied"]
    suggested = [c.title for c in r.changes if c.kind == "suggestion"]
    for t in case.get("apply", []):
        assert any(t in a for a in applied), f"{case['id']}: expected applied '{t}', got {applied}"
    for t in case.get("suggest", []):
        assert any(t in s for s in suggested), f"{case['id']}: expected suggestion '{t}', got {suggested}"
    for t in case.get("forbid", []):
        assert not any(t in x for x in applied + suggested), f"{case['id']}: '{t}' must not appear"
    if case.get("keep"):
        assert not r.changed and not applied, f"{case['id']}: must not be rewritten, applied {applied}"
        assert r.optimized_sql == case["sql"].strip()
    if case.get("apply"):
        assert r.changed
    # an applied change must never be listed as a suggestion and vice versa
    assert r.changed == bool(applied)
    idx = next((c.before for c in r.changes if c.title.startswith("Indexes")), "")
    for t in case.get("index_has", []):
        assert t in idx, f"{case['id']}: index advice lacks {t}: {idx}"
    for t in case.get("index_lacks", []):
        assert t not in idx, f"{case['id']}: index advice must not contain {t}: {idx}"


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_top_level_order_by_is_never_changed(case):
    r = _run(case)
    if not r.changed:
        return
    a = sqlglot.parse_one(case["sql"], read="mysql")
    b = sqlglot.parse_one(r.optimized_sql, read="mysql")
    for x, y in ((a, b),):
        oa, ob = x.args.get("order"), y.args.get("order")
        assert (oa.sql() if oa else None) == (ob.sql() if ob else None)
        la, lb = x.args.get("limit"), y.args.get("limit")
        assert (la.sql() if la else None) == (lb.sql() if lb else None)


def _rewritten_pairs():
    pairs = []
    for c in CASES:
        r = optimize_sql(c["sql"], "mysql", SCHEMA)
        if r.changed:
            pairs.append(pytest.param(c["sql"], r.optimized_sql, id=c["id"]))
        if c.get("verify_suggestion"):
            s = next((x for x in r.changes if x.title.startswith(c["verify_suggestion"])), None)
            if s is not None:  # absence is caught by test_static_expectations
                pairs.append(pytest.param(c["sql"], s.after, id=c["id"] + "-suggestion"))
    return pairs


PAIRS = _rewritten_pairs()


@pytest.mark.parametrize("original,optimized", PAIRS)
def test_rewrite_equivalent_duckdb(original, optimized):
    v = check_equivalent(SCHEMA, original, optimized, backend=DUCK)
    assert v.equivalent, v.detail


@pytest.mark.skipif(not HAVE_MYSQL, reason="no MySQL server configured (set QUERYOPT_MYSQL_SOCKET or _HOST)")
@pytest.mark.parametrize("original,optimized", PAIRS)
def test_rewrite_equivalent_mysql(original, optimized, mysql_backend):
    v = check_equivalent(SCHEMA, original, optimized, backend=mysql_backend)
    assert v.equivalent, v.detail


DUCK = DuckDBBackend()


@pytest.fixture(scope="module")
def mysql_backend():
    from sqlopt.verify import MySQLBackend

    be = MySQLBackend("qo_corpus")
    yield be
    be.close()


def test_corpus_covers_every_scope():
    cats = {c["cat"] for c in CASES}
    assert cats >= {"basic", "functions", "joins", "aggregation", "subqueries", "indexes", "sorting", "advanced"}
    assert sum(1 for c in CASES if c.get("keep")) >= 30  # plenty of "must NOT rewrite" cases
    assert len({c["id"] for c in CASES}) == len(CASES)
    assert not [c for c in CASES if isinstance(sqlglot.parse_one(c["sql"], read="mysql"), exp.Command)]
