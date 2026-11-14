"""
sqlopt.verify — Execute original and optimized SQL on the same synthetic data and compare.

Used by the test-suite (every automatic rewrite must survive it) and available to
the API when a schema is supplied. Data is generated to provoke semantic bugs:

* NULLs in every nullable column (default 20%)
* heavy duplication (small value domains)
* timestamps on and around year/day boundaries (23:59:59, 00:00:00)
* mixed-case strings (MySQL's default collation is case-insensitive)
* empty tables (size 0) as well as small and medium tables

Backends: a real MySQL server when QUERYOPT_MYSQL_SOCKET or QUERYOPT_MYSQL_HOST is
set (pymysql), otherwise DuckDB with the SQL transpiled by sqlglot. Results are
compared as multisets. Row order is not compared: rows that tie on the ORDER BY
key may legitimately come back in a different order under a different plan, so
the static tests instead assert that rewrites never touch a top-level ORDER BY.
"""
from __future__ import annotations

import datetime as dt
import decimal
import os
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Protocol

import sqlglot
from sqlglot import exp

from sqlopt.schema import Schema, parse_schema

STRINGS = ["a", "A", "b", "B", "india", "India", "INDIA", "paid", "Paid", "shipped", "cancelled",
           "electronics", "Electronics", "x%y", "", " a"]
BOUNDARY_TIMES = [
    dt.datetime(2024, 12, 31, 23, 59, 59), dt.datetime(2025, 1, 1, 0, 0, 0), dt.datetime(2025, 1, 1, 0, 0, 1),
    dt.datetime(2025, 6, 15, 12, 0, 0), dt.datetime(2025, 12, 31, 0, 0, 0), dt.datetime(2025, 12, 31, 23, 59, 59),
    dt.datetime(2026, 1, 1, 0, 0, 0), dt.datetime(2024, 2, 29, 10, 30, 0), dt.datetime(2023, 3, 1, 8, 0, 0),
]


class Backend(Protocol):
    name: str

    def reset(self, ddl_statements: list[str]) -> None: ...
    def insert(self, table: str, cols: list[str], rows: list[tuple]) -> None: ...
    def query(self, sql: str) -> list[tuple]: ...


# ── backends ───────────────────────────────────────────────────────────────


class MySQLBackend:
    name = "mysql"

    def __init__(self, database: str = "qo_verify") -> None:
        import pymysql

        kw: dict[str, Any] = {"user": os.environ.get("QUERYOPT_MYSQL_USER", "root"),
                              "password": os.environ.get("QUERYOPT_MYSQL_PASSWORD", ""), "autocommit": True}
        if os.environ.get("QUERYOPT_MYSQL_SOCKET"):
            kw["unix_socket"] = os.environ["QUERYOPT_MYSQL_SOCKET"]
        else:
            kw["host"] = os.environ.get("QUERYOPT_MYSQL_HOST", "127.0.0.1")
            kw["port"] = int(os.environ.get("QUERYOPT_MYSQL_PORT", "3306"))
        self.conn = pymysql.connect(**kw)
        self.db = database
        with self.conn.cursor() as c:
            c.execute(f"DROP DATABASE IF EXISTS {database}")
            c.execute(f"CREATE DATABASE {database}")
            c.execute(f"USE {database}")

    def version(self) -> str:
        with self.conn.cursor() as c:
            c.execute("SELECT VERSION()")
            return c.fetchone()[0]

    def reset(self, ddl_statements: list[str]) -> None:
        with self.conn.cursor() as c:
            c.execute("SET FOREIGN_KEY_CHECKS = 0")
            c.execute("SHOW TABLES")
            for (t,) in c.fetchall():
                c.execute(f"DROP TABLE `{t}`")
            for s in ddl_statements:
                c.execute(s)

    def insert(self, table: str, cols: list[str], rows: list[tuple]) -> None:
        if not rows:
            return
        ph = ", ".join(["%s"] * len(cols))
        with self.conn.cursor() as c:
            c.executemany(f"INSERT INTO `{table}` ({', '.join(cols)}) VALUES ({ph})", rows)

    def query(self, sql: str) -> list[tuple]:
        with self.conn.cursor() as c:
            c.execute(sql)
            return list(c.fetchall())

    def close(self) -> None:
        self.conn.close()


class DuckDBBackend:
    name = "duckdb"

    def __init__(self, read_dialect: str = "mysql") -> None:
        import duckdb

        self.con = duckdb.connect()
        self.read = read_dialect

    def _t(self, sql: str) -> str:
        return sqlglot.transpile(sql, read=self.read, write="duckdb")[0]

    def reset(self, ddl_statements: list[str]) -> None:
        for (t,) in self.con.execute("SELECT table_name FROM information_schema.tables").fetchall():
            self.con.execute(f'DROP TABLE IF EXISTS "{t}"')
        for s in ddl_statements:
            tree = sqlglot.parse_one(s, read=self.read)
            if isinstance(tree, exp.Create) and isinstance(tree.this, exp.Schema):
                # keep only column definitions: DuckDB has no inline KEY/INDEX syntax
                tree.this.set("expressions", [e for e in tree.this.expressions if isinstance(e, exp.ColumnDef)])
                for cd in tree.this.expressions:
                    kind = cd.args.get("kind")
                    if kind is not None and kind.this not in (exp.DataType.Type.DECIMAL, exp.DataType.Type.VARCHAR,
                                                               exp.DataType.Type.CHAR):
                        kind.set("expressions", [])  # INT(11), TINYINT(1): display widths only
                    cd.set("constraints", [c for c in cd.args.get("constraints") or []
                                           if isinstance(c.args.get("kind"), exp.NotNullColumnConstraint)])
                self.con.execute(tree.sql(dialect="duckdb"))

    def insert(self, table: str, cols: list[str], rows: list[tuple]) -> None:
        if rows:
            ph = ", ".join(["?"] * len(cols))
            self.con.executemany(f'INSERT INTO "{table}" ({", ".join(cols)}) VALUES ({ph})', rows)

    def query(self, sql: str) -> list[tuple]:
        return self.con.execute(self._t(sql)).fetchall()


def default_backend() -> Backend:
    if os.environ.get("QUERYOPT_MYSQL_SOCKET") or os.environ.get("QUERYOPT_MYSQL_HOST"):
        return MySQLBackend()
    return DuckDBBackend()


# ── data ──────────────────────────────────────────────────────────────────


def _value(typ: str, col: str, rng: random.Random, i: int, pk: bool, null_rate: float) -> Any:
    if pk:
        return i + 1
    if rng.random() < null_rate:
        return None
    t = typ.upper()
    if t.startswith(("TINYINT(1)", "BOOL", "BIT")):
        return rng.choice([0, 1])
    if t.startswith(("INT", "BIGINT", "SMALLINT", "TINYINT", "MEDIUMINT")):
        return rng.randint(1, 12) if col.endswith("_id") else rng.choice([-1, 0, 1, 2, 3, 5, 10, 1000, 1001])
    if t.startswith(("DECIMAL", "NUMERIC")):
        return decimal.Decimal(rng.choice(["0.00", "0.10", "0.20", "0.30", "999.99", "1000.00", "1000.01", "5.55"]))
    if t.startswith(("FLOAT", "DOUBLE", "REAL")):
        return rng.choice([0.0, 0.1, 0.2, 1000.0, 1000.5])
    if t.startswith(("DATETIME", "TIMESTAMP")):
        return rng.choice(BOUNDARY_TIMES)
    if t.startswith("DATE"):
        return rng.choice(BOUNDARY_TIMES).date()
    return rng.choice(STRINGS)


@dataclass
class Dataset:
    tables: dict[str, tuple[list[str], list[tuple]]] = field(default_factory=dict)


def generate(schema: Schema, size: int, seed: int, null_rate: float = 0.2) -> Dataset:
    rng = random.Random(seed)
    ds = Dataset()
    for table, cols in schema.columns.items():
        pks = {u[0] for u in schema.unique.get(table, []) if len(u) == 1}
        rows = []
        for i in range(size):
            rows.append(tuple(
                _value(schema.types.get((table, c.lower()), "VARCHAR"), c.lower(), rng, i, c.lower() in pks, null_rate)
                for c in cols
            ))
        ds.tables[table] = (cols, rows)
    return ds


# ── comparison ────────────────────────────────────────────────────────────


def _norm(v: Any) -> Any:
    if isinstance(v, float):
        return round(v, 6)
    if isinstance(v, decimal.Decimal):
        return round(float(v), 6)
    if isinstance(v, dt.datetime):
        return v.replace(tzinfo=None)
    return v


def _rows(rows: list[tuple]) -> list[tuple]:
    return [tuple(_norm(v) for v in r) for r in rows]


@dataclass
class Verdict:
    equivalent: bool
    backend: str
    runs: int
    detail: str = ""


def check_equivalent(
    ddl: str,
    original: str,
    optimized: str,
    dialect: str = "mysql",
    sizes: tuple[int, ...] = (0, 8, 60),
    seeds: tuple[int, ...] = (1, 2, 3),
    backend: Backend | None = None,
) -> Verdict:
    schema, err = parse_schema(ddl, dialect)
    if err or not schema.columns:
        return Verdict(False, "none", 0, err or "schema has no tables")
    be = backend or default_backend()
    ddl_statements = [s.sql(dialect=dialect) for s in sqlglot.parse(ddl, read=dialect)
                      if isinstance(s, exp.Create)]
    runs = 0
    for size in sizes:
        for seed in seeds if size else (0,):
            be.reset(ddl_statements)
            for table, (cols, rows) in generate(schema, size, seed).tables.items():
                be.insert(table, cols, rows)
            try:
                a = _rows(be.query(original))
            except Exception as e:  # noqa: BLE001
                return Verdict(False, be.name, runs, f"original failed: {e}")
            try:
                b = _rows(be.query(optimized))
            except Exception as e:  # noqa: BLE001
                return Verdict(False, be.name, runs, f"optimized failed: {e}")
            runs += 1
            if Counter(a) != Counter(b):
                missing = list((Counter(a) - Counter(b)).elements())[:3]
                extra = list((Counter(b) - Counter(a)).elements())[:3]
                return Verdict(False, be.name, runs,
                               f"size={size} seed={seed}: rows differ; only in original {missing}, only in optimized {extra}")
    return Verdict(True, be.name, runs)
