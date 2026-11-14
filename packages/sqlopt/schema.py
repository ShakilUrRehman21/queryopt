"""
sqlopt.schema — Optional schema context pasted by the user (CREATE TABLE / CREATE INDEX).

With it the optimizer can skip indexes that already exist (or are covered by an
existing index with the same leading columns), treat primary keys as indexed,
and spell out the column list that should replace SELECT *. Nothing connects
to a database; the DDL is only parsed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp


@dataclass
class Schema:
    columns: dict[str, list[str]] = field(default_factory=dict)  # table -> columns (declared order)
    indexes: dict[str, list[tuple[str, ...]]] = field(default_factory=dict)  # table -> leading-ordered cols
    unique: dict[str, list[tuple[str, ...]]] = field(default_factory=dict)  # PRIMARY KEY / UNIQUE only
    types: dict[tuple[str, str], str] = field(default_factory=dict)  # (table, column) -> SQL type

    def add_index(self, table: str, cols: list[str], unique: bool = False) -> None:
        if cols:
            key = tuple(c.lower() for c in cols)
            self.indexes.setdefault(table.lower(), []).append(key)
            if unique:
                self.unique.setdefault(table.lower(), []).append(key)

    def unique_prefix(self, table: str, cols: tuple[str, ...]) -> tuple[str, ...] | None:
        """A PK/UNIQUE key made of leading columns of `cols`: lookups on it already return one row."""
        want = tuple(c.lower() for c in cols)
        return next((u for u in self.unique.get(table.lower(), []) if want[: len(u)] == u), None)

    def low_cardinality(self, table: str, col: str) -> bool:
        t = self.types.get((table.lower(), col.lower()), "")
        return t.startswith(("BOOL", "BIT")) or t in ("TINYINT", "TINYINT(1)")

    def covered(self, table: str, cols: tuple[str, ...]) -> tuple[str, ...] | None:
        """An existing index whose leading columns already equal `cols` (or vice versa)."""
        want = tuple(c.lower() for c in cols)
        for idx in self.indexes.get(table.lower(), []):
            if idx[: len(want)] == want:
                return idx
        return None

    def __bool__(self) -> bool:
        return bool(self.columns or self.indexes)


def _names(node: exp.Expression) -> list[str]:
    cols = [c.name for c in node.find_all(exp.Column)]
    if not cols:
        cols = [i.name for i in node.find_all(exp.Identifier)][1:] if isinstance(node, exp.IndexColumnConstraint) and node.this else \
               [i.name for i in node.find_all(exp.Identifier)]
    return cols


def parse_schema(ddl: str, dialect: str) -> tuple[Schema, str | None]:
    sch = Schema()
    if not (ddl or "").strip():
        return sch, None
    try:
        stmts = [s for s in sqlglot.parse(ddl, read=dialect) if s is not None]
    except Exception as e:  # noqa: BLE001
        return sch, f"Could not read the schema: {str(e).splitlines()[0]}"
    for s in stmts:
        if not isinstance(s, exp.Create):
            continue
        kind = (s.args.get("kind") or "").upper()
        if kind == "TABLE" and isinstance(s.this, exp.Schema):
            table = s.this.this.name.lower()
            sch.columns[table] = [cd.name for cd in s.this.expressions if isinstance(cd, exp.ColumnDef)]
            for cd in s.this.expressions:
                if isinstance(cd, exp.ColumnDef):
                    if cd.args.get("kind") is not None:
                        sch.types[(table, cd.name.lower())] = cd.args["kind"].sql(dialect=dialect).upper()
                    if any(isinstance(c.args.get("kind"), (exp.PrimaryKeyColumnConstraint, exp.UniqueColumnConstraint))
                           for c in cd.args.get("constraints") or []):
                        sch.add_index(table, [cd.name], unique=True)
                elif isinstance(cd, (exp.PrimaryKey, exp.UniqueColumnConstraint)):
                    sch.add_index(table, _names(cd), unique=True)
                elif isinstance(cd, exp.IndexColumnConstraint):
                    sch.add_index(table, _names(cd))
        elif kind in ("INDEX", "UNIQUE INDEX") or isinstance(s.this, exp.Index):
            idx = s.this
            tbl = idx.args.get("table")
            if tbl is not None:
                params = idx.args.get("params")
                sch.add_index(tbl.name, [c.name for c in (params or idx).find_all(exp.Column)],
                              unique=bool(idx.args.get("unique")) or "UNIQUE" in kind)
    return sch, None
