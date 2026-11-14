"""
parser.parser — SQL parser: SQL string → internal AST.

Uses SQLGlot to tokenize and parse SQL, then converts SQLGlot's AST
into QueryOpt Lab's internal AST types (defined in ast_types.py).

Supported statements (Phase 1):
    SELECT ... FROM ... [JOIN ...] [WHERE ...] [GROUP BY ...] [ORDER BY ...] [LIMIT ...]
    CREATE TABLE ... (column definitions)
    INSERT INTO ... [(columns)] VALUES (...)

The SQLGlot dependency is entirely contained within this module.
The planner has zero SQLGlot imports.
"""
from __future__ import annotations

from typing import Any

import sqlglot
import sqlglot.expressions as exp

from common.errors import SQLParseError, UnsupportedQueryError
from common.logging import get_logger
from common.types import DataType, JoinType, SortDirection
from parser.ast_types import (
    BinaryOp,
    ColumnDefinition,
    ColumnRef,
    CreateTableStatement,
    Expr,
    FunctionCall,
    InsertStatement,
    JoinClause,
    Literal,
    ParseResult,
    ProjectionItem,
    SelectStatement,
    SortItem,
    Star,
    Statement,
    TableSource,
    UnaryOp,
)

logger = get_logger(__name__)


# ── SQLGlot dialect for QueryOpt Lab ─────────────────────────────────────────
# We use the default (no dialect) for maximum compatibility.
_DIALECT = None


# ── Data type mapping ─────────────────────────────────────────────────────────

_SQLGLOT_TYPE_MAP: dict[str, DataType] = {
    "INT": DataType.INTEGER,
    "INTEGER": DataType.INTEGER,
    "BIGINT": DataType.INTEGER,
    "SMALLINT": DataType.INTEGER,
    "TINYINT": DataType.INTEGER,
    "FLOAT": DataType.FLOAT,
    "DOUBLE": DataType.FLOAT,
    "REAL": DataType.FLOAT,
    "DECIMAL": DataType.FLOAT,
    "NUMERIC": DataType.FLOAT,
    "BOOLEAN": DataType.BOOLEAN,
    "BOOL": DataType.BOOLEAN,
    "TEXT": DataType.TEXT,
    "VARCHAR": DataType.TEXT,
    "CHAR": DataType.TEXT,
    "STRING": DataType.TEXT,
    "DATE": DataType.DATE,
    "TIMESTAMP": DataType.TIMESTAMP,
    "DATETIME": DataType.TIMESTAMP,
}


def _map_data_type(sqlglot_dtype: exp.DataType) -> DataType:
    """Convert a SQLGlot DataType node to a QueryOpt Lab DataType."""
    type_str = sqlglot_dtype.this.name.upper() if sqlglot_dtype.this else "TEXT"
    if type_str in _SQLGLOT_TYPE_MAP:
        return _SQLGLOT_TYPE_MAP[type_str]
    # Unknown type → treat as TEXT with a warning
    logger.warning(f"Unknown SQL type '{type_str}'; treating as TEXT")
    return DataType.TEXT


# ── Expression conversion ─────────────────────────────────────────────────────

_BINARY_OP_MAP: dict[type, str] = {
    exp.EQ: "=",
    exp.NEQ: "!=",
    exp.LT: "<",
    exp.LTE: "<=",
    exp.GT: ">",
    exp.GTE: ">=",
    exp.And: "AND",
    exp.Or: "OR",
    exp.Add: "+",
    exp.Sub: "-",
    exp.Mul: "*",
    exp.Div: "/",
}


def _convert_expr(node: exp.Expression) -> Expr:
    """Recursively convert a SQLGlot expression to a QueryOpt Lab Expr."""

    match type(node):
        case exp.Column:
            table = node.table if hasattr(node, "table") and node.table else None
            return ColumnRef(column=node.name, table=table or None)

        case exp.Literal:
            if node.is_string:
                return Literal(value=node.this, data_type=DataType.TEXT)
            if node.is_int:
                return Literal(value=int(node.this), data_type=DataType.INTEGER)
            # Float
            try:
                return Literal(value=float(node.this), data_type=DataType.FLOAT)
            except ValueError:
                return Literal(value=node.this, data_type=DataType.TEXT)

        case exp.Boolean:
            return Literal(value=node.this, data_type=DataType.BOOLEAN)

        case exp.Null:
            return Literal(value=None, data_type=DataType.NULL)

        case exp.Star:
            table = node.table if hasattr(node, "table") and node.table else None
            return Star(table=table or None)

        case exp.Is:
            # IS NULL / IS NOT NULL
            right = node.args.get("expression")
            if isinstance(right, exp.Null):
                return UnaryOp(op="IS NULL", expr=_convert_expr(node.this))
            return UnaryOp(op="IS NOT NULL", expr=_convert_expr(node.this))

        case exp.Not:
            return UnaryOp(op="NOT", expr=_convert_expr(node.this))

        case exp.Neg:
            return UnaryOp(op="-", expr=_convert_expr(node.this))

        case exp.Anonymous | exp.Count | exp.Sum | exp.Avg | exp.Min | exp.Max:
            func_name: str
            if isinstance(node, exp.Anonymous):
                func_name = node.name.upper()
            else:
                func_name = type(node).__name__.upper()
            args = tuple(_convert_expr(a) for a in node.args.get("expressions", []) or [])
            # COUNT(*) has a Star child
            if isinstance(node, exp.Count) and not args:
                args = (Star(),)
            distinct = bool(node.args.get("distinct"))
            return FunctionCall(name=func_name, args=args, distinct=distinct)

        case _:
            # Check binary ops
            node_type = type(node)
            if node_type in _BINARY_OP_MAP:
                return BinaryOp(
                    left=_convert_expr(node.left),
                    op=_BINARY_OP_MAP[node_type],
                    right=_convert_expr(node.right),
                )

            # Try to handle as binary with left/right
            if hasattr(node, "left") and hasattr(node, "right"):
                op_name = type(node).__name__.upper()
                return BinaryOp(
                    left=_convert_expr(node.left),
                    op=op_name,
                    right=_convert_expr(node.right),
                )

            raise UnsupportedQueryError(
                message=f"Unsupported expression type: {type(node).__name__}",
                feature=type(node).__name__,
                context={"node_sql": node.sql() if hasattr(node, "sql") else str(node)},
            )


# ── Statement Converters ──────────────────────────────────────────────────────

def _convert_select(node: exp.Select) -> SelectStatement:
    """Convert a SQLGlot Select node to a SelectStatement."""

    # FROM clause
    from_expr = node.args.get("from_") or node.args.get("from")
    if from_expr is None:
        raise UnsupportedQueryError(
            message="SELECT without FROM is not supported in QueryOpt Lab.",
            feature="SELECT_WITHOUT_FROM",
        )
    from_table = _extract_table_source(from_expr.this)

    # JOINs
    joins: list[JoinClause] = []
    for join_node in node.args.get("joins") or []:
        join_table = _extract_table_source(join_node.this)
        join_kind = join_node.args.get("kind")
        join_type_str = join_kind.upper() if join_kind else "INNER"
        if "LEFT" in join_type_str:
            jtype = JoinType.LEFT
        elif "RIGHT" in join_type_str:
            raise UnsupportedQueryError(message="RIGHT JOIN is not yet supported.", feature="RIGHT_JOIN")
        elif "FULL" in join_type_str:
            raise UnsupportedQueryError(message="FULL JOIN is not yet supported.", feature="FULL_JOIN")
        else:
            jtype = JoinType.INNER

        on_expr = join_node.args.get("on")
        if on_expr is None:
            raise UnsupportedQueryError(
                message="JOIN without ON condition is not supported.",
                feature="JOIN_WITHOUT_ON",
            )
        condition = _convert_expr(on_expr)
        joins.append(JoinClause(table=join_table, join_type=jtype, condition=condition))

    # WHERE
    where_node = node.args.get("where")
    where: Expr | None = None
    if where_node is not None:
        where = _convert_expr(where_node.this)

    # SELECT projections
    projections: list[ProjectionItem] = []
    for sel_expr in node.expressions:
        if isinstance(sel_expr, exp.Star):
            projections.append(ProjectionItem(expr=Star()))
        elif isinstance(sel_expr, exp.Alias):
            inner = _convert_expr(sel_expr.this)
            alias = sel_expr.alias
            projections.append(ProjectionItem(expr=inner, alias=alias or None))
        else:
            projections.append(ProjectionItem(expr=_convert_expr(sel_expr)))

    # GROUP BY
    group_by: list[Expr] = []
    group_node = node.args.get("group")
    if group_node is not None:
        gb_exprs = group_node.expressions if hasattr(group_node, "expressions") else [group_node]
        for gb in gb_exprs:
            group_by.append(_convert_expr(gb))

    # HAVING
    having_node = node.args.get("having")
    having: Expr | None = None
    if having_node is not None:
        having = _convert_expr(having_node.this)

    # ORDER BY
    order_by: list[SortItem] = []
    order_node = node.args.get("order")
    if order_node is not None:
        ob_exprs = order_node.expressions if hasattr(order_node, "expressions") else [order_node]
        for ob in ob_exprs:
            expr = _convert_expr(ob.this)
            direction = SortDirection.DESC if ob.args.get("desc") else SortDirection.ASC
            order_by.append(SortItem(expr=expr, direction=direction))

    # LIMIT / OFFSET
    limit: int | None = None
    offset: int | None = None
    limit_node = node.args.get("limit")
    if limit_node is not None:
        try:
            val_node = limit_node.expression if hasattr(limit_node, "expression") else limit_node.this
            limit = int(val_node.this)
        except (AttributeError, ValueError, TypeError):
            pass
    offset_node = node.args.get("offset")
    if offset_node is not None:
        try:
            val_node = offset_node.expression if hasattr(offset_node, "expression") else offset_node.this
            offset = int(val_node.this)
        except (AttributeError, ValueError, TypeError):
            pass

    distinct = bool(node.args.get("distinct"))

    return SelectStatement(
        projections=projections,
        from_table=from_table,
        joins=joins,
        where=where,
        group_by=group_by,
        having=having,
        order_by=order_by,
        limit=limit,
        offset=offset,
        distinct=distinct,
    )


def _extract_table_source(node: exp.Expression) -> TableSource:
    """Extract a TableSource from a FROM or JOIN table reference."""
    if isinstance(node, exp.Alias):
        inner = node.this
        alias = node.alias
        if isinstance(inner, exp.Table):
            return TableSource(table_name=inner.name, alias=alias or None)
    if isinstance(node, exp.Table):
        return TableSource(table_name=node.name, alias=node.alias or None)
    raise UnsupportedQueryError(
        message=f"Unsupported table reference: {type(node).__name__}",
        feature=type(node).__name__,
    )


def _convert_create_table(node: exp.Create) -> CreateTableStatement:
    """Convert a SQLGlot Create node to a CreateTableStatement."""
    table_name = node.this.this.name
    if_not_exists = bool(node.args.get("exists"))

    column_defs: list[ColumnDefinition] = []
    schema_node = node.this
    for col_def in schema_node.expressions:
        if not isinstance(col_def, exp.ColumnDef):
            continue
        col_name = col_def.name
        dtype_node = col_def.args.get("kind")
        dtype = _map_data_type(dtype_node) if dtype_node else DataType.TEXT

        nullable = True
        primary_key = False
        for constraint in col_def.args.get("constraints") or []:
            constraint_kind = constraint.args.get("kind")
            if isinstance(constraint_kind, exp.NotNullColumnConstraint):
                nullable = False
            elif isinstance(constraint_kind, exp.PrimaryKeyColumnConstraint):
                primary_key = True
                nullable = False

        column_defs.append(ColumnDefinition(
            name=col_name,
            data_type=dtype,
            nullable=nullable,
            primary_key=primary_key,
        ))

    return CreateTableStatement(
        table_name=table_name,
        columns=column_defs,
        if_not_exists=if_not_exists,
    )


def _convert_insert(node: exp.Insert) -> InsertStatement:
    """Convert a SQLGlot Insert node to an InsertStatement."""
    table_name = node.this.this.name if isinstance(node.this, exp.Schema) else node.this.name

    # Column list (optional)
    columns: list[str] | None = None
    if isinstance(node.this, exp.Schema) and node.this.expressions:
        columns = [col.name for col in node.this.expressions]

    # Values
    values_rows: list[list[Any]] = []
    values_node = node.args.get("expression")
    if isinstance(values_node, exp.Values):
        for tuple_node in values_node.expressions:
            row_values: list[Any] = []
            for val in tuple_node.expressions:
                if isinstance(val, exp.Null):
                    row_values.append(None)
                elif isinstance(val, exp.Literal):
                    if val.is_string:
                        row_values.append(val.this)
                    elif val.is_int:
                        row_values.append(int(val.this))
                    else:
                        try:
                            row_values.append(float(val.this))
                        except ValueError:
                            row_values.append(val.this)
                elif isinstance(val, exp.Boolean):
                    row_values.append(val.this)
                elif isinstance(val, exp.Neg):
                    inner = val.this
                    if isinstance(inner, exp.Literal):
                        if inner.is_int:
                            row_values.append(-int(inner.this))
                        else:
                            row_values.append(-float(inner.this))
                    else:
                        row_values.append(None)
                else:
                    row_values.append(None)
            values_rows.append(row_values)

    return InsertStatement(
        table_name=table_name,
        columns=columns,
        values_rows=values_rows,
    )


# ── Public API ────────────────────────────────────────────────────────────────

_BLOCKED_KEYWORDS = frozenset([
    "DROP", "TRUNCATE", "ALTER", "CREATE INDEX", "CREATE VIEW",
    "LOAD DATA", "COPY", "CALL", "EXECUTE", "GRANT", "REVOKE",
])


def parse(sql: str) -> ParseResult:
    """
    Parse a SQL string into a QueryOpt Lab internal AST.

    Supported statements: SELECT, CREATE TABLE, INSERT INTO ... VALUES.
    All other statements raise UnsupportedQueryError.

    Raises:
        SQLParseError: if the SQL cannot be parsed at all.
        UnsupportedQueryError: if the SQL is valid but uses unsupported features.
    """
    sql = sql.strip()
    if not sql:
        raise SQLParseError(message="Empty SQL string.")

    # Rough pre-check for blocked statements
    first_word = sql.split()[0].upper() if sql.split() else ""
    if first_word in ("DROP", "TRUNCATE", "ALTER"):
        raise UnsupportedQueryError(
            message=f"Statement type '{first_word}' is not supported in QueryOpt Lab.",
            feature=first_word,
        )

    # Parse with SQLGlot
    try:
        statements = sqlglot.parse(sql, dialect=_DIALECT, error_level=sqlglot.ErrorLevel.RAISE)
    except sqlglot.errors.ParseError as e:
        raise SQLParseError(
            message=f"SQL parse error: {e}",
            context={"sql": sql, "sqlglot_error": str(e)},
        ) from e

    if not statements:
        raise SQLParseError(message="No statements found in SQL string.")
    if len(statements) > 1:
        raise UnsupportedQueryError(
            message="Multiple statements in one call are not supported. Submit one statement at a time.",
            feature="MULTI_STATEMENT",
        )

    node = statements[0]

    try:
        statement = _dispatch(node, sql)
    except (UnsupportedQueryError, SQLParseError):
        raise
    except Exception as e:
        raise SQLParseError(
            message=f"Failed to convert parsed SQL to internal AST: {e}",
            context={"sql": sql},
        ) from e

    logger.debug(f"Parsed: {type(statement).__name__}")
    return ParseResult(sql=sql, statement=statement)


def _dispatch(node: exp.Expression, sql: str) -> Statement:
    """Route a SQLGlot node to the appropriate converter."""
    if isinstance(node, exp.Select):
        return _convert_select(node)
    if isinstance(node, exp.Create) and node.kind == "TABLE":
        return _convert_create_table(node)
    if isinstance(node, exp.Insert):
        return _convert_insert(node)

    stmt_type = type(node).__name__
    raise UnsupportedQueryError(
        message=f"Statement type '{stmt_type}' is not supported in QueryOpt Lab Phase 1.",
        feature=stmt_type,
    )
