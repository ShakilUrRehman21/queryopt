"""
optimizer.rules.predicate_pushdown — Predicate pushdown optimization rule.

Pushes filter predicates down the operator tree as close as possible to the
leaf scan operators to reduce the number of tuples passed through joins and projects.
"""
from __future__ import annotations

from common.types import JoinType
from optimizer.rules.base import Rule
from parser.ast_types import BinaryOp, ColumnRef, Expr, UnaryOp
from planner.nodes import (
    LogicalFilter,
    LogicalJoin,
    LogicalNode,
    LogicalProject,
    LogicalScan,
)


def _get_node_tables(node: LogicalNode) -> set[str]:
    """Collect all table names and aliases present in a logical subtree."""
    tables: set[str] = set()
    if isinstance(node, LogicalScan):
        if node.table_name:
            tables.add(node.table_name.lower())
        if node.alias:
            tables.add(node.alias.lower())
    for ch in node.children:
        tables.update(_get_node_tables(ch))
    return tables


def _can_push_predicate(predicate: Expr, node: LogicalNode) -> bool:
    """Check if all column references in predicate can be fully resolved in node."""
    node_tables = _get_node_tables(node)
    schema_cols = {c.name.lower().split(".")[-1] for c in node.schema.columns}

    refs: list[ColumnRef] = []

    def walk(e: Expr) -> None:
        match e:
            case ColumnRef():
                refs.append(e)
            case BinaryOp():
                walk(e.left)
                walk(e.right)
            case UnaryOp():
                walk(e.expr)
            case _:
                pass

    walk(predicate)

    if not refs:
        return True

    for r in refs:
        col = r.column.lower()
        if r.table:
            tbl = r.table.lower()
            if tbl not in node_tables:
                return False
        if col not in schema_cols:
            return False

    return True


def _split_and(expr: Expr) -> list[Expr]:
    if isinstance(expr, BinaryOp) and expr.op.upper() == "AND":
        return _split_and(expr.left) + _split_and(expr.right)
    return [expr]


def _and_all(parts: list[Expr]) -> Expr:
    out = parts[0]
    for p in parts[1:]:
        out = BinaryOp(left=out, op="AND", right=p)
    return out


class PredicatePushdownRule(Rule):
    """
    Pushes LogicalFilter nodes below joins and projects where valid.
    """

    @property
    def name(self) -> str:
        return "PredicatePushdown"

    def apply(self, node: LogicalNode) -> tuple[LogicalNode, bool]:
        modified = False

        # First recursively apply down the tree
        new_children = []
        for child in node.children:
            new_child, child_mod = self.apply(child)
            new_children.append(new_child)
            if child_mod:
                modified = True

        current = node.with_children(new_children) if modified else node

        # Now check if current node is a LogicalFilter that can be pushed down
        if isinstance(current, LogicalFilter):
            child = current.child

            # Case 1: Filter above Project
            if isinstance(child, LogicalProject):
                if _can_push_predicate(current.predicate, child.child):
                    pushed_filter = LogicalFilter(
                        schema=child.child.schema,
                        child=child.child,
                        predicate=current.predicate,
                    )
                    new_project = LogicalProject(
                        schema=current.schema,
                        child=pushed_filter,
                        projections=child.projections,
                    )
                    return new_project, True

            # Case 2: Filter above Join.
            # Split the predicate into AND-conjuncts and push each one to the
            # side that can evaluate it on its own; conjuncts that need both
            # sides (or a non-pushable side of an outer join) stay above.
            if isinstance(child, LogicalJoin):
                conjuncts = _split_and(current.predicate)
                inner = child.join_type in (JoinType.INNER, "INNER JOIN")
                left_parts: list[Expr] = []
                right_parts: list[Expr] = []
                remaining: list[Expr] = []
                for c in conjuncts:
                    if _can_push_predicate(c, child.left):
                        left_parts.append(c)
                    elif inner and _can_push_predicate(c, child.right):
                        right_parts.append(c)
                    else:
                        remaining.append(c)

                if left_parts or right_parts:
                    new_left = child.left
                    new_right = child.right
                    if left_parts:
                        new_left = LogicalFilter(
                            schema=child.left.schema,
                            child=child.left,
                            predicate=_and_all(left_parts),
                        )
                    if right_parts:
                        new_right = LogicalFilter(
                            schema=child.right.schema,
                            child=child.right,
                            predicate=_and_all(right_parts),
                        )
                    new_join = child.with_children([new_left, new_right])
                    if remaining:
                        return (
                            LogicalFilter(
                                schema=current.schema,
                                child=new_join,
                                predicate=_and_all(remaining),
                            ),
                            True,
                        )
                    return new_join, True

        return current, modified
