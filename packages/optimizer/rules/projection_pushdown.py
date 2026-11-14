"""
optimizer.rules.projection_pushdown — Column pruning / projection pushdown rule.

Identifies the subset of columns required by all upstream operators and
pushes column pruning into leaf LogicalScan nodes.
"""
from __future__ import annotations

from optimizer.rules.base import Rule
from parser.ast_types import BinaryOp, ColumnRef, Expr, FunctionCall, UnaryOp
from planner.nodes import (
    LogicalAggregate,
    LogicalFilter,
    LogicalJoin,
    LogicalNode,
    LogicalProject,
    LogicalScan,
    LogicalSort,
)


def _collect_columns(expr: Expr, out_set: set[str]) -> None:
    match expr:
        case ColumnRef():
            out_set.add(expr.column.lower())
        case BinaryOp():
            _collect_columns(expr.left, out_set)
            _collect_columns(expr.right, out_set)
        case UnaryOp():
            _collect_columns(expr.expr, out_set)
        case FunctionCall():
            for a in expr.args:
                _collect_columns(a, out_set)
        case _:
            pass


class ProjectionPushdownRule(Rule):
    """
    Prunes unreferenced columns by updating LogicalScan.projection_columns.
    """

    @property
    def name(self) -> str:
        return "ProjectionPushdown"

    def apply(self, node: LogicalNode) -> tuple[LogicalNode, bool]:
        # Collect all referenced columns across the entire query tree.
        # Include root output schema columns so the final projected columns are never omitted.
        referenced: set[str] = {c.name.lower().split(".")[-1] for c in node.schema.columns}
        self._collect_all_referenced(node, referenced)

        # Recursively update LogicalScan nodes
        return self._prune_scans(node, referenced)

    def _collect_all_referenced(self, node: LogicalNode, out: set[str]) -> None:
        match node:
            case LogicalFilter():
                _collect_columns(node.predicate, out)
            case LogicalProject():
                for p in node.projections:
                    _collect_columns(p.expr, out)
            case LogicalJoin():
                _collect_columns(node.condition, out)
            case LogicalAggregate():
                for g in node.group_by:
                    _collect_columns(g, out)
                for a in node.aggregates:
                    _collect_columns(a.expr, out)
            case LogicalSort():
                for s in node.sort_items:
                    _collect_columns(s.expr, out)
            case _:
                pass

        for child in node.children:
            self._collect_all_referenced(child, out)

    def _prune_scans(self, node: LogicalNode, referenced: set[str]) -> tuple[LogicalNode, bool]:
        if isinstance(node, LogicalScan):
            all_cols = [c.name for c in node.schema.columns]
            needed = [c for c in all_cols if c.lower() in referenced]
            # Ensure at least one column is scanned (e.g. for COUNT(*))
            if not needed and all_cols:
                needed = [all_cols[0]]

            if set(needed) != set(node.projection_columns or all_cols):
                from common.types import Schema
                needed_lower = {c.lower() for c in needed}
                pruned_cols = [c for c in node.schema.columns if c.name.lower() in needed_lower]
                new_scan = LogicalScan(
                    schema=Schema(pruned_cols),
                    table_name=node.table_name,
                    alias=node.alias,
                    projection_columns=needed,
                    estimated_rows=node.estimated_rows,
                    estimated_cost=node.estimated_cost,
                )
                return new_scan, True
            return node, False

        modified = False
        new_children = []
        for child in node.children:
            new_child, child_mod = self._prune_scans(child, referenced)
            new_children.append(new_child)
            if child_mod:
                modified = True

        if modified:
            return node.with_children(new_children), True
        return node, False
