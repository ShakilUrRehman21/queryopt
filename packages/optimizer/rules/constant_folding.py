"""
optimizer.rules.constant_folding — Evaluates compile-time constant expressions.

Replaces subexpressions containing only literals with their precomputed literal value.
For example: `WHERE age > 18 + 2` -> `WHERE age > 20`.
"""
from __future__ import annotations

from common.types import DataType
from executor.base import eval_expr
from optimizer.rules.base import Rule
from parser.ast_types import BinaryOp, Expr, Literal, ProjectionItem, UnaryOp
from planner.nodes import LogicalFilter, LogicalNode, LogicalProject


def _is_constant(expr: Expr) -> bool:
    match expr:
        case Literal():
            return True
        case BinaryOp():
            return _is_constant(expr.left) and _is_constant(expr.right)
        case UnaryOp():
            return _is_constant(expr.expr)
        case _:
            return False


def _fold_expr(expr: Expr) -> tuple[Expr, bool]:
    match expr:
        case BinaryOp():
            left_folded, left_mod = _fold_expr(expr.left)
            right_folded, right_mod = _fold_expr(expr.right)
            mod = left_mod or right_mod

            if _is_constant(left_folded) and _is_constant(right_folded):
                # Evaluate constant expression
                from common.types import Schema
                dummy_schema = Schema([])
                val = eval_expr(BinaryOp(left_folded, expr.op, right_folded), None, dummy_schema)
                dtype = DataType.FLOAT if isinstance(val, float) else (
                    DataType.INTEGER if isinstance(val, int) else (
                        DataType.BOOLEAN if isinstance(val, bool) else DataType.TEXT
                    )
                )
                return Literal(value=val, data_type=dtype), True

            if mod:
                return BinaryOp(left_folded, expr.op, right_folded), True
            return expr, False

        case UnaryOp():
            inner_folded, inner_mod = _fold_expr(expr.expr)
            if _is_constant(inner_folded):
                from common.types import Schema
                dummy_schema = Schema([])
                val = eval_expr(UnaryOp(expr.op, inner_folded), None, dummy_schema)
                dtype = DataType.BOOLEAN if isinstance(val, bool) else DataType.INTEGER
                return Literal(value=val, data_type=dtype), True
            if inner_mod:
                return UnaryOp(expr.op, inner_folded), True
            return expr, False

        case _:
            return expr, False


class ConstantFoldingRule(Rule):
    """Folds constant expressions in Filter predicates and Projection expressions."""

    @property
    def name(self) -> str:
        return "ConstantFolding"

    def apply(self, node: LogicalNode) -> tuple[LogicalNode, bool]:
        modified = False

        # Apply to children first
        new_children = []
        for child in node.children:
            new_child, child_mod = self.apply(child)
            new_children.append(new_child)
            if child_mod:
                modified = True

        current = node.with_children(new_children) if modified else node

        if isinstance(current, LogicalFilter):
            folded_pred, pred_mod = _fold_expr(current.predicate)
            if pred_mod:
                return (
                    LogicalFilter(
                        schema=current.schema,
                        child=current.child,
                        predicate=folded_pred,
                        estimated_rows=current.estimated_rows,
                        estimated_cost=current.estimated_cost,
                    ),
                    True,
                )

        if isinstance(current, LogicalProject):
            new_projs = []
            proj_mod = False
            for p in current.projections:
                f_expr, e_mod = _fold_expr(p.expr)
                if e_mod:
                    proj_mod = True
                    new_projs.append(ProjectionItem(expr=f_expr, alias=p.alias))
                else:
                    new_projs.append(p)
            if proj_mod:
                return (
                    LogicalProject(
                        schema=current.schema,
                        child=current.child,
                        projections=new_projs,
                        estimated_rows=current.estimated_rows,
                        estimated_cost=current.estimated_cost,
                    ),
                    True,
                )

        return current, modified
