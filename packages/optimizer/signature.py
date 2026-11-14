"""
optimizer.signature — Order-independent signatures for relational sub-expressions.

A *signature* identifies "the rows produced by joining this set of base tables
under this set of predicates", independently of join order or operator shape:

    sig( (A ⋈ B) ⋈ σ_p(C) )  ==  sig( σ_p(C) ⋈ (B ⋈ A) )

Signatures are the key used by the cardinality feedback loop (Phase 6): once an
execution has observed the true row count of a sub-expression, any later plan —
including every candidate the DP join enumerator builds for a table subset — can
look that number up instead of trusting a modelled estimate. This mirrors the
LEO learning optimizer in DB2 (Stillger et al., VLDB 2001) in a deliberately
simple, transparent form.

Both logical and physical trees are supported so the same key is produced when
an estimate is made (logical, during optimization) and when it is checked
(physical, after execution).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from parser.ast_types import BinaryOp, ColumnRef, Expr, FunctionCall, Literal, Star, UnaryOp

if TYPE_CHECKING:
    from executor.base import PhysicalOperator
    from planner.nodes import LogicalNode

_COMMUTATIVE = {"=", "!=", "AND", "OR", "+", "*"}
_FLIP = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}


@dataclass(frozen=True)
class RelSignature:
    """Canonical (tables, predicates) pair; hashable and JSON-friendly via `key`."""

    tables: frozenset[str]
    predicates: frozenset[str]

    @property
    def key(self) -> str:
        return "|".join(sorted(self.tables)) + "#" + " AND ".join(sorted(self.predicates))

    def union(self, other: RelSignature, extra_predicates: list[str] | None = None) -> RelSignature:
        preds = set(self.predicates) | set(other.predicates)
        if extra_predicates:
            preds.update(extra_predicates)
        return RelSignature(self.tables | other.tables, frozenset(preds))

    def with_predicates(self, extra: list[str]) -> RelSignature:
        return RelSignature(self.tables, frozenset(set(self.predicates) | set(extra)))

    def to_dict(self) -> dict[str, Any]:
        return {"tables": sorted(self.tables), "predicates": sorted(self.predicates)}


# ── Expression canonicalisation ─────────────────────────────────────────────


def canonical_expr(expr: Expr | None) -> str:
    """Deterministic, whitespace-free text for an expression.

    Operands of commutative operators are sorted, and `5 < col` is rewritten to
    `col > 5`, so logically identical predicates map to the same string.
    """
    if expr is None:
        return ""
    match expr:
        case ColumnRef():
            return f"{expr.table.lower()}.{expr.column.lower()}" if expr.table else expr.column.lower()
        case Literal():
            v = expr.value
            if isinstance(v, str):
                return "'" + v.replace("'", "''") + "'"
            if isinstance(v, bool):
                return "TRUE" if v else "FALSE"
            if isinstance(v, float) and v.is_integer():
                return str(int(v))
            return str(v)
        case BinaryOp():
            op = expr.op.upper()
            left, right = canonical_expr(expr.left), canonical_expr(expr.right)
            if op in _FLIP and isinstance(expr.left, Literal) and not isinstance(expr.right, Literal):
                left, right, op = right, left, _FLIP[op]
            elif op in _COMMUTATIVE and right < left:
                left, right = right, left
            return f"({left}{op}{right})"
        case UnaryOp():
            return f"({expr.op.upper()} {canonical_expr(expr.expr)})"
        case FunctionCall():
            args = ",".join(canonical_expr(a) for a in expr.args)
            return f"{expr.name.upper()}({'DISTINCT ' if expr.distinct else ''}{args})"
        case Star():
            return f"{expr.table}.*" if expr.table else "*"
        case _:
            return repr(expr)


def split_conjuncts(expr: Expr | None) -> list[Expr]:
    if expr is None:
        return []
    if isinstance(expr, BinaryOp) and expr.op.upper() == "AND":
        return split_conjuncts(expr.left) + split_conjuncts(expr.right)
    return [expr]


def predicate_strings(expr: Expr | None) -> list[str]:
    """Canonical conjuncts of a predicate, with trivially-true terms removed."""
    out = []
    for c in split_conjuncts(expr):
        if isinstance(c, Literal) and bool(c.value) is True:
            continue
        out.append(canonical_expr(c))
    return out


# ── Logical plans ───────────────────────────────────────────────────────────


def logical_signature(node: LogicalNode) -> RelSignature | None:
    """Signature of a logical subtree, or None if it is not a select-join block
    (aggregates and limits change the row set in ways a signature can't name)."""
    from planner.nodes import (
        LogicalFilter,
        LogicalJoin,
        LogicalProject,
        LogicalScan,
        LogicalSort,
    )

    match node:
        case LogicalScan():
            return RelSignature(frozenset([(node.alias or node.table_name).lower()]), frozenset())
        case LogicalFilter():
            child = logical_signature(node.child)
            return child.with_predicates(predicate_strings(node.predicate)) if child else None
        case LogicalJoin():
            left, right = logical_signature(node.left), logical_signature(node.right)
            if left is None or right is None:
                return None
            if node.join_type.name not in ("INNER", "CROSS"):
                # Outer joins are not order-independent; keep them distinct.
                return RelSignature(
                    left.tables | right.tables,
                    frozenset(
                        set(left.predicates) | set(right.predicates)
                        | {f"{node.join_type.name}:{p}" for p in predicate_strings(node.condition)}
                    ),
                )
            return left.union(right, predicate_strings(node.condition))
        case LogicalProject() | LogicalSort():
            return logical_signature(node.children[0])
        case _:
            return None


# ── Physical plans ──────────────────────────────────────────────────────────


def physical_observations(root: PhysicalOperator) -> list[tuple[RelSignature, float, int]]:
    """Walk an *executed* physical tree and return (signature, estimated, actual)
    for every operator whose actual row count is trustworthy.

    Operators beneath a LIMIT may have been cut off early, so they are skipped
    unless a blocking operator (Sort, HashAggregate, a join's build side) in
    between guarantees its input was fully consumed.
    """
    from executor.aggregate import HashAggregate
    from executor.filter import Filter
    from executor.join import HashJoin, NestedLoopJoin
    from executor.limit import Limit
    from executor.project import Project
    from executor.scan import SeqScan
    from executor.sort import Sort

    out: list[tuple[RelSignature, float, int]] = []

    def visit(op: PhysicalOperator, reliable: bool) -> RelSignature | None:
        sig: RelSignature | None
        if isinstance(op, SeqScan):
            sig = RelSignature(frozenset([(op.alias or op.table_name).lower()]), frozenset())
        elif isinstance(op, Filter):
            child = visit(op.child, reliable)
            sig = child.with_predicates(predicate_strings(op.predicate)) if child else None
        elif isinstance(op, (HashJoin, NestedLoopJoin)):
            # Build/inner side is always fully materialised.
            left = visit(op.left, reliable)
            right = visit(op.right, True)
            if left is None or right is None:
                sig = None
            elif op.join_type.name in ("INNER", "CROSS"):
                sig = left.union(right, predicate_strings(op.condition))
            else:
                sig = RelSignature(
                    left.tables | right.tables,
                    frozenset(
                        set(left.predicates) | set(right.predicates)
                        | {f"{op.join_type.name}:{p}" for p in predicate_strings(op.condition)}
                    ),
                )
        elif isinstance(op, (Project,)):
            sig = visit(op.child, reliable)
            return sig  # same rows as child; don't double-record
        elif isinstance(op, Sort):
            return visit(op.child, True)
        elif isinstance(op, HashAggregate):
            visit(op.child, True)
            return None
        elif isinstance(op, Limit):
            visit(op.child, False)
            return None
        else:
            for ch in op.children:
                visit(ch, reliable)
            return None

        if sig is not None and reliable:
            out.append((sig, float(op.estimated_rows or 0.0), int(op.actual_rows)))
        return sig

    visit(root, True)
    return out
