"""
planner.nodes — Logical plan tree node definitions for QueryOpt Lab.

Logical plan nodes represent algebraic operators over relation streams.
Each node has:
- `children`: List of child LogicalNodes (empty for leaf nodes like Scan).
- `schema`: The Schema of rows emitted by this operator.
- `estimated_rows`: Cardinality estimate (populated by stats/cost model, initially None).
- `estimated_cost`: Cost estimate (populated by optimizer, initially None).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from common.types import Column, JoinType, Schema
from parser.ast_types import Expr, ProjectionItem, SortItem


@dataclass
class LogicalNode(ABC):
    """Abstract base class for all logical plan nodes."""
    schema: Schema
    estimated_rows: float | None = None
    estimated_cost: float | None = None

    @property
    @abstractmethod
    def children(self) -> list[LogicalNode]:
        """Return all child operators of this node."""
        ...

    @abstractmethod
    def with_children(self, new_children: list[LogicalNode]) -> LogicalNode:
        """Return a copy of this node with updated children."""
        ...

    @abstractmethod
    def node_name(self) -> str:
        """Short operator name for visualization and debugging."""
        ...

    @abstractmethod
    def details(self) -> dict[str, Any]:
        """Operator-specific attributes for visualization and plan output."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """Serialize plan node and its subtree to a dictionary."""
        return {
            "node_type": self.node_name(),
            "schema": [c.name for c in self.schema.columns],
            "estimated_rows": self.estimated_rows,
            "estimated_cost": self.estimated_cost,
            "details": self.details(),
            "children": [c.to_dict() for c in self.children],
        }


@dataclass
class LogicalScan(LogicalNode):
    """
    Leaf operator: reads data from a named base table.
    """
    table_name: str = ""
    alias: str | None = None
    projection_columns: list[str] = field(default_factory=list)  # pruned column names

    @property
    def children(self) -> list[LogicalNode]:
        return []

    def with_children(self, new_children: list[LogicalNode]) -> LogicalScan:
        if new_children:
            raise ValueError("LogicalScan cannot have children")
        return LogicalScan(
            schema=self.schema,
            estimated_rows=self.estimated_rows,
            estimated_cost=self.estimated_cost,
            table_name=self.table_name,
            alias=self.alias,
            projection_columns=list(self.projection_columns),
        )

    def node_name(self) -> str:
        return "LogicalScan"

    def details(self) -> dict[str, Any]:
        return {
            "table_name": self.table_name,
            "alias": self.alias,
            "columns": self.projection_columns or [c.name for c in self.schema.columns],
        }


@dataclass
class LogicalFilter(LogicalNode):
    """
    Unary operator: evaluates a boolean predicate against each incoming row.
    """
    child: LogicalNode = field(default=None)  # type: ignore
    predicate: Expr = field(default=None)  # type: ignore

    @property
    def children(self) -> list[LogicalNode]:
        return [self.child]

    def with_children(self, new_children: list[LogicalNode]) -> LogicalFilter:
        if len(new_children) != 1:
            raise ValueError("LogicalFilter requires exactly one child")
        return LogicalFilter(
            schema=new_children[0].schema,
            estimated_rows=self.estimated_rows,
            estimated_cost=self.estimated_cost,
            child=new_children[0],
            predicate=self.predicate,
        )

    def node_name(self) -> str:
        return "LogicalFilter"

    def details(self) -> dict[str, Any]:
        return {"predicate": str(self.predicate)}


@dataclass
class LogicalProject(LogicalNode):
    """
    Unary operator: projects and computes expressions from incoming rows.
    """
    child: LogicalNode = field(default=None)  # type: ignore
    projections: list[ProjectionItem] = field(default_factory=list)

    @property
    def children(self) -> list[LogicalNode]:
        return [self.child]

    def with_children(self, new_children: list[LogicalNode]) -> LogicalProject:
        if len(new_children) != 1:
            raise ValueError("LogicalProject requires exactly one child")
        return LogicalProject(
            schema=self.schema,
            estimated_rows=self.estimated_rows,
            estimated_cost=self.estimated_cost,
            child=new_children[0],
            projections=list(self.projections),
        )

    def node_name(self) -> str:
        return "LogicalProject"

    def details(self) -> dict[str, Any]:
        return {
            "projections": [
                p.alias if p.alias else str(p.expr) for p in self.projections
            ]
        }


@dataclass
class LogicalJoin(LogicalNode):
    """
    Binary operator: joins rows from left and right children on a condition.
    """
    left: LogicalNode = field(default=None)  # type: ignore
    right: LogicalNode = field(default=None)  # type: ignore
    join_type: JoinType = JoinType.INNER
    condition: Expr = field(default=None)  # type: ignore

    @property
    def children(self) -> list[LogicalNode]:
        return [self.left, self.right]

    def with_children(self, new_children: list[LogicalNode]) -> LogicalJoin:
        if len(new_children) != 2:
            raise ValueError("LogicalJoin requires exactly two children")
        left_child, right_child = new_children
        if (len(left_child.schema.columns) + len(right_child.schema.columns)) != len(self.schema.columns):
            combined_cols = list(left_child.schema.columns)
            existing_names = {c.name.lower() for c in combined_cols}

            def _get_node_alias(node: LogicalNode) -> str | None:
                if isinstance(node, LogicalScan):
                    return node.alias or node.table_name
                if hasattr(node, "child"):
                    return _get_node_alias(node.child)
                return None

            right_alias = _get_node_alias(right_child)

            for c in right_child.schema.columns:
                col_name = c.name
                if col_name.lower() in existing_names:
                    if right_alias:
                        col_name = f"{right_alias}.{col_name}"
                    else:
                        matched_in_self = [
                            sc.name for sc in self.schema.columns
                            if sc.name.lower().endswith(f".{col_name.lower()}")
                        ]
                        if matched_in_self:
                            col_name = matched_in_self[-1]
                combined_cols.append(
                    Column(
                        name=col_name,
                        data_type=c.data_type,
                        nullable=c.nullable,
                        default_value=c.default_value,
                    )
                )
            new_schema = Schema(combined_cols)
        else:
            new_schema = self.schema

        return LogicalJoin(
            schema=new_schema,
            estimated_rows=self.estimated_rows,
            estimated_cost=self.estimated_cost,
            left=left_child,
            right=right_child,
            join_type=self.join_type,
            condition=self.condition,
        )

    def node_name(self) -> str:
        return "LogicalJoin"

    def details(self) -> dict[str, Any]:
        return {
            "join_type": self.join_type.value,
            "condition": str(self.condition),
        }


@dataclass
class LogicalAggregate(LogicalNode):
    """
    Unary operator: groups rows by group_by keys and calculates aggregates.
    """
    child: LogicalNode = field(default=None)  # type: ignore
    group_by: list[Expr] = field(default_factory=list)
    aggregates: list[ProjectionItem] = field(default_factory=list)

    @property
    def children(self) -> list[LogicalNode]:
        return [self.child]

    def with_children(self, new_children: list[LogicalNode]) -> LogicalAggregate:
        if len(new_children) != 1:
            raise ValueError("LogicalAggregate requires exactly one child")
        return LogicalAggregate(
            schema=self.schema,
            estimated_rows=self.estimated_rows,
            estimated_cost=self.estimated_cost,
            child=new_children[0],
            group_by=list(self.group_by),
            aggregates=list(self.aggregates),
        )

    def node_name(self) -> str:
        return "LogicalAggregate"

    def details(self) -> dict[str, Any]:
        return {
            "group_by": [str(g) for g in self.group_by],
            "aggregates": [p.alias or str(p.expr) for p in self.aggregates],
        }


@dataclass
class LogicalSort(LogicalNode):
    """
    Unary operator: orders rows by sort expressions.
    """
    child: LogicalNode = field(default=None)  # type: ignore
    sort_items: list[SortItem] = field(default_factory=list)

    @property
    def children(self) -> list[LogicalNode]:
        return [self.child]

    def with_children(self, new_children: list[LogicalNode]) -> LogicalSort:
        if len(new_children) != 1:
            raise ValueError("LogicalSort requires exactly one child")
        return LogicalSort(
            schema=new_children[0].schema,
            estimated_rows=self.estimated_rows,
            estimated_cost=self.estimated_cost,
            child=new_children[0],
            sort_items=list(self.sort_items),
        )

    def node_name(self) -> str:
        return "LogicalSort"

    def details(self) -> dict[str, Any]:
        return {
            "order_by": [f"{item.expr} {item.direction.value}" for item in self.sort_items]
        }


@dataclass
class LogicalLimit(LogicalNode):
    """
    Unary operator: limits the number of rows produced, with optional offset.
    """
    child: LogicalNode = field(default=None)  # type: ignore
    limit: int = 0
    offset: int | None = None

    @property
    def children(self) -> list[LogicalNode]:
        return [self.child]

    def with_children(self, new_children: list[LogicalNode]) -> LogicalLimit:
        if len(new_children) != 1:
            raise ValueError("LogicalLimit requires exactly one child")
        return LogicalLimit(
            schema=new_children[0].schema,
            estimated_rows=self.estimated_rows,
            estimated_cost=self.estimated_cost,
            child=new_children[0],
            limit=self.limit,
            offset=self.offset,
        )

    def node_name(self) -> str:
        return "LogicalLimit"

    def details(self) -> dict[str, Any]:
        return {"limit": self.limit, "offset": self.offset}
