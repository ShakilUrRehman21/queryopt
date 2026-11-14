"""
optimizer.physical_planner — Compiles a LogicalPlan into an executable PhysicalPlan.

Maps each logical algebraic operator to a concrete Volcano physical operator
using the database catalog to bind table handles and schemas.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from common.errors import PlanningError
from common.types import JoinType
from executor.aggregate import HashAggregate
from executor.base import PhysicalOperator
from executor.filter import Filter
from executor.join import HashJoin, NestedLoopJoin, _extract_equi_join_keys
from executor.limit import Limit
from executor.project import Project
from executor.scan import SeqScan
from executor.sort import Sort
from planner.nodes import (
    LogicalAggregate,
    LogicalFilter,
    LogicalJoin,
    LogicalLimit,
    LogicalNode,
    LogicalProject,
    LogicalScan,
    LogicalSort,
)

if TYPE_CHECKING:
    from storage.catalog import Catalog
    from storage.page_store import PageStore
    from storage.table import TableHandle


def _get_logical_table_names(node: LogicalNode) -> set[str]:
    names: set[str] = set()
    if isinstance(node, LogicalScan):
        if node.table_name:
            names.add(node.table_name.lower())
        if node.alias:
            names.add(node.alias.lower())
    for child in node.children:
        names.update(_get_logical_table_names(child))
    return names


class PhysicalPlanner:
    """
    Translates LogicalNode trees to PhysicalOperator trees.
    """

    def __init__(self, catalog: Catalog, page_store: PageStore, db_name: str = "default") -> None:
        self.catalog = catalog
        self.page_store = page_store
        self.db_name = db_name

    def plan(self, logical_node: LogicalNode) -> PhysicalOperator:
        """
        Recursively compile a LogicalNode tree into a PhysicalOperator tree.
        """
        match logical_node:
            case LogicalScan():
                from storage.table import TableHandle
                table_meta = self.catalog.get_table(logical_node.table_name)
                table_handle = TableHandle(table_meta, self.page_store, self.catalog)
                return SeqScan(
                    table_handle=table_handle,
                    schema=logical_node.schema,
                    projection_columns=logical_node.projection_columns,
                    estimated_rows=logical_node.estimated_rows or 0.0,
                    alias=logical_node.alias,
                )

            case LogicalFilter():
                child_op = self.plan(logical_node.child)
                return Filter(
                    child=child_op,
                    predicate=logical_node.predicate,
                    estimated_rows=logical_node.estimated_rows or 0.0,
                )

            case LogicalProject():
                child_op = self.plan(logical_node.child)
                return Project(
                    child=child_op,
                    projections=logical_node.projections,
                    schema=logical_node.schema,
                    estimated_rows=logical_node.estimated_rows or 0.0,
                )

            case LogicalJoin():
                left_op = self.plan(logical_node.left)
                right_op = self.plan(logical_node.right)

                left_tables = _get_logical_table_names(logical_node.left)
                right_tables = _get_logical_table_names(logical_node.right)

                # Check if HashJoin is applicable (equi-join)
                equi_keys = _extract_equi_join_keys(
                    logical_node.condition, left_op.schema, right_op.schema, left_tables, right_tables
                )
                if equi_keys is not None and logical_node.join_type in (JoinType.INNER, JoinType.LEFT):
                    return HashJoin(
                        left=left_op,
                        right=right_op,
                        condition=logical_node.condition,
                        schema=logical_node.schema,
                        join_type=logical_node.join_type,
                        estimated_rows=logical_node.estimated_rows or 0.0,
                    )
                else:
                    return NestedLoopJoin(
                        left=left_op,
                        right=right_op,
                        condition=logical_node.condition,
                        schema=logical_node.schema,
                        join_type=logical_node.join_type,
                        estimated_rows=logical_node.estimated_rows or 0.0,
                    )

            case LogicalAggregate():
                child_op = self.plan(logical_node.child)
                return HashAggregate(
                    child=child_op,
                    group_by=logical_node.group_by,
                    aggregates=logical_node.aggregates,
                    schema=logical_node.schema,
                    estimated_rows=logical_node.estimated_rows or 0.0,
                )

            case LogicalSort():
                child_op = self.plan(logical_node.child)
                return Sort(
                    child=child_op,
                    sort_items=logical_node.sort_items,
                    estimated_rows=logical_node.estimated_rows or 0.0,
                )

            case LogicalLimit():
                child_op = self.plan(logical_node.child)
                return Limit(
                    child=child_op,
                    limit=logical_node.limit,
                    offset=logical_node.offset,
                    estimated_rows=logical_node.estimated_rows or 0.0,
                )

            case _:
                raise PlanningError(f"No physical mapping for logical node: {type(logical_node).__name__}")
