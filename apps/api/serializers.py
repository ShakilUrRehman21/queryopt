"""
apps.api.serializers — Plan tree serializers for interactive graph visualization.
"""
from __future__ import annotations

from typing import Any
from apps.api.models import GraphEdge, GraphNode, PlanGraph
from executor.base import PhysicalOperator
from planner.nodes import LogicalNode


def logical_plan_to_graph(root: LogicalNode) -> PlanGraph:
    """Converts a LogicalNode tree into a visual PlanGraph."""
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    counter = 0

    def walk(node: LogicalNode) -> str:
        nonlocal counter
        counter += 1
        node_id = f"node_{counter}"

        try:
            details = node.details()
        except Exception:
            details = {}

        # Ensure all detail values are JSON serializable
        clean_details = {k: str(v) for k, v in details.items()}

        g_node = GraphNode(
            id=node_id,
            label=node.node_name(),
            operator_type=node.node_name(),
            estimated_rows=node.estimated_rows,
            estimated_cost=node.estimated_cost,
            details=clean_details,
        )
        nodes.append(g_node)

        for child in node.children:
            child_id = walk(child)
            edges.append(
                GraphEdge(
                    id=f"edge_{child_id}_{node_id}",
                    source=child_id,
                    target=node_id,
                )
            )

        return node_id

    walk(root)
    return PlanGraph(nodes=nodes, edges=edges)


def physical_plan_to_graph(root: PhysicalOperator) -> PlanGraph:
    """Converts a PhysicalOperator tree into a visual PlanGraph."""
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    counter = 0

    def walk(op: PhysicalOperator) -> str:
        nonlocal counter
        counter += 1
        node_id = f"phys_{counter}"

        details: dict[str, Any] = {}
        for attr in ["table_name", "condition", "predicate", "projections", "limit", "offset", "group_by"]:
            if hasattr(op, attr):
                val = getattr(op, attr)
                if val is not None:
                    details[attr] = str(val)

        g_node = GraphNode(
            id=node_id,
            label=op.name,
            operator_type=op.name,
            estimated_rows=getattr(op, "estimated_rows", None),
            actual_rows=getattr(op, "actual_rows", None),
            details=details,
        )
        nodes.append(g_node)

        for child in op.children:
            child_id = walk(child)
            edges.append(
                GraphEdge(
                    id=f"edge_{child_id}_{node_id}",
                    source=child_id,
                    target=node_id,
                )
            )

        return node_id

    walk(root)
    return PlanGraph(nodes=nodes, edges=edges)
