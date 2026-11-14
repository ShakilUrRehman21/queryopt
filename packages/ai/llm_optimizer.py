"""
ai.llm_optimizer — LLM-guided join ordering, verified by the engine (RQ5).

Research question 5 asks how a language model's optimization advice compares
with what the engine actually measures. This optimizer makes that a controlled
experiment: everything is identical to `CostBasedOptimizer` — heuristic
rewrites, cardinality estimates, physical operator choice, executor — except
the *join order*, which is proposed by an LLM.

The model never touches execution directly. Its proposal is

1. **validated** — must be a permutation of exactly the relations in the
   join cluster, otherwise we fall back to the DP plan and record why;
2. **costed** — with the same cost model as the DP enumerator, so every
   result carries `estimated_cost_ratio_vs_dp` (model regret, ≥ 1.0);
3. **executed** — by the normal engine, so the oracle result-equivalence
   check and real latencies apply.

All prompts/answers go through the reproducibility cache in `ai.llm`.

Offline stand-in
----------------
Without a configured model the optimizer uses Greedy Operator Ordering
(Fegaras, 1998: repeatedly join the pair with the smallest estimated result).
It is clearly labelled `strategy: greedy_offline` in the metadata, and is a
useful third data point between "DP-optimal" and "LLM intuition".
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ai.llm import LLMCallError, LLMClient, LLMUnavailableError, extract_json, get_llm_client
from common.logging import get_logger
from common.types import DataType, JoinType, Schema
from optimizer.annotator import CardinalityAnnotator
from optimizer.optimizer_cost_based import (
    CostBasedOptimizer,
    _combine_conditions,
    _get_expr_scans,
    _get_node_tables,
    _qualify_scan_node,
)
from optimizer.signature import canonical_expr
from parser.ast_types import Expr, Literal
from planner.nodes import LogicalFilter, LogicalJoin, LogicalNode, LogicalScan

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a query optimizer choosing a join order for a relational engine.
The engine builds LEFT-DEEP plans: the first two relations are joined, then each next
relation is joined to the running result. Every join is a hash join when the condition
is an equality: the NEW relation is the build side (hash table), the running result is
streamed. Cost grows with the size of every intermediate result and of the build inputs,
and joining two relations with no join condition between them is a Cartesian product.
Reply with ONLY a JSON object:
{"order": ["<relation>", ...], "rationale": "<one short sentence>"}
The order must list every relation exactly once, using the names given."""


def _relation_name(node: LogicalNode, i: int) -> str:
    scan = node
    while not isinstance(scan, LogicalScan) and scan.children:
        if len(scan.children) != 1:
            return f"subplan_{i}"
        scan = scan.children[0]
    if isinstance(scan, LogicalScan):
        return (scan.alias or scan.table_name).lower()
    return f"subplan_{i}"


def _filters_of(node: LogicalNode) -> list[str]:
    out = []
    n = node
    while n is not None and not isinstance(n, LogicalScan):
        if isinstance(n, LogicalFilter):
            out.append(canonical_expr(n.predicate))
        n = n.children[0] if len(n.children) == 1 else None
    return out


class LLMGuidedOptimizer(CostBasedOptimizer):
    def __init__(self, client: LLMClient | None = None, model: str | None = None,
                 provider: str | None = None, **kw: Any) -> None:
        super().__init__(**kw)
        self._client = client or get_llm_client(provider=provider, model=model)
        self._llm_meta: dict[str, Any] = {}

    @property
    def optimizer_id(self) -> str:
        return "llm_guided"

    def optimize(self, logical_plan, catalog, page_store, db_name="default"):  # type: ignore[override]
        self._llm_meta = {
            "used": False,
            "provider": self._client.provider,
            "model": self._client.model,
            "fallback_reason": "query has no multi-table join",
        }
        return super().optimize(logical_plan, catalog, page_store, db_name)

    def _result_metadata(self, annotator: CardinalityAnnotator) -> dict:
        meta = super()._result_metadata(annotator)
        meta["llm"] = dict(self._llm_meta)
        return meta

    # ── core ────────────────────────────────────────────────────────────
    def _dp_best_join_order(
        self,
        scans: list[LogicalNode],
        conditions: list[Expr],
        outer_schema: Schema,
        annotator: CardinalityAnnotator,
    ) -> LogicalNode:
        dp_plan = super()._dp_best_join_order(scans, conditions, outer_schema, annotator)
        qualified = [_qualify_scan_node(s) for s in scans]
        names = [_relation_name(s, i) for i, s in enumerate(qualified)]
        if len(set(names)) != len(names):
            names = [f"{n}#{i}" for i, n in enumerate(names)]
        scan_tables = [_get_node_tables(s) for s in qualified]
        scan_cols = [{c.name.lower().split(".")[-1] for c in s.schema.columns} for s in qualified]
        cond_scans = [(c, _get_expr_scans(c, scan_tables, scan_cols)) for c in conditions]

        dp_order = self._leaf_order(dp_plan, qualified)
        _, dp_cost = self._build_left_deep(dp_order, qualified, cond_scans, annotator)

        order: list[int] | None = None
        if self._client.available:
            try:
                order, rationale, prov = self._ask_llm(names, qualified, cond_scans)
                self._llm_meta.update(
                    used=True, strategy="llm", rationale=rationale, fallback_reason=None,
                    order=[names[i] for i in order], **{k: v for k, v in prov.items() if k != "text"},
                )
            except (LLMCallError, LLMUnavailableError, ValueError) as exc:
                self._llm_meta.update(used=False, strategy="dp_fallback", fallback_reason=str(exc)[:300])
                order = None
        else:
            order = self._greedy_order(qualified, cond_scans, annotator)
            self._llm_meta.update(
                used=False,
                strategy="greedy_offline",
                fallback_reason="no LLM configured — used Greedy Operator Ordering as an offline stand-in",
                order=[names[i] for i in order],
            )

        self._llm_meta["dp_order"] = [names[i] for i in dp_order]
        self._llm_meta["dp_estimated_join_cost"] = round(dp_cost, 4)
        if order is None:
            self._llm_meta["order"] = self._llm_meta["dp_order"]
            self._llm_meta["estimated_cost_ratio_vs_dp"] = 1.0
            return dp_plan

        plan, cost = self._build_left_deep(order, qualified, cond_scans, annotator)
        self._llm_meta["proposed_estimated_join_cost"] = round(cost, 4)
        self._llm_meta["estimated_cost_ratio_vs_dp"] = round(cost / dp_cost, 4) if dp_cost > 0 else 1.0
        self._llm_meta["matches_dp"] = order == dp_order
        return plan

    # ── LLM call ────────────────────────────────────────────────────────
    def _ask_llm(
        self, names: list[str], scans: list[LogicalNode], cond_scans: list[tuple[Expr, set[int]]]
    ) -> tuple[list[int], str, dict[str, Any]]:
        lines = ["RELATIONS (estimated rows after their own filters):"]
        for n, s in zip(names, scans):
            filt = _filters_of(s)
            lines.append(
                f"- {n}: ~{(s.estimated_rows or 0):,.0f} rows"
                + (f"; filters: {' AND '.join(filt)}" if filt else "")
            )
        lines.append("JOIN CONDITIONS:")
        for c, inv in cond_scans:
            lines.append(f"- {canonical_expr(c)}   (relations: {', '.join(sorted(names[i] for i in inv))})")
        resp = self._client.complete(SYSTEM_PROMPT, "\n".join(lines), max_tokens=300)
        obj = extract_json(resp.text)
        proposed = [str(x).lower().strip() for x in obj.get("order", [])]
        lookup = {n.lower(): i for i, n in enumerate(names)}
        if sorted(proposed) != sorted(lookup):
            raise ValueError(f"model proposed {proposed}, expected a permutation of {sorted(lookup)}")
        return [lookup[p] for p in proposed], str(obj.get("rationale", ""))[:400], resp.provenance()

    # ── plan construction ───────────────────────────────────────────────
    def _leaf_order(self, plan: LogicalNode, qualified: list[LogicalNode]) -> list[int]:
        """Recover the left-deep order of a DP plan as indices into `qualified`."""
        keys = [frozenset(_get_node_tables(s)) for s in qualified]
        order: list[int] = []

        def visit(n: LogicalNode) -> None:
            if isinstance(n, LogicalJoin):
                visit(n.left)
                visit(n.right)
                return
            k = frozenset(_get_node_tables(n))
            for i, key in enumerate(keys):
                if key == k and i not in order:
                    order.append(i)
                    return

        visit(plan)
        if sorted(order) != list(range(len(qualified))):
            return list(range(len(qualified)))
        return order

    def _build_left_deep(
        self,
        order: list[int],
        scans: list[LogicalNode],
        cond_scans: list[tuple[Expr, set[int]]],
        annotator: CardinalityAnnotator,
    ) -> tuple[LogicalNode, float]:
        plan = scans[order[0]]
        joined = {order[0]}
        used: set[int] = set()
        total = 0.0
        for idx in order[1:]:
            new = scans[idx]
            applicable = [
                (ci, c) for ci, (c, inv) in enumerate(cond_scans)
                if ci not in used and inv and idx in inv and inv <= joined | {idx}
            ]
            connected = any(cond_scans[ci][1] & joined for ci, _ in applicable)
            left_rows = plan.estimated_rows or 100.0
            right_rows = new.estimated_rows or 100.0
            if applicable and connected:
                cond = _combine_conditions([c for _, c in applicable])
                used.update(ci for ci, _ in applicable)
                out_rows = self._join_output_rows(plan, new, cond, left_rows, right_rows, annotator)
                total += self._cost_model_join_cost(left_rows, right_rows, out_rows)
            else:
                cond = Literal(value=True, data_type=DataType.BOOLEAN)
                out_rows = left_rows * right_rows
                total += left_rows * right_rows * 1000.0
            plan = LogicalJoin(
                schema=Schema(list(plan.schema.columns) + list(new.schema.columns)),
                left=plan,
                right=new,
                join_type=JoinType.INNER,
                condition=cond,
                estimated_rows=out_rows,
            )
            joined.add(idx)
        leftover = [c for ci, (c, _inv) in enumerate(cond_scans) if ci not in used]
        if leftover:
            plan = LogicalFilter(
                schema=plan.schema, child=plan, predicate=_combine_conditions(leftover),
                estimated_rows=plan.estimated_rows,
            )
        return plan, total

    def _greedy_order(
        self, scans: list[LogicalNode], cond_scans: list[tuple[Expr, set[int]]],
        annotator: CardinalityAnnotator,
    ) -> list[int]:
        n = len(scans)
        start = min(range(n), key=lambda i: scans[i].estimated_rows or 100.0)
        order, joined = [start], {start}
        cur_rows = scans[start].estimated_rows or 100.0
        while len(order) < n:
            best, best_rows, best_connected = None, float("inf"), False
            for i in range(n):
                if i in joined:
                    continue
                conds = [c for c, inv in cond_scans if inv and i in inv and inv <= joined | {i} and inv & joined]
                r_rows = scans[i].estimated_rows or 100.0
                if conds:
                    rows = annotator.estimator.estimate_join(
                        cur_rows, r_rows, _combine_conditions(conds), annotator.table_stats
                    ).estimated_rows
                else:
                    rows = cur_rows * r_rows
                key = (not bool(conds), rows)
                if best is None or key < (not best_connected, best_rows):
                    best, best_rows, best_connected = i, rows, bool(conds)
            order.append(best)  # type: ignore[arg-type]
            joined.add(best)  # type: ignore[arg-type]
            cur_rows = best_rows
        return order
