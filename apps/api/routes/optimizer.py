"""
apps.api.routes.optimizer — Side-by-side optimizer comparison endpoints.
"""
from __future__ import annotations

import json
import time

from fastapi import APIRouter, HTTPException

from apps.api.models import (
    OptimizerCompareRequest,
    OptimizerCompareResponse,
    OptimizerEvaluation,
)
from apps.api.serializers import physical_plan_to_graph
from common.types import ExecutionStatus
from parser.parser import parse
from parser.validator import validate
from planner.builder import build_logical_plan
from queryopt.engine import QueryEngine

router = APIRouter(prefix="/optimizer", tags=["optimizer"])


@router.post("/compare", response_model=OptimizerCompareResponse)
def compare_optimizers(req: OptimizerCompareRequest) -> OptimizerCompareResponse:
    """
    Executes the query across Naive, Rule-Based, and Cost-Based optimizers side-by-side.
    Returns plan details, performance metrics, and graphs for comparison.
    """
    engine = QueryEngine(db_name=req.db_name)

    try:
        parse_result = parse(req.sql)
        stmt = parse_result.statement
        validate(stmt, engine.db.catalog, req.db_name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    optimizers_to_run = req.optimizers or ["naive", "rule_based", "cost_based"]
    for spec in optimizers_to_run:
        try:
            engine.get_optimizer(spec)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    evaluations: list[OptimizerEvaluation] = []
    result_rows_list: list[list[tuple]] = []

    for opt_id in optimizers_to_run:
        # Re-build fresh logical plan for clean optimization
        logical_plan = build_logical_plan(stmt, engine.db.catalog, req.db_name)
        opt = engine.get_optimizer(opt_id)

        plan_start = time.perf_counter()
        opt_result = opt.optimize(
            logical_plan,
            catalog=engine.db.catalog,
            page_store=engine.db._page_store,
            db_name=req.db_name,
        )
        plan_time_ms = (time.perf_counter() - plan_start) * 1000.0

        res = engine.run_optimized(opt_result, plan_time_ms)

        if res.status == ExecutionStatus.SUCCESS:
            result_rows_list.append(res.rows)
        graph = physical_plan_to_graph(opt_result.physical_plan)

        evaluations.append(
            OptimizerEvaluation(
                optimizer_id=opt_id,
                status=res.status.name,
                execution_time_ms=round(res.execution_time_ms, 3),
                planning_time_ms=round(res.planning_time_ms, 3),
                total_time_ms=round(res.execution_time_ms + res.planning_time_ms, 3),
                rows_returned=res.rows_returned,
                estimated_cost=opt_result.estimated_cost,
                estimated_rows=opt_result.estimated_rows,
                rules_applied=opt_result.rules_applied,
                physical_plan_ascii=str(opt_result.physical_plan),
                graph=graph,
                error_message=res.error_message,
                optimizer_metadata=json.loads(json.dumps(opt_result.metadata, default=str)),
            )
        )

    # Oracle check: do all optimizers produce identical semantic results?
    results_match = len(result_rows_list) == len(optimizers_to_run)
    if len(result_rows_list) > 1:
        first_sorted = sorted(result_rows_list[0], key=repr)
        for other in result_rows_list[1:]:
            if sorted(other, key=repr) != first_sorted:
                results_match = False
                break

    return OptimizerCompareResponse(
        sql=req.sql,
        evaluations=evaluations,
        results_match=results_match,
    )
