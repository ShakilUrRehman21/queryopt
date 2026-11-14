"""
apps.api.routes.query — SQL query execution and explain endpoints.
"""
from __future__ import annotations

import time
from fastapi import APIRouter, HTTPException

from apps.api.models import (
    AntiPatternModel,
    IndexRecommendationModel,
    OptimizationActionModel,
    QueryExplainResponse,
    QueryRewriteRequest,
    QueryRewriteResponse,
    QueryRunRequest,
    QueryRunResponse,
)
from apps.api.serializers import logical_plan_to_graph, physical_plan_to_graph
from common.types import ExecutionStatus
from packages.optimizer.rewriter import QueryRewriteAdvisor
from parser.parser import parse
from parser.validator import validate
from planner.builder import build_logical_plan
from planner.printer import explain_logical_plan
from queryopt.engine import QueryEngine

router = APIRouter(prefix="/query", tags=["query"])


def _jsonable(obj):
    """Round-trip through json so numpy scalars etc. never break the response."""
    import json

    return json.loads(json.dumps(obj or {}, default=str))


@router.post("/run", response_model=QueryRunResponse)
def run_query(req: QueryRunRequest) -> QueryRunResponse:
    """Executes a SQL query string using the specified optimizer."""
    engine = QueryEngine(db_name=req.db_name)
    result = engine.execute(req.sql, optimizer_id=req.optimizer_id)

    columns = [c.name for c in result.schema.columns] if result.schema else []
    # Convert Row tuples to lists for JSON compatibility
    rows_json = [list(r) for r in result.rows]

    return QueryRunResponse(
        status=result.status.name,
        columns=columns,
        rows=rows_json,
        rows_returned=result.rows_returned,
        execution_time_ms=round(result.execution_time_ms, 3),
        planning_time_ms=round(result.planning_time_ms, 3),
        error_message=result.error_message,
        optimizer_metadata=_jsonable(result.optimizer_metadata),
    )


@router.post("/explain", response_model=QueryExplainResponse)
def explain_query(req: QueryRunRequest) -> QueryExplainResponse:
    """Generates ASCII plan trees, optimization metadata, and visual graphs."""
    engine = QueryEngine(db_name=req.db_name)

    try:
        parse_result = parse(req.sql)
        stmt = parse_result.statement
        validate(stmt, engine.db.catalog, req.db_name)
        logical_plan = build_logical_plan(stmt, engine.db.catalog, req.db_name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        opt = engine.get_optimizer(req.optimizer_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    opt_result = opt.optimize(
        logical_plan,
        catalog=engine.db.catalog,
        page_store=engine.db._page_store,
        db_name=req.db_name,
    )

    logical_graph = logical_plan_to_graph(opt_result.optimized_logical_plan)
    physical_graph = physical_plan_to_graph(opt_result.physical_plan)

    return QueryExplainResponse(
        status="SUCCESS",
        optimizer_id=req.optimizer_id,
        logical_plan_ascii=explain_logical_plan(logical_plan),
        optimized_plan_ascii=explain_logical_plan(opt_result.optimized_logical_plan),
        physical_plan_ascii=str(opt_result.physical_plan),
        rules_applied=opt_result.rules_applied,
        estimated_cost=opt_result.estimated_cost,
        estimated_rows=opt_result.estimated_rows,
        alternatives_considered=opt_result.alternatives_considered,
        logical_graph=logical_graph,
        physical_graph=physical_graph,
        optimizer_metadata=_jsonable(opt_result.metadata),
    )


@router.post("/optimize-rewrite", response_model=QueryRewriteResponse)
def optimize_and_rewrite(req: QueryRewriteRequest) -> QueryRewriteResponse:
    """
    Automated SQL Query Optimizer & Rewrite Advisor.
    Analyzes raw input SQL queries for bottlenecks and produces an optimized rewritten query.
    """
    advisor = QueryRewriteAdvisor()
    res = advisor.analyze_and_rewrite(req.sql, dialect=req.dialect)

    return QueryRewriteResponse(
        original_sql=res.original_sql,
        optimized_sql=res.optimized_sql,
        has_improvements=res.has_improvements,
        anti_patterns=[
            AntiPatternModel(
                id=a.id,
                title=a.title,
                severity=a.severity,
                description=a.description,
                impact=a.impact,
            )
            for a in res.anti_patterns
        ],
        optimizations=[
            OptimizationActionModel(
                rule_name=o.rule_name,
                summary=o.summary,
                before_snippet=o.before_snippet,
                after_snippet=o.after_snippet,
                explanation=o.explanation,
            )
            for o in res.optimizations
        ],
        index_recommendations=[
            IndexRecommendationModel(
                table=idx.table,
                columns=idx.columns,
                ddl=idx.ddl,
                reason=idx.reason,
            )
            for idx in res.index_recommendations
        ],
        score_before=res.score_before,
        score_after=res.score_after,
        summary_heading=res.summary_heading,
    )

