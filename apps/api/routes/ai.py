"""
apps.api.routes.ai — AI Lab endpoints.

    GET    /api/ai/status            which model (if any) is configured; optimizer catalogue
    POST   /api/ai/explain           run a query and explain its plan (facts + narrative [+ LLM])
    POST   /api/ai/arena             race every optimizer family on one query, oracle-checked
    POST   /api/ai/nl2sql            question → validated SQL (→ optional execution)
    POST   /api/ai/rewrite/verify    propose and/or empirically verify a rewrite
    GET    /api/ai/feedback          adaptive optimizer's observed-cardinality store
    DELETE /api/ai/feedback          reset it (start a fresh RQ4 run)
    GET    /api/ai/learned           learned-estimator models and their held-out accuracy
    POST   /api/ai/learned/train     (re)train learned-estimator models
"""
from __future__ import annotations

import json
import time
from typing import Any

from fastapi import APIRouter, HTTPException

from ai.llm import LLMCallError, LLMUnavailableError, get_llm_client
from ai.narrator import collect_facts, llm_narrate, narrate, q_error
from apps.api.models import (
    AIExplainRequest,
    AIExplainResponse,
    AIStatusResponse,
    ArenaRequest,
    LearnedTrainRequest,
    NL2SQLRequest,
    RewriteVerifyRequest,
)
from apps.api.serializers import physical_plan_to_graph
from common.types import ExecutionStatus
from optimizer.registry import list_optimizers
from queryopt.engine import QueryEngine

router = APIRouter(prefix="/ai", tags=["ai"])

ARENA_DEFAULT = ["naive", "rule_based", "cost_based", "adaptive", "learned", "llm_guided"]


def _j(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=str))


def _optimize_and_run(engine: QueryEngine, sql: str, optimizer_id: str):
    try:
        t0 = time.perf_counter()
        opt_result = engine.optimize_select(sql, optimizer_id)
        plan_ms = (time.perf_counter() - t0) * 1000.0
    except ValueError as e:  # unknown optimizer
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=getattr(e, "message", None) or str(e))
    return opt_result, engine.run_optimized(opt_result, plan_ms)


@router.get("/status", response_model=AIStatusResponse)
def ai_status() -> AIStatusResponse:
    client = get_llm_client()
    return AIStatusResponse(
        provider=client.provider,
        model=client.model,
        available=client.available,
        optimizers=list_optimizers(),
        features={
            "plan_narrator": True,
            "llm_narrator": client.available,
            "nl2sql": client.available,
            "llm_rewrites": client.available,
            "heuristic_rewrites": True,
            "llm_join_ordering": client.available,
            "adaptive_feedback": True,
            "learned_estimator": True,
        },
    )


@router.post("/explain", response_model=AIExplainResponse)
def explain(req: AIExplainRequest) -> AIExplainResponse:
    engine = QueryEngine(db_name=req.db_name)
    opt_result, res = _optimize_and_run(engine, req.sql, req.optimizer_id)
    facts = collect_facts(req.sql, opt_result, res)
    llm_text, llm_error = None, None
    if req.use_llm:
        try:
            llm_text = llm_narrate(facts, req.question)
        except LLMUnavailableError:
            llm_error = "No language model configured — showing the engine's own explanation."
        except LLMCallError as e:
            llm_error = str(e)
    return AIExplainResponse(
        status=res.status.name,
        facts=_j(facts.to_dict()),
        narrative=narrate(facts),
        llm_narrative=llm_text,
        llm_error=llm_error,
        graph=physical_plan_to_graph(opt_result.physical_plan),
        columns=[c.name for c in res.schema.columns] if res.schema else [],
        rows=[list(r) for r in res.rows[:50]],
    )


@router.post("/arena")
def arena(req: ArenaRequest) -> dict[str, Any]:
    """Every optimizer family on the same query, data and executor."""
    engine = QueryEngine(db_name=req.db_name)
    specs = req.optimizers or ARENA_DEFAULT
    entries: list[dict[str, Any]] = []
    rowsets: list[list[tuple]] = []
    for spec in specs:
        opt_result, res = _optimize_and_run(engine, req.sql, spec)
        facts = collect_facts(req.sql, opt_result, res)
        ok = res.status == ExecutionStatus.SUCCESS
        if ok:
            rowsets.append(sorted(res.rows, key=repr))
        root_q = q_error(opt_result.physical_plan.estimated_rows or 0.0, res.rows_returned)
        entries.append(
            {
                "optimizer_id": spec,
                "status": res.status.name,
                "error_message": res.error_message,
                "execution_time_ms": round(res.execution_time_ms, 3),
                "planning_time_ms": round(res.planning_time_ms, 3),
                "total_time_ms": round(res.execution_time_ms + res.planning_time_ms, 3),
                "rows_returned": res.rows_returned,
                "estimated_cost": round(opt_result.estimated_cost, 4),
                "root_q_error": round(root_q, 3) if facts.has_estimates else None,
                "max_q_error": (facts.worst.q_error if facts.worst else 1.0) if facts.has_estimates else None,
                "join_order": facts.join_order,
                "join_algorithms": facts.join_algorithms,
                "metadata": _j(opt_result.metadata),
                "graph": physical_plan_to_graph(opt_result.physical_plan).model_dump(),
            }
        )
    results_match = len(rowsets) == len(specs) and all(r == rowsets[0] for r in rowsets[1:])
    ok_entries = [e for e in entries if e["status"] == "SUCCESS"]
    winner = min(ok_entries, key=lambda e: e["execution_time_ms"]) if ok_entries else None
    estimating = [e for e in ok_entries if e["max_q_error"] is not None]
    best_est = min(estimating, key=lambda e: e["max_q_error"]) if estimating else None
    distinct_plans = len({tuple(e["join_order"]) for e in ok_entries if e["join_order"]})
    headline = []
    if winner:
        headline.append(
            f"Fastest execution: {winner['optimizer_id']} ({winner['execution_time_ms']:.2f} ms)."
        )
    if best_est:
        headline.append(
            f"Most accurate estimates: {best_est['optimizer_id']} (worst q-error {best_est['max_q_error']:.2f})."
        )
    if distinct_plans:
        headline.append(f"{distinct_plans} distinct join order(s) across optimizers.")
    headline.append(
        "All optimizers returned identical results." if results_match
        else "WARNING: results differ between optimizers — see the oracle check."
    )
    return {
        "sql": req.sql,
        "entries": entries,
        "results_match": results_match,
        "winner": winner["optimizer_id"] if winner else None,
        "headline": " ".join(headline),
        "timing_note": (
            "Single run per optimizer, so small differences are noise; use Experiments for repeated, "
            "warmed-up measurements. Planning time includes one-off model training for 'learned' "
            "the first time a table is seen."
        ),
    }


@router.post("/nl2sql")
def nl2sql(req: NL2SQLRequest) -> dict[str, Any]:
    from ai.nl2sql import nl_to_sql

    engine = QueryEngine(db_name=req.db_name)
    try:
        out = nl_to_sql(engine, req.question)
    except LLMUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except LLMCallError as e:
        raise HTTPException(status_code=502, detail=str(e))
    if out["valid"] and req.execute:
        res = engine.execute(out["sql"], optimizer_id=req.optimizer_id)
        out["result"] = {
            "status": res.status.name,
            "columns": [c.name for c in res.schema.columns] if res.schema else [],
            "rows": [list(r) for r in res.rows[:100]],
            "rows_returned": res.rows_returned,
            "execution_time_ms": round(res.execution_time_ms, 3),
            "error_message": res.error_message,
        }
    return _j(out)


@router.post("/rewrite/verify")
def rewrite_verify(req: RewriteVerifyRequest) -> dict[str, Any]:
    from ai.rewrite_verifier import heuristic_rewrite, propose_rewrite, verify_rewrite

    engine = QueryEngine(db_name=req.db_name)
    proposal: dict[str, Any]
    if req.candidate_sql:
        proposal = {"sql": req.candidate_sql, "rationale": "Provided by user.", "source": "user"}
    else:
        use_llm = req.source == "llm" or (req.source == "auto" and get_llm_client().available)
        proposal = {}
        if use_llm:
            try:
                proposal = propose_rewrite(engine, req.sql)
            except LLMUnavailableError as e:
                if req.source == "llm":
                    raise HTTPException(status_code=503, detail=str(e))
            except LLMCallError as e:
                if req.source == "llm":
                    raise HTTPException(status_code=502, detail=str(e))
        if not proposal:
            proposal = heuristic_rewrite(req.sql)
    verification = verify_rewrite(
        engine, req.sql, proposal["sql"], optimizer_id=req.optimizer_id, repetitions=req.repetitions
    )
    return _j({"original_sql": req.sql, "proposal": proposal, "verification": verification})


@router.get("/feedback")
def feedback(db_name: str = "default", limit: int = 50) -> dict[str, Any]:
    engine = QueryEngine(db_name=db_name, record_feedback=False)
    return engine.feedback_store.summary(limit=limit)


@router.delete("/feedback")
def reset_feedback(db_name: str = "default") -> dict[str, str]:
    engine = QueryEngine(db_name=db_name, record_feedback=False)
    engine.feedback_store.reset()
    return {"status": "SUCCESS", "message": "Cardinality feedback cleared."}


@router.get("/learned")
def learned_models(db_name: str = "default") -> dict[str, Any]:
    from query_stats.learned import TableSelectivityModel

    engine = QueryEngine(db_name=db_name, record_feedback=False)
    out = {}
    for t in engine.db.catalog.list_tables():
        m = TableSelectivityModel.load(engine.db.db_dir, t.lower())
        if m is not None:
            out[t] = {"columns": m.columns, "row_count": m.row_count, "metrics": m.metrics}
    return {"models": out}


@router.post("/learned/train")
def train_learned(req: LearnedTrainRequest) -> dict[str, Any]:
    from optimizer.optimizer_learned import ensure_model

    engine = QueryEngine(db_name=req.db_name, record_feedback=False)
    tables = req.tables or engine.db.catalog.list_tables()
    out: dict[str, Any] = {}
    for t in tables:
        if not engine.db.catalog.has_table(t):
            raise HTTPException(status_code=404, detail=f"Unknown table '{t}'")
        m = ensure_model(engine.db.catalog, engine.db._page_store, t, force=True)
        out[t] = m.metrics if m else {"skipped": "no numeric columns or too few rows"}
    return {"models": _j(out)}
