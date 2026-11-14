"""
apps.api.models — Pydantic schemas for the QueryOpt Lab REST API.
"""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class QueryRunRequest(BaseModel):
    sql: str = Field(..., description="SQL query string to execute")
    optimizer_id: str = Field(
        "cost_based",
        description=(
            "Optimizer id or parametrised spec: naive, rule_based, cost_based, adaptive, "
            "learned, llm_guided, e.g. cost_based[qerr=10,mode=under]"
        ),
    )
    db_name: str = Field("default", description="Target database name")


class ColumnInfo(BaseModel):
    name: str
    data_type: str
    nullable: bool = True
    primary_key: bool = False


class QueryRunResponse(BaseModel):
    status: str
    columns: list[str] = Field(default_factory=list)
    rows: list[list[Any]] = Field(default_factory=list)
    rows_returned: int = 0
    execution_time_ms: float = 0.0
    planning_time_ms: float = 0.0
    error_message: str | None = None
    optimizer_metadata: dict[str, Any] = Field(default_factory=dict)


class GraphNode(BaseModel):
    id: str
    label: str
    operator_type: str
    estimated_rows: float | None = None
    estimated_cost: float | None = None
    actual_rows: int | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    id: str
    source: str
    target: str


class PlanGraph(BaseModel):
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)


class QueryExplainResponse(BaseModel):
    status: str
    optimizer_id: str
    logical_plan_ascii: str
    optimized_plan_ascii: str
    physical_plan_ascii: str
    rules_applied: list[str] = Field(default_factory=list)
    estimated_cost: float | None = None
    estimated_rows: float | None = None
    alternatives_considered: int = 1
    logical_graph: PlanGraph
    physical_graph: PlanGraph
    error_message: str | None = None
    optimizer_metadata: dict[str, Any] = Field(default_factory=dict)


class OptimizerCompareRequest(BaseModel):
    sql: str
    db_name: str = "default"
    optimizers: list[str] = Field(
        default_factory=lambda: ["naive", "rule_based", "cost_based"],
        description="Optimizer ids/specs to race on the same query.",
    )


class OptimizerEvaluation(BaseModel):
    optimizer_id: str
    status: str
    execution_time_ms: float
    planning_time_ms: float
    total_time_ms: float
    rows_returned: int
    estimated_cost: float | None = None
    estimated_rows: float | None = None
    rules_applied: list[str] = Field(default_factory=list)
    physical_plan_ascii: str
    graph: PlanGraph
    error_message: str | None = None
    optimizer_metadata: dict[str, Any] = Field(default_factory=dict)


class OptimizerCompareResponse(BaseModel):
    sql: str
    evaluations: list[OptimizerEvaluation]
    results_match: bool


class TableSchemaInfo(BaseModel):
    name: str
    row_count: int
    columns: list[ColumnInfo]


class DatabaseSchemaResponse(BaseModel):
    db_name: str
    tables: list[TableSchemaInfo]


class WorkloadInfo(BaseModel):
    workload_id: str
    name: str
    description: str
    table_names: list[str]
    query_count: int


class GenerateWorkloadRequest(BaseModel):
    workload_id: str = "W1"
    scale: int = 1
    db_name: str = "default"
    seed: int = 42


class ExperimentRunApiRequest(BaseModel):
    config_yaml: str | None = None
    config_name: str | None = None


class QueryRewriteRequest(BaseModel):
    sql: str = Field(..., description="Raw unoptimized SQL query")
    dialect: str = Field("postgres", description="SQL dialect (postgres, mysql, sqlite, etc.)")


class AntiPatternModel(BaseModel):
    id: str
    title: str
    severity: str
    description: str
    impact: str


class OptimizationActionModel(BaseModel):
    rule_name: str
    summary: str
    before_snippet: str
    after_snippet: str
    explanation: str


class IndexRecommendationModel(BaseModel):
    table: str
    columns: list[str]
    ddl: str
    reason: str


class QueryRewriteResponse(BaseModel):
    original_sql: str
    optimized_sql: str
    has_improvements: bool
    anti_patterns: list[AntiPatternModel] = Field(default_factory=list)
    optimizations: list[OptimizationActionModel] = Field(default_factory=list)
    index_recommendations: list[IndexRecommendationModel] = Field(default_factory=list)
    score_before: int
    score_after: int
    summary_heading: str



# ── AI Lab ──────────────────────────────────────────────────────────────────


class AIStatusResponse(BaseModel):
    provider: str
    model: str
    available: bool
    optimizers: list[dict[str, str]]
    features: dict[str, bool]


class AIExplainRequest(BaseModel):
    sql: str
    optimizer_id: str = "cost_based"
    db_name: str = "default"
    question: str | None = None
    use_llm: bool = True


class AIExplainResponse(BaseModel):
    status: str
    facts: dict[str, Any]
    narrative: dict[str, Any]
    llm_narrative: dict[str, Any] | None = None
    llm_error: str | None = None
    graph: PlanGraph
    columns: list[str] = Field(default_factory=list)
    rows: list[list[Any]] = Field(default_factory=list)


class NL2SQLRequest(BaseModel):
    question: str
    db_name: str = "default"
    execute: bool = True
    optimizer_id: str = "cost_based"


class RewriteVerifyRequest(BaseModel):
    sql: str
    candidate_sql: str | None = Field(
        None, description="Rewrite to verify. If omitted, one is proposed (LLM, else heuristic)."
    )
    source: str = Field("auto", description="auto | llm | heuristic (when candidate_sql is omitted)")
    optimizer_id: str = "cost_based"
    repetitions: int = Field(5, ge=1, le=50)
    db_name: str = "default"


class LearnedTrainRequest(BaseModel):
    tables: list[str] | None = None
    db_name: str = "default"


class ArenaRequest(BaseModel):
    sql: str
    db_name: str = "default"
    optimizers: list[str] | None = Field(
        None, description="Defaults to every optimizer family (classical, learning, AI)."
    )
