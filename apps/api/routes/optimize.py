"""apps.api.routes.optimize — the product endpoint: SQL in, optimized SQL + reasons out."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from sqlopt import DIALECTS, optimize_sql
from sqlopt.ai import add_ai_pass, ai_available

router = APIRouter(tags=["optimize"])


class OptimizeRequest(BaseModel):
    sql: str = Field(..., max_length=100_000)
    dialect: str = "postgres"
    use_ai: bool = True
    schema_ddl: str = Field("", max_length=200_000, description="Optional CREATE TABLE / CREATE INDEX statements")


@router.get("/optimize/meta")
def meta() -> dict[str, Any]:
    return {"dialects": DIALECTS, "ai": ai_available()}


@router.post("/optimize")
def optimize(req: OptimizeRequest) -> dict[str, Any]:
    result = optimize_sql(req.sql, req.dialect, req.schema_ddl)
    if req.use_ai:
        add_ai_pass(result)
    return result.to_dict()
