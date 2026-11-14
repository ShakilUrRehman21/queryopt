"""
apps.api.routes.benchmark — Benchmark workloads and dataset generation endpoints.
"""
from __future__ import annotations

from fastapi import APIRouter

from apps.api.models import GenerateWorkloadRequest, WorkloadInfo
from benchmark.generator import DatasetGenerator
from benchmark.workloads import WORKLOAD_IDS, create_workload
from queryopt.engine import QueryEngine

router = APIRouter(prefix="/benchmark", tags=["benchmark"])


@router.get("/workloads", response_model=list[WorkloadInfo])
def list_workloads() -> list[WorkloadInfo]:
    """Returns the catalog of standard benchmark workloads (W1–W9)."""
    workloads = []
    for wid in WORKLOAD_IDS:
        wl = create_workload(wid)
        workloads.append(
            WorkloadInfo(
                workload_id=wl.workload_id,
                name=wl.name,
                description=wl.description,
                table_names=[t.name for t in wl.table_specs],
                query_count=len(wl.queries),
            )
        )
    return workloads


@router.post("/generate")
def generate_workload(req: GenerateWorkloadRequest) -> dict[str, str]:
    """Populates a database with the specified workload tables and collects statistics."""
    wl = create_workload(req.workload_id, scale=req.scale)
    engine = QueryEngine(db_name=req.db_name)

    gen = DatasetGenerator(seed=req.seed)
    gen.populate_database(engine.db, wl.table_specs, collect_statistics=True)

    return {
        "status": "SUCCESS",
        "message": f"Successfully generated workload {wl.workload_id} ({wl.name}) into database '{req.db_name}'",
    }
