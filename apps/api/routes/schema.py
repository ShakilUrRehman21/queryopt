"""
apps.api.routes.schema — Schema inspection and demo database seeding endpoints.
"""
from __future__ import annotations

from fastapi import APIRouter

from apps.api.models import ColumnInfo, DatabaseSchemaResponse, TableSchemaInfo
from benchmark.generator import DatasetGenerator
from benchmark.workloads import get_standard_tables
from queryopt.engine import QueryEngine

router = APIRouter(prefix="/schema", tags=["schema"])


@router.get("", response_model=DatabaseSchemaResponse)
def get_schema(db_name: str = "default") -> DatabaseSchemaResponse:
    """Returns database schema information including table schemas and row counts."""
    engine = QueryEngine(db_name=db_name)
    catalog = engine.db.catalog

    tables_info: list[TableSchemaInfo] = []
    for t_name in catalog.list_tables():
        meta = catalog.get_table(t_name)
        cols = [
            ColumnInfo(
                name=c.name,
                data_type=c.data_type.value,
                nullable=c.nullable,
                primary_key=c.primary_key,
            )
            for c in meta.schema.columns
        ]
        tables_info.append(
            TableSchemaInfo(
                name=t_name,
                row_count=meta.row_count,
                columns=cols,
            )
        )

    return DatabaseSchemaResponse(
        db_name=db_name,
        tables=tables_info,
    )


@router.post("/seed-demo", response_model=DatabaseSchemaResponse)
def seed_demo_database(db_name: str = "default", scale: int = 1) -> DatabaseSchemaResponse:
    """Populates the database with standard benchmark tables and statistics."""
    engine = QueryEngine(db_name=db_name)
    table_specs = get_standard_tables(scale=scale)

    gen = DatasetGenerator(seed=42)
    gen.populate_database(engine.db, table_specs, collect_statistics=True)

    return get_schema(db_name=db_name)
