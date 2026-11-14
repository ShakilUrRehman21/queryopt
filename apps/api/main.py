"""
apps.api.main — FastAPI application entrypoint for QueryOpt Lab.
"""
from __future__ import annotations

import sys
import platform
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.api.routes import ai, benchmark, experiment, optimize, optimizer, query, schema
from common.config import get_config
from common.logging import get_logger
from queryopt.engine import QueryEngine

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Seeds demo tables on startup for any demo schema that is not present yet."""
    logger.info("Initializing QueryOpt Lab API service...")
    try:
        engine = QueryEngine(db_name="default")
        existing = {t.lower() for t in engine.db.catalog.list_tables()}
        from benchmark.generator import DatasetGenerator
        from benchmark.workloads import get_correlated_tables, get_standard_tables

        gen = DatasetGenerator(seed=42)
        # Standard W1–W7 schema plus the correlated W8/W9 schema used by the
        # AI Lab demos. Each group is only seeded if none of its tables exist,
        # so user data is never overwritten.
        for label, specs in (
            ("standard (W1–W7)", get_standard_tables(scale=1)),
            ("correlated (W8–W9)", get_correlated_tables(scale=1)),
        ):
            if not existing & {s.name for s in specs}:
                logger.info(f"Seeding {label} demo tables...")
                gen.populate_database(engine.db, specs, collect_statistics=True)
    except Exception as e:
        logger.warning(f"Could not auto-seed demo tables: {e}")
    yield
    logger.info("Shutting down QueryOpt Lab API service.")


app = FastAPI(
    title="QueryOpt Lab API",
    description="REST API for studying, executing, visualizing, and evaluating query optimization strategies.",
    version="0.2.0",
    lifespan=lifespan,
)

# Enable CORS for local dev servers (Vite: 5173, Next/CRA: 3000, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
app.include_router(query.router, prefix="/api")
app.include_router(optimizer.router, prefix="/api")
app.include_router(schema.router, prefix="/api")
app.include_router(benchmark.router, prefix="/api")
app.include_router(experiment.router, prefix="/api")
app.include_router(ai.router, prefix="/api")
app.include_router(optimize.router, prefix="/api")


@app.get("/health")
def health_check():
    """Health check endpoint providing runtime and system status."""
    return {
        "status": "healthy",
        "service": "queryopt-lab-api",
        "version": "0.2.0",
        "python_version": sys.version,
        "platform": platform.platform(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("apps.api.main:app", host="0.0.0.0", port=8000, reload=True)
