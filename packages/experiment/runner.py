"""
experiment.runner — Core execution orchestrator for query optimization experiments.

Coordinates:
1. Environment setup and dataset preparation
2. Repetitions and warm-up cycles
3. Metrics collection (planning time, execution time, q-error, plan cost)
4. Statistical aggregation and provenance persistence
"""
from __future__ import annotations

import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmark.generator import DatasetGenerator
from benchmark.workloads import create_workload, get_standard_tables
from common.errors import QueryOptError
from common.logging import get_logger
from common.types import ExecutionStatus
from experiment.config import ExperimentConfig
from experiment.metrics import (
    MetricSummary,
    QueryRunMetric,
    QuerySummary,
    calculate_q_error,
)
from experiment.storage import ResultWriter, build_reproducibility_record
from queryopt.engine import QueryEngine
from storage.table import Database

logger = get_logger(__name__)


@dataclass
class ExperimentResult:
    """Complete results of an experiment execution."""
    experiment_id: str
    name: str
    output_dir: Path
    raw_metrics: list[QueryRunMetric] = field(default_factory=list)
    summaries: list[QuerySummary] = field(default_factory=list)
    reproducibility_record: dict[str, Any] = field(default_factory=dict)
    total_duration_seconds: float = 0.0


class ExperimentRunner:
    """
    Executes an experiment configuration against the QueryOpt engine.
    """

    def __init__(self, base_data_dir: Path | None = None) -> None:
        self.base_data_dir = base_data_dir

    def run(self, config: ExperimentConfig) -> ExperimentResult:
        start_wall_time = time.perf_counter()
        experiment_id = f"exp_{uuid.uuid4().hex[:8]}"
        logger.info(f"Starting experiment '{config.experiment.name}' [{experiment_id}]")

        # 1. Prepare data directory and engine
        temp_dir: tempfile.TemporaryDirectory | None = None
        if self.base_data_dir is not None:
            data_dir = self.base_data_dir / experiment_id
            data_dir.mkdir(parents=True, exist_ok=True)
        else:
            temp_dir = tempfile.TemporaryDirectory(prefix="queryopt_exp_")
            data_dir = Path(temp_dir.name)

        db_name = "benchmark_db"
        engine = QueryEngine(data_dir=data_dir, db_name=db_name)

        # 2. Populate dataset
        self._prepare_dataset(engine.db, config)

        # 3. Resolve queries to run
        queries = self._resolve_queries(config)

        raw_metrics: list[QueryRunMetric] = []
        summaries: list[QuerySummary] = []

        # Fail fast on unknown optimizer ids instead of silently substituting one.
        for opt_id in config.optimizers:
            engine.get_optimizer(opt_id)

        # 4. Matrix execution: query x optimizer
        for q in queries:
            for opt_id in config.optimizers:
                logger.info(f"Running query '{q.id}' on optimizer '{opt_id}'...")
                if config.execution.feedback == "isolated":
                    engine.feedback_store.reset()

                # Warmup runs (discarded from statistical summaries)
                for w in range(config.execution.warmup_runs):
                    self._execute_single_run(
                        engine=engine,
                        experiment_id=experiment_id,
                        query_id=q.id,
                        sql=q.sql,
                        optimizer_id=opt_id,
                        repetition=w + 1,
                        is_warmup=True,
                    )

                # Measured repetitions
                measured_metrics: list[QueryRunMetric] = []
                for rep in range(1, config.execution.repetitions + 1):
                    m = self._execute_single_run(
                        engine=engine,
                        experiment_id=experiment_id,
                        query_id=q.id,
                        sql=q.sql,
                        optimizer_id=opt_id,
                        repetition=rep,
                        is_warmup=False,
                    )
                    raw_metrics.append(m)
                    measured_metrics.append(m)

                # Compute summary for this (query, optimizer)
                if measured_metrics:
                    exec_times = [m.execution_time_ms for m in measured_metrics if m.status == "SUCCESS"]
                    plan_times = [m.planning_time_ms for m in measured_metrics if m.status == "SUCCESS"]
                    tot_times = [m.total_time_ms for m in measured_metrics if m.status == "SUCCESS"]

                    rep_last = measured_metrics[-1]
                    summary = QuerySummary(
                        query_id=q.id,
                        optimizer_id=opt_id,
                        execution_time_summary=MetricSummary.compute(exec_times),
                        planning_time_summary=MetricSummary.compute(plan_times),
                        total_time_summary=MetricSummary.compute(tot_times),
                        estimated_cost=rep_last.estimated_cost,
                        estimated_rows=rep_last.estimated_rows,
                        actual_rows=rep_last.actual_rows,
                        q_error=rep_last.q_error,
                        rules_applied=rep_last.rules_applied,
                        alternatives_considered=rep_last.alternatives_considered,
                    )
                    summaries.append(summary)

        total_duration = time.perf_counter() - start_wall_time

        # 5. Persist results
        out_dir = Path(config.result_storage.directory) / config.experiment.name
        writer = ResultWriter(output_dir=out_dir, overwrite=config.result_storage.overwrite)

        if config.result_storage.save_raw:
            writer.write_raw_metrics(raw_metrics, config.result_storage.formats)
        if config.result_storage.save_summary:
            writer.write_summary(summaries, config.result_storage.formats)

        reproducibility = build_reproducibility_record(
            experiment_id=experiment_id,
            experiment_name=config.experiment.name,
            config_dict=self._provenance(config),
        )
        writer.write_reproducibility(reproducibility)

        logger.info(f"Experiment '{config.experiment.name}' completed in {total_duration:.2f}s. Results in {out_dir}")

        return ExperimentResult(
            experiment_id=experiment_id,
            name=config.experiment.name,
            output_dir=out_dir,
            raw_metrics=raw_metrics,
            summaries=summaries,
            reproducibility_record=reproducibility,
            total_duration_seconds=total_duration,
        )

    @staticmethod
    def _provenance(config: ExperimentConfig) -> dict[str, Any]:
        cfg: dict[str, Any] = {
            "optimizers": config.optimizers,
            "dataset": config.dataset.__dict__,
            "execution": config.execution.__dict__,
        }
        if any(o.startswith("llm") for o in config.optimizers):
            # LLM answers are replayed from the response cache, but record which
            # model produced them so results can be attributed and re-run.
            from ai.llm import get_llm_client

            cfg["llm"] = get_llm_client().describe()
        return cfg

    def _prepare_dataset(self, db: Database, config: ExperimentConfig) -> None:
        """Populate database using workload or generator configuration."""
        generator = DatasetGenerator(seed=config.dataset.seed)

        if config.dataset.workload_id:
            workload = create_workload(config.dataset.workload_id, scale=config.dataset.scale)
            table_specs = workload.table_specs
        else:
            table_specs = get_standard_tables(scale=config.dataset.scale)

        generator.populate_database(db, table_specs, collect_statistics=True)

    def _resolve_queries(self, config: ExperimentConfig) -> list[Any]:
        """Collect queries from config or workload."""
        if config.queries:
            return config.queries
        if config.dataset.workload_id:
            wl = create_workload(config.dataset.workload_id, scale=config.dataset.scale)
            from experiment.config import QueryConfig
            return [QueryConfig(id=q.query_id, sql=q.sql, description=q.description) for q in wl.queries]
        return []

    def _execute_single_run(
        self,
        engine: QueryEngine,
        experiment_id: str,
        query_id: str,
        sql: str,
        optimizer_id: str,
        repetition: int,
        is_warmup: bool,
    ) -> QueryRunMetric:
        """Execute one query with a given optimizer, capturing metrics."""
        # 1. Explain phase (get estimates and plan stats)
        estimated_cost = 0.0
        rules_applied: list[str] = []
        alternatives = 1
        try:
            exp_info = engine.explain(sql, optimizer_id=optimizer_id)
            estimated_cost = float(exp_info.get("estimated_cost") or 0.0)
            rules_applied = list(exp_info.get("rules_applied") or [])
            alternatives = int(exp_info.get("alternatives_considered") or 1)
        except Exception:
            pass

        # 2. Execution phase
        start = time.perf_counter()
        result = engine.execute(sql, optimizer_id=optimizer_id)
        wall_time_ms = (time.perf_counter() - start) * 1000.0

        actual_rows = len(result.rows) if result.status == ExecutionStatus.SUCCESS else 0
        est_rows = 0.0
        if result.operator_stats:
            root_stat = next(iter(result.operator_stats.values()))
            est_rows = float(root_stat.estimated_rows or 0.0)

        q_err = calculate_q_error(est_rows, actual_rows)

        return QueryRunMetric(
            experiment_id=experiment_id,
            query_id=query_id,
            optimizer_id=optimizer_id,
            repetition=repetition,
            is_warmup=is_warmup,
            planning_time_ms=result.planning_time_ms,
            execution_time_ms=result.execution_time_ms,
            total_time_ms=wall_time_ms,
            estimated_cost=estimated_cost,
            estimated_rows=est_rows,
            actual_rows=actual_rows,
            q_error=q_err,
            status=result.status.name,
            error_message=result.error_message,
            rules_applied=rules_applied,
            alternatives_considered=alternatives,
        )
