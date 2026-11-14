"""
queryopt.engine — Unified query processing engine.

Connects all QueryOpt Lab engine layers:
  SQL String ──► Parser ──► Validator ──► Planner ──► Optimizer ──► Executor ──► QueryResult
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from common.config import get_config
from common.errors import QueryOptError
from common.logging import get_logger
from common.types import Column, DataType, ExecutionStatus, QueryResult, Schema
from executor.executor import VolcanoExecutor
from optimizer.base import BaseOptimizer
from optimizer.optimizer_cost_based import CostBasedOptimizer
from optimizer.optimizer_naive import NaiveOptimizer
from optimizer.optimizer_rule_based import RuleBasedOptimizer
from optimizer.registry import create_optimizer
from parser.ast_types import (
    CreateTableStatement,
    InsertStatement,
    SelectStatement,
)
from parser.parser import parse
from parser.validator import validate
from planner.builder import build_logical_plan
from planner.printer import explain_logical_plan
from storage.table import Database

logger = get_logger(__name__)


class QueryEngine:
    """
    Top-level database engine for executing SQL queries and managing storage.
    """

    def __init__(
        self,
        data_dir: Path | str | None = None,
        db_name: str = "default",
        record_feedback: bool = True,
    ) -> None:
        if data_dir is None:
            data_dir = get_config().data_dir
        self.data_dir = Path(data_dir)
        self.db_name = db_name
        self.db = Database.open_or_create(self.data_dir / self.db_name)
        self.executor = VolcanoExecutor()

        self.optimizers: dict[str, BaseOptimizer] = {
            "rule_based": RuleBasedOptimizer(),
            "naive": NaiveOptimizer(),
            "cost_based": CostBasedOptimizer(),
        }
        # Every successful SELECT feeds observed cardinalities to the adaptive
        # optimizer's store (Phase 6). Cheap; disable for pure timing studies.
        self.record_feedback = record_feedback
        self._feedback_store = None

    # ── Optimizer resolution ────────────────────────────────────────────────
    def get_optimizer(self, spec: str) -> BaseOptimizer:
        """Resolve an optimizer id or parametrised spec (see optimizer.registry)."""
        if spec in self.optimizers:
            return self.optimizers[spec]
        opt = create_optimizer(spec)  # raises ValueError for unknown ids
        self.optimizers[spec] = opt
        return opt

    @property
    def feedback_store(self):
        from optimizer.feedback import CardinalityFeedbackStore

        if self._feedback_store is None:
            self._feedback_store = CardinalityFeedbackStore(self.db.db_dir)
        return self._feedback_store

    def optimize_select(self, sql: str, optimizer_id: str = "rule_based"):
        """Parse, validate, plan and optimize a SELECT; returns OptimizationResult."""
        parse_result = parse(sql)
        stmt = parse_result.statement
        if not isinstance(stmt, SelectStatement):
            raise QueryOptError("Only SELECT statements can be optimized.")
        validate(stmt, self.db.catalog, self.db_name)
        logical_plan = build_logical_plan(stmt, self.db.catalog, self.db_name)
        return self.get_optimizer(optimizer_id).optimize(
            logical_plan,
            catalog=self.db.catalog,
            page_store=self.db._page_store,
            db_name=self.db_name,
        )

    def run_optimized(self, opt_result, planning_time_ms: float = 0.0) -> QueryResult:
        """Execute an already-optimized plan and record cardinality feedback."""
        result = self.executor.execute(opt_result.physical_plan)
        result.planning_time_ms = planning_time_ms
        if self.record_feedback and result.status == ExecutionStatus.SUCCESS:
            try:
                self.feedback_store.record_execution(opt_result.physical_plan)
            except Exception as exc:  # feedback must never break a query
                logger.warning(f"Could not record cardinality feedback: {exc}")
        return result

    def execute(self, sql: str, optimizer_id: str = "rule_based") -> QueryResult:
        """
        Execute a SQL statement from raw string to QueryResult.
        """
        start_time = time.perf_counter()
        try:
            # 1. Parse
            parse_result = parse(sql)
            stmt = parse_result.statement

            # 2. Handle DDL: CREATE TABLE
            if isinstance(stmt, CreateTableStatement):
                validate(stmt, self.db.catalog, self.db_name)
                cols = [
                    Column(
                        name=c.name,
                        data_type=c.data_type,
                        nullable=c.nullable,
                        primary_key=c.primary_key,
                    )
                    for c in stmt.columns
                ]
                self.db.create_table(stmt.table_name, Schema(cols))
                total_time = (time.perf_counter() - start_time) * 1000.0
                return QueryResult(
                    status=ExecutionStatus.SUCCESS,
                    rows=[],
                    schema=None,
                    execution_time_ms=total_time,
                )

            # 3. Handle DML: INSERT
            if isinstance(stmt, InsertStatement):
                validate(stmt, self.db.catalog, self.db_name)
                table = self.db.get_table(stmt.table_name)
                for row_vals in stmt.values_rows:
                    if stmt.columns:
                        val_dict = dict(zip(stmt.columns, row_vals))
                        table.insert(val_dict)
                    else:
                        table.insert(row_vals)
                total_time = (time.perf_counter() - start_time) * 1000.0
                return QueryResult(
                    status=ExecutionStatus.SUCCESS,
                    rows=[],
                    schema=None,
                    execution_time_ms=total_time,
                    rows_returned=len(stmt.values_rows),
                )

            # 4. Handle DQL: SELECT
            if isinstance(stmt, SelectStatement):
                # Validate
                validate(stmt, self.db.catalog, self.db_name)

                # Logical plan
                plan_start = time.perf_counter()
                logical_plan = build_logical_plan(stmt, self.db.catalog, self.db_name)

                # Optimize (unknown optimizer ids are an error, not a silent fallback)
                try:
                    opt = self.get_optimizer(optimizer_id)
                except ValueError as e:
                    raise QueryOptError(str(e)) from e

                opt_result = opt.optimize(
                    logical_plan,
                    catalog=self.db.catalog,
                    page_store=self.db._page_store,
                    db_name=self.db_name,
                )
                plan_time_ms = (time.perf_counter() - plan_start) * 1000.0

                # Execute (+ feedback collection)
                result = self.run_optimized(opt_result, plan_time_ms)
                result.optimizer_metadata = opt_result.metadata
                return result

            raise QueryOptError(f"Unsupported statement: {type(stmt).__name__}")

        except QueryOptError as e:
            total_time = (time.perf_counter() - start_time) * 1000.0
            return QueryResult(
                status=ExecutionStatus.ERROR,
                rows=[],
                schema=None,
                error_message=e.message,
                error_type=e.error_code,
                execution_time_ms=total_time,
            )
        except Exception as e:
            total_time = (time.perf_counter() - start_time) * 1000.0
            return QueryResult(
                status=ExecutionStatus.ERROR,
                rows=[],
                schema=None,
                error_message=str(e),
                error_type="INTERNAL_ERROR",
                execution_time_ms=total_time,
            )

    def explain(self, sql: str, optimizer_id: str = "rule_based") -> dict[str, Any]:
        """
        Explain a SELECT query without executing it.
        """
        parse_result = parse(sql)
        stmt = parse_result.statement
        if not isinstance(stmt, SelectStatement):
            raise QueryOptError("EXPLAIN is only supported for SELECT statements.")

        validate(stmt, self.db.catalog, self.db_name)
        logical_plan = build_logical_plan(stmt, self.db.catalog, self.db_name)

        try:
            opt = self.get_optimizer(optimizer_id)
        except ValueError as e:
            raise QueryOptError(str(e)) from e
        opt_result = opt.optimize(
            logical_plan,
            catalog=self.db.catalog,
            page_store=self.db._page_store,
            db_name=self.db_name,
        )

        return {
            "initial_plan": explain_logical_plan(opt_result.initial_logical_plan),
            "optimized_plan": explain_logical_plan(opt_result.optimized_logical_plan),
            "rules_applied": opt_result.rules_applied,
            "optimization_time_ms": opt_result.optimization_time_ms,
            "estimated_cost": opt_result.estimated_cost,
            "alternatives_considered": opt_result.alternatives_considered,
            "metadata": opt_result.metadata,
        }
