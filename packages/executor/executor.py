"""
executor.executor — Volcano query executor entry point.

Drives execution of a PhysicalOperator tree using the Volcano iterator model,
tracks metrics, collects timing, and enforces configured resource limits.
"""
from __future__ import annotations

import time

from common.config import ResourceLimits, get_config
from common.errors import QueryOptError, ResourceLimitError
from common.logging import get_logger
from common.types import ExecutionStatus, QueryResult, Row
from executor.base import PhysicalOperator

logger = get_logger(__name__)


class VolcanoExecutor:
    """
    Volcano iterator engine driving a physical operator pipeline to completion.
    """

    def __init__(self, limits: ResourceLimits | None = None) -> None:
        self.limits = limits or get_config().limits

    def execute(self, root_operator: PhysicalOperator) -> QueryResult:
        """
        Execute the physical plan tree and return a QueryResult.
        """
        rows: list[Row] = []
        start_time = time.perf_counter()
        timeout_sec = self.limits.max_query_time_seconds
        max_rows = self.limits.max_rows_returned

        try:
            root_operator.open()
            while True:
                # Check timeout
                elapsed_sec = time.perf_counter() - start_time
                if elapsed_sec > timeout_sec:
                    raise ResourceLimitError(
                        message=f"Query exceeded maximum execution time of {timeout_sec}s.",
                        limit_type="time",
                        limit_value=timeout_sec,
                        actual_value=elapsed_sec,
                    )

                row = root_operator.next()
                if row is None:
                    break

                rows.append(row)
                if len(rows) > max_rows:
                    raise ResourceLimitError(
                        message=f"Query exceeded maximum result rows limit of {max_rows}.",
                        limit_type="rows",
                        limit_value=max_rows,
                        actual_value=len(rows),
                    )

            root_operator.close()
            total_execution_ms = (time.perf_counter() - start_time) * 1000.0

            # Collect stats
            all_stats = root_operator.collect_all_stats()
            # Calculate total scanned rows from scan operators
            rows_scanned = sum(
                s.actual_rows for s in all_stats.values() if "Scan" in s.operator_type
            )

            return QueryResult(
                status=ExecutionStatus.SUCCESS,
                rows=rows,
                schema=root_operator.schema,
                execution_time_ms=total_execution_ms,
                rows_returned=len(rows),
                rows_scanned=rows_scanned,
                operator_stats=all_stats,
            )

        except ResourceLimitError as e:
            try:
                root_operator.close()
            except Exception:
                pass
            total_execution_ms = (time.perf_counter() - start_time) * 1000.0
            return QueryResult(
                status=ExecutionStatus.RESOURCE_LIMIT,
                rows=[],
                schema=root_operator.schema,
                error_message=e.message,
                error_type=e.error_code,
                execution_time_ms=total_execution_ms,
            )

        except QueryOptError as e:
            try:
                root_operator.close()
            except Exception:
                pass
            total_execution_ms = (time.perf_counter() - start_time) * 1000.0
            return QueryResult(
                status=ExecutionStatus.ERROR,
                rows=[],
                schema=root_operator.schema,
                error_message=e.message,
                error_type=e.error_code,
                execution_time_ms=total_execution_ms,
            )

        except Exception as e:
            try:
                root_operator.close()
            except Exception:
                pass
            total_execution_ms = (time.perf_counter() - start_time) * 1000.0
            logger.exception(f"Unhandled error during query execution: {e}")
            return QueryResult(
                status=ExecutionStatus.ERROR,
                rows=[],
                schema=root_operator.schema,
                error_message=str(e),
                error_type="INTERNAL_ERROR",
                execution_time_ms=total_execution_ms,
            )
