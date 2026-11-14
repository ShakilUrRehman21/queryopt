"""
experiment — Experiment orchestration, metrics collection, and provenance for QueryOpt Lab.
"""
from experiment.config import (
    DatasetConfig,
    ExecutionConfig,
    ExperimentConfig,
    ExperimentConfigError,
    ExperimentMetadata,
    QueryConfig,
    ResultStorageConfig,
)
from experiment.metrics import (
    MetricSummary,
    QueryRunMetric,
    QuerySummary,
    calculate_q_error,
)
from experiment.runner import (
    ExperimentResult,
    ExperimentRunner,
)
from experiment.storage import (
    ResultWriter,
    build_reproducibility_record,
)

__all__ = [
    "DatasetConfig",
    "ExecutionConfig",
    "ExperimentConfig",
    "ExperimentConfigError",
    "ExperimentMetadata",
    "ExperimentResult",
    "ExperimentRunner",
    "MetricSummary",
    "QueryConfig",
    "QueryRunMetric",
    "QuerySummary",
    "ResultStorageConfig",
    "ResultWriter",
    "build_reproducibility_record",
    "calculate_q_error",
]
