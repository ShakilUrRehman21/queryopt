"""
common.config — Runtime configuration for QueryOpt Lab.

All configuration is read from environment variables.
Defaults are suitable for local development.
Production values must be set explicitly via environment or .env file.

See .env.example for all available variables.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum


class Environment(Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


def _bool_env(key: str, default: bool) -> bool:
    """Parse a boolean from an environment variable."""
    val = os.environ.get(key, "").lower()
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off"):
        return False
    return default


def _int_env(key: str, default: int) -> int:
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _float_env(key: str, default: float) -> float:
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


@dataclass(frozen=True)
class ResourceLimits:
    """Configurable resource limits enforced by the executor."""
    max_query_time_seconds: float
    max_rows_returned: int
    max_rows_scanned: int
    max_memory_mb: int
    max_plan_candidates: int
    max_join_count: int


@dataclass(frozen=True)
class QueryOptConfig:
    """
    Complete runtime configuration for the QueryOpt Lab engine.

    Build via QueryOptConfig.from_env() to read from environment variables.
    For tests, use QueryOptConfig.for_testing() for fast, minimal defaults.
    """
    environment: Environment
    debug: bool
    version: str

    # Storage
    data_dir: str
    results_dir: str
    max_data_dir_size_mb: int

    # Execution limits (normal mode)
    limits: ResourceLimits

    # Public Demo Mode (restricted limits)
    public_demo_mode: bool
    demo_limits: ResourceLimits

    # Logging
    log_level: str
    log_format: str  # "json" | "text"

    @classmethod
    def from_env(cls) -> "QueryOptConfig":
        """Build configuration from environment variables."""
        env_str = os.environ.get("QUERYOPT_ENV", "development").lower()
        environment = Environment(env_str) if env_str in Environment._value2member_map_ else Environment.DEVELOPMENT

        return cls(
            environment=environment,
            debug=_bool_env("QUERYOPT_DEBUG", environment == Environment.DEVELOPMENT),
            version=os.environ.get("QUERYOPT_VERSION", "0.2.0"),

            data_dir=os.environ.get("DATA_DIR", "/tmp/data" if (os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME")) else "./data"),
            results_dir=os.environ.get("RESULTS_DIR", "/tmp/results" if (os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME")) else "./experiments/results"),
            max_data_dir_size_mb=_int_env("MAX_DATA_DIR_SIZE_MB", 500),

            limits=ResourceLimits(
                max_query_time_seconds=_float_env("MAX_QUERY_TIME_SECONDS", 30.0),
                max_rows_returned=_int_env("MAX_ROWS_RETURNED", 100_000),
                max_rows_scanned=_int_env("MAX_ROWS_SCANNED", 10_000_000),
                max_memory_mb=_int_env("MAX_MEMORY_MB", 512),
                max_plan_candidates=_int_env("MAX_PLAN_CANDIDATES", 50),
                max_join_count=_int_env("MAX_JOIN_COUNT", 6),
            ),

            public_demo_mode=_bool_env("PUBLIC_DEMO_MODE", False),
            demo_limits=ResourceLimits(
                max_query_time_seconds=_float_env("DEMO_MAX_QUERY_TIME_SECONDS", 10.0),
                max_rows_returned=_int_env("DEMO_MAX_ROWS", 10_000),
                max_rows_scanned=_int_env("DEMO_MAX_ROWS_SCANNED", 1_000_000),
                max_memory_mb=_int_env("DEMO_MAX_MEMORY_MB", 256),
                max_plan_candidates=20,
                max_join_count=4,
            ),

            log_level=os.environ.get("LOG_LEVEL", "INFO"),
            log_format=os.environ.get("LOG_FORMAT", "text"),
        )

    @classmethod
    def for_testing(cls) -> "QueryOptConfig":
        """Return a config suitable for unit and integration tests."""
        return cls(
            environment=Environment.TESTING,
            debug=True,
            version="0.1.0-test",
            data_dir="./test_data",
            results_dir="./test_results",
            max_data_dir_size_mb=100,
            limits=ResourceLimits(
                max_query_time_seconds=60.0,
                max_rows_returned=1_000_000,
                max_rows_scanned=100_000_000,
                max_memory_mb=2048,
                max_plan_candidates=100,
                max_join_count=8,
            ),
            public_demo_mode=False,
            demo_limits=ResourceLimits(
                max_query_time_seconds=5.0,
                max_rows_returned=1_000,
                max_rows_scanned=100_000,
                max_memory_mb=64,
                max_plan_candidates=10,
                max_join_count=3,
            ),
            log_level="DEBUG",
            log_format="text",
        )

    @property
    def active_limits(self) -> ResourceLimits:
        """Return demo_limits in public demo mode, otherwise normal limits."""
        return self.demo_limits if self.public_demo_mode else self.limits

    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION


# Module-level default config (can be overridden in tests)
_default_config: QueryOptConfig | None = None


def get_config() -> QueryOptConfig:
    """Return the module-level config, building from environment if not set."""
    global _default_config
    if _default_config is None:
        _default_config = QueryOptConfig.from_env()
    return _default_config


def set_config(config: QueryOptConfig) -> None:
    """Override the module-level config. Used in tests."""
    global _default_config
    _default_config = config
