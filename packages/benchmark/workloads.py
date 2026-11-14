"""
benchmark.workloads — Standard benchmark workloads (W1–W7) for query optimization experiments.

Defines reproducible schemas and query sets designed to isolate specific
optimizer behaviors:
- W1: Point queries, range scans, and constant folding
- W2: Multi-predicate selectivity and predicate pushdown
- W3: Multi-table join reordering (asymmetric relation sizes)
- W4: Aggregations and GROUP BY cardinality reduction
- W5: Skewed distributions (Zipfian) for histogram vs uniform evaluation
- W6: Top-K ordering and LIMIT pushdown
- W7: Comprehensive mixed analytical queries
- W8: Correlated predicates (breaks attribute independence; learned estimator testbed)
- W9: Repeated join workload for between-query adaptive feedback (RQ4)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from benchmark.generator import ColumnSpec, DataDistribution, TableSpec
from common.types import DataType


@dataclass(frozen=True)
class BenchmarkQuery:
    """A single benchmark query instance."""
    query_id: str
    sql: str
    description: str
    tags: list[str] = field(default_factory=list)


@dataclass
class BenchmarkWorkload:
    """A packaged workload containing table specifications and queries."""
    workload_id: str
    name: str
    description: str
    table_specs: list[TableSpec]
    queries: list[BenchmarkQuery]


def get_standard_tables(scale: int = 1) -> list[TableSpec]:
    """
    Standard benchmark schema: customers, orders, lineitem, products.
    Scales row counts by scale multiplier.
    """
    return [
        TableSpec(
            name="customers",
            row_count=50 * scale,
            columns=[
                ColumnSpec("id", DataType.INTEGER, DataDistribution.SEQUENTIAL, min_val=1),
                ColumnSpec("name", DataType.TEXT, DataDistribution.UNIFORM, min_val=1, max_val=50, text_prefix="cust_"),
                ColumnSpec("region", DataType.TEXT, DataDistribution.UNIFORM, min_val=1, max_val=5, text_prefix="region_"),
                ColumnSpec("credit_rating", DataType.FLOAT, DataDistribution.NORMAL, min_val=300, max_val=850, mean=650, std=50),
            ],
        ),
        TableSpec(
            name="orders",
            row_count=200 * scale,
            columns=[
                ColumnSpec("id", DataType.INTEGER, DataDistribution.SEQUENTIAL, min_val=1),
                ColumnSpec("customer_id", DataType.INTEGER, DataDistribution.UNIFORM, min_val=1, max_val=50 * scale),
                ColumnSpec("amount", DataType.FLOAT, DataDistribution.NORMAL, min_val=10.0, max_val=1000.0, mean=250.0, std=150.0),
                ColumnSpec("status", DataType.TEXT, DataDistribution.UNIFORM, min_val=1, max_val=3, text_prefix="status_"),
            ],
        ),
        TableSpec(
            name="lineitem",
            row_count=500 * scale,
            columns=[
                ColumnSpec("id", DataType.INTEGER, DataDistribution.SEQUENTIAL, min_val=1),
                ColumnSpec("order_id", DataType.INTEGER, DataDistribution.UNIFORM, min_val=1, max_val=200 * scale),
                ColumnSpec("product_id", DataType.INTEGER, DataDistribution.UNIFORM, min_val=1, max_val=20 * scale),
                ColumnSpec("quantity", DataType.INTEGER, DataDistribution.UNIFORM, min_val=1, max_val=10),
                ColumnSpec("discount", DataType.FLOAT, DataDistribution.UNIFORM, min_val=0.0, max_val=0.3),
            ],
        ),
        TableSpec(
            name="products",
            row_count=20 * scale,
            columns=[
                ColumnSpec("id", DataType.INTEGER, DataDistribution.SEQUENTIAL, min_val=1),
                ColumnSpec("name", DataType.TEXT, DataDistribution.UNIFORM, min_val=1, max_val=20, text_prefix="prod_"),
                ColumnSpec("category", DataType.TEXT, DataDistribution.UNIFORM, min_val=1, max_val=4, text_prefix="cat_"),
                ColumnSpec("price", DataType.FLOAT, DataDistribution.UNIFORM, min_val=5.0, max_val=200.0),
            ],
        ),
    ]


def get_skewed_tables(scale: int = 1) -> list[TableSpec]:
    """Tables with high Zipfian skew for histogram vs uniform evaluation."""
    return [
        TableSpec(
            name="skewed_orders",
            row_count=1000 * scale,
            columns=[
                ColumnSpec("id", DataType.INTEGER, DataDistribution.SEQUENTIAL, min_val=1),
                # Zipfian customer_id: a few hot customers have huge order volumes
                ColumnSpec("customer_id", DataType.INTEGER, DataDistribution.ZIPFIAN, min_val=1, max_val=50, zipf_param=1.8),
                ColumnSpec("amount", DataType.FLOAT, DataDistribution.ZIPFIAN, min_val=1.0, max_val=1000.0, zipf_param=1.5),
            ],
        ),
    ]


def get_correlated_tables(scale: int = 1) -> list[TableSpec]:
    """
    IoT-style schema whose filter columns are strongly correlated.

    `humidity` follows `temperature` (noise 5% of range) and `pressure` moves
    inversely to it. Any estimator that multiplies per-column selectivities
    (System R independence) is wrong by an order of magnitude on conjunctions
    over these columns, which is exactly what W8/W9 and RQ3-RQ5 probe.
    """
    return [
        TableSpec(
            name="sensors",
            row_count=2000 * scale,
            columns=[
                ColumnSpec("id", DataType.INTEGER, DataDistribution.SEQUENTIAL, min_val=1),
                ColumnSpec("zone_id", DataType.INTEGER, DataDistribution.UNIFORM, min_val=1, max_val=20),
                ColumnSpec("temperature", DataType.FLOAT, DataDistribution.UNIFORM, min_val=0.0, max_val=100.0),
                ColumnSpec(
                    "humidity", DataType.FLOAT, DataDistribution.CORRELATED,
                    min_val=0.0, max_val=100.0, correlated_with="temperature", correlation_noise=0.05,
                ),
                ColumnSpec(
                    "pressure", DataType.FLOAT, DataDistribution.CORRELATED,
                    min_val=900.0, max_val=1100.0, correlated_with="temperature",
                    correlation_sign=-1.0, correlation_noise=0.08,
                ),
            ],
        ),
        TableSpec(
            name="zones",
            row_count=20,
            columns=[
                ColumnSpec("id", DataType.INTEGER, DataDistribution.SEQUENTIAL, min_val=1),
                ColumnSpec("name", DataType.TEXT, DataDistribution.SEQUENTIAL, min_val=1, text_prefix="zone_"),
                ColumnSpec("floor", DataType.INTEGER, DataDistribution.UNIFORM, min_val=1, max_val=5),
            ],
        ),
        TableSpec(
            name="alerts",
            row_count=600 * scale,
            columns=[
                ColumnSpec("id", DataType.INTEGER, DataDistribution.SEQUENTIAL, min_val=1),
                ColumnSpec("sensor_id", DataType.INTEGER, DataDistribution.UNIFORM, min_val=1, max_val=2000 * scale),
                ColumnSpec("severity", DataType.INTEGER, DataDistribution.UNIFORM, min_val=1, max_val=5),
            ],
        ),
    ]


def create_workload(workload_id: str, scale: int = 1) -> BenchmarkWorkload:
    """Factory for standard benchmark workloads."""
    match workload_id.upper():
        case "W1":
            return BenchmarkWorkload(
                workload_id="W1",
                name="Point and Range Scans",
                description="Tests scan cost, constant folding, and single-predicate filtering.",
                table_specs=get_standard_tables(scale),
                queries=[
                    BenchmarkQuery(
                        "W1_Q1",
                        "SELECT id, amount FROM orders WHERE id = 10",
                        "Single-row point lookup",
                        ["point", "scan"],
                    ),
                    BenchmarkQuery(
                        "W1_Q2",
                        "SELECT id, amount FROM orders WHERE amount > 500",
                        "Range scan with 50% selectivity",
                        ["range", "filter"],
                    ),
                    BenchmarkQuery(
                        "W1_Q3",
                        "SELECT id, amount FROM orders WHERE id = 10 + 20",
                        "Predicate with constant expression (tests constant folding)",
                        ["constant_folding"],
                    ),
                ],
            )

        case "W2":
            return BenchmarkWorkload(
                workload_id="W2",
                name="Multi-Predicate Filtering",
                description="Tests conjunction, disjunction, and predicate pushdown through joins.",
                table_specs=get_standard_tables(scale),
                queries=[
                    BenchmarkQuery(
                        "W2_Q1",
                        "SELECT id, amount FROM orders WHERE amount > 200 AND status = 'status_1'",
                        "Conjunctive filter testing attribute independence",
                        ["conjunction", "filter"],
                    ),
                    BenchmarkQuery(
                        "W2_Q2",
                        "SELECT orders.id, customers.name FROM orders JOIN customers ON orders.customer_id = customers.id WHERE orders.amount > 300",
                        "Predicate pushdown through inner join",
                        ["pushdown", "join"],
                    ),
                ],
            )

        case "W3":
            return BenchmarkWorkload(
                workload_id="W3",
                name="Multi-Table Join Ordering",
                description="Tests CBO dynamic-programming join reordering across 3 tables with asymmetric sizes.",
                table_specs=get_standard_tables(scale),
                queries=[
                    BenchmarkQuery(
                        "W3_Q1",
                        "SELECT customers.name, orders.amount FROM customers JOIN orders ON customers.id = orders.customer_id",
                        "Two-table equi-join",
                        ["join", "2-way"],
                    ),
                    BenchmarkQuery(
                        "W3_Q2",
                        "SELECT customers.name, lineitem.quantity FROM lineitem JOIN orders ON lineitem.order_id = orders.id JOIN customers ON orders.customer_id = customers.id WHERE customers.credit_rating > 700",
                        "Three-table join reordering: selective customer filter should pull customers first",
                        ["join", "3-way", "cbo"],
                    ),
                ],
            )

        case "W4":
            return BenchmarkWorkload(
                workload_id="W4",
                name="Aggregation and Grouping",
                description="Tests HashAggregate with varying reduction ratios.",
                table_specs=get_standard_tables(scale),
                queries=[
                    BenchmarkQuery(
                        "W4_Q1",
                        "SELECT COUNT(id), SUM(amount) FROM orders",
                        "Scalar aggregation (reduces to 1 row)",
                        ["aggregate", "scalar"],
                    ),
                    BenchmarkQuery(
                        "W4_Q2",
                        "SELECT status, COUNT(id), AVG(amount) FROM orders GROUP BY status",
                        "Low-cardinality GROUP BY",
                        ["aggregate", "group_by"],
                    ),
                ],
            )

        case "W5":
            return BenchmarkWorkload(
                workload_id="W5",
                name="Skewed Data and Histograms",
                description="Tests estimation accuracy on heavy-tailed Zipfian distributions.",
                table_specs=get_skewed_tables(scale),
                queries=[
                    BenchmarkQuery(
                        "W5_Q1",
                        "SELECT id, amount FROM skewed_orders WHERE customer_id = 1",
                        "Point query on the most frequent Zipfian key",
                        ["skew", "zipf", "point"],
                    ),
                    BenchmarkQuery(
                        "W5_Q2",
                        "SELECT id, amount FROM skewed_orders WHERE amount < 50",
                        "Range query on skewed amount distribution",
                        ["skew", "zipf", "range"],
                    ),
                ],
            )

        case "W6":
            return BenchmarkWorkload(
                workload_id="W6",
                name="Ordering and Limit Pushdown",
                description="Tests Sort and Limit operator interaction.",
                table_specs=get_standard_tables(scale),
                queries=[
                    BenchmarkQuery(
                        "W6_Q1",
                        "SELECT id, amount FROM orders ORDER BY amount DESC LIMIT 10",
                        "Top-10 highest value orders",
                        ["sort", "limit", "top_k"],
                    ),
                    BenchmarkQuery(
                        "W6_Q2",
                        "SELECT id, amount FROM orders ORDER BY id ASC LIMIT 5 OFFSET 10",
                        "Pagination with LIMIT and OFFSET",
                        ["sort", "limit", "pagination"],
                    ),
                ],
            )

        case "W7":
            return BenchmarkWorkload(
                workload_id="W7",
                name="Mixed Analytical Pipeline",
                description="End-to-end analytical query combining join, filter, group by, and sort.",
                table_specs=get_standard_tables(scale),
                queries=[
                    BenchmarkQuery(
                        "W7_Q1",
                        "SELECT customers.region, COUNT(orders.id), SUM(orders.amount) "
                        "FROM customers JOIN orders ON customers.id = orders.customer_id "
                        "WHERE orders.amount > 100 "
                        "GROUP BY customers.region "
                        "ORDER BY customers.region ASC",
                        "Full analytical report",
                        ["analytical", "mixed", "e2e"],
                    ),
                ],
            )

        case "W8":
            return BenchmarkWorkload(
                workload_id="W8",
                name="Correlated Predicates",
                description=(
                    "Conjunctive filters over correlated columns. Independence-based "
                    "estimators under-estimate positively correlated conjunctions and "
                    "over-estimate contradictory ones."
                ),
                table_specs=get_correlated_tables(scale),
                queries=[
                    BenchmarkQuery(
                        "W8_Q1",
                        "SELECT id, temperature, humidity FROM sensors WHERE temperature > 85 AND humidity > 85",
                        "Positively correlated conjunction (independence under-estimates)",
                        ["correlation", "filter", "underestimate"],
                    ),
                    BenchmarkQuery(
                        "W8_Q2",
                        "SELECT id, temperature, humidity FROM sensors WHERE temperature > 85 AND humidity < 15",
                        "Contradictory conjunction (independence over-estimates; true answer ~0 rows)",
                        ["correlation", "filter", "overestimate"],
                    ),
                    BenchmarkQuery(
                        "W8_Q3",
                        "SELECT id, pressure FROM sensors WHERE temperature < 20 AND pressure > 1050",
                        "Inverse correlation across two columns",
                        ["correlation", "filter", "inverse"],
                    ),
                ],
            )

        case "W9":
            return BenchmarkWorkload(
                workload_id="W9",
                name="Feedback-Driven Join Ordering",
                description=(
                    "Joins whose driving filter is mis-estimated because of correlation. "
                    "Run repeatedly so the adaptive optimizer can learn from observed cardinalities."
                ),
                table_specs=get_correlated_tables(scale),
                queries=[
                    BenchmarkQuery(
                        "W9_Q1",
                        "SELECT zones.name, alerts.severity FROM alerts "
                        "JOIN sensors ON alerts.sensor_id = sensors.id "
                        "JOIN zones ON sensors.zone_id = zones.id "
                        "WHERE sensors.temperature > 80 AND sensors.humidity > 80",
                        "3-way join driven by a correlated (under-estimated) filter",
                        ["join", "3-way", "correlation", "adaptive"],
                    ),
                    BenchmarkQuery(
                        "W9_Q2",
                        "SELECT zones.name, sensors.temperature FROM sensors "
                        "JOIN zones ON sensors.zone_id = zones.id "
                        "JOIN alerts ON alerts.sensor_id = sensors.id "
                        "WHERE sensors.temperature > 90 AND sensors.humidity < 10 AND alerts.severity >= 4",
                        "3-way join driven by a contradictory (over-estimated) filter",
                        ["join", "3-way", "correlation", "adaptive"],
                    ),
                ],
            )

        case _:
            raise ValueError(
                f"Unknown workload ID: '{workload_id}'. Supported: {', '.join(WORKLOAD_IDS)}"
            )


WORKLOAD_IDS = ["W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "W9"]
