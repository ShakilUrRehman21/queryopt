"""
queryopt.cli — Command-line interface for QueryOpt Lab.

Commands:
  queryopt query run     --sql "..." [--optimizer ...]
  queryopt query explain --sql "..." [--optimizer ...]
  queryopt db create     [--db-name ...]
  queryopt db status     [--db-name ...]
  queryopt ai status
  queryopt ai explain    --sql "..." [--optimizer ...] [--question "..."]
  queryopt ai arena      --sql "..."
  queryopt ai train      [--table ...]
  queryopt ai feedback   [--reset]
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from common.config import get_config
from common.types import ExecutionStatus
from queryopt.engine import QueryEngine

console = Console()


@click.group()
def main() -> None:
    """QueryOpt Lab — Experimental Platform for Relational Query Optimization."""
    pass


@main.group()
def query() -> None:
    """Run and analyze SQL queries."""
    pass


@query.command(name="run")
@click.option("--sql", "-s", required=True, help="SQL statement to execute.")
@click.option(
    "--optimizer", "-o", default="rule_based",
    help="Optimizer id/spec: naive, rule_based, cost_based, adaptive, learned, llm_guided, "
         "or e.g. 'cost_based[qerr=10,mode=under]'.",
)
@click.option("--db-name", "-d", default="default", help="Target database name.")
@click.option("--data-dir", type=click.Path(), default=None, help="Custom data directory.")
def run_query(sql: str, optimizer: str, db_name: str, data_dir: str | None) -> None:
    """Execute a SQL statement."""
    engine = QueryEngine(data_dir=data_dir, db_name=db_name)
    result = engine.execute(sql, optimizer_id=optimizer)

    if result.status == ExecutionStatus.SUCCESS:
        if result.schema is not None:
            # Render tabular results
            table = Table(title=f"Query Result ({len(result.rows)} rows in {result.execution_time_ms:.2f}ms)")
            for col in result.schema.columns:
                table.add_column(col.name, style="cyan")
            for row in result.rows:
                table.add_row(*(str(val) if val is not None else "[dim]NULL[/dim]" for val in row))
            console.print(table)
        else:
            console.print(f"[green][OK] Statement executed successfully in {result.execution_time_ms:.2f}ms.[/green]")
    else:
        console.print(f"[red][ERROR] Query failed [{result.error_type}]: {result.error_message}[/red]")
        sys.exit(1)


@query.command(name="explain")
@click.option("--sql", "-s", required=True, help="SQL SELECT query to explain.")
@click.option(
    "--optimizer", "-o", default="rule_based",
    help="Optimizer id/spec: naive, rule_based, cost_based, adaptive, learned, llm_guided, "
         "or e.g. 'cost_based[qerr=10,mode=under]'.",
)
@click.option("--db-name", "-d", default="default", help="Target database name.")
@click.option("--data-dir", type=click.Path(), default=None, help="Custom data directory.")
def explain_query(sql: str, optimizer: str, db_name: str, data_dir: str | None) -> None:
    """Display the initial and optimized logical plan trees."""
    engine = QueryEngine(data_dir=data_dir, db_name=db_name)
    try:
        explanation = engine.explain(sql, optimizer_id=optimizer)
        console.print(Panel(explanation["initial_plan"], title="[bold yellow]Initial Logical Plan[/bold yellow]"))
        rules = ", ".join(explanation["rules_applied"]) or "None"
        cost_str = f" | Estimated Cost: {explanation['estimated_cost']:.2f}" if explanation.get("estimated_cost") else ""
        alts_str = f" | Alternatives: {explanation['alternatives_considered']}" if explanation.get("alternatives_considered") else ""
        console.print(f"[cyan]Rules Applied:[/cyan] {rules} (Time: {explanation['optimization_time_ms']:.2f}ms{cost_str}{alts_str})")
        console.print(Panel(explanation["optimized_plan"], title="[bold green]Optimized Logical Plan[/bold green]"))
    except Exception as e:
        console.print(f"[red]Explain failed: {e}[/red]")
        sys.exit(1)


@main.group()
def db() -> None:
    """Manage database storage and metadata."""
    pass


@db.command(name="create")
@click.option("--db-name", "-d", default="default", help="Database name to create.")
@click.option("--data-dir", type=click.Path(), default=None, help="Custom data directory.")
def create_database(db_name: str, data_dir: str | None) -> None:
    """Create a new database catalog."""
    from storage.table import Database
    d_dir = Path(data_dir) if data_dir else get_config().data_dir
    Database.create(d_dir / db_name)
    console.print(f"[green][OK] Created database '{db_name}' at {d_dir / db_name}[/green]")


@db.command(name="status")
@click.option("--db-name", "-d", default="default", help="Database name.")
@click.option("--data-dir", type=click.Path(), default=None, help="Custom data directory.")
def database_status(db_name: str, data_dir: str | None) -> None:
    """Show tables and row counts for a database."""
    engine = QueryEngine(data_dir=data_dir, db_name=db_name)
    catalog = engine.db.catalog
    table_names = catalog.list_tables()

    if not table_names:
        console.print(f"[yellow]Database '{db_name}' has no tables.[/yellow]")
        return

    table = Table(title=f"Database '{db_name}' Tables")
    table.add_column("Table Name", style="bold")
    table.add_column("Columns", style="cyan")
    table.add_column("Rows", style="green")
    table.add_column("Pages", style="magenta")

    for t_name in table_names:
        t_meta = catalog.get_table(t_name)
        col_list = ", ".join(f"{c.name}:{c.data_type.value}" for c in t_meta.schema.columns)
        table.add_row(t_name, col_list, str(t_meta.row_count), str(len(t_meta.page_ids)))

    console.print(table)


# ── Benchmark Commands ────────────────────────────────────────────────────────

@main.group()
def benchmark() -> None:
    """Generate and manage benchmark datasets and workloads."""
    pass


@benchmark.command(name="list")
def list_workloads() -> None:
    """List standard workloads (W1-W7)."""
    from benchmark.workloads import create_workload

    table = Table(title="Standard QueryOpt Lab Benchmark Workloads (W1-W7)")
    table.add_column("Workload", style="bold cyan")
    table.add_column("Name", style="yellow")
    table.add_column("Tables", style="green")
    table.add_column("Queries", style="magenta")
    table.add_column("Description")

    for wid in ["W1", "W2", "W3", "W4", "W5", "W6", "W7"]:
        wl = create_workload(wid)
        t_names = ", ".join(t.name for t in wl.table_specs)
        q_count = str(len(wl.queries))
        table.add_row(wl.workload_id, wl.name, t_names, q_count, wl.description)

    console.print(table)


@benchmark.command(name="generate")
@click.option("--workload", "-w", default="W1", help="Workload ID (W1..W9).")
@click.option("--scale", "-s", default=1, type=int, help="Scale multiplier for row counts.")
@click.option("--db-name", "-d", default="benchmark_db", help="Target database name.")
@click.option("--data-dir", type=click.Path(), default=None, help="Custom data directory.")
@click.option("--seed", default=42, type=int, help="Random seed for reproducibility.")
def generate_benchmark(workload: str, scale: int, db_name: str, data_dir: str | None, seed: int) -> None:
    """Generate and populate a benchmark workload database."""
    from benchmark.generator import DatasetGenerator
    from benchmark.workloads import create_workload

    wl = create_workload(workload, scale=scale)
    engine = QueryEngine(data_dir=data_dir, db_name=db_name)
    gen = DatasetGenerator(seed=seed)

    with console.status(f"[bold green]Generating workload {wl.workload_id} ({wl.name})...[/bold green]"):
        gen.populate_database(engine.db, wl.table_specs, collect_statistics=True)

    console.print(f"[bold green][OK] Successfully generated workload {wl.workload_id} in database '{db_name}'.[/bold green]")


# ── Experiment Commands ───────────────────────────────────────────────────────

@main.group()
def experiment() -> None:
    """Run and analyze reproducible query optimization experiments."""
    pass


@experiment.command(name="run")
@click.argument("config_file", type=click.Path(exists=True))
def run_experiment(config_file: str) -> None:
    """Execute an experiment defined in a YAML configuration file."""
    from experiment.config import ExperimentConfig
    from experiment.runner import ExperimentRunner

    cfg = ExperimentConfig.from_yaml(config_file)
    console.print(f"[bold cyan]Launching Experiment:[/bold cyan] {cfg.experiment.name}")
    if cfg.experiment.description:
        console.print(f"[dim]{cfg.experiment.description}[/dim]")

    runner = ExperimentRunner()
    result = runner.run(cfg)

    # Render summary table
    table = Table(title=f"Experiment Results: {result.name} ({result.total_duration_seconds:.2f}s)")
    table.add_column("Query", style="bold")
    table.add_column("Optimizer", style="cyan")
    table.add_column("Exec Mean (ms)", style="green")
    table.add_column("Plan Mean (ms)", style="yellow")
    table.add_column("Est. Cost", style="magenta")
    table.add_column("Est. Rows", style="blue")
    table.add_column("Act. Rows", style="bold white")
    table.add_column("Q-Error", style="red")

    for s in result.summaries:
        table.add_row(
            s.query_id,
            s.optimizer_id,
            f"{s.execution_time_summary.mean:.2f}",
            f"{s.planning_time_summary.mean:.2f}",
            f"{s.estimated_cost:.2f}",
            f"{s.estimated_rows:.1f}",
            str(s.actual_rows),
            f"{s.q_error:.2f}",
        )

    console.print(table)
    console.print(f"[green][OK] Artifacts and reproducibility metadata saved to: {result.output_dir}[/green]")


# ── AI Lab ──────────────────────────────────────────────────────────────────


@main.group()
def ai() -> None:
    """AI-assisted optimization: explanations, learned/adaptive optimizers, LLM guidance."""
    pass


@ai.command(name="status")
def ai_status() -> None:
    """Show which language model (if any) is configured."""
    from ai.llm import get_llm_client
    from optimizer.registry import list_optimizers

    c = get_llm_client()
    state = "[green]available[/green]" if c.available else "[yellow]offline (deterministic fallbacks)[/yellow]"
    console.print(f"LLM provider: [bold]{c.provider}[/bold]  model: [bold]{c.model}[/bold]  {state}")
    t = Table(title="Optimizers")
    for col in ("id", "family", "description"):
        t.add_column(col)
    for o in list_optimizers():
        t.add_row(o["id"], o["family"], o["description"])
    console.print(t)


@ai.command(name="explain")
@click.option("--sql", "-s", required=True)
@click.option("--optimizer", "-o", default="cost_based")
@click.option("--question", "-q", default=None, help="Free-form question for the LLM narrator.")
@click.option("--db-name", "-d", default="default")
def ai_explain(sql: str, optimizer: str, question: str | None, db_name: str) -> None:
    """Run a query and explain its plan in plain language."""
    from ai.llm import LLMCallError, LLMUnavailableError
    from ai.narrator import collect_facts, llm_narrate, narrate

    import time

    engine = QueryEngine(db_name=db_name)
    t0 = time.perf_counter()
    opt_result = engine.optimize_select(sql, optimizer)
    res = engine.run_optimized(opt_result, (time.perf_counter() - t0) * 1000.0)
    facts = collect_facts(sql, opt_result, res)
    n = narrate(facts)
    console.print(Panel(n["summary"], title="Summary", border_style="green"))
    console.print("\n".join(n["steps"]))
    for section in ("why", "misestimates", "suggestions"):
        if n[section]:
            console.print(f"\n[bold]{section.capitalize()}[/bold]")
            for line in n[section]:
                console.print(f"  • {line}")
    try:
        out = llm_narrate(facts, question)
        console.print(Panel(out["text"], title=f"LLM ({out['model']})", border_style="cyan"))
    except (LLMUnavailableError, LLMCallError) as e:
        console.print(f"[dim]LLM narration skipped: {e}[/dim]")


@ai.command(name="arena")
@click.option("--sql", "-s", required=True)
@click.option("--db-name", "-d", default="default")
def ai_arena(sql: str, db_name: str) -> None:
    """Race every optimizer family on one query."""
    from ai.narrator import collect_facts

    engine = QueryEngine(db_name=db_name)
    t = Table(title="Optimizer Arena")
    for col in ("optimizer", "exec ms", "plan ms", "rows", "worst q-error", "join order"):
        t.add_column(col)
    import time

    for spec in ("naive", "rule_based", "cost_based", "adaptive", "learned", "llm_guided"):
        t0 = time.perf_counter()
        r = engine.optimize_select(sql, spec)
        res = engine.run_optimized(r, (time.perf_counter() - t0) * 1000.0)
        f = collect_facts(sql, r, res)
        t.add_row(
            spec, f"{res.execution_time_ms:.2f}", f"{res.planning_time_ms:.2f}", str(res.rows_returned),
            f"{f.worst.q_error:.2f}" if f.worst else "—", " → ".join(f.join_order) or "—",
        )
    console.print(t)


@ai.command(name="train")
@click.option("--table", "-t", multiple=True, help="Table(s) to train; default all.")
@click.option("--db-name", "-d", default="default")
def ai_train(table: tuple[str, ...], db_name: str) -> None:
    """(Re)train learned cardinality models and compare with histograms."""
    from optimizer.optimizer_learned import ensure_model

    engine = QueryEngine(db_name=db_name)
    t = Table(title="Learned estimator — held-out q-error (lower is better)")
    for col in ("table", "learned median", "learned p95", "histogram median", "histogram p95", "seconds"):
        t.add_column(col)
    for name in table or engine.db.catalog.list_tables():
        m = ensure_model(engine.db.catalog, engine.db._page_store, name, force=True)
        if m is None:
            continue
        x = m.metrics
        t.add_row(name, str(x["learned_median_qerror"]), str(x["learned_p95_qerror"]),
                  str(x.get("histogram_median_qerror", "—")), str(x.get("histogram_p95_qerror", "—")),
                  str(x["training_seconds"]))
    console.print(t)


@ai.command(name="feedback")
@click.option("--reset", is_flag=True, help="Clear all recorded cardinalities.")
@click.option("--db-name", "-d", default="default")
def ai_feedback(reset: bool, db_name: str) -> None:
    """Inspect or reset the adaptive optimizer's feedback store."""
    engine = QueryEngine(db_name=db_name, record_feedback=False)
    if reset:
        engine.feedback_store.reset()
        console.print("[green]Feedback cleared.[/green]")
        return
    s = engine.feedback_store.summary(limit=15)
    t = Table(title=f"Cardinality feedback ({s['entries']} sub-expressions)")
    for col in ("tables", "predicates", "estimated", "actual", "q-error", "seen"):
        t.add_column(col)
    for e in s["worst_misestimates"]:
        t.add_row(",".join(e["tables"]), " AND ".join(e["predicates"])[:70], str(e["last_estimate"]),
                  str(e["actual_rows"]), str(e["q_error"]), str(e["observations"]))
    console.print(t)


if __name__ == "__main__":
    main()
